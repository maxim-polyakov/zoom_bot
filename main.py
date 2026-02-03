#!/usr/bin/env python3
"""
Главный скрипт запуска Zoom агента
"""

import asyncio
import signal
import sys
import time

from config import settings
from utils.logger import setup_logger

try:
    from zoom_integration.zoom_client import ZoomClient
    from dashboard.server import DashboardServer
    from transcription.webhook_server import WebhookServer
    from state_manager.meeting_state import MeetingState
    from llm_processing.analyzer import TranscriptAnalyzer
    from news_fetcher.news_agent import NewsAgent
    from transcription.processor import TranscriptProcessor, get_transcript_processor

    COMPONENTS_LOADED = True
except ImportError as e:
    print(f"Import error: {e}")
    COMPONENTS_LOADED = False

logger = setup_logger(__name__)


class ZoomAgentBot:
    def __init__(self):
        self.running = False
        self.state = None
        self.zoom_client = None
        self.dashboard_server = None
        self.webhook_server = None
        self.news_agent = None
        self.analyzer = None
        self.processor = None

        if not COMPONENTS_LOADED:
            logger.error("Failed to load required components")
            raise ImportError("Required components failed to load")

    async def initialize(self):
        """Инициализация всех компонентов"""
        logger.info("Initializing Zoom Agent Bot...")

        # Инициализация состояния встречи
        self.state = MeetingState()

        # Инициализация компонентов
        self.news_agent = NewsAgent()
        self.analyzer = TranscriptAnalyzer()
        self.processor = get_transcript_processor(self.state)

        # Запуск веб-сервера дашборда
        self.dashboard_server = DashboardServer(self.state)
        dashboard_task = asyncio.create_task(
            self.dashboard_server.start()
        )

        # Запуск сервера для приема транскрипта
        self.webhook_server = WebhookServer(self.state)
        webhook_task = asyncio.create_task(
            self.webhook_server.start()
        )

        # Инициализация Zoom клиента
        self.zoom_client = ZoomClient(self.state)

        # Запуск фоновой обработки транскрипта
        if self.processor:
            self.processor.start_processing()

        logger.info("All components initialized successfully")
        return dashboard_task, webhook_task

    async def start_meeting_flow(self):
        """Запуск основного потока работы в встрече"""
        logger.info("Starting meeting flow...")

        try:
            # 1. Подключение к Zoom
            await self.zoom_client.connect()

            # 2. Запуск демонстрации экрана
            await self.zoom_client.start_screen_share()

            # 3. Запуск мониторинга соединения в фоне
            monitoring_task = asyncio.create_task(
                self.zoom_client.monitor_connection()
            )

            logger.info("Meeting flow started successfully")
            return monitoring_task

        except Exception as e:
            logger.error(f"Failed to start meeting flow: {e}")
            # Если не удалось подключиться, используем мок-данные
            if settings.USE_MOCK_TRANSCRIPT:
                logger.info("Falling back to mock transcript mode")
                # Можно добавить генерацию мок-данных здесь
            raise

    async def run(self):
        """Основной цикл работы бота"""
        self.running = True

        # Настройка обработчиков сигналов
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Инициализация
            dashboard_task, webhook_task = await self.initialize()

            # Основной цикл
            logger.info("Bot is running. Press Ctrl+C to stop.")

            # Если не используем мок, запускаем процесс встречи
            if not settings.USE_MOCK_TRANSCRIPT and settings.ZOOM_MEETING_URL:
                try:
                    meeting_task = await self.start_meeting_flow()
                    if meeting_task:
                        await meeting_task
                except Exception as e:
                    logger.error(f"Failed to start meeting flow: {e}")
                    logger.info("Continuing without Zoom connection")
            else:
                logger.info("Using mock transcript mode or no Zoom URL configured")

            # Основной цикл ожидания
            while self.running:
                try:
                    # Проверяем состояние дашборда
                    if self.processor and hasattr(self.processor, 'force_process_sync'):
                        # Периодически запускаем обработку
                        result = self.processor.force_process_sync()
                        if result and result.get("status") == "success":
                            logger.debug("Dashboard updated via force process")

                    await asyncio.sleep(10)  # Проверка каждые 10 секунд

                except Exception as e:
                    logger.error(f"Error in main wait loop: {e}")
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Корректное завершение работы"""
        logger.info("Shutting down...")

        self.running = False

        # Останавливаем компоненты в правильном порядке
        try:
            if self.processor:
                self.processor.stop_processing()
                await self.processor.shutdown()

            if self.zoom_client:
                await self.zoom_client.leave_meeting()
                await self.zoom_client.disconnect()

            if self.dashboard_server:
                await self.dashboard_server.stop()

            if self.webhook_server:
                await self.webhook_server.stop()

            if self.news_agent:
                await self.news_agent.cleanup()

            logger.info("Shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def main():
    """Точка входа"""
    try:
        bot = ZoomAgentBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()