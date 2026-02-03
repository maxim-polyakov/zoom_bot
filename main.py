#!/usr/bin/env python3
"""
Главный скрипт запуска Zoom агента с использованием Zoom Desktop SDK API
"""

import asyncio
import signal
import sys
import time
import traceback
from typing import Optional

from config import settings
from utils.logger import setup_logger

try:
    # Используем наш ctypes клиент
    from zoom_integration.zoom_sdk_ctypes import ZoomSDKCtypesClient as ZoomDesktopSDKClient
    from zoom_integration.zoom_sdk_ctypes import ZoomMeetingInfo, ZoomSDKStatus
    from dashboard.server import DashboardServer
    from transcription.webhook_server import WebhookServer
    from state_manager.meeting_state import MeetingState, MeetingStatus
    from llm_processing.analyzer import TranscriptAnalyzer
    from news_fetcher.news_agent import NewsAgent
    from transcription.processor import TranscriptProcessor, get_transcript_processor

    COMPONENTS_LOADED = True
except ImportError as e:
    print(f"Import error: {e}")
    traceback.print_exc()
    COMPONENTS_LOADED = False

logger = setup_logger(__name__)


class ZoomAgentBot:
    def __init__(self):
        self.running = False
        self.state = None
        self.zoom_client: Optional[ZoomDesktopSDKClient] = None
        self.dashboard_server = None
        self.webhook_server = None
        self.news_agent = None
        self.analyzer = None
        self.processor = None
        self._shutdown_flag = False

        if not COMPONENTS_LOADED:
            logger.error("Failed to load required components")
            raise ImportError("Required components failed to load")

    def _extract_meeting_info_from_url(self, zoom_url: str) -> tuple[str, str]:
        """Извлечение ID встречи и пароля из Zoom URL"""
        import re

        # Паттерны для Zoom URL
        patterns = [
            r'https://[a-z]+\.zoom\.us/j/(\d+)\?pwd=([a-zA-Z0-9]+)',  # С паролем
            r'https://[a-z]+\.zoom\.us/j/(\d+)',  # Без пароля
            r'zoom\.us/j/(\d+)\?pwd=([a-zA-Z0-9]+)',
            r'zoom\.us/j/(\d+)',
            r'^\d{9,11}$',  # Просто ID встречи
        ]

        for pattern in patterns:
            match = re.search(pattern, zoom_url)
            if match:
                if len(match.groups()) == 2:
                    return match.group(1), match.group(2)
                elif len(match.groups()) == 1:
                    return match.group(1), ""

        # Если URL не распознан, попробуем извлечь числовой ID
        numbers = re.findall(r'\d{9,11}', zoom_url)
        if numbers:
            return numbers[0], ""

        raise ValueError(f"Не удалось извлечь ID встречи из URL: {zoom_url}")

    async def initialize(self):
        """Инициализация всех компонентов"""
        logger.info("Initializing Zoom Agent Bot with Desktop SDK...")

        # Инициализация состояния встречи
        self.state = MeetingState()

        # Обновляем статус
        self.state.update_status(MeetingStatus.DISCONNECTED)

        # Инициализация компонентов
        self.news_agent = NewsAgent()
        self.analyzer = TranscriptAnalyzer()
        self.processor = get_transcript_processor(self.state)

        # Инициализация Zoom Desktop SDK клиента
        self.zoom_client = ZoomDesktopSDKClient(self.state)

        # Инициализация SDK
        if not await self.zoom_client.initialize_sdk():
            logger.error("Failed to initialize Zoom SDK")
            if not settings.USE_MOCK_TRANSCRIPT:
                raise RuntimeError("Zoom SDK initialization failed")
            else:
                logger.info("Continuing in mock mode despite SDK failure")

        # Запуск фоновой обработки транскрипта
        if self.processor:
            self.processor.start_processing()

        logger.info("Core components initialized successfully")
        return True

    async def start_servers(self):
        """Запуск серверов (дашборда и вебхуков)"""
        tasks = []

        # Запуск веб-сервера дашборда
        if settings.DASHBOARD_PORT:
            try:
                self.dashboard_server = DashboardServer(self.state)
                dashboard_task = asyncio.create_task(
                    self.dashboard_server.start(),
                    name="dashboard_server"
                )
                tasks.append(dashboard_task)
                logger.info(f"Dashboard server starting on port {settings.DASHBOARD_PORT}")
            except Exception as e:
                logger.error(f"Failed to start dashboard server: {e}")

        # Запуск сервера для приема транскрипта
        if settings.TRANSCRIPT_WEBHOOK_PORT:
            try:
                self.webhook_server = WebhookServer(self.state)
                webhook_task = asyncio.create_task(
                    self.webhook_server.start(),
                    name="webhook_server"
                )
                tasks.append(webhook_task)
                logger.info(f"Webhook server starting on port {settings.TRANSCRIPT_WEBHOOK_PORT}")
            except Exception as e:
                logger.error(f"Failed to start webhook server: {e}")

        return tasks

    async def prepare_meeting_info(self) -> Optional[ZoomMeetingInfo]:
        """Подготовка информации о встрече из настроек"""
        try:
            # Используем URL если он указан, иначе используем отдельные поля
            if settings.ZOOM_MEETING_URL:
                # Извлекаем ID встречи и пароль из URL
                meeting_id, password = self._extract_meeting_info_from_url(
                    settings.ZOOM_MEETING_URL
                )
            elif settings.ZOOM_MEETING_ID:
                # Используем отдельные поля
                meeting_id = settings.ZOOM_MEETING_ID
                password = settings.ZOOM_PASSWORD or ""
            else:
                logger.error("No Zoom meeting configured")
                return None

            # Создаем объект с информацией о встрече
            meeting_info = ZoomMeetingInfo(
                meeting_id=meeting_id,
                password=password,
                display_name=settings.ZOOM_DISPLAY_NAME or "AI Assistant",
                no_audio=settings.ZOOM_JOIN_WITHOUT_AUDIO,
                no_video=settings.ZOOM_JOIN_WITHOUT_VIDEO
            )

            logger.info(f"Prepared meeting info: ID={meeting_id}, "
                        f"Display Name={meeting_info.display_name}")

            return meeting_info

        except Exception as e:
            logger.error(f"Failed to prepare meeting info: {e}")
            return None

    async def start_meeting_flow(self):
        """Запуск основного потока работы в встрече через Desktop SDK"""
        logger.info("Starting meeting flow with Desktop SDK...")

        try:
            # 1. Подготовка информации о встрече
            meeting_info = await self.prepare_meeting_info()
            if not meeting_info:
                if not settings.USE_MOCK_TRANSCRIPT:
                    logger.error("No valid meeting info available")
                    return False
                else:
                    logger.warning("No meeting info, continuing in mock mode")
                    return False

            # 2. Присоединение к встрече через SDK
            logger.info(f"Joining meeting: {meeting_info.meeting_id}")

            join_success = await self.zoom_client.join_meeting(meeting_info)
            if not join_success:
                if not settings.USE_MOCK_TRANSCRIPT:
                    logger.error("Failed to join meeting")
                    return False
                else:
                    logger.warning("Failed to join meeting, continuing in mock mode")
                    return False

            logger.info("Meeting flow started successfully via SDK")

            # Обновляем состояние
            self.state.update_status(MeetingStatus.IN_MEETING.value)  # Используем .value
            self.state.update({
                'meeting_id': meeting_info.meeting_id,
                'display_name': meeting_info.display_name,
                'joined_time': time.time(),
                'zoom_status': 'in_meeting',
                'meeting_start_time': time.time()
            })

            return True

        except Exception as e:
            logger.error(f"Failed to start meeting flow: {e}")
            traceback.print_exc()

            # Если не удалось подключиться, используем мок-данные
            if settings.USE_MOCK_TRANSCRIPT:
                logger.info("Falling back to mock transcript mode")
                self.state.update({
                    'mock_mode': True,
                    'error': str(e)
                })
                return True
            else:
                logger.error("Cannot proceed without Zoom connection")
                return False

    async def monitor_zoom_connection(self):
        """Мониторинг состояния подключения к Zoom"""
        logger.info("Starting Zoom connection monitoring...")

        check_interval = 30  # Проверка каждые 30 секунд

        while self.running and self.zoom_client and not self._shutdown_flag:
            try:
                current_status = self.zoom_client.status

                # Логируем изменения статуса
                if hasattr(self, '_last_zoom_status') and self._last_zoom_status != current_status:
                    logger.info(f"Zoom status changed: {self._last_zoom_status} -> {current_status}")

                self._last_zoom_status = current_status

                # Обновляем состояние в дашборде
                self.state.update({
                    'zoom_status': current_status.value,
                    'last_status_check': time.time()
                })

                # Проверяем критические состояния
                if current_status == ZoomSDKStatus.ERROR:
                    logger.error("Zoom SDK reported error state")
                    # Можно попробовать восстановить соединение
                    if settings.ZOOM_AUTO_RECONNECT:
                        logger.info("Attempting to reconnect...")
                        await asyncio.sleep(5)

                        # Пытаемся переподключиться
                        if hasattr(self.zoom_client, 'meeting_info') and self.zoom_client.meeting_info:
                            await self.zoom_client.leave_meeting()
                            await asyncio.sleep(2)
                            await self.zoom_client.join_meeting(self.zoom_client.meeting_info)

                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                logger.info("Connection monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")
                await asyncio.sleep(10)

    async def run_main_loop(self):
        """Основной цикл работы бота"""
        logger.info("Starting main loop...")

        while self.running and not self._shutdown_flag:
            try:
                # Обновляем дашборд с текущим статусом
                if self.zoom_client:
                    status_info = self.zoom_client.get_status_info()
                    self.state.update(status_info)

                # Периодически запускаем обработку транскрипта
                if self.processor and hasattr(self.processor, 'force_process_sync'):
                    result = self.processor.force_process_sync()
                    if result and result.get("status") == "success":
                        logger.debug("Dashboard updated via force process")

                await asyncio.sleep(10)  # Проверка каждые 10 секунд

            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main wait loop: {e}")
                await asyncio.sleep(5)

    async def run(self):
        """Основной метод запуска бота"""
        self.running = True

        # Настройка обработчиков сигналов
        def signal_handler(sig, frame):
            logger.info(f"Shutdown signal received: {sig}")
            self.running = False
            self._shutdown_flag = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # 1. Инициализация
            if not await self.initialize():
                logger.error("Failed to initialize bot")
                return

            # 2. Запуск серверов
            server_tasks = await self.start_servers()

            # Даем серверам время запуститься
            await asyncio.sleep(2)

            # 3. Проверяем работу серверов
            servers_running = len(server_tasks) > 0

            if not servers_running:
                logger.warning("No servers started, dashboard may not be available")

            # 4. Если не используем мок, запускаем процесс встречи
            meeting_started = False
            if not settings.USE_MOCK_TRANSCRIPT and (settings.ZOOM_MEETING_URL or settings.ZOOM_MEETING_ID):
                try:
                    meeting_started = await self.start_meeting_flow()

                    if meeting_started:
                        logger.info("Zoom meeting flow started successfully")

                        # Запускаем мониторинг соединения
                        monitoring_task = asyncio.create_task(
                            self.monitor_zoom_connection(),
                            name="zoom_monitoring"
                        )
                    else:
                        logger.warning("Meeting flow not started")

                except Exception as e:
                    logger.error(f"Failed to start meeting flow: {e}")
                    traceback.print_exc()

                    if not settings.USE_MOCK_TRANSCRIPT:
                        logger.error("Exiting due to Zoom connection failure")
                        raise
                    else:
                        logger.info("Continuing in mock mode despite error")
            else:
                logger.info("Using mock transcript mode or no Zoom meeting configured")
                # Обновляем статус для мок-режима
                self.state.update_status(MeetingStatus.CONNECTED)
                self.state.update({
                    'mock_mode': True,
                    'status': 'mock_mode'
                })

            # 5. Основной цикл работы
            logger.info("Bot is running. Press Ctrl+C to stop.")

            main_loop_task = asyncio.create_task(
                self.run_main_loop(),
                name="main_loop"
            )

            # 6. Ожидание завершения задач
            tasks_to_wait = [main_loop_task] + server_tasks

            try:
                # Ожидаем завершения основной задачи
                done, pending = await asyncio.wait(
                    tasks_to_wait,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Если какая-то задача завершилась, отменяем остальные
                for task in pending:
                    task.cancel()

                # Ждем завершения отмененных задач
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

            except asyncio.CancelledError:
                logger.info("Main tasks cancelled")
            except Exception as e:
                logger.error(f"Error in task management: {e}")

        except Exception as e:
            logger.error(f"Error in main bot execution: {e}")
            traceback.print_exc()
        finally:
            # Дополнительная гарантия завершения
            await self.shutdown()

    async def shutdown(self):
        """Корректное завершение работы"""
        if self._shutdown_flag:
            return

        self._shutdown_flag = True
        self.running = False

        logger.info("Shutting down Zoom Agent Bot...")

        # Останавливаем компоненты в правильном порядке
        shutdown_errors = []

        try:
            if self.processor:
                logger.info("Stopping transcript processor...")
                try:
                    self.processor.stop_processing()
                    if hasattr(self.processor, 'shutdown'):
                        await self.processor.shutdown()
                except Exception as e:
                    shutdown_errors.append(f"processor: {e}")

            if self.zoom_client:
                logger.info("Disconnecting from Zoom...")
                try:
                    # Сначала выходим из встречи, если мы в ней
                    if hasattr(self.zoom_client, 'status'):
                        if self.zoom_client.status in [ZoomSDKStatus.IN_MEETING,
                                                       ZoomSDKStatus.SCREEN_SHARING]:
                            await self.zoom_client.leave_meeting()

                    # Затем очищаем ресурсы SDK
                    if hasattr(self.zoom_client, 'cleanup'):
                        await self.zoom_client.cleanup()
                except Exception as e:
                    shutdown_errors.append(f"zoom_client: {e}")

            if self.dashboard_server:
                logger.info("Stopping dashboard server...")
                try:
                    await self.dashboard_server.stop()
                except Exception as e:
                    shutdown_errors.append(f"dashboard_server: {e}")

            if self.webhook_server:
                logger.info("Stopping webhook server...")
                try:
                    await self.webhook_server.stop()
                except Exception as e:
                    shutdown_errors.append(f"webhook_server: {e}")

            if self.news_agent:
                logger.info("Cleaning up news agent...")
                try:
                    await self.news_agent.cleanup()
                except Exception as e:
                    shutdown_errors.append(f"news_agent: {e}")

            if shutdown_errors:
                logger.warning(f"Some errors during shutdown: {shutdown_errors}")
            else:
                logger.info("Shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            traceback.print_exc()

        # Небольшая задержка для завершения всех операций
        await asyncio.sleep(1)


def main():
    """Точка входа"""
    try:
        # Проверяем настройки перед запуском
        logger.info("=" * 50)
        logger.info("Starting Zoom Agent Bot")
        logger.info("=" * 50)

        if not settings.ZOOM_MEETING_URL and not settings.ZOOM_MEETING_ID and not settings.USE_MOCK_TRANSCRIPT:
            logger.error("ZOOM_MEETING_URL or ZOOM_MEETING_ID not configured and USE_MOCK_TRANSCRIPT is False")
            logger.info("Please set Zoom meeting configuration or enable USE_MOCK_TRANSCRIPT")
            sys.exit(1)

        # Проверяем учетные данные Zoom
        if not settings.ZOOM_CLIENT_ID or not settings.ZOOM_CLIENT_SECRET:
            logger.warning("Zoom OAuth credentials not fully configured")
            logger.info("Client ID and Secret are required for Desktop SDK authentication")

            if not settings.USE_MOCK_TRANSCRIPT:
                logger.error("Cannot proceed without Zoom credentials")
                sys.exit(1)
            else:
                logger.info("Continuing in mock mode without Zoom credentials")

        bot = ZoomAgentBot()

        # Запускаем асинхронный код
        asyncio.run(bot.run())

    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()