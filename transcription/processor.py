"""
Процессор транскрипта - основной координационный центр анализа
Обрабатывает транскрипт, управляет LLM анализами, координирует поиск новостей
"""

import asyncio
import json
import re
import time
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ProcessingConfig:
    """Конфигурация обработки транскрипта"""
    update_interval_seconds: int = 60
    transcript_window_minutes: int = 3
    min_words_for_analysis: int = 50
    entity_importance_threshold: float = 0.4
    max_concurrent_llm_requests: int = 3
    news_search_delay_seconds: int = 5
    enable_news_search: bool = True
    enable_realtime_updates: bool = True
    llm_timeout_seconds: int = 30
    llm_retry_count: int = 2
    llm_backoff_factor: float = 1.5
    cache_analysis_results: bool = True
    analysis_cache_ttl_seconds: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}


@dataclass
class CachedAnalysis:
    """Кэшированный результат анализа"""
    transcript_hash: str
    analysis_result: Dict[str, Any]
    timestamp: datetime
    ttl_seconds: int = 300

    def is_valid(self) -> bool:
        return (datetime.now() - self.timestamp).total_seconds() < self.ttl_seconds


class TranscriptProcessor:
    """Основной процессор транскрипта"""

    def __init__(self, meeting_state):
        self.meeting_state = meeting_state
        self.config = ProcessingConfig()

        # Инициализация всех необходимых атрибутов
        self._transcript_queue = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TranscriptProcessor")
        self._running = False
        self._processing = False
        self._last_processed_time = 0
        self._processed_count = 0
        self._last_force_process_time = 0
        self._transcript_history = []

        # Для совместимости с существующим кодом
        self.processing_queue = deque(maxlen=100)
        self.processing_lock = threading.RLock()
        self.is_processing = False
        self.last_processing_time = None
        self.stats = {
            "total_analyses": 0,
            "cache_hits": 0,
            "llm_requests": 0,
            "news_searches": 0,
            "errors": 0,
            "avg_processing_time_ms": 0
        }

        self.analysis_cache = {}
        self.entity_search_cache = set()
        self.shutdown_event = threading.Event()

        logger.info("TranscriptProcessor initialized with all attributes")

    def add_transcript(self, transcript_data: Dict[str, Any]):
        """Добавление транскрипта в очередь обработки"""
        with self._lock:
            self._transcript_queue.append(transcript_data)
            logger.debug(f"Added transcript to queue, size: {len(self._transcript_queue)}")

    def start_processing(self):
        """Запуск фоновой обработки транскрипта"""
        if self._running:
            logger.warning("Processing already started")
            return

        self._running = True
        self._processing = True

        # Запускаем фоновый поток для обработки
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="TranscriptProcessingThread"
        )
        self._processing_thread.start()
        logger.info("Processing loop started")

    def stop_processing(self):
        """Остановка фоновой обработки"""
        self._running = False
        self._processing = False
        self.shutdown_event.set()
        logger.info("Background processing stopped")

        # Останавливаем executor
        self._executor.shutdown(wait=False)

    def _processing_loop(self):
        """Основной цикл обработки транскрипта"""
        logger.info("Processing loop started")

        while self._running and not self.shutdown_event.is_set():
            try:
                # Проверяем, есть ли данные для обработки
                with self._lock:
                    has_data = len(self._transcript_queue) > 0

                if has_data:
                    # Запускаем обработку
                    future = self._executor.submit(self._process_transcript_cycle)
                    try:
                        # Ждем завершения обработки
                        result = future.result(timeout=10)
                        logger.debug(f"Processing cycle completed: {result}")
                    except Exception as e:
                        logger.warning(f"Processing cycle error: {e}")

                # Пауза между циклами обработки
                time.sleep(5)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(5)

    def _process_transcript_cycle(self) -> Dict[str, Any]:
        """Один цикл обработки транскрипта"""
        try:
            # Проверяем, есть ли новые данные для обработки
            with self._lock:
                if not self._transcript_queue:
                    # Если нет данных, просто возвращаем текущее состояние
                    return {
                        "status": "success",
                        "message": "No new transcript data",
                        "processed": False,
                        "queue_size": 0
                    }

                # Берем данные из очереди
                transcript_data = list(self._transcript_queue)
                self._transcript_queue.clear()
                queue_size = len(transcript_data)

            if not transcript_data:
                return {
                    "status": "success",
                    "message": "No transcript data to process",
                    "processed": False
                }

            logger.debug(f"Processing {queue_size} transcript items")

            # Обработка данных (упрощенная версия)
            processed_count = 0
            for data in transcript_data:
                try:
                    # Базовая обработка транскрипта
                    if self._process_single_transcript(data):
                        processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing transcript item: {e}")

            # Обновляем статистику
            with self._lock:
                self._last_processed_time = time.time()
                self._processed_count += processed_count

            return {
                "status": "success",
                "message": f"Processed {processed_count}/{queue_size} items",
                "processed": True,
                "processed_count": processed_count
            }

        except Exception as e:
            logger.error(f"Error in transcript processing cycle: {e}")
            return {
                "status": "error",
                "message": str(e),
                "processed": False
            }

    def _process_single_transcript(self, transcript_data: Dict[str, Any]) -> bool:
        """Обработка одного элемента транскрипта"""
        try:
            # Проверяем структуру данных
            if not isinstance(transcript_data, dict):
                logger.warning(f"Invalid transcript data type: {type(transcript_data)}")
                return False

            # Проверяем наличие текста
            text = transcript_data.get('text', '')
            if not text or not isinstance(text, str):
                logger.debug("No valid text in transcript data")
                return False

            # Простая обработка для теста
            logger.debug(f"Processing transcript text: {text[:100]}...")

            # Сохраняем в историю
            with self._lock:
                self._transcript_history.append({
                    'text': text,
                    'timestamp': time.time(),
                    'metadata': transcript_data.get('metadata', {})
                })

                # Ограничиваем размер истории
                if len(self._transcript_history) > 1000:
                    self._transcript_history = self._transcript_history[-1000:]

            return True

        except Exception as e:
            logger.error(f"Error processing single transcript: {e}")
            return False

    def force_process_sync(self, timeout: int = 5) -> Dict[str, Any]:
        """Синхронный запуск обработки с ограничением времени"""
        try:
            # Проверяем флаги
            if not self._processing:
                return {"status": "error", "message": "Processing not running"}

            if not self._running:
                return {"status": "error", "message": "Processor not running"}

            # Запускаем обработку в фоновом потоке
            with self._lock:
                self._last_force_process_time = time.time()

            future = self._executor.submit(self._process_transcript_cycle)

            try:
                # Ждем завершения с таймаутом
                result = future.result(timeout=timeout)
                logger.debug(f"Force process completed: {result}")
                return result

            except TimeoutError:
                logger.warning(f"Force process timeout after {timeout} seconds")
                # Отменяем задачу если она еще выполняется
                if not future.done():
                    future.cancel()
                return {"status": "timeout", "message": f"Processing timed out after {timeout} seconds"}

            except Exception as e:
                logger.error(f"Error in force_process_sync future: {e}")
                return {"status": "error", "message": str(e)}

        except Exception as e:
            logger.error(f"Error in force_process_sync: {e}")
            return {"status": "error", "message": str(e)}

    def get_processing_stats(self) -> Dict[str, Any]:
        """Получение статистики обработки"""
        with self._lock:
            return {
                "running": self._running,
                "processing": self._processing,
                "last_processed_time": self._last_processed_time,
                "processed_count": self._processed_count,
                "queue_size": len(self._transcript_queue),
                "history_size": len(self._transcript_history)
            }

    async def shutdown(self):
        """Корректное завершение работы"""
        self.stop_processing()
        logger.info("TranscriptProcessor shutdown complete")


def get_transcript_processor(meeting_state):
    """Фабричная функция для получения процессора транскрипта"""
    try:
        # Создаем процессор
        processor = TranscriptProcessor(meeting_state)

        # Запускаем обработку
        processor.start_processing()
        logger.info("Background processing started")

        return processor
    except Exception as e:
        logger.error(f"Failed to create transcript processor: {e}")
        raise