"""
Процессор транскрипта - основной координационный центр анализа
Обрабатывает транскрипт, управляет LLM анализами, координирует поиск новостей
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import threading
from collections import deque, defaultdict

from config import settings
from utils.logger import setup_logger
from llm_processing.analyzer import TranscriptAnalyzer
from news_fetcher.news_agent import NewsAgent
from state_manager.meeting_state import MeetingState, EntityType

logger = setup_logger(__name__)


@dataclass
class ProcessingConfig:
    """Конфигурация обработки транскрипта"""
    update_interval_seconds: int = 60
    transcript_window_minutes: int = 3
    min_words_for_analysis: int = 50
    entity_importance_threshold: float = 0.4
    max_concurrent_llm_requests: int = 3
    news_search_delay_seconds: int = 5  # Задержка перед поиском новостей для новых сущностей
    enable_news_search: bool = True
    enable_realtime_updates: bool = True

    # Настройки LLM
    llm_timeout_seconds: int = 30
    llm_retry_count: int = 2
    llm_backoff_factor: float = 1.5

    # Кэширование
    cache_analysis_results: bool = True
    analysis_cache_ttl_seconds: int = 300  # 5 минут

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_interval_seconds": self.update_interval_seconds,
            "transcript_window_minutes": self.transcript_window_minutes,
            "min_words_for_analysis": self.min_words_for_analysis,
            "entity_importance_threshold": self.entity_importance_threshold,
            "max_concurrent_llm_requests": self.max_concurrent_llm_requests,
            "news_search_delay_seconds": self.news_search_delay_seconds,
            "enable_news_search": self.enable_news_search,
            "enable_realtime_updates": self.enable_realtime_updates,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "llm_retry_count": self.llm_retry_count,
            "llm_backoff_factor": self.llm_backoff_factor,
            "cache_analysis_results": self.cache_analysis_results,
            "analysis_cache_ttl_seconds": self.analysis_cache_ttl_seconds
        }


@dataclass
class CachedAnalysis:
    """Кэшированный результат анализа"""
    transcript_hash: str
    analysis_result: Dict[str, Any]
    timestamp: datetime
    ttl_seconds: int = 300

    def is_valid(self) -> bool:
        """Проверка валидности кэша"""
        return (datetime.now() - self.timestamp).total_seconds() < self.ttl_seconds


class TranscriptProcessor:
    """Основной процессор транскрипта"""

    def __init__(self, meeting_state: MeetingState):
        self.meeting_state = meeting_state
        self.config = ProcessingConfig()

        # Инициализация компонентов
        self.analyzer = TranscriptAnalyzer()
        self.news_agent = NewsAgent()

        # Кэширование
        self.analysis_cache: Dict[str, CachedAnalysis] = {}
        self.entity_search_cache: Set[str] = set()  # Сущности, для которых уже искали новости

        # Очереди и состояния
        self.processing_queue = deque(maxlen=100)
        self.processing_lock = threading.RLock()
        self.is_processing = False
        self.last_processing_time: Optional[datetime] = None

        # Статистика
        self.stats = {
            "total_analyses": 0,
            "cache_hits": 0,
            "llm_requests": 0,
            "news_searches": 0,
            "errors": 0,
            "avg_processing_time_ms": 0
        }

        # Семафор для ограничения одновременных LLM запросов
        self.llm_semaphore = asyncio.Semaphore(self.config.max_concurrent_llm_requests)

        # Поток для периодической обработки
        self.processing_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()

        logger.info("TranscriptProcessor initialized")

    def start_processing(self):
        """Запуск фоновой обработки транскрипта"""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("Processing already running")
            return

        self.shutdown_event.clear()
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="TranscriptProcessor"
        )
        self.processing_thread.start()
        logger.info("Background processing started")

    def stop_processing(self):
        """Остановка фоновой обработки"""
        self.shutdown_event.set()

        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            self.processing_thread = None

        logger.info("Background processing stopped")

    def _processing_loop(self):
        """Основной цикл обработки в фоновом потоке"""
        import time

        logger.info("Processing loop started")

        while not self.shutdown_event.is_set():
            try:
                # Проверяем, нужно ли обновить дашборд
                if self._should_update_dashboard():
                    # Создаем новый event loop для этого потока и запускаем обработку
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.process_recent_transcript())
                        loop.close()
                    except Exception as e:
                        logger.error(f"Error running processing in thread: {e}")
                        self.stats["errors"] += 1

                # Обрабатываем очередь синхронно
                self._process_queue_sync()

                # Очищаем устаревший кэш
                self._cleanup_cache()

                # Задержка перед следующей проверкой
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.stats["errors"] += 1
                time.sleep(5)

    def _process_queue_sync(self):
        """Синхронная обработка элементов из очереди"""
        if not self.processing_queue:
            return

        try:
            # Создаем event loop для обработки очереди
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            tasks = []
            while self.processing_queue:
                task = self.processing_queue.popleft()
                if task["type"] == "news_search":
                    task_coro = self._search_news_for_entity(
                        task["entity_name"],
                        task["entity_type"]
                    )
                    tasks.append(task_coro)

            # Запускаем все задачи параллельно
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

            loop.close()

        except Exception as e:
            logger.error(f"Error processing queue: {e}")

    def force_process_sync(self):
        """Синхронный метод для принудительной обработки"""
        try:
            # Пытаемся получить работающий цикл
            try:
                # Если есть работающий цикл
                loop = asyncio.get_running_loop()
                logger.debug(f"Found running event loop, using run_coroutine_threadsafe")

                # Важно: создаем корутину и передаем ее
                coroutine = self.process_recent_transcript()

                # Используем run_coroutine_threadsafe для потокобезопасности
                future = asyncio.run_coroutine_threadsafe(coroutine, loop)
                result = future.result(timeout=30)  # 30 секунд таймаут
                return result

            except RuntimeError:
                # Нет работающего цикла - создаем новый
                logger.debug(f"No running event loop, creating new one with asyncio.run")
                return asyncio.run(self.process_recent_transcript())

        except asyncio.TimeoutError:
            logger.error("Timeout in force_process_sync")
            return {"status": "error", "error": "Processing timeout"}

        except Exception as e:
            logger.error(f"Error in force_process_sync: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    def _should_update_dashboard(self) -> bool:
        """Определение, нужно ли обновить дашборд"""
        with self.processing_lock:
            if not self.last_processing_time:
                return True

            time_since_last = datetime.now() - self.last_processing_time
            return time_since_last.total_seconds() >= self.config.update_interval_seconds

    def _process_queue(self):
        """Обработка элементов из очереди"""
        if not self.processing_queue:
            return

        try:
            # Берем первый элемент из очереди
            task = self.processing_queue.popleft()

            if task["type"] == "news_search":
                asyncio.run(self._search_news_for_entity(
                    task["entity_name"],
                    task["entity_type"]
                ))

        except Exception as e:
            logger.error(f"Error processing queue item: {e}")

    async def process_recent_transcript(self) -> Dict[str, Any]:
        """Основной метод обработки последнего транскрипта"""
        start_time = datetime.now()

        try:
            with self.processing_lock:
                self.is_processing = True

                # Получаем последний транскрипт
                transcript_text = self.meeting_state.get_recent_transcript(
                    minutes=self.config.transcript_window_minutes
                )

                if not transcript_text:
                    logger.debug("No transcript to process")
                    return {"status": "no_transcript"}

                # Проверяем минимальную длину
                word_count = len(transcript_text.split())
                if word_count < self.config.min_words_for_analysis:
                    logger.debug(f"Transcript too short: {word_count} words")
                    return {"status": "too_short", "word_count": word_count}

                # Проверяем кэш
                transcript_hash = self._generate_transcript_hash(transcript_text)
                cached_result = self._get_cached_analysis(transcript_hash)

                if cached_result and self.config.cache_analysis_results:
                    logger.info("Using cached analysis")
                    self.stats["cache_hits"] += 1
                    analysis_result = cached_result.analysis_result
                else:
                    # Анализируем с помощью LLM
                    logger.info(f"Analyzing transcript ({word_count} words)")
                    analysis_result = await self._analyze_with_llm(transcript_text)

                    # Кэшируем результат
                    if self.config.cache_analysis_results:
                        self._cache_analysis(transcript_hash, analysis_result)

                # Обновляем состояние встречи
                self.meeting_state.update_analysis(analysis_result)

                # Извлекаем сущности для поиска новостей
                entities = analysis_result.get("entities", [])
                await self._process_entities_for_news(entities)

                # Обновляем статистику
                self._update_stats(start_time)

                logger.info(f"Processing completed in {(datetime.now() - start_time).total_seconds():.2f}s")

                return {
                    "status": "success",
                    "analysis": analysis_result,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }

        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
            self.stats["errors"] += 1
            return {"status": "error", "error": str(e)}

        finally:
            with self.processing_lock:
                self.is_processing = False
                self.last_processing_time = datetime.now()

    async def _analyze_with_llm(self, transcript_text: str) -> Dict[str, Any]:
        """Анализ транскрипта с помощью LLM с повторными попытками"""
        async with self.llm_semaphore:
            for attempt in range(self.config.llm_retry_count + 1):
                try:
                    logger.debug(f"LLM analysis attempt {attempt + 1}")

                    # Создаем таймаут для LLM запроса
                    analysis_task = self.analyzer.analyze_transcript(transcript_text)
                    analysis_result = await asyncio.wait_for(
                        analysis_task,
                        timeout=self.config.llm_timeout_seconds
                    )

                    self.stats["llm_requests"] += 1

                    # Валидируем результат
                    if self._validate_analysis_result(analysis_result):
                        return analysis_result
                    else:
                        logger.warning("LLM returned invalid analysis format")
                        raise ValueError("Invalid analysis format")

                except asyncio.TimeoutError:
                    logger.warning(f"LLM request timed out (attempt {attempt + 1})")
                    if attempt == self.config.llm_retry_count:
                        raise

                except Exception as e:
                    logger.error(f"LLM analysis error (attempt {attempt + 1}): {e}")
                    if attempt == self.config.llm_retry_count:
                        raise

                # Экспоненциальная задержка перед повторной попыткой
                if attempt < self.config.llm_retry_count:
                    delay = self.config.llm_backoff_factor ** attempt
                    await asyncio.sleep(delay)

            # Если все попытки не удались
            return self._get_fallback_analysis(transcript_text)

    def _validate_analysis_result(self, analysis: Dict[str, Any]) -> bool:
        """Валидация результата анализа"""
        required_keys = ["current_topic", "summary", "entities", "decisions", "open_questions"]

        # Проверяем наличие обязательных ключей
        if not all(key in analysis for key in required_keys):
            return False

        # Проверяем типы данных
        if not isinstance(analysis["summary"], list):
            return False

        if not isinstance(analysis["entities"], list):
            return False

        if not isinstance(analysis["decisions"], list):
            return False

        if not isinstance(analysis["open_questions"], list):
            return False

        return True

    def _get_fallback_analysis(self, transcript_text: str) -> Dict[str, Any]:
        """Резервный анализ при ошибках LLM"""
        logger.warning("Using fallback analysis")

        # Простая эвристика для извлечения сущностей
        entities = self._extract_entities_heuristic(transcript_text)

        # Определяем тему по ключевым словам
        topic = self._extract_topic_heuristic(transcript_text)

        return {
            "current_topic": {
                "topic": topic,
                "subtopics": [],
                "context": "Fallback analysis",
                "confidence": 0.3,
                "topic_shift": False,
                "keywords": []
            },
            "summary": ["Analysis temporarily unavailable. Using basic processing."],
            "entities": entities,
            "decisions": [],
            "open_questions": ["LLM analysis service is temporarily unavailable"],
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _extract_entities_heuristic(self, text: str) -> List[Dict[str, Any]]:
        """Эвристическое извлечение сущностей из текста"""
        entities = []

        # Поиск заглавных слов (возможные имена/названия)
        capital_words = re.findall(r'\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+)*\b', text)

        for word in set(capital_words):
            # Пропускаем короткие слова и общеупотребительные
            if len(word.split()) == 1 and len(word) < 4:
                continue

            # Определяем тип по контексту
            entity_type = self._guess_entity_type(word, text)

            entities.append({
                "name": word,
                "type": entity_type,
                "context": f"Mentioned in discussion",
                "importance": 0.4,
                "first_mentioned_at": "unknown"
            })

        return entities[:10]  # Ограничиваем количество

    def _guess_entity_type(self, word: str, context: str) -> str:
        """Определение типа сущности по контексту"""
        context_lower = context.lower()
        word_lower = word.lower()

        # Медицинские термины
        medical_keywords = ["drug", "treatment", "therapy", "dose", "mg", "clinical"]
        if any(keyword in context_lower for keyword in medical_keywords):
            return "medicine"

        # Компании
        company_keywords = ["inc", "corp", "ltd", "co", "company", "firm"]
        if any(f" {keyword}" in word_lower for keyword in company_keywords):
            return "company"

        # Проекты
        project_keywords = ["project", "initiative", "program", "trial", "study"]
        if any(keyword in context_lower for keyword in project_keywords):
            return "project"

        # По умолчанию
        return "other"

    def _extract_topic_heuristic(self, text: str) -> str:
        """Эвристическое определение темы"""
        # Ищем наиболее частые существительные
        words = text.lower().split()
        nouns = [w for w in words if len(w) > 4]  # Простая эвристика

        if not nouns:
            return "General discussion"

        # Считаем частоту
        from collections import Counter
        freq = Counter(nouns)
        top_words = [word for word, _ in freq.most_common(3)]

        return "Discussion about " + ", ".join(top_words)

    async def _process_entities_for_news(self, entities: List[Dict[str, Any]]):
        """Обработка сущностей для поиска новостей"""
        if not self.config.enable_news_search:
            return

        for entity in entities:
            entity_name = entity.get("name")
            entity_type = entity.get("type")
            importance = entity.get("importance", 0.0)

            if not entity_name:
                continue

            # Проверяем порог важности
            if importance < self.config.entity_importance_threshold:
                logger.debug(f"Entity {entity_name} below importance threshold: {importance}")
                continue

            # Проверяем, не искали ли уже новости для этой сущности
            cache_key = f"{entity_name}_{entity_type}"
            if cache_key in self.entity_search_cache:
                logger.debug(f"News already searched for {entity_name}")
                continue

            # Проверяем, есть ли уже сущность в состоянии встречи
            if self.meeting_state.has_searched_entity(entity_name):
                logger.debug(f"Entity {entity_name} already has news in meeting state")
                self.entity_search_cache.add(cache_key)
                continue

            # Добавляем в очередь для поиска новостей
            self.processing_queue.append({
                "type": "news_search",
                "entity_name": entity_name,
                "entity_type": entity_type,
                "importance": importance,
                "added_at": datetime.now()
            })

            logger.info(f"Queued news search for entity: {entity_name}")

    async def _search_news_for_entity(self, entity_name: str, entity_type: str):
        """Поиск новостей для сущности"""
        try:
            logger.info(f"Starting news search for: {entity_name}")

            # Задержка перед поиском (чтобы дать время на накопление контекста)
            await asyncio.sleep(self.config.news_search_delay_seconds)

            # Получаем контекст сущности из состояния встречи
            entity_context = self._get_entity_context(entity_name)

            # Ищем новости
            news_items = await self.news_agent.fetch_news_for_entity({
                "name": entity_name,
                "type": entity_type,
                "context": entity_context
            })

            # Добавляем новости в состояние встречи
            if news_items:
                self.meeting_state.add_news_items(entity_name, news_items)
                logger.info(f"Found {len(news_items)} news items for {entity_name}")
            else:
                logger.info(f"No news found for {entity_name}")

            # Добавляем в кэш поиска
            cache_key = f"{entity_name}_{entity_type}"
            self.entity_search_cache.add(cache_key)

            self.stats["news_searches"] += 1

        except Exception as e:
            logger.error(f"Error searching news for {entity_name}: {e}")

    def _get_entity_context(self, entity_name: str) -> str:
        """Получение контекста сущности из транскрипта"""
        # Получаем последний транскрипт
        transcript = self.meeting_state.get_recent_transcript(minutes=5)

        if not transcript:
            return ""

        # Ищем упоминания сущности в транскрипте
        pattern = re.compile(rf'.*?({re.escape(entity_name)}).*?', re.IGNORECASE)
        matches = pattern.findall(transcript)

        if not matches:
            return ""

        # Берем контекст вокруг первого упоминания
        text_lower = transcript.lower()
        entity_lower = entity_name.lower()

        if entity_lower in text_lower:
            idx = text_lower.index(entity_lower)
            start = max(0, idx - 100)
            end = min(len(transcript), idx + 100)
            context = transcript[start:end]
            return context

        return ""

    # ==================== УПРАВЛЕНИЕ КЭШЕМ ====================

    def _generate_transcript_hash(self, text: str) -> str:
        """Генерация хэша для транскрипта"""
        import hashlib

        # Нормализуем текст (убираем лишние пробелы, приводим к нижнему регистру)
        normalized = re.sub(r'\s+', ' ', text.strip().lower())

        # Хэшируем
        return hashlib.md5(normalized.encode()).hexdigest()

    def _get_cached_analysis(self, transcript_hash: str) -> Optional[Dict[str, Any]]:
        """Получение кэшированного анализа"""
        if transcript_hash in self.analysis_cache:
            cached = self.analysis_cache[transcript_hash]
            if cached.is_valid():
                return cached.analysis_result
            else:
                del self.analysis_cache[transcript_hash]

        return None

    def _cache_analysis(self, transcript_hash: str, analysis: Dict[str, Any]):
        """Кэширование анализа"""
        cached = CachedAnalysis(
            transcript_hash=transcript_hash,
            analysis_result=analysis,
            timestamp=datetime.now(),
            ttl_seconds=self.config.analysis_cache_ttl_seconds
        )
        self.analysis_cache[transcript_hash] = cached

        # Ограничиваем размер кэша
        if len(self.analysis_cache) > 100:
            # Удаляем самые старые записи
            oldest_keys = sorted(
                self.analysis_cache.keys(),
                key=lambda k: self.analysis_cache[k].timestamp
            )[:20]
            for key in oldest_keys:
                del self.analysis_cache[key]

    def _cleanup_cache(self):
        """Очистка устаревшего кэша"""
        current_time = datetime.now()
        expired_keys = []

        for key, cached in self.analysis_cache.items():
            if not cached.is_valid():
                expired_keys.append(key)

        for key in expired_keys:
            del self.analysis_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    # ==================== СТАТИСТИКА И МОНИТОРИНГ ====================

    def _update_stats(self, start_time: datetime):
        """Обновление статистики обработки"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000  # мс

        # Обновляем скользящее среднее
        self.stats["avg_processing_time_ms"] = (
                self.stats["avg_processing_time_ms"] * 0.8 + processing_time * 0.2
        )

        self.stats["total_analyses"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики процессора"""
        with self.processing_lock:
            return {
                **self.stats,
                "config": self.config.to_dict(),
                "cache_size": len(self.analysis_cache),
                "queue_size": len(self.processing_queue),
                "is_processing": self.is_processing,
                "last_processing_time": self.last_processing_time.isoformat() if self.last_processing_time else None,
                "entity_search_cache_size": len(self.entity_search_cache)
            }

    def get_processing_status(self) -> Dict[str, Any]:
        """Получение статуса обработки"""
        with self.processing_lock:
            return {
                "is_running": self.processing_thread is not None and self.processing_thread.is_alive(),
                "is_processing": self.is_processing,
                "last_processing_time": self.last_processing_time.isoformat() if self.last_processing_time else None,
                "next_scheduled_update": self._get_next_scheduled_update(),
                "queue_size": len(self.processing_queue),
                "cache_hit_rate": self._get_cache_hit_rate()
            }

    def _get_next_scheduled_update(self) -> Optional[str]:
        """Получение времени следующего запланированного обновления"""
        if not self.last_processing_time:
            return None

        next_time = self.last_processing_time + timedelta(seconds=self.config.update_interval_seconds)
        return next_time.isoformat()

    def _get_cache_hit_rate(self) -> float:
        """Получение процента попаданий в кэш"""
        total = self.stats["total_analyses"]
        hits = self.stats["cache_hits"]

        if total == 0:
            return 0.0

        return hits / total * 100

    # ==================== РУЧНОЕ УПРАВЛЕНИЕ ====================

    async def force_update(self):
        """Принудительное обновление дашборда"""
        logger.info("Forcing dashboard update")
        return await self.process_recent_transcript()

    async def search_news_for_entity_manual(self, entity_name: str, entity_type: str = None):
        """Ручной запуск поиска новостей для сущности"""
        logger.info(f"Manual news search requested for: {entity_name}")

        # Если тип не указан, пытаемся определить из состояния встречи
        if not entity_type:
            # Ищем сущность в состоянии встречи
            entities = self.meeting_state.get_entities()
            for entity in entities:
                if entity["name"] == entity_name:
                    entity_type = entity["type"]
                    break

        if not entity_type:
            entity_type = "other"

        # Запускаем поиск
        await self._search_news_for_entity(entity_name, entity_type)

    def clear_cache(self):
        """Очистка кэша анализа"""
        with self.processing_lock:
            self.analysis_cache.clear()
            self.entity_search_cache.clear()
            logger.info("Processor cache cleared")

    def update_config(self, **kwargs):
        """Обновление конфигурации процессора"""
        with self.processing_lock:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    old_value = getattr(self.config, key)
                    setattr(self.config, key, value)
                    logger.info(f"Config updated: {key}={old_value} -> {value}")
                else:
                    logger.warning(f"Invalid config key: {key}")

    # ==================== ЭКСПОРТ/ИМПОРТ ====================

    def export_state(self) -> Dict[str, Any]:
        """Экспорт состояния процессора"""
        with self.processing_lock:
            return {
                "config": self.config.to_dict(),
                "stats": self.stats.copy(),
                "cache_info": {
                    "analysis_cache_size": len(self.analysis_cache),
                    "entity_search_cache_size": len(self.entity_search_cache),
                    "cache_hit_rate": self._get_cache_hit_rate()
                },
                "processing_info": {
                    "is_processing": self.is_processing,
                    "last_processing_time": self.last_processing_time.isoformat() if self.last_processing_time else None,
                    "queue_size": len(self.processing_queue)
                },
                "exported_at": datetime.now().isoformat()
            }

    # ==================== ЗАВЕРШЕНИЕ РАБОТЫ ====================

    async def shutdown(self):
        """Корректное завершение работы процессора"""
        logger.info("Shutting down TranscriptProcessor...")

        # Останавливаем фоновую обработку
        self.stop_processing()

        # Очищаем ресурсы
        await self.news_agent.cleanup()

        # Очищаем кэш
        self.clear_cache()

        logger.info("TranscriptProcessor shutdown complete")


# Глобальный экземпляр для использования во всем приложении
processor_instance: Optional[TranscriptProcessor] = None


def get_transcript_processor(meeting_state: MeetingState = None) -> TranscriptProcessor:
    """Получение глобального экземпляра TranscriptProcessor"""
    global processor_instance

    if processor_instance is None:
        if meeting_state is None:
            from state_manager.meeting_state import get_meeting_state
            meeting_state = get_meeting_state()

        processor_instance = TranscriptProcessor(meeting_state)

    return processor_instance


def reset_transcript_processor():
    """Сброс глобального экземпляра TranscriptProcessor"""
    global processor_instance

    if processor_instance:
        processor_instance.shutdown()
        processor_instance = None