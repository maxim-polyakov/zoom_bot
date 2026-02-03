"""
Управление состоянием встречи Zoom агента
Отслеживание транскриптов, сущностей, новостей и состояния дашборда
"""

import json
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import heapq
from collections import defaultdict, deque

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MeetingStatus(Enum):
    """Статусы встречи"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    IN_MEETING = "in_meeting"
    SCREEN_SHARING = "screen_sharing"
    ERROR = "error"


class EntityType(Enum):
    """Типы сущностей"""
    MEDICINE = "medicine"
    COMPANY = "company"
    PERSON = "person"
    PROJECT = "project"
    DISEASE = "disease"
    METRIC = "metric"
    LOCATION = "location"
    OTHER = "other"


@dataclass
class TranscriptSegment:
    """Сегмент транскрипта"""
    text: str
    speaker: str
    timestamp: datetime
    duration: float = 0.0  # в секундах
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "speaker": self.speaker,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
            "confidence": self.confidence
        }


@dataclass
class Entity:
    """Сущность, упомянутая во встрече"""
    name: str
    entity_type: EntityType
    first_mentioned: datetime
    last_mentioned: datetime
    mention_count: int = 1
    context: str = ""
    importance: float = 0.5  # от 0 до 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.entity_type.value,
            "first_mentioned": self.first_mentioned.isoformat(),
            "last_mentioned": self.last_mentioned.isoformat(),
            "mention_count": self.mention_count,
            "context": self.context,
            "importance": self.importance
        }


@dataclass
class NewsItem:
    """Новость по сущности"""
    title: str
    source: str
    date: str
    snippet: str
    url: str
    freshness: str
    entity_name: str
    entity_type: str
    added_at: datetime = field(default_factory=datetime.now)
    thumbnail: str = ""
    relevance_score: float = 0.5  # от 0 до 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "date": self.date,
            "snippet": self.snippet,
            "url": self.url,
            "freshness": self.freshness,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "added_at": self.added_at.isoformat(),
            "thumbnail": self.thumbnail,
            "relevance_score": self.relevance_score
        }


@dataclass
class DecisionItem:
    """Решение или следующий шаг"""
    text: str
    decision_type: str  # decision/action/question/information
    assignee: str = ""
    deadline: str = ""
    priority: str = "medium"  # high/medium/low
    status: str = "proposed"  # proposed/agreed/rejected
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.decision_type,
            "assignee": self.assignee,
            "deadline": self.deadline,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class TopicAnalysis:
    """Анализ текущей темы"""
    topic: str
    subtopics: List[str]
    context: str
    confidence: float
    topic_shift: bool
    keywords: List[str]
    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "subtopics": self.subtopics,
            "context": self.context,
            "confidence": self.confidence,
            "topic_shift": self.topic_shift,
            "keywords": self.keywords,
            "analyzed_at": self.analyzed_at.isoformat()
        }


class MeetingState:
    """Основной класс управления состоянием встречи"""


    def __init__(self):
        # Блокировка для потокобезопасности
        self.lock = threading.RLock()

        # Статус встречи
        self.status: MeetingStatus = MeetingStatus.DISCONNECTED
        self.meeting_start_time: Optional[datetime] = None
        self.meeting_id: str = ""
        self.meeting_url: str = ""

        # Транскрипт
        self.transcript_segments: List[TranscriptSegment] = []
        self.transcript_buffer: deque = deque(maxlen=1000)  # Ограничиваем размер
        self.last_transcript_update: Optional[datetime] = None

        # Анализ
        self.current_topic: Optional[TopicAnalysis] = None
        self.topic_history: List[TopicAnalysis] = []
        self.summary_points: List[str] = []
        self.decisions: List[DecisionItem] = []
        self.open_questions: List[str] = []

        # Сущности
        self.entities: Dict[str, Entity] = {}  # name -> Entity
        self.entity_mentions: Dict[str, List[datetime]] = defaultdict(list)

        # Новости
        self.news_items: List[NewsItem] = []
        self.entity_news_map: Dict[str, List[NewsItem]] = defaultdict(list)
        self.searched_entities: Set[str] = set()

        # Статистика
        self.stats = {
            "word_count": 0,
            "speaker_count": 0,
            "unique_entities": 0,
            "topic_changes": 0,
            "decisions_made": 0,
            "questions_raised": 0,
            "news_found": 0
        }

        # История изменений (для графика)
        self.timeline_data: List[Dict[str, Any]] = []

        # Настройки
        self.update_interval = settings.UPDATE_INTERVAL_SECONDS

        logger.info("MeetingState initialized")

    # ==================== УПРАВЛЕНИЕ СТАТУСОМ ====================

    def update(self, data: Dict[str, Any]):
        """Обновление состояния простым присваиванием значений по ключам"""
        with self.lock:
            for key, value in data.items():
                # Проверяем, существует ли атрибут
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    # Если атрибута нет, добавляем в stats или создаем динамически
                    if key in self.stats:
                        self.stats[key] = value
                    else:
                        # Создаем динамический атрибут
                        setattr(self, key, value)

            logger.debug(f"State updated with {len(data)} fields")

    def update_status(self, status):
        """Обновление статуса встречи"""
        with self.lock:
            old_status = self.status

            # Если пришла строка, логируем как есть
            if isinstance(status, str):
                # Просто сохраняем строку
                self.status_str = status
                logger.info(f"Status changed: {old_status} -> {status}")
            elif hasattr(status, 'value'):
                # Если это Enum с value
                self.status = status
                logger.info(f"Status changed: {old_status.value} -> {status.value}")
            else:
                # Любой другой случай
                self.status = status
                logger.info(f"Status changed: {old_status} -> {status}")

    def get_status(self) -> str:
        """Получение текущего статуса"""
        with self.lock:
            return self.status.value

    def is_screen_sharing(self) -> bool:
        """Проверка, идет ли демонстрация экрана"""
        with self.lock:
            return self.status == MeetingStatus.SCREEN_SHARING

    # ==================== УПРАВЛЕНИЕ ТРАНСКРИПТОМ ====================

    def add_transcript_segment(self, text: str, speaker: str,
                               timestamp: datetime = None,
                               metadata: Dict[str, Any] = None):
        """Добавление сегмента транскрипта"""
        with self.lock:
            if not timestamp:
                timestamp = datetime.now()

            segment = TranscriptSegment(
                text=text,
                speaker=speaker,
                timestamp=timestamp,
                metadata=metadata or {}
            )

            self.transcript_segments.append(segment)
            self.transcript_buffer.append(segment)
            self.last_transcript_update = timestamp

            # Обновляем статистику
            self.stats["word_count"] += len(text.split())

            # Отслеживаем спикеров
            if speaker not in self.stats:
                self.stats["speaker_count"] = len(set(
                    s.speaker for s in self.transcript_segments
                ))

            logger.debug(f"Added transcript segment: {speaker}: {text[:50]}...")

    def get_full_transcript(self, limit: int = None) -> List[Dict[str, Any]]:
        """Получение полного транскрипта"""
        with self.lock:
            segments = self.transcript_segments
            if limit:
                segments = segments[-limit:]
            return [s.to_dict() for s in segments]

    def get_recent_transcript(self, minutes: int = 2) -> str:
        """Получение последних минут транскрипта"""
        with self.lock:
            if not self.transcript_segments:
                return ""

            cutoff_time = datetime.now() - timedelta(minutes=minutes)

            recent_segments = [
                s for s in self.transcript_segments
                if s.timestamp >= cutoff_time
            ]

            # Объединяем текст
            transcript_text = " ".join(s.text for s in recent_segments)

            return transcript_text

    def get_transcript_stats(self) -> Dict[str, Any]:
        """Получение статистики транскрипта"""
        with self.lock:
            if not self.transcript_segments:
                return {"total_segments": 0, "total_words": 0}

            total_words = sum(len(s.text.split()) for s in self.transcript_segments)
            speakers = set(s.speaker for s in self.transcript_segments)

            return {
                "total_segments": len(self.transcript_segments),
                "total_words": total_words,
                "unique_speakers": len(speakers),
                "last_update": self.last_transcript_update.isoformat() if self.last_transcript_update else None,
                "duration_minutes": self._get_meeting_duration_minutes()
            }

    # ==================== УПРАВЛЕНИЕ СУЩНОСТЯМИ ====================

    def add_entity(self, name: str, entity_type: EntityType,
                   context: str = "", importance: float = 0.5):
        """Добавление или обновление сущности"""
        with self.lock:
            current_time = datetime.now()

            if name in self.entities:
                # Обновляем существующую сущность
                entity = self.entities[name]
                entity.last_mentioned = current_time
                entity.mention_count += 1
                entity.importance = max(entity.importance, importance)

                if context and not entity.context:
                    entity.context = context
            else:
                # Создаем новую сущность
                entity = Entity(
                    name=name,
                    entity_type=entity_type,
                    first_mentioned=current_time,
                    last_mentioned=current_time,
                    context=context,
                    importance=importance
                )
                self.entities[name] = entity
                self.stats["unique_entities"] += 1

            # Записываем упоминание
            self.entity_mentions[name].append(current_time)

            # Записываем в timeline
            self._add_timeline_event("entity_mentioned", {
                "entity": name,
                "type": entity_type.value,
                "importance": importance
            })

            logger.info(f"Added/updated entity: {name} ({entity_type.value})")

            return entity

    def get_entities(self, min_importance: float = 0.3,
                     limit: int = 15) -> List[Dict[str, Any]]:
        """Получение сущностей с фильтрацией по важности"""
        with self.lock:
            entities = [
                e for e in self.entities.values()
                if e.importance >= min_importance
            ]

            # Сортируем по важности и количеству упоминаний
            entities.sort(key=lambda x: (x.importance, x.mention_count), reverse=True)

            return [e.to_dict() for e in entities[:limit]]

    def get_new_entities_since_last_check(self,
                                          current_entities: List[Dict[str, Any]],
                                          since_minutes: int = 5) -> List[Dict[str, Any]]:
        """Получение новых сущностей с момента последней проверки"""
        with self.lock:
            current_names = {e["name"] for e in current_entities}
            cutoff_time = datetime.now() - timedelta(minutes=since_minutes)

            new_entities = []

            for name, entity in self.entities.items():
                if (name not in current_names and
                        entity.last_mentioned >= cutoff_time):
                    new_entities.append(entity.to_dict())

            return new_entities

    def get_entity_mentions_timeline(self, entity_name: str) -> List[datetime]:
        """Получение временной линии упоминаний сущности"""
        with self.lock:
            return self.entity_mentions.get(entity_name, [])

    # ==================== УПРАВЛЕНИЕ НОВОСТЯМИ ====================

    def add_news_items(self, entity_name: str, news_items: List[Dict[str, Any]]):
        """Добавление новостей для сущности"""
        with self.lock:
            added_count = 0

            for item in news_items:
                # Проверяем дубликаты по URL
                if any(n.url == item.get("url") for n in self.news_items):
                    continue

                news_item = NewsItem(
                    title=item.get("title", ""),
                    source=item.get("source", "Unknown"),
                    date=item.get("date", ""),
                    snippet=item.get("snippet", ""),
                    url=item.get("url", ""),
                    freshness=item.get("freshness", "unknown"),
                    entity_name=entity_name,
                    entity_type=item.get("entity_type", "unknown"),
                    thumbnail=item.get("thumbnail", ""),
                    relevance_score=item.get("relevance_score", 0.5)
                )

                self.news_items.append(news_item)
                self.entity_news_map[entity_name].append(news_item)
                added_count += 1

            self.stats["news_found"] += added_count
            self.searched_entities.add(entity_name)

            if added_count > 0:
                logger.info(f"Added {added_count} news items for {entity_name}")

                # Записываем в timeline
                self._add_timeline_event("news_added", {
                    "entity": entity_name,
                    "count": added_count
                })

    def get_recent_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение последних новостей"""
        with self.lock:
            # Сортируем по времени добавления
            sorted_news = sorted(
                self.news_items,
                key=lambda x: x.added_at,
                reverse=True
            )

            return [n.to_dict() for n in sorted_news[:limit]]

    def get_news_for_entity(self, entity_name: str,
                            limit: int = 5) -> List[Dict[str, Any]]:
        """Получение новостей для конкретной сущности"""
        with self.lock:
            news_items = self.entity_news_map.get(entity_name, [])

            # Сортируем по свежести и релевантности
            news_items.sort(key=lambda x: (x.relevance_score, x.added_at),
                            reverse=True)

            return [n.to_dict() for n in news_items[:limit]]

    def has_searched_entity(self, entity_name: str) -> bool:
        """Проверка, искались ли новости для сущности"""
        with self.lock:
            return entity_name in self.searched_entities

    # ==================== УПРАВЛЕНИЕ АНАЛИЗОМ ====================

    def update_analysis(self, analysis: Dict[str, Any]):
        """Обновление анализа встречи"""
        with self.lock:
            # Обновляем тему
            if "current_topic" in analysis:
                topic_data = analysis["current_topic"]
                if isinstance(topic_data, dict):
                    topic_analysis = TopicAnalysis(
                        topic=topic_data.get("topic", ""),
                        subtopics=topic_data.get("subtopics", []),
                        context=topic_data.get("context", ""),
                        confidence=topic_data.get("confidence", 0.5),
                        topic_shift=topic_data.get("topic_shift", False),
                        keywords=topic_data.get("keywords", [])
                    )

                    if (self.current_topic and
                            topic_analysis.topic != self.current_topic.topic):
                        self.stats["topic_changes"] += 1

                    self.current_topic = topic_analysis
                    self.topic_history.append(topic_analysis)

            # Обновляем справку
            if "summary" in analysis:
                self.summary_points = analysis["summary"]

            # Обновляем решения
            if "decisions" in analysis:
                self._update_decisions(analysis["decisions"])

            # Обновляем вопросы
            if "open_questions" in analysis:
                self.open_questions = analysis["open_questions"]
                self.stats["questions_raised"] = len(self.open_questions)

            logger.info("Analysis updated")

    def _update_decisions(self, decisions_data: List[Dict[str, Any]]):
        """Обновление списка решений"""
        new_decisions = []

        for decision_data in decisions_data:
            if isinstance(decision_data, dict):
                decision = DecisionItem(
                    text=decision_data.get("text", ""),
                    decision_type=decision_data.get("type", "action"),
                    assignee=decision_data.get("assignee", ""),
                    deadline=decision_data.get("deadline", ""),
                    priority=decision_data.get("priority", "medium"),
                    status=decision_data.get("status", "proposed")
                )
                new_decisions.append(decision)
            elif isinstance(decision_data, str):
                decision = DecisionItem(
                    text=decision_data,
                    decision_type="action"
                )
                new_decisions.append(decision)

        # Добавляем только новые решения
        existing_texts = {d.text for d in self.decisions}

        for decision in new_decisions:
            if decision.text not in existing_texts:
                self.decisions.append(decision)
                existing_texts.add(decision.text)

                if decision.status == "agreed":
                    self.stats["decisions_made"] += 1

    def get_current_topic(self) -> Optional[Dict[str, Any]]:
        """Получение текущей темы"""
        with self.lock:
            if self.current_topic:
                return self.current_topic.to_dict()
            return None

    def get_topic_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение истории тем"""
        with self.lock:
            topics = self.topic_history[-limit:] if self.topic_history else []
            return [t.to_dict() for t in topics]

    # ==================== УПРАВЛЕНИЕ РЕШЕНИЯМИ И ВОПРОСАМИ ====================

    def add_decision(self, text: str, decision_type: str = "action",
                     assignee: str = "", priority: str = "medium"):
        """Ручное добавление решения"""
        with self.lock:
            decision = DecisionItem(
                text=text,
                decision_type=decision_type,
                assignee=assignee,
                priority=priority,
                status="proposed"
            )
            self.decisions.append(decision)

            logger.info(f"Added decision: {text}")

    def update_decision_status(self, decision_index: int, status: str):
        """Обновление статуса решения"""
        with self.lock:
            if 0 <= decision_index < len(self.decisions):
                self.decisions[decision_index].status = status
                self.decisions[decision_index].updated_at = datetime.now()

                if status == "agreed":
                    self.stats["decisions_made"] += 1

    def add_open_question(self, question: str):
        """Добавление открытого вопроса"""
        with self.lock:
            if question not in self.open_questions:
                self.open_questions.append(question)
                self.stats["questions_raised"] += 1

    def mark_question_resolved(self, question_index: int):
        """Пометить вопрос как решенный"""
        with self.lock:
            if 0 <= question_index < len(self.open_questions):
                self.open_questions.pop(question_index)

    # ==================== ДАННЫЕ ДЛЯ ДАШБОРДА ====================

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Получение всех данных для дашборда"""
        with self.lock:
            # Основные данные
            data = {
                "meeting_status": self.status.value,
                "meeting_duration": self._get_meeting_duration(),
                "last_update": datetime.now().isoformat(),

                # Текущая тема
                "current_topic": self.get_current_topic() or {
                    "topic": "Waiting for discussion to start...",
                    "subtopics": [],
                    "context": "",
                    "confidence": 0.0,
                    "topic_shift": False,
                    "keywords": []
                },

                # Краткая справка
                "summary": self.summary_points or [
                    "Discussion summary will appear here",
                    "Key arguments and context",
                    "Main points being discussed"
                ],

                # Сущности
                "entities": self.get_entities(min_importance=0.3, limit=15) or [],

                # Решения
                "decisions": [d.to_dict() for d in self.decisions[-5:]] or [],

                # Вопросы
                "open_questions": self.open_questions[-5:] or [],

                # Новости
                "news": self.get_recent_news(limit=8) or [],

                # Статистика
                "stats": {
                    **self.stats,
                    **self.get_transcript_stats()
                },

                # Timeline для графика
                "discussion_timeline": self._prepare_timeline_data(),

                # Активные сущности
                "active_entities": self._get_active_entities(),

                # Время начала встречи
                "meeting_start_time": self.meeting_start_time.isoformat() if self.meeting_start_time else None
            }

            return data

    def _prepare_timeline_data(self) -> Dict[str, Any]:
        """Подготовка данных для временной линии"""
        with self.lock:
            # Группируем события по 5-минутным интервалам
            if not self.timeline_data:
                return {"labels": [], "datasets": []}

            # Создаем интервалы
            if not self.meeting_start_time:
                return {"labels": [], "datasets": []}

            intervals = []
            current_time = self.meeting_start_time
            now = datetime.now()

            while current_time <= now:
                intervals.append(current_time)
                current_time += timedelta(minutes=5)

            # Подсчитываем события в интервалах
            topic_counts = [0] * (len(intervals) - 1)
            entity_counts = [0] * (len(intervals) - 1)
            decision_counts = [0] * (len(intervals) - 1)

            for event in self.timeline_data:
                event_time = datetime.fromisoformat(event["timestamp"])

                for i in range(len(intervals) - 1):
                    if intervals[i] <= event_time < intervals[i + 1]:
                        if event["type"] == "topic_change":
                            topic_counts[i] += 1
                        elif event["type"] == "entity_mentioned":
                            entity_counts[i] += 1
                        elif event["type"] == "decision_added":
                            decision_counts[i] += 1
                        break

            # Форматируем метки времени
            labels = [
                interval.strftime("%H:%M")
                for interval in intervals[:-1]
            ]

            return {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Topic Changes",
                        "data": topic_counts,
                        "borderColor": "#2d8cff",
                        "backgroundColor": "rgba(45, 140, 255, 0.1)"
                    },
                    {
                        "label": "Entities Mentioned",
                        "data": entity_counts,
                        "borderColor": "#ffc107",
                        "backgroundColor": "rgba(255, 193, 7, 0.1)"
                    },
                    {
                        "label": "Decisions Made",
                        "data": decision_counts,
                        "borderColor": "#28a745",
                        "backgroundColor": "rgba(40, 167, 69, 0.1)"
                    }
                ]
            }

    def _get_active_entities(self) -> List[Dict[str, Any]]:
        """Получение активных сущностей (упоминались в последние 10 минут)"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=10)

            active_entities = []

            for name, entity in self.entities.items():
                if entity.last_mentioned >= cutoff_time:
                    active_entities.append({
                        "name": name,
                        "type": entity.entity_type.value,
                        "last_mentioned": entity.last_mentioned.isoformat(),
                        "mention_count": entity.mention_count,
                        "has_news": name in self.entity_news_map
                    })

            # Сортируем по времени последнего упоминания
            active_entities.sort(
                key=lambda x: x["last_mentioned"],
                reverse=True
            )

            return active_entities[:10]

    # ==================== УТИЛИТНЫЕ МЕТОДЫ ====================

    def _add_timeline_event(self, event_type: str, data: Dict[str, Any] = None):
        """Добавление события в timeline"""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        self.timeline_data.append(event)

        # Ограничиваем размер timeline
        if len(self.timeline_data) > 1000:
            self.timeline_data = self.timeline_data[-500:]

    def _get_meeting_duration(self) -> str:
        """Получение продолжительности встречи в формате строки"""
        if not self.meeting_start_time:
            return "00:00:00"

        duration = datetime.now() - self.meeting_start_time
        total_seconds = int(duration.total_seconds())

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _get_meeting_duration_minutes(self) -> float:
        """Получение продолжительности встречи в минутах"""
        if not self.meeting_start_time:
            return 0.0

        duration = datetime.now() - self.meeting_start_time
        return duration.total_seconds() / 60.0

    # ==================== СОХРАНЕНИЕ И ЗАГРУЗКА ====================

    def save_state(self, filepath: str):
        """Сохранение состояния в файл"""
        with self.lock:
            try:
                state_data = {
                    "meeting_id": self.meeting_id,
                    "meeting_url": self.meeting_url,
                    "status": self.status.value,
                    "meeting_start_time": self.meeting_start_time.isoformat() if self.meeting_start_time else None,

                    "transcript_segments": [
                        segment.to_dict()
                        for segment in self.transcript_segments[-100:]  # Сохраняем последние 100 сегментов
                    ],

                    "entities": {
                        name: entity.to_dict()
                        for name, entity in self.entities.items()
                    },

                    "decisions": [d.to_dict() for d in self.decisions],
                    "open_questions": self.open_questions,
                    "summary_points": self.summary_points,

                    "current_topic": self.current_topic.to_dict() if self.current_topic else None,

                    "news_items": [n.to_dict() for n in self.news_items],

                    "stats": self.stats,
                    "timeline_data": self.timeline_data[-100:],  # Сохраняем последние 100 событий

                    "saved_at": datetime.now().isoformat()
                }

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, ensure_ascii=False, indent=2)

                logger.info(f"State saved to {filepath}")

            except Exception as e:
                logger.error(f"Error saving state: {e}")

    def load_state(self, filepath: str):
        """Загрузка состояния из файла"""
        with self.lock:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)

                # Восстанавливаем основные данные
                self.meeting_id = state_data.get("meeting_id", "")
                self.meeting_url = state_data.get("meeting_url", "")

                status_str = state_data.get("status", "disconnected")
                self.status = MeetingStatus(status_str)

                start_time_str = state_data.get("meeting_start_time")
                if start_time_str:
                    self.meeting_start_time = datetime.fromisoformat(start_time_str)

                # Восстанавливаем транскрипт
                self.transcript_segments = []
                for segment_data in state_data.get("transcript_segments", []):
                    segment = TranscriptSegment(
                        text=segment_data["text"],
                        speaker=segment_data["speaker"],
                        timestamp=datetime.fromisoformat(segment_data["timestamp"]),
                        duration=segment_data.get("duration", 0.0),
                        confidence=segment_data.get("confidence", 1.0),
                        metadata=segment_data.get("metadata", {})
                    )
                    self.transcript_segments.append(segment)

                # Восстанавливаем сущности
                self.entities = {}
                for name, entity_data in state_data.get("entities", {}).items():
                    entity = Entity(
                        name=name,
                        entity_type=EntityType(entity_data["type"]),
                        first_mentioned=datetime.fromisoformat(entity_data["first_mentioned"]),
                        last_mentioned=datetime.fromisoformat(entity_data["last_mentioned"]),
                        mention_count=entity_data["mention_count"],
                        context=entity_data["context"],
                        importance=entity_data["importance"],
                        metadata=entity_data.get("metadata", {})
                    )
                    self.entities[name] = entity

                # Восстанавливаем решения и вопросы
                self.decisions = []
                for decision_data in state_data.get("decisions", []):
                    decision = DecisionItem(
                        text=decision_data["text"],
                        decision_type=decision_data["type"],
                        assignee=decision_data.get("assignee", ""),
                        deadline=decision_data.get("deadline", ""),
                        priority=decision_data.get("priority", "medium"),
                        status=decision_data.get("status", "proposed"),
                        created_at=datetime.fromisoformat(decision_data["created_at"]),
                        updated_at=datetime.fromisoformat(decision_data["updated_at"])
                    )
                    self.decisions.append(decision)

                self.open_questions = state_data.get("open_questions", [])
                self.summary_points = state_data.get("summary_points", [])

                # Восстанавливаем тему
                topic_data = state_data.get("current_topic")
                if topic_data:
                    self.current_topic = TopicAnalysis(
                        topic=topic_data["topic"],
                        subtopics=topic_data["subtopics"],
                        context=topic_data["context"],
                        confidence=topic_data["confidence"],
                        topic_shift=topic_data["topic_shift"],
                        keywords=topic_data["keywords"],
                        analyzed_at=datetime.fromisoformat(topic_data["analyzed_at"])
                    )

                # Восстанавливаем новости
                self.news_items = []
                self.entity_news_map = defaultdict(list)
                for news_data in state_data.get("news_items", []):
                    news_item = NewsItem(
                        title=news_data["title"],
                        source=news_data["source"],
                        date=news_data["date"],
                        snippet=news_data["snippet"],
                        url=news_data["url"],
                        freshness=news_data["freshness"],
                        entity_name=news_data["entity_name"],
                        entity_type=news_data["entity_type"],
                        added_at=datetime.fromisoformat(news_data["added_at"]),
                        thumbnail=news_data.get("thumbnail", ""),
                        relevance_score=news_data.get("relevance_score", 0.5)
                    )
                    self.news_items.append(news_item)
                    self.entity_news_map[news_data["entity_name"]].append(news_item)

                # Восстанавливаем статистику
                self.stats = state_data.get("stats", self.stats.copy())

                # Восстанавливаем timeline
                self.timeline_data = state_data.get("timeline_data", [])

                logger.info(f"State loaded from {filepath}")

            except Exception as e:
                logger.error(f"Error loading state: {e}")

    # ==================== ОЧИСТКА И СБРОС ====================

    def clear_state(self):
        """Очистка состояния (для новой встречи)"""
        with self.lock:
            self.status = MeetingStatus.DISCONNECTED
            self.meeting_start_time = None

            self.transcript_segments.clear()
            self.transcript_buffer.clear()
            self.last_transcript_update = None

            self.current_topic = None
            self.topic_history.clear()
            self.summary_points.clear()
            self.decisions.clear()
            self.open_questions.clear()

            self.entities.clear()
            self.entity_mentions.clear()

            self.news_items.clear()
            self.entity_news_map.clear()
            self.searched_entities.clear()

            # Сбрасываем статистику
            self.stats = {
                "word_count": 0,
                "speaker_count": 0,
                "unique_entities": 0,
                "topic_changes": 0,
                "decisions_made": 0,
                "questions_raised": 0,
                "news_found": 0
            }

            self.timeline_data.clear()

            logger.info("State cleared")

    def reset_meeting(self):
        """Сброс встречи (но сохраняем некоторые данные)"""
        with self.lock:
            # Сохраняем сущности и новости для возможного повторного использования
            entities_backup = dict(self.entities)
            news_backup = list(self.news_items)
            entity_news_backup = dict(self.entity_news_map)

            # Очищаем состояние
            self.clear_state()

            # Восстанавливаем сущности и новости
            self.entities = entities_backup
            self.news_items = news_backup
            self.entity_news_map = entity_news_backup

            logger.info("Meeting reset (entities and news preserved)")

    # ==================== КОНТЕКСТНЫЙ МЕНЕДЖЕР ====================

    def __enter__(self):
        """Поддержка контекстного менеджера"""
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Поддержка контекстного менеджера"""
        self.lock.release()


# Глобальный экземпляр для использования во всем приложении
meeting_state_instance: Optional[MeetingState] = None


def get_meeting_state() -> MeetingState:
    """Получение глобального экземпляра MeetingState"""
    global meeting_state_instance

    if meeting_state_instance is None:
        meeting_state_instance = MeetingState()

    return meeting_state_instance


def reset_meeting_state():
    """Сброс глобального экземпляра MeetingState"""
    global meeting_state_instance
    meeting_state_instance = None