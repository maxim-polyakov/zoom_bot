"""
Вебхук-сервер для приема транскрипта из внешних источников
Поддерживает различные форматы: Zoom API, Recall.ai, AssemblyAI и т.д.
"""

import asyncio
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, ValidationError

from config import settings
from utils.logger import setup_logger
from state_manager.meeting_state import MeetingState, get_meeting_state

logger = setup_logger(__name__)


class TranscriptFormat(Enum):
    """Форматы транскрипта"""
    ZOOM_API = "zoom_api"
    RECALL_AI = "recall_ai"
    ASSEMBLY_AI = "assembly_ai"
    DEEPGRAM = "deepgram"
    SIMPLE_TEXT = "simple_text"
    CUSTOM_JSON = "custom_json"


class TranscriptSegment(BaseModel):
    """Модель сегмента транскрипта"""
    text: str
    speaker: str
    start_time: Optional[float] = None  # в секундах
    end_time: Optional[float] = None  # в секундах
    confidence: Optional[float] = None
    language: Optional[str] = "en"
    metadata: Optional[Dict[str, Any]] = None


class WebhookPayload(BaseModel):
    """Базовая модель вебхук пэйлоада"""
    format: TranscriptFormat = TranscriptFormat.SIMPLE_TEXT
    meeting_id: Optional[str] = None
    meeting_topic: Optional[str] = None
    segments: List[TranscriptSegment] = []
    raw_text: Optional[str] = None  # Для простого текста
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class WebhookServer:
    """Сервер для приема транскрипта через вебхуки"""

    def __init__(self, meeting_state: MeetingState = None):
        self.meeting_state = meeting_state or get_meeting_state()
        self.app = FastAPI(title="Transcript Webhook Server")
        self.parsers = self._init_parsers()
        self.webhook_secret = settings.WEBHOOK_SECRET if hasattr(settings, 'WEBHOOK_SECRET') else None

        # Статистика
        self.stats = {
            "total_requests": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "last_received": None,
            "formats_received": {}
        }

        # Подписчики на события транскрипта
        self.subscribers: List[Callable] = []

        # Настройка маршрутов
        self._setup_routes()

        logger.info("WebhookServer initialized")

    def _init_parsers(self) -> Dict[TranscriptFormat, Callable]:
        """Инициализация парсеров для разных форматов"""
        return {
            TranscriptFormat.ZOOM_API: self._parse_zoom_api,
            TranscriptFormat.RECALL_AI: self._parse_recall_ai,
            TranscriptFormat.ASSEMBLY_AI: self._parse_assembly_ai,
            TranscriptFormat.DEEPGRAM: self._parse_deepgram,
            TranscriptFormat.SIMPLE_TEXT: self._parse_simple_text,
            TranscriptFormat.CUSTOM_JSON: self._parse_custom_json
        }

    def _setup_routes(self):
        """Настройка API маршрутов"""

        @self.app.get("/")
        async def root():
            """Корневой endpoint"""
            return {
                "status": "running",
                "service": "transcript_webhook_server",
                "version": "1.0.0",
                "endpoints": ["/webhook", "/health", "/stats"]
            }

        @self.app.get("/health")
        async def health_check():
            """Проверка здоровья сервера"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": "todo",  # Можно добавить расчет времени работы
                "memory_usage": "todo"
            }

        @self.app.get("/stats")
        async def get_stats():
            """Получение статистики сервера"""
            return {
                "stats": self.stats,
                "subscribers_count": len(self.subscribers),
                "meeting_state_status": self.meeting_state.get_status()
            }

        @self.app.post("/webhook")
        async def receive_webhook(
                request: Request,
                x_webhook_signature: Optional[str] = Header(None),
                x_webhook_format: Optional[str] = Header(None)
        ):
            """Основной endpoint для приема вебхуков"""
            self.stats["total_requests"] += 1

            try:
                # Проверка подписи (если настроена)
                if self.webhook_secret and x_webhook_signature:
                    if not self._verify_signature(request, x_webhook_signature):
                        raise HTTPException(status_code=401, detail="Invalid signature")

                # Определяем формат
                format_str = x_webhook_format or "simple_text"
                try:
                    transcript_format = TranscriptFormat(format_str.lower())
                except ValueError:
                    # Пытаемся определить формат автоматически
                    transcript_format = await self._detect_format(request)

                # Парсим тело запроса
                body = await request.json()

                # Обрабатываем транскрипт
                processed = await self.process_transcript(body, transcript_format)

                # Обновляем статистику
                format_name = transcript_format.value
                self.stats["formats_received"][format_name] = \
                    self.stats["formats_received"].get(format_name, 0) + 1
                self.stats["successful_parses"] += 1
                self.stats["last_received"] = datetime.now().isoformat()

                # Уведомляем подписчиков
                await self._notify_subscribers(processed)

                return {
                    "status": "success",
                    "processed_segments": len(processed),
                    "format": format_name,
                    "timestamp": datetime.now().isoformat()
                }

            except ValidationError as e:
                self.stats["failed_parses"] += 1
                logger.error(f"Validation error: {e}")
                raise HTTPException(status_code=400, detail=str(e))

            except Exception as e:
                self.stats["failed_parses"] += 1
                logger.error(f"Webhook processing error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/webhook/raw")
        async def receive_raw_transcript(
                payload: Dict[str, Any],
                x_webhook_signature: Optional[str] = Header(None)
        ):
            """Endpoint для сырых данных транскрипта"""
            self.stats["total_requests"] += 1

            try:
                # Проверка подписи
                if self.webhook_secret and x_webhook_signature:
                    body_str = json.dumps(payload, sort_keys=True)
                    expected_sig = self._calculate_signature(body_str)
                    if x_webhook_signature != expected_sig:
                        raise HTTPException(status_code=401, detail="Invalid signature")

                # Обрабатываем как простой текст
                if isinstance(payload, dict) and "text" in payload:
                    text = payload["text"]
                    speaker = payload.get("speaker", "Unknown")

                    self.meeting_state.add_transcript_segment(
                        text=text,
                        speaker=speaker,
                        metadata={"source": "raw_webhook"}
                    )

                    self.stats["successful_parses"] += 1

                    return {"status": "success", "text_received": len(text)}
                else:
                    raise HTTPException(status_code=400, detail="Invalid payload format")

            except Exception as e:
                self.stats["failed_parses"] += 1
                logger.error(f"Raw webhook error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/webhook/test")
        async def test_webhook(payload: Dict[str, Any]):
            """Тестовый endpoint для проверки работы"""
            test_text = payload.get("text", "Test transcription from webhook")
            test_speaker = payload.get("speaker", "Test User")

            self.meeting_state.add_transcript_segment(
                text=test_text,
                speaker=test_speaker,
                metadata={"source": "test_webhook"}
            )

            return {
                "status": "test_success",
                "message": "Test transcription added",
                "text": test_text[:100] + "..." if len(test_text) > 100 else test_text,
                "timestamp": datetime.now().isoformat()
            }

    async def _detect_format(self, request: Request) -> TranscriptFormat:
        """Автоматическое определение формата транскрипта"""
        try:
            body = await request.json()

            # Проверяем формат Zoom API
            if "event" in body and body.get("event") == "meeting.recording_transcript_completed":
                return TranscriptFormat.ZOOM_API

            # Проверяем формат Recall.ai
            if "recording" in body and "transcript" in body:
                return TranscriptFormat.RECALL_AI

            # Проверяем формат AssemblyAI
            if "utterances" in body or "words" in body:
                return TranscriptFormat.ASSEMBLY_AI

            # По умолчанию - простой текст
            return TranscriptFormat.SIMPLE_TEXT

        except:
            return TranscriptFormat.SIMPLE_TEXT

    def _verify_signature(self, request: Request, signature: str) -> bool:
        """Проверка подписи вебхука"""
        try:
            # Получаем тело запроса
            body_bytes = request.body()

            # Вычисляем ожидаемую подпись
            expected_sig = hashlib.sha256(
                body_bytes + self.webhook_secret.encode()
            ).hexdigest()

            return signature == expected_sig

        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False

    def _calculate_signature(self, body_str: str) -> str:
        """Вычисление подписи для тела запроса"""
        if not self.webhook_secret:
            return ""

        return hashlib.sha256(
            (body_str + self.webhook_secret).encode()
        ).hexdigest()

    async def process_transcript(
            self,
            payload: Dict[str, Any],
            format_type: TranscriptFormat
    ) -> List[Dict[str, Any]]:
        """Обработка транскрипта в зависимости от формата"""
        parser = self.parsers.get(format_type)
        if not parser:
            logger.error(f"No parser for format: {format_type}")
            raise ValueError(f"Unsupported format: {format_type}")

        # Парсим данные
        segments = parser(payload)

        # Добавляем в состояние встречи
        for segment in segments:
            self.meeting_state.add_transcript_segment(
                text=segment["text"],
                speaker=segment["speaker"],
                metadata={
                    **segment.get("metadata", {}),
                    "format": format_type.value,
                    "parsed_at": datetime.now().isoformat()
                }
            )

        logger.info(f"Processed {len(segments)} segments in {format_type.value} format")
        return segments

    def _parse_zoom_api(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Парсинг формата Zoom API"""
        segments = []

        # Zoom API может отправлять транскрипт в разных форматах
        if payload.get("event") == "meeting.recording_transcript_completed":
            # Обработка завершенного транскрипта
            transcript_data = payload.get("payload", {}).get("object", {})
            transcript_url = transcript_data.get("download_url")

            # Здесь можно добавить логику загрузки файла
            # Пока возвращаем заглушку
            segments.append({
                "text": f"Zoom transcript completed: {transcript_data.get('id', 'unknown')}",
                "speaker": "Zoom System",
                "metadata": {
                    "zoom_event": payload["event"],
                    "meeting_id": transcript_data.get("meeting_id"),
                    "transcript_id": transcript_data.get("id")
                }
            })

        elif "transcript" in payload:
            # Прямой транскрипт в пэйлоаде
            transcript_text = payload.get("transcript", "")
            speaker = payload.get("speaker", "Unknown")

            segments.append({
                "text": transcript_text,
                "speaker": speaker,
                "metadata": {"source": "zoom_api_direct"}
            })

        return segments

    def _parse_recall_ai(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Парсинг формата Recall.ai"""
        segments = []

        try:
            recording_data = payload.get("recording", {})
            transcript_data = payload.get("transcript", {})

            # Recall.ai может отправлять сегменты или полный текст
            if "segments" in transcript_data:
                for segment in transcript_data["segments"]:
                    segments.append({
                        "text": segment.get("text", ""),
                        "speaker": segment.get("speaker", "Unknown"),
                        "start_time": segment.get("start_time"),
                        "end_time": segment.get("end_time"),
                        "metadata": {
                            "source": "recall_ai",
                            "segment_id": segment.get("id"),
                            "confidence": segment.get("confidence")
                        }
                    })
            elif "text" in transcript_data:
                # Полный текст
                segments.append({
                    "text": transcript_data["text"],
                    "speaker": recording_data.get("speaker", "Unknown"),
                    "metadata": {
                        "source": "recall_ai_full",
                        "recording_id": recording_data.get("id")
                    }
                })

        except Exception as e:
            logger.error(f"Error parsing Recall.ai format: {e}")

        return segments

    def _parse_assembly_ai(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Парсинг формата AssemblyAI"""
        segments = []

        try:
            # AssemblyAI может отправлять utterances (высказывания)
            if "utterances" in payload:
                for utterance in payload["utterances"]:
                    segments.append({
                        "text": utterance.get("text", ""),
                        "speaker": utterance.get("speaker", "Speaker"),
                        "start_time": utterance.get("start"),
                        "end_time": utterance.get("end"),
                        "metadata": {
                            "source": "assembly_ai",
                            "confidence": utterance.get("confidence")
                        }
                    })

            # Или words (слова) которые нужно собрать
            elif "words" in payload:
                current_speaker = None
                current_text = []
                current_start = None

                for word in payload["words"]:
                    speaker = word.get("speaker")

                    if speaker != current_speaker and current_text:
                        # Завершаем предыдущий сегмент
                        segments.append({
                            "text": " ".join(current_text),
                            "speaker": current_speaker or "Unknown",
                            "start_time": current_start,
                            "end_time": word.get("start"),
                            "metadata": {"source": "assembly_ai_words"}
                        })
                        current_text = []

                    if not current_start:
                        current_start = word.get("start")

                    current_speaker = speaker
                    current_text.append(word.get("text", ""))

                # Добавляем последний сегмент
                if current_text:
                    segments.append({
                        "text": " ".join(current_text),
                        "speaker": current_speaker or "Unknown",
                        "start_time": current_start,
                        "metadata": {"source": "assembly_ai_words"}
                    })

        except Exception as e:
            logger.error(f"Error parsing AssemblyAI format: {e}")

        return segments

    def _parse_deepgram(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Парсинг формата Deepgram"""
        segments = []

        try:
            # Deepgram отправляет результаты в results.channels
            results = payload.get("results", {})
            channels = results.get("channels", [])

            for channel in channels:
                alternatives = channel.get("alternatives", [])
                for alt in alternatives:
                    text = alt.get("transcript", "")
                    if text:
                        segments.append({
                            "text": text,
                            "speaker": f"Channel_{channel.get('channel', 0)}",
                            "metadata": {
                                "source": "deepgram",
                                "confidence": alt.get("confidence"),
                                "words_count": len(alt.get("words", []))
                            }
                        })

        except Exception as e:
            logger.error(f"Error parsing Deepgram format: {e}")

        return segments

    def _parse_simple_text(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Парсинг простого текстового формата"""
        segments = []

        try:
            # Простой текст
            if isinstance(payload, str):
                text = payload
                speaker = "Unknown"
            else:
                text = payload.get("text", "")
                speaker = payload.get("speaker", "Unknown")

            if text:
                segments.append({
                    "text": text,
                    "speaker": speaker,
                    "metadata": {"source": "simple_text"}
                })

        except Exception as e:
            logger.error(f"Error parsing simple text: {e}")

        return segments

    def _parse_custom_json(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Парсинг кастомного JSON формата"""
        segments = []

        try:
            # Пытаемся найти стандартные поля
            text = payload.get("transcript") or payload.get("text") or payload.get("content")
            speaker = payload.get("speaker") or payload.get("author") or "Unknown"

            if text:
                segments.append({
                    "text": text,
                    "speaker": speaker,
                    "metadata": {
                        "source": "custom_json",
                        "original_payload_keys": list(payload.keys())
                    }
                })

            # Или ищем массив сообщений
            elif "messages" in payload:
                for msg in payload["messages"]:
                    if isinstance(msg, dict):
                        msg_text = msg.get("text") or msg.get("content")
                        msg_speaker = msg.get("speaker") or msg.get("author") or "Unknown"

                        if msg_text:
                            segments.append({
                                "text": msg_text,
                                "speaker": msg_speaker,
                                "metadata": {"source": "custom_json_messages"}
                            })

        except Exception as e:
            logger.error(f"Error parsing custom JSON: {e}")

        return segments

    async def _notify_subscribers(self, segments: List[Dict[str, Any]]):
        """Уведомление подписчиков о новых сегментах"""
        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(segments)
                else:
                    subscriber(segments)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")

    def subscribe(self, callback: Callable):
        """Подписка на события транскрипта"""
        if callback not in self.subscribers:
            self.subscribers.append(callback)
            logger.info(f"New subscriber added. Total: {len(self.subscribers)}")

    def unsubscribe(self, callback: Callable):
        """Отписка от событий транскрипта"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Subscriber removed. Total: {len(self.subscribers)}")

    async def start(self):
        """Запуск веб-сервера"""
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",  # Принимаем запросы со всех интерфейсов
            port=settings.TRANSCRIPT_WEBHOOK_PORT,
            log_level="info" if settings.DEBUG else "warning"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def stop(self):
        """Остановка сервера"""
        # Очищаем подписчиков
        self.subscribers.clear()

        logger.info("WebhookServer stopped")


# Глобальный экземпляр
webhook_server_instance: Optional[WebhookServer] = None