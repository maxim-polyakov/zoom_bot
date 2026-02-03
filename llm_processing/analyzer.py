"""
Анализатор транскрипта с использованием LLM
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from config import settings
from utils.logger import setup_logger
from llm_processing.prompts import (
    TOPIC_ANALYSIS_PROMPT,
    SUMMARY_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    DECISION_EXTRACTION_PROMPT,
    QUESTION_EXTRACTION_PROMPT
)

logger = setup_logger(__name__)


class TranscriptAnalyzer:
    def __init__(self):
        self.llm_client = self._init_llm_client()
        self.conversation_history = []
        self.last_analysis_time = None

    # Заменить в методе _init_llm_client:
    def _init_llm_client(self):
        """Инициализация клиента LLM в зависимости от провайдера"""
        if settings.LLM_PROVIDER == "openai":
            from openai import AsyncOpenAI
            return AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                # Убрать параметр proxies, если он есть
                # proxies=settings.PROXY_URL if hasattr(settings, 'PROXY_URL') else None
            )
        elif settings.LLM_PROVIDER == "anthropic":
            from anthropic import AsyncAnthropic
            return AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

    async def analyze_transcript(self, transcript_text: str) -> Dict[str, Any]:
        """
        Полный анализ транскрипта
        Возвращает структурированные данные для дашборда
        """
        logger.info("Starting transcript analysis...")

        try:
            # Добавляем в историю
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "text": transcript_text[-5000:]  # Берем последние ~5000 символов
            })

            # Ограничиваем историю
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            # Параллельный запуск всех анализов
            tasks = [
                self._extract_current_topic(transcript_text),
                self._generate_summary(transcript_text),
                self._extract_entities(transcript_text),
                self._extract_decisions(transcript_text),
                self._extract_open_questions(transcript_text)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Обработка результатов
            analysis_result = {
                "current_topic": self._safe_get_result(results[0], "No topic detected"),
                "summary": self._safe_get_result(results[1], []),
                "entities": self._safe_get_result(results[2], []),
                "decisions": self._safe_get_result(results[3], []),
                "open_questions": self._safe_get_result(results[4], []),
                "analysis_timestamp": datetime.now().isoformat()
            }

            self.last_analysis_time = datetime.now()
            logger.info("Transcript analysis completed successfully")

            return analysis_result

        except Exception as e:
            logger.error(f"Error in transcript analysis: {e}")
            return self._get_default_analysis()

    async def _extract_current_topic(self, text: str) -> Dict[str, Any]:
        """Извлечение текущей темы обсуждения"""
        try:
            if settings.LLM_PROVIDER == "openai":
                response = await self.llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": TOPIC_ANALYSIS_PROMPT},
                        {"role": "user", "content": text[-2000:]}
                    ],
                    temperature=0.3,
                    max_tokens=200,
                    response_format={"type": "json_object"}  # <-- ДОБАВЬТЕ ЭТУ СТРОКУ
                )
                content = response.choices[0].message.content
            elif settings.LLM_PROVIDER == "anthropic":
                response = await self.llm_client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=200,
                    temperature=0.3,
                    system=TOPIC_ANALYSIS_PROMPT,
                    messages=[{"role": "user", "content": text[-2000:]}]
                )
                content = response.content[0].text
            else:
                # Мок-ответ для тестирования
                return {
                    "topic": "Test discussion topic",
                    "subtopics": ["subtopic 1", "subtopic 2"],
                    "context": "Test context",
                    "confidence": 0.8,
                    "topic_shift": False,
                    "keywords": ["test", "discussion"]
                }

            # Парсинг JSON ответа
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON: {content[:100]}")
                return {"topic": content.strip(), "confidence": 0.5}

        except Exception as e:
            logger.error(f"Error extracting topic: {e}")
            return {"topic": "Error analyzing topic", "confidence": 0}

    async def _generate_summary(self, text: str) -> List[str]:
        """Генерация краткой справки"""
        try:
            if settings.LLM_PROVIDER == "openai":
                response = await self.llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SUMMARY_PROMPT},
                        {"role": "user", "content": text[-3000:]}
                    ],
                    temperature=0.4,
                    max_tokens=500,
                    response_format={"type": "json_object"}  # <-- ДОБАВЬТЕ
                )
                content = response.choices[0].message.content
            elif settings.LLM_PROVIDER == "anthropic":
                response = await self.llm_client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=500,
                    temperature=0.4,
                    system=SUMMARY_PROMPT,
                    messages=[{"role": "user", "content": text[-3000:]}]
                )
                content = response.content[0].text
            else:
                # Мок-ответ для тестирования
                return ["Summary point 1", "Summary point 2", "Summary point 3"]

            # Парсинг JSON
            try:
                result = json.loads(content)
                if isinstance(result, dict) and "summary_points" in result:
                    return result["summary_points"][:8]
                elif isinstance(result, list):
                    return result[:8]
                else:
                    return ["Could not parse summary"]
            except json.JSONDecodeError:
                # Ручной парсинг
                summary_points = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('```'):
                        if line.startswith(('-', '•', '*', '1.', '2.', '3.', '4.', '5.')):
                            line = line[1:].strip() if line[0] in '-•*' else line[2:].strip()
                        if line:
                            summary_points.append(line)

                return summary_points[:8]

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ["Error generating summary"]

    async def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Извлечение ключевых сущностей"""
        try:
            if settings.LLM_PROVIDER == "openai":
                response = await self.llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
                        {"role": "user", "content": text[-3000:]}
                    ],
                    temperature=0.2,
                    max_tokens=400
                )
                content = response.choices[0].message.content
            else:  # anthropic
                response = await self.llm_client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=400,
                    temperature=0.2,
                    system=ENTITY_EXTRACTION_PROMPT,
                    messages=[{"role": "user", "content": text[-3000:]}]
                )
                content = response.content[0].text

            # Парсинг JSON
            try:
                entities = json.loads(content)
                if isinstance(entities, list):
                    return entities[:15]  # Ограничиваем количеством
                return []
            except json.JSONDecodeError:
                # Ручной парсинг если не JSON
                return self._parse_entities_from_text(content)

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    async def _extract_decisions(self, text: str) -> List[Dict[str, Any]]:
        """Извлечение решений и next steps"""
        try:
            if settings.LLM_PROVIDER == "openai":
                response = await self.llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": DECISION_EXTRACTION_PROMPT},
                        {"role": "user", "content": text[-3000:]}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                content = response.choices[0].message.content
            else:  # anthropic
                response = await self.llm_client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=300,
                    temperature=0.3,
                    system=DECISION_EXTRACTION_PROMPT,
                    messages=[{"role": "user", "content": text[-3000:]}]
                )
                content = response.content[0].text

            try:
                decisions = json.loads(content)
                if isinstance(decisions, list):
                    return decisions[:5]
                return []
            except json.JSONDecodeError:
                return self._parse_list_from_text(content, "decision")

        except Exception as e:
            logger.error(f"Error extracting decisions: {e}")
            return []

    async def _extract_open_questions(self, text: str) -> List[str]:
        """Извлечение открытых вопросов"""
        try:
            if settings.LLM_PROVIDER == "openai":
                response = await self.llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": QUESTION_EXTRACTION_PROMPT},
                        {"role": "user", "content": text[-3000:]}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                content = response.choices[0].message.content
            else:  # anthropic
                response = await self.llm_client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=300,
                    temperature=0.3,
                    system=QUESTION_EXTRACTION_PROMPT,
                    messages=[{"role": "user", "content": text[-3000:]}]
                )
                content = response.content[0].text

            try:
                questions = json.loads(content)
                if isinstance(questions, list):
                    return questions[:5]
                return []
            except json.JSONDecodeError:
                return self._parse_list_from_text(content, "question")

        except Exception as e:
            logger.error(f"Error extracting questions: {e}")
            return []

    def _parse_entities_from_text(self, text: str) -> List[Dict[str, str]]:
        """Парсинг сущностей из текстового ответа"""
        entities = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('```'):
                continue

            # Пытаемся извлечь тип и название
            parts = line.split(':')
            if len(parts) >= 2:
                entity_type = parts[0].strip().lower()
                entity_name = ':'.join(parts[1:]).strip()
                entities.append({
                    "name": entity_name,
                    "type": entity_type
                })
            elif line:
                entities.append({
                    "name": line,
                    "type": "unknown"
                })

        return entities[:15]

    def _parse_list_from_text(self, text: str, item_type: str) -> List:
        """Парсинг списка из текстового ответа"""
        items = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('```'):
                continue

            # Убираем маркеры списка
            if line.startswith(('-', '•', '*', '1.', '2.', '3.', '4.', '5.')):
                line = line[1:].strip() if line[0] in '-•*' else line[2:].strip()

            if line:
                if item_type == "decision":
                    items.append({"text": line, "completed": False})
                else:
                    items.append(line)

        return items[:5]  # Ограничиваем 5 пунктами

    def _safe_get_result(self, result, default):
        """Безопасное получение результата"""
        if isinstance(result, Exception):
            logger.error(f"Analysis task failed: {result}")
            return default
        return result

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Возвращает анализ по умолчанию при ошибке"""
        return {
            "current_topic": "Error in analysis",
            "summary": ["Unable to generate summary due to error"],
            "entities": [],
            "decisions": [],
            "open_questions": ["Why is the analysis not working?"],
            "analysis_timestamp": datetime.now().isoformat()
        }