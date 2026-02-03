"""
Агент для автоматического поиска новостей по упомянутым сущностям
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class NewsAgent:
    def __init__(self):
        self.session = None
        self.searched_entities = set()
        self.news_cache = {}  # Кэш для дедупликации
        self.last_search_time = {}

    async def fetch_news_for_entity(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Поиск новостей для сущности
        """
        entity_name = entity.get('name', '')
        entity_type = entity.get('type', '')

        if not entity_name:
            return []

        # Проверяем, не искали ли мы недавно эту сущность
        cache_key = f"{entity_name}_{entity_type}"

        if cache_key in self.searched_entities:
            # Проверяем время последнего поиска
            if cache_key in self.last_search_time:
                time_since_last = datetime.now() - self.last_search_time[cache_key]
                if time_since_last < timedelta(minutes=30):
                    logger.info(f"Entity {entity_name} searched recently, skipping")
                    return self.news_cache.get(cache_key, [])

        logger.info(f"Searching news for entity: {entity_name} ({entity_type})")

        try:
            # Выбираем провайдера поиска
            if settings.SEARCH_PROVIDER == "google":
                news_items = await self._search_google_news(entity_name, entity_type)
            elif settings.SEARCH_PROVIDER == "serpapi":
                news_items = await self._search_serpapi(entity_name, entity_type)
            else:
                news_items = await self._search_bing_news(entity_name, entity_type)

            # Фильтруем по свежести
            filtered_news = self._filter_by_freshness(news_items)

            # Ограничиваем количество
            limited_news = filtered_news[:settings.MAX_NEWS_PER_ENTITY]

            # Кэшируем результаты
            self.searched_entities.add(cache_key)
            self.last_search_time[cache_key] = datetime.now()
            self.news_cache[cache_key] = limited_news

            logger.info(f"Found {len(limited_news)} news items for {entity_name}")

            return limited_news

        except Exception as e:
            logger.error(f"Error fetching news for {entity_name}: {e}")
            return []

    async def _search_google_news(self, entity: str, entity_type: str) -> List[Dict[str, Any]]:
        """Поиск через Google News (пример через SerpAPI)"""
        # Если есть SerpAPI ключ
        if settings.SEARCH_API_KEY and "serpapi" in settings.SEARCH_API_KEY:
            return await self._search_serpapi(entity, entity_type)

        # Имитация поиска (в реальном проекте нужна реальная интеграция)
        return await self._mock_news_search(entity, entity_type)

    async def _search_serpapi(self, entity: str, entity_type: str) -> List[Dict[str, Any]]:
        """Поиск через SerpAPI"""
        if not settings.SEARCH_API_KEY:
            logger.warning("No SerpAPI key provided, using mock data")
            return await self._mock_news_search(entity, entity_type)

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Формируем запрос
            query = self._build_search_query(entity, entity_type)
            encoded_query = quote_plus(query)

            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "tbm": "nws",  # News search
                "api_key": settings.SEARCH_API_KEY,
                "num": 10,
                "hl": "en",
                "gl": "us"
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_serpapi_results(data)
                else:
                    logger.error(f"SerpAPI error: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error with SerpAPI: {e}")
            return []

    async def _search_bing_news(self, entity: str, entity_type: str) -> List[Dict[str, Any]]:
        """Поиск через Bing News API"""
        # Заглушка - в реальном проекте нужна интеграция с Bing API
        return await self._mock_news_search(entity, entity_type)

    def _build_search_query(self, entity: str, entity_type: str) -> str:
        """Формирование поискового запроса"""
        base_query = entity

        # Добавляем контекст в зависимости от типа сущности
        type_context = {
            "medicine": "drug treatment clinical trial",
            "company": "business news earnings",
            "person": "interview statement biography",
            "project": "update development launch",
            "disease": "research treatment symptoms",
            "metric": "analysis report statistics"
        }

        if entity_type in type_context:
            base_query += f" {type_context[entity_type]}"

        # Добавляем ограничение по времени
        days_ago = (datetime.now() - timedelta(days=settings.NEWS_FRESHNESS_DAYS)).strftime("%m/%d/%Y")
        base_query += f" after:{days_ago}"

        return base_query

    def _parse_serpapi_results(self, data: Dict) -> List[Dict[str, Any]]:
        """Парсинг результатов SerpAPI"""
        news_items = []

        if "news_results" not in data:
            return news_items

        for item in data["news_results"][:10]:  # Берем первые 10
            try:
                news_item = {
                    "title": item.get("title", ""),
                    "source": item.get("source", "Unknown"),
                    "date": item.get("date", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "thumbnail": item.get("thumbnail", ""),
                    "freshness": self._calculate_freshness(item.get("date", ""))
                }

                # Проверяем дедупликацию
                if not self._is_duplicate_news(news_item):
                    news_items.append(news_item)

            except Exception as e:
                logger.error(f"Error parsing news item: {e}")
                continue

        return news_items

    def _filter_by_freshness(self, news_items: List[Dict]) -> List[Dict]:
        """Фильтрация новостей по свежести"""
        filtered = []

        for item in news_items:
            freshness = item.get("freshness", "unknown")

            # Проверяем, что новость за последние N дней
            if freshness == "unknown":
                # Пытаемся извлечь из даты
                date_str = item.get("date", "")
                if self._is_recent_date(date_str):
                    filtered.append(item)
            elif "day" in freshness:
                days = int(freshness.split()[0])
                if days <= settings.NEWS_FRESHNESS_DAYS:
                    filtered.append(item)
            else:
                # Если "just now" или "hours", добавляем
                filtered.append(item)

        return filtered

    def _calculate_freshness(self, date_str: str) -> str:
        """Расчет свежести новости"""
        if not date_str:
            return "unknown"

        try:
            # Парсинг различных форматов дат
            date_formats = [
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%B %d, %Y",
                "%d %B %Y"
            ]

            news_date = None
            for fmt in date_formats:
                try:
                    news_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue

            if not news_date:
                return "unknown"

            # Сравнение с текущей датой
            delta = datetime.now() - news_date

            if delta.days == 0:
                if delta.seconds < 3600:
                    return "just now"
                elif delta.seconds < 7200:
                    return "1 hour ago"
                else:
                    return f"{delta.seconds // 3600} hours ago"
            elif delta.days == 1:
                return "1 day ago"
            else:
                return f"{delta.days} days ago"

        except Exception:
            return "unknown"

    def _is_recent_date(self, date_str: str) -> bool:
        """Проверка, является ли дата недавней"""
        try:
            # Упрощенная проверка
            recent_keywords = [
                "hour", "day", "today", "yesterday",
                "minute", "just now", "recent"
            ]

            date_lower = date_str.lower()
            return any(keyword in date_lower for keyword in recent_keywords)

        except Exception:
            return False

    def _is_duplicate_news(self, news_item: Dict) -> bool:
        """Проверка на дубликаты новостей"""
        # Простая проверка по URL и заголовку
        url = news_item.get("url", "")
        title = news_item.get("title", "").lower()

        for cached in self.news_cache.values():
            for item in cached:
                if (item.get("url") == url or
                        item.get("title", "").lower() == title):
                    return True

        return False

    async def _mock_news_search(self, entity: str, entity_type: str) -> List[Dict[str, Any]]:
        """Мок-поиск новостей для демонстрации"""
        # Имитация задержки сети
        await asyncio.sleep(1)

        mock_news = [
            {
                "title": f"Latest developments in {entity}",
                "source": "Example News",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "snippet": f"Recent news and updates about {entity} show promising developments in the field.",
                "url": f"https://example.com/news/{entity.replace(' ', '-').lower()}",
                "freshness": "1 day ago"
            },
            {
                "title": f"{entity}: New research findings",
                "source": "Science Daily",
                "date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                "snippet": f"Researchers have published new findings related to {entity} in a recent study.",
                "url": f"https://sciencedaily.com/{entity.replace(' ', '_')}",
                "freshness": "2 days ago"
            },
            {
                "title": f"Industry analysis: {entity} market trends",
                "source": "Business Insider",
                "date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                "snippet": f"Market analysis shows interesting trends in the {entity} sector.",
                "url": f"https://businessinsider.com/analysis/{entity.replace(' ', '-')}",
                "freshness": "5 days ago"
            }
        ]

        return mock_news

    async def cleanup(self):
        """Очистка ресурсов"""
        if self.session:
            await self.session.close()