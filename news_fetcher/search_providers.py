"""
Поставщики поиска новостей для агента
Поддержка различных API: Google, Bing, SerpAPI, NewsAPI
"""

import asyncio
import aiohttp
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus, urlencode
import time

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SearchProvider:
    """Базовый класс для всех провайдеров поиска"""

    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_ttl = 300  # 5 минут в секундах
        self.request_delay = 1  # Задержка между запросами
        self.last_request_time = 0

    async def search(self, query: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """Основной метод поиска"""
        raise NotImplementedError

    async def _make_request(self, url: str, params: Dict = None, headers: Dict = None) -> Dict:
        """Выполнение HTTP запроса с задержкой"""
        # Соблюдаем задержку между запросами
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)

        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                self.last_request_time = time.time()

                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        return await response.json()
                    else:
                        text = await response.text()
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return {"text": text}
                else:
                    logger.error(f"Request failed: {response.status}")
                    return {"error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"Request error: {e}")
            return {"error": str(e)}

    def _generate_cache_key(self, query: str, provider: str) -> str:
        """Генерация ключа кэша"""
        key_str = f"{provider}:{query}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Проверка кэша"""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
            else:
                del self.cache[cache_key]
        return None

    def _store_cache(self, cache_key: str, data: List[Dict]):
        """Сохранение в кэш"""
        self.cache[cache_key] = (data, time.time())

    def _parse_news_item(self, item: Dict) -> Dict[str, Any]:
        """Парсинг новости в стандартный формат"""
        raise NotImplementedError

    async def cleanup(self):
        """Очистка ресурсов"""
        if self.session:
            await self.session.close()


class GoogleSearchProvider(SearchProvider):
    """Провайдер для Google Search API"""

    async def search(self, query: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """Поиск через Google Custom Search API"""
        cache_key = self._generate_cache_key(query, "google")
        cached = self._check_cache(cache_key)
        if cached:
            logger.info(f"Returning cached results for: {query}")
            return cached

        if not settings.SEARCH_API_KEY:
            logger.error("Google Search API key not configured")
            return []

        # Формируем запрос с учетом типа сущности
        search_query = self._build_query(query, entity_type)

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": settings.SEARCH_API_KEY,
            "cx": self._get_search_engine_id(),  # Search Engine ID
            "q": search_query,
            "num": 10,
            "sort": "date",  # Сортировка по дате
            "dateRestrict": f"d{settings.NEWS_FRESHNESS_DAYS}",
            "safe": "active",
            "lr": "lang_en|lang_ru"
        }

        logger.info(f"Searching Google for: {search_query}")
        results = await self._make_request(url, params=params)

        news_items = self._parse_results(results)
        self._store_cache(cache_key, news_items)

        return news_items

    def _get_search_engine_id(self) -> str:
        """Получение ID поискового движка"""
        # Можно хранить в настройках или использовать дефолтный
        engine_id = getattr(settings, "GOOGLE_SEARCH_ENGINE_ID", "")
        if not engine_id:
            # По умолчанию используем общий поиск
            engine_id = "d4c9a62d3b9e343a5"
        return engine_id

    def _build_query(self, query: str, entity_type: str = None) -> str:
        """Построение поискового запроса"""
        base_query = query

        # Добавляем ключевые слова в зависимости от типа
        type_keywords = {
            "medicine": "news research study clinical trial",
            "company": "news earnings report business",
            "person": "interview statement biography",
            "project": "update progress development",
            "disease": "research treatment study",
            "metric": "analysis report data"
        }

        if entity_type in type_keywords:
            base_query += f" {type_keywords[entity_type]}"

        # Добавляем ограничение по новостям
        base_query += " news"

        return base_query

    def _parse_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Парсинг результатов Google Search"""
        news_items = []

        if "items" not in results:
            return news_items

        for item in results["items"]:
            try:
                # Проверяем, что это новость
                if not self._is_news_item(item):
                    continue

                news_item = {
                    "title": item.get("title", ""),
                    "source": self._extract_source(item),
                    "date": self._extract_date(item),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "freshness": self._calculate_freshness(item),
                    "thumbnail": item.get("pagemap", {}).get("cse_thumbnail", [{}])[0].get("src", "")
                }

                news_items.append(news_item)

            except Exception as e:
                logger.error(f"Error parsing Google result: {e}")
                continue

        return news_items[:settings.MAX_NEWS_PER_ENTITY]

    def _is_news_item(self, item: Dict) -> bool:
        """Проверка, является ли результат новостью"""
        # Проверяем по формату URL или наличию новостных доменов
        url = item.get("link", "").lower()
        news_domains = [
            "news.", "reuters", "bloomberg", "cnn", "bbc",
            "nytimes", "wsj", "forbes", "techcrunch"
        ]

        return any(domain in url for domain in news_domains)

    def _extract_source(self, item: Dict) -> str:
        """Извлечение источника из результата"""
        display_link = item.get("displayLink", "")
        if display_link:
            return display_link

        # Пытаемся извлечь из URL
        url = item.get("link", "")
        if "://" in url:
            return url.split("://")[1].split("/")[0]

        return "Unknown source"

    def _extract_date(self, item: Dict) -> str:
        """Извлечение даты из результата"""
        # Google иногда возвращает дату в snippet
        snippet = item.get("snippet", "")

        # Ищем паттерны даты
        import re
        date_patterns = [
            r"(\d{1,2}\s+\w+\s+\d{4})",
            r"(\w+\s+\d{1,2},\s+\d{4})",
            r"(\d{4}-\d{2}-\d{2})"
        ]

        for pattern in date_patterns:
            match = re.search(pattern, snippet)
            if match:
                return match.group(1)

        return datetime.now().strftime("%Y-%m-%d")

    def _calculate_freshness(self, item: Dict) -> str:
        """Расчет свежести новости"""
        date_str = self._extract_date(item)
        return self._parse_freshness(date_str)


class SerpApiProvider(SearchProvider):
    """Провайдер для SerpAPI"""

    async def search(self, query: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """Поиск через SerpAPI"""
        cache_key = self._generate_cache_key(query, "serpapi")
        cached = self._check_cache(cache_key)
        if cached:
            logger.info(f"Returning cached results for: {query}")
            return cached

        if not settings.SEARCH_API_KEY:
            logger.error("SerpAPI key not configured")
            return []

        search_query = self._build_query(query, entity_type)

        url = "https://serpapi.com/search"
        params = {
            "q": search_query,
            "api_key": settings.SEARCH_API_KEY,
            "engine": "google",
            "tbm": "nws",  # News search
            "num": 10,
            "hl": "en",
            "gl": "us",
            "tbs": f"qdr:d{settings.NEWS_FRESHNESS_DAYS}"  # Last N days
        }

        logger.info(f"Searching via SerpAPI for: {search_query}")
        results = await self._make_request(url, params=params)

        news_items = self._parse_results(results)
        self._store_cache(cache_key, news_items)

        return news_items

    def _build_query(self, query: str, entity_type: str = None) -> str:
        """Построение запроса для SerpAPI"""
        base_query = query

        # SerpAPI хорошо работает с прямыми запросами
        if entity_type:
            base_query += f" {entity_type}"

        return base_query

    def _parse_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Парсинг результатов SerpAPI"""
        news_items = []

        if "news_results" not in results:
            return news_items

        for item in results["news_results"]:
            try:
                news_item = {
                    "title": item.get("title", ""),
                    "source": item.get("source", {}).get("name", "Unknown"),
                    "date": item.get("date", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "freshness": self._parse_freshness(item.get("date", "")),
                    "thumbnail": item.get("thumbnail", "")
                }

                news_items.append(news_item)

            except Exception as e:
                logger.error(f"Error parsing SerpAPI result: {e}")
                continue

        return news_items[:settings.MAX_NEWS_PER_ENTITY]


class BingSearchProvider(SearchProvider):
    """Провайдер для Bing News Search API"""

    async def search(self, query: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """Поиск через Bing News API"""
        cache_key = self._generate_cache_key(query, "bing")
        cached = self._check_cache(cache_key)
        if cached:
            logger.info(f"Returning cached results for: {query}")
            return cached

        if not settings.SEARCH_API_KEY:
            logger.error("Bing API key not configured")
            return []

        search_query = self._build_query(query, entity_type)

        url = "https://api.bing.microsoft.com/v7.0/news/search"
        headers = {
            "Ocp-Apim-Subscription-Key": settings.SEARCH_API_KEY
        }
        params = {
            "q": search_query,
            "count": 10,
            "mkt": "en-US",
            "freshness": f"Day{settings.NEWS_FRESHNESS_DAYS}",
            "sortBy": "Date"
        }

        logger.info(f"Searching Bing for: {search_query}")
        results = await self._make_request(url, params=params, headers=headers)

        news_items = self._parse_results(results)
        self._store_cache(cache_key, news_items)

        return news_items

    def _build_query(self, query: str, entity_type: str = None) -> str:
        """Построение запроса для Bing"""
        # Bing хорошо обрабатывает сложные запросы
        base_query = f'"{query}"'

        if entity_type:
            base_query += f" {entity_type}"

        return base_query

    def _parse_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Парсинг результатов Bing"""
        news_items = []

        if "value" not in results:
            return news_items

        for item in results["value"]:
            try:
                news_item = {
                    "title": item.get("name", ""),
                    "source": item.get("provider", [{}])[0].get("name", "Unknown"),
                    "date": item.get("datePublished", ""),
                    "snippet": item.get("description", ""),
                    "url": item.get("url", ""),
                    "freshness": self._parse_freshness(item.get("datePublished", "")),
                    "thumbnail": item.get("image", {}).get("thumbnail", {}).get("contentUrl", "")
                }

                news_items.append(news_item)

            except Exception as e:
                logger.error(f"Error parsing Bing result: {e}")
                continue

        return news_items[:settings.MAX_NEWS_PER_ENTITY]


class NewsApiProvider(SearchProvider):
    """Провайдер для NewsAPI"""

    async def search(self, query: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """Поиск через NewsAPI"""
        cache_key = self._generate_cache_key(query, "newsapi")
        cached = self._check_cache(cache_key)
        if cached:
            logger.info(f"Returning cached results for: {query}")
            return cached

        if not settings.SEARCH_API_KEY:
            logger.error("NewsAPI key not configured")
            return []

        search_query = self._build_query(query, entity_type)

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": search_query,
            "apiKey": settings.SEARCH_API_KEY,
            "pageSize": 10,
            "language": "en",
            "sortBy": "publishedAt",
            "from": (datetime.now() - timedelta(days=settings.NEWS_FRESHNESS_DAYS)).strftime("%Y-%m-%d")
        }

        logger.info(f"Searching NewsAPI for: {search_query}")
        results = await self._make_request(url, params=params)

        news_items = self._parse_results(results)
        self._store_cache(cache_key, news_items)

        return news_items

    def _build_query(self, query: str, entity_type: str = None) -> str:
        """Построение запроса для NewsAPI"""
        base_query = query

        # NewsAPI поддерживает расширенный синтаксис
        if entity_type:
            base_query += f" AND ({entity_type})"

        return base_query

    def _parse_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Парсинг результатов NewsAPI"""
        news_items = []

        if results.get("status") != "ok" or "articles" not in results:
            return news_items

        for article in results["articles"]:
            try:
                news_item = {
                    "title": article.get("title", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "date": article.get("publishedAt", ""),
                    "snippet": article.get("description", ""),
                    "url": article.get("url", ""),
                    "freshness": self._parse_freshness(article.get("publishedAt", "")),
                    "thumbnail": article.get("urlToImage", "")
                }

                news_items.append(news_item)

            except Exception as e:
                logger.error(f"Error parsing NewsAPI article: {e}")
                continue

        return news_items[:settings.MAX_NEWS_PER_ENTITY]


class MockSearchProvider(SearchProvider):
    """Мок-провайдер для тестирования"""

    async def search(self, query: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """Мок-поиск новостей"""
        logger.info(f"Mock search for: {query} ({entity_type})")

        # Имитируем задержку сети
        await asyncio.sleep(0.5)

        mock_news = [
            {
                "title": f"Latest research on {query} shows promising results",
                "source": "Scientific Journal",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "snippet": f"A recent study published in Nature explores new applications of {query} in modern medicine. Researchers found significant improvements in key metrics.",
                "url": f"https://example.com/research/{query.replace(' ', '-').lower()}",
                "freshness": "Today",
                "thumbnail": "https://example.com/thumb1.jpg"
            },
            {
                "title": f"{query}: Industry trends and market analysis",
                "source": "Business Review",
                "date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                "snippet": f"Market analysts report growing interest in {query} technologies. Investment has increased by 45% in the last quarter.",
                "url": f"https://business.example.com/analysis/{query.replace(' ', '_')}",
                "freshness": "2 days ago",
                "thumbnail": "https://example.com/thumb2.jpg"
            },
            {
                "title": f"Expert interview: The future of {query}",
                "source": "Tech Insights",
                "date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                "snippet": f"Leading expert Dr. Smith discusses the potential impact of {query} on future technologies and ethical considerations.",
                "url": f"https://tech.example.com/interview/{query.replace(' ', '-')}",
                "freshness": "5 days ago",
                "thumbnail": "https://example.com/thumb3.jpg"
            },
            {
                "title": f"Regulatory update for {query} applications",
                "source": "Regulatory News",
                "date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                "snippet": f"New guidelines have been issued for the use of {query} in clinical settings. The changes aim to improve safety protocols.",
                "url": f"https://regulatory.example.com/updates/{query.replace(' ', '-')}",
                "freshness": "1 week ago",
                "thumbnail": ""
            },
            {
                "title": f"Comparative analysis: {query} vs traditional approaches",
                "source": "Research Digest",
                "date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                "snippet": f"A comprehensive review compares the effectiveness of {query} with established methods, highlighting advantages and limitations.",
                "url": f"https://research.example.com/comparison/{query.replace(' ', '-')}",
                "freshness": "10 days ago",
                "thumbnail": "https://example.com/thumb5.jpg"
            }
        ]

        return mock_news


class SearchProviderFactory:
    """Фабрика для создания провайдеров поиска"""

    @staticmethod
    def create_provider(provider_name: str = None) -> SearchProvider:
        """Создание провайдера поиска"""
        if not provider_name:
            provider_name = settings.SEARCH_PROVIDER

        providers = {
            "google": GoogleSearchProvider,
            "serpapi": SerpApiProvider,
            "bing": BingSearchProvider,
            "newsapi": NewsApiProvider,
            "mock": MockSearchProvider
        }

        provider_class = providers.get(provider_name.lower())
        if not provider_class:
            logger.warning(f"Provider {provider_name} not found, using mock")
            provider_class = MockSearchProvider

        return provider_class()

    @staticmethod
    def get_available_providers() -> List[str]:
        """Получение списка доступных провайдеров"""
        return ["google", "serpapi", "bing", "newsapi", "mock"]


# Утилитарные функции для всех провайдеров

def _parse_freshness(date_str: str) -> str:
    """Универсальная функция расчета свежести"""
    if not date_str:
        return "unknown"

    try:
        # Пытаемся распарсить различные форматы дат
        date_formats = [
            "%Y-%m-%dT%H:%M:%SZ",  # ISO format
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d %b %Y",
            "%b %d, %Y",
            "%d/%m/%Y",
            "%m/%d/%Y"
        ]

        parsed_date = None
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue

        if not parsed_date:
            # Пытаемся найти дату в строке
            import re
            patterns = [
                r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})",
                r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})",
                r"(\w+)\s+(\d{1,2}),\s+(\d{4})"
            ]

            for pattern in patterns:
                match = re.search(pattern, date_str)
                if match:
                    # Упрощенный парсинг
                    parsed_date = datetime.now() - timedelta(days=1)
                    break

        if parsed_date:
            delta = datetime.now() - parsed_date

            if delta.days == 0:
                if delta.seconds < 3600:
                    return "just now"
                elif delta.seconds < 7200:
                    return "1 hour ago"
                else:
                    hours = delta.seconds // 3600
                    return f"{hours} hours ago"
            elif delta.days == 1:
                return "yesterday"
            elif delta.days < 7:
                return f"{delta.days} days ago"
            elif delta.days < 30:
                weeks = delta.days // 7
                return f"{weeks} weeks ago"
            else:
                months = delta.days // 30
                return f"{months} months ago"

        return "recent"

    except Exception as e:
        logger.error(f"Error parsing freshness: {e}")
        return "unknown"


def _filter_duplicates(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Фильтрация дубликатов новостей"""
    seen_urls = set()
    unique_items = []

    for item in news_items:
        url = item.get("url", "")
        title = item.get("title", "").lower()

        # Создаем ключ для проверки дубликатов
        url_key = url
        title_key = hashlib.md5(title.encode()).hexdigest()

        if url_key not in seen_urls and title_key not in seen_urls:
            seen_urls.add(url_key)
            seen_urls.add(title_key)
            unique_items.append(item)

    return unique_items


def _filter_by_freshness(news_items: List[Dict[str, Any]], max_days: int = 30) -> List[Dict[str, Any]]:
    """Фильтрация новостей по свежести"""
    filtered_items = []

    for item in news_items:
        freshness = item.get("freshness", "").lower()

        # Пропускаем старые новости
        if "month" in freshness or "year" in freshness:
            continue

        # Проверяем дни
        if "day" in freshness:
            try:
                days = int(freshness.split()[0])
                if days > max_days:
                    continue
            except (ValueError, IndexError):
                pass

        filtered_items.append(item)

    return filtered_items