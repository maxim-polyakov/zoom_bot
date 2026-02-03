"""
Вспомогательные функции для проекта Zoom агента
Утилиты для работы со временем, текстом, файлами и API
"""

import re
import json
import hashlib
import random
import string
import time
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import html
import urllib.parse
from urllib.parse import urlparse, parse_qs, urlencode
import base64

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ СО ВРЕМЕНЕМ ====================

def get_current_timestamp() -> str:
    """Получение текущей метки времени в ISO формате"""
    return datetime.now(timezone.utc).isoformat()


def parse_iso_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Парсинг ISO строки времени"""
    try:
        # Пробуем разные форматы ISO
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d"
        ]:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        # Если не нашли подходящий формат
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return None

    except Exception as e:
        logger.error(f"Error parsing timestamp {timestamp_str}: {e}")
        return None


def format_duration(seconds: float) -> str:
    """Форматирование продолжительности в читаемый формат"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_time_ago(timestamp: datetime) -> str:
    """Форматирование времени в формате 'X времени назад'"""
    now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    diff = now - timestamp

    if diff.days > 365:
        years = diff.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
    elif diff.days > 30:
        months = diff.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "just now"


def is_recent(timestamp: datetime, hours: int = 24) -> bool:
    """Проверка, является ли время недавним"""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    return (now - timestamp) < timedelta(hours=hours)


# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С ТЕКСТОМ ====================

def clean_text(text: str, max_length: int = None) -> str:
    """Очистка текста от лишних пробелов и символов"""
    if not text:
        return ""

    # Убираем HTML теги
    text = re.sub(r'<[^>]+>', '', text)

    # Убираем лишние пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text)

    # Убираем начальные и конечные пробелы
    text = text.strip()

    # Экранируем HTML символы
    text = html.escape(text)

    # Ограничиваем длину
    if max_length and len(text) > max_length:
        text = text[:max_length] + "..."

    return text


def extract_first_sentence(text: str, max_length: int = 200) -> str:
    """Извлечение первого предложения из текста"""
    if not text:
        return ""

    # Находим конец первого предложения
    sentence_endings = ['.', '!', '?', '。', '！', '？']

    for end_char in sentence_endings:
        idx = text.find(end_char)
        if idx != -1:
            sentence = text[:idx + 1]

            # Очищаем и ограничиваем длину
            sentence = clean_text(sentence)
            if len(sentence) > max_length:
                sentence = sentence[:max_length] + "..."

            return sentence

    # Если не нашли конец предложения
    text = clean_text(text)
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text


def count_words(text: str) -> int:
    """Подсчет слов в тексте"""
    if not text:
        return 0

    words = re.findall(r'\b\w+\b', text)
    return len(words)


def extract_hashtags(text: str) -> List[str]:
    """Извлечение хэштегов из текста"""
    hashtags = re.findall(r'#(\w+)', text, re.IGNORECASE)
    return list(set(hashtags))  # Убираем дубликаты


def extract_urls(text: str) -> List[str]:
    """Извлечение URL из текста"""
    # Простой regex для URL
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    urls = re.findall(url_pattern, text)

    # Нормализуем URL
    normalized_urls = []
    for url in urls:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        normalized_urls.append(url)

    return normalized_urls


def generate_summary(text: str, max_sentences: int = 3) -> str:
    """Генерация краткого содержания текста"""
    if not text:
        return ""

    # Разделяем на предложения
    sentences = re.split(r'[.!?]+', text)

    # Фильтруем пустые предложения и короткие
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    # Берем первые N предложений
    summary_sentences = sentences[:max_sentences]

    # Объединяем
    summary = '. '.join(summary_sentences)
    if summary and not summary.endswith('.'):
        summary += '.'

    return clean_text(summary)


# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С ФАЙЛАМИ ====================

def ensure_directory(path: str) -> Path:
    """Создание директории, если она не существует"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> bool:
    """Сохранение данных в JSON файл"""
    try:
        path = Path(filepath)
        ensure_directory(path.parent)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

        logger.debug(f"JSON saved to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        return False


def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """Загрузка данных из JSON файла"""
    try:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"JSON file not found: {filepath}")
            return None

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.debug(f"JSON loaded from {filepath}")
        return data

    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        return None


def append_to_file(filepath: str, content: str):
    """Добавление контента в файл"""
    try:
        path = Path(filepath)
        ensure_directory(path.parent)

        with open(path, 'a', encoding='utf-8') as f:
            f.write(content + '\n')

    except Exception as e:
        logger.error(f"Error appending to file {filepath}: {e}")


def read_last_lines(filepath: str, num_lines: int = 10) -> List[str]:
    """Чтение последних N строк файла"""
    try:
        path = Path(filepath)
        if not path.exists():
            return []

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        return lines[-num_lines:] if lines else []

    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return []


# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С API ====================

def create_headers(
        content_type: str = "application/json",
        auth_token: str = None,
        custom_headers: Dict[str, str] = None
) -> Dict[str, str]:
    """Создание заголовков для HTTP запросов"""
    headers = {
        "Content-Type": content_type,
        "User-Agent": f"ZoomAgentBot/1.0 ({settings.BOT_NAME})"
    }

    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    if custom_headers:
        headers.update(custom_headers)

    return headers


def build_url(
        base_url: str,
        params: Dict[str, Any] = None,
        path: str = ""
) -> str:
    """Построение URL с параметрами"""
    url = base_url.rstrip('/')

    if path:
        url += '/' + path.lstrip('/')

    if params:
        # Фильтруем None значения
        filtered_params = {k: v for k, v in params.items() if v is not None}

        # Кодируем параметры
        encoded_params = urlencode(filtered_params, doseq=True)
        url += '?' + encoded_params

    return url


def parse_query_params(url: str) -> Dict[str, List[str]]:
    """Парсинг параметров запроса из URL"""
    parsed = urlparse(url)
    return parse_qs(parsed.query)


def validate_response(response_data: Dict[str, Any], required_fields: List[str] = None) -> bool:
    """Валидация ответа API"""
    if not isinstance(response_data, dict):
        logger.warning("Response is not a dictionary")
        return False

    if "error" in response_data:
        logger.warning(f"API error: {response_data.get('error')}")
        return False

    if required_fields:
        for field in required_fields:
            if field not in response_data:
                logger.warning(f"Missing required field: {field}")
                return False

    return True


# ==================== ФУНКЦИИ БЕЗОПАСНОСТИ ====================

def sanitize_input(input_str: str, max_length: int = 500) -> str:
    """Санобработка пользовательского ввода"""
    if not input_str:
        return ""

    # Ограничиваем длину
    input_str = input_str[:max_length]

    # Убираем опасные символы
    input_str = html.escape(input_str)

    # Убираем лишние пробелы
    input_str = re.sub(r'\s+', ' ', input_str).strip()

    return input_str


def generate_random_string(length: int = 16) -> str:
    """Генерация случайной строки"""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def generate_hash(data: str, algorithm: str = "sha256") -> str:
    """Генерация хэша для данных"""
    hash_func = getattr(hashlib, algorithm, hashlib.sha256)
    return hash_func(data.encode()).hexdigest()


def mask_sensitive_data(data: Dict[str, Any], sensitive_fields: List[str] = None) -> Dict[str, Any]:
    """Маскировка чувствительных данных"""
    if sensitive_fields is None:
        sensitive_fields = ["password", "token", "api_key", "secret", "key"]

    masked_data = data.copy()

    for key, value in masked_data.items():
        if isinstance(value, dict):
            masked_data[key] = mask_sensitive_data(value, sensitive_fields)
        elif isinstance(value, str):
            for sensitive in sensitive_fields:
                if sensitive.lower() in key.lower():
                    masked_data[key] = "***MASKED***"
                    break

    return masked_data


# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С АСИНХРОННОСТЬЮ ====================

async def run_with_timeout(coro, timeout: float, default_value=None):
    """Выполнение корутины с таймаутом"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default_value


def retry_async(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0,
                exceptions: tuple = (Exception,)):
    """
    Декоратор для повторных попыток выполнения асинхронной функции

    Пример использования:
    @retry_async(max_retries=3, delay=1.0)
    async def my_function():
        ...
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        break

                    # Экспоненциальная задержка
                    wait_time = delay * (backoff_factor ** attempt)
                    logger = setup_logger(__name__)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )

                    await asyncio.sleep(wait_time)

            # Если все попытки не удались
            raise last_exception or Exception(f"All {max_retries} retry attempts failed for {func.__name__}")

        return wrapper

    return decorator


def run_in_background(coro):
    """Запуск корутины в фоновом режиме"""
    task = asyncio.create_task(coro)

    # Добавляем обработку ошибок
    def handle_task_result(task: asyncio.Task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass  # Задача была отменена
        except Exception as e:
            logger.error(f"Background task failed: {e}")

    task.add_done_callback(handle_task_result)
    return task


# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С URL И ССЫЛКАМИ ====================

def is_valid_url(url: str) -> bool:
    """Проверка валидности URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def normalize_url(url: str) -> str:
    """Нормализация URL"""
    if not url:
        return ""

    # Добавляем протокол если отсутствует
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Убираем слеш в конце
    url = url.rstrip('/')

    return url


def extract_domain(url: str) -> str:
    """Извлечение домена из URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return ""


def shorten_url(url: str, max_length: int = 50) -> str:
    """Сокращение URL для отображения"""
    if len(url) <= max_length:
        return url

    # Оставляем начало и конец URL
    part_length = max_length // 2 - 3
    start = url[:part_length]
    end = url[-part_length:]

    return f"{start}...{end}"


# ==================== УТИЛИТЫ ДЛЯ ЛОГИРОВАНИЯ И ОТЛАДКИ ====================

def format_dict_for_log(data: Dict[str, Any], max_depth: int = 3) -> str:
    """Форматирование словаря для логирования"""

    def format_value(value, depth):
        if depth >= max_depth:
            return "..."

        if isinstance(value, dict):
            items = []
            for k, v in value.items():
                formatted = format_value(v, depth + 1)
                items.append(f"{k}: {formatted}")
            return "{" + ", ".join(items) + "}"
        elif isinstance(value, list):
            if len(value) > 3:
                return f"[{format_value(value[0], depth + 1)}, ... ({len(value)} items)]"
            items = [format_value(v, depth + 1) for v in value[:3]]
            return "[" + ", ".join(items) + "]"
        elif isinstance(value, str):
            if len(value) > 100:
                return f'"{value[:100]}..."'
            return f'"{value}"'
        else:
            return str(value)

    return format_value(data, 0)


def benchmark(func):
    """Декоратор для бенчмаркинга функций"""
    if asyncio.iscoroutinefunction(func):
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            end = time.time()
            logger.debug(f"{func.__name__} took {end - start:.3f}s")
            return result

        return async_wrapper
    else:
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.debug(f"{func.__name__} took {end - start:.3f}s")
            return result

        return sync_wrapper


# ==================== СПЕЦИФИЧНЫЕ ДЛЯ ZOOM ФУНКЦИИ ====================

def extract_zoom_meeting_id(url_or_id: str) -> Optional[str]:
    """Извлечение ID встречи Zoom из URL или строки"""
    if not url_or_id:
        return None

    # Если это уже чистый ID (только цифры)
    if re.match(r'^\d+$', url_or_id):
        return url_or_id

    # Пробуем извлечь из URL
    patterns = [
        r'zoom\.us/j/(\d+)',
        r'zoom\.us/my/(\w+)',
        r'meeting_id=(\d+)',
        r'/(\d{9,11})(?:\?|$|/)'
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    return None


def validate_zoom_settings(settings: Dict[str, Any]) -> List[str]:
    """Валидация настроек Zoom"""
    errors = []

    required_fields = ["ZOOM_MEETING_URL", "BOT_EMAIL"]

    for field in required_fields:
        if not settings.get(field):
            errors.append(f"Missing required field: {field}")

    # Проверка формата URL
    zoom_url = settings.get("ZOOM_MEETING_URL", "")
    if zoom_url and not is_valid_url(zoom_url):
        errors.append(f"Invalid Zoom URL: {zoom_url}")

    # Проверка email
    email = settings.get("BOT_EMAIL", "")
    if email and not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        errors.append(f"Invalid email format: {email}")

    return errors


def get_zoom_meeting_info(meeting_id: str) -> Dict[str, Any]:
    """Получение информации о встрече Zoom (заглушка)"""
    # В реальной реализации здесь будет API вызов к Zoom
    return {
        "meeting_id": meeting_id,
        "topic": "Zoom Meeting",
        "start_time": get_current_timestamp(),
        "duration": 60,  # минуты
        "timezone": "UTC",
        "join_url": f"https://zoom.us/j/{meeting_id}"
    }


# ==================== ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ ====================

class RateLimiter:
    """Ограничитель частоты запросов"""

    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
        self.lock = asyncio.Lock()

    async def wait(self):
        """Ожидание следующего разрешенного времени вызова"""
        async with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_call_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self.last_call_time = time.time()


class Cache:
    """Простой кэш в памяти"""

    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.default_ttl = default_ttl

    def set(self, key: str, value: Any, ttl: int = None):
        """Установка значения в кэш"""
        if ttl is None:
            ttl = self.default_ttl

        expire_time = time.time() + ttl
        self.cache[key] = (value, expire_time)

    def get(self, key: str) -> Any:
        """Получение значения из кэша"""
        if key not in self.cache:
            return None

        value, expire_time = self.cache[key]

        if time.time() > expire_time:
            del self.cache[key]
            return None

        return value

    def delete(self, key: str):
        """Удаление значения из кэша"""
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        """Очистка кэша"""
        self.cache.clear()

    def cleanup(self):
        """Очистка просроченных записей"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expire_time) in self.cache.items()
            if current_time > expire_time
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)