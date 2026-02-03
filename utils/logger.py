"""
Настройка логирования для проекта Zoom агента
Кастомные форматеры, обработчики и утилиты для логирования
"""

import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
import asyncio
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from config import settings


class JSONFormatter(logging.Formatter):
    """Форматтер для JSON логов"""

    def format(self, record: logging.LogRecord) -> str:
        """Форматирование записи лога в JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Добавляем исключение если есть
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }

        # Добавляем дополнительные поля
        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Цветной форматтер для консоли"""

    # Цветовые коды ANSI
    COLORS = {
        'DEBUG': '\033[0;36m',      # Cyan
        'INFO': '\033[0;32m',       # Green
        'WARNING': '\033[0;33m',    # Yellow
        'ERROR': '\033[0;31m',      # Red
        'CRITICAL': '\033[1;31m',   # Bold Red
        'RESET': '\033[0m'          # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Форматирование с цветами"""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])

        # Форматируем сообщение
        message = super().format(record)

        # Добавляем цвет
        colored_message = f"{color}{message}{self.COLORS['RESET']}"

        return colored_message


def setup_logger(
    name: str = "zoom_agent",
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    json_format: bool = False
) -> logging.Logger:
    """
    Настройка логгера с файловым и консольным выводом

    Args:
        name: Имя логгера
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Логировать в файл
        log_to_console: Логировать в консоль
        json_format: Использовать JSON формат

    Returns:
        Настроенный логгер
    """
    if level is None:
        level = settings.DEBUG_LEVEL if hasattr(settings, 'DEBUG_LEVEL') else 'INFO'

    # Создаем логгер
    logger = logging.getLogger(name)

    # Устанавливаем уровень
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Убираем дублирующие обработчики
    if logger.handlers:
        return logger

    # Создаем форматтеры
    if json_format:
        formatter = JSONFormatter()
        console_formatter = JSONFormatter()
    else:
        detailed_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        simple_format = '%(asctime)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(detailed_format)
        console_formatter = ColoredFormatter(simple_format)

    # Файловый обработчик
    if log_to_file:
        try:
            # Создаем директорию для логов если не существует
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            # Файл с ротацией по размеру (10MB)
            file_handler = RotatingFileHandler(
                log_dir / f"{name}.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

            # Отдельный файл для ошибок
            error_handler = RotatingFileHandler(
                log_dir / f"{name}_error.log",
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3,
                encoding='utf-8'
            )
            error_handler.setFormatter(formatter)
            error_handler.setLevel(logging.ERROR)
            logger.addHandler(error_handler)

        except Exception as e:
            print(f"Failed to setup file logging: {e}")

    # Консольный обработчик
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
        logger.addHandler(console_handler)

    # Добавляем обработчик для отправки логов в Sentry (если настроено)
    if hasattr(settings, 'SENTRY_DSN') and settings.SENTRY_DSN:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration

            sentry_logging = LoggingIntegration(
                level=logging.INFO,           # Уровень для отправки в Sentry
                event_level=logging.ERROR     # Уровень для отправки событий
            )

            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                integrations=[sentry_logging],
                traces_sample_rate=1.0 if settings.DEBUG else 0.1
            )

            logger.info("Sentry logging initialized")

        except ImportError:
            logger.warning("Sentry SDK not installed, skipping Sentry integration")
        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")

    # Предотвращаем распространение логов на корневой логгер
    logger.propagate = False

    return logger


def log_exception(logger: logging.Logger, exception: Exception, context: Dict[str, Any] = None):
    """Логирование исключения с контекстом"""
    if context is None:
        context = {}

    exc_info = (
        type(exception),
        exception,
        exception.__traceback__
    )

    logger.error(
        f"Exception occurred: {str(exception)}",
        exc_info=exc_info,
        extra={"context": context}
    )


def log_with_context(logger: logging.Logger, level: str, message: str, **kwargs):
    """Логирование с дополнительным контекстом"""
    log_func = getattr(logger, level.lower(), logger.info)

    # Маскируем чувствительные данные
    safe_kwargs = {}
    sensitive_keys = ['password', 'token', 'secret', 'key', 'api_key']

    for key, value in kwargs.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            safe_kwargs[key] = '***MASKED***'
        else:
            safe_kwargs[key] = value

    extra_data = {"context": safe_kwargs}
    log_func(message, extra=extra_data)


def create_task_logger(task_name: str) -> logging.Logger:
    """Создание логгера для конкретной задачи"""
    return setup_logger(f"zoom_agent.task.{task_name}")


class LogCapture:
    """Контекстный менеджер для захвата логов"""

    def __init__(self, logger_name: str = "zoom_agent", level: str = "DEBUG"):
        self.logger_name = logger_name
        self.level = getattr(logging, level.upper())
        self.captured_records = []
        self.original_handlers = []
        self.capture_handler = None

    def __enter__(self):
        """Начало захвата логов"""
        logger = logging.getLogger(self.logger_name)

        # Сохраняем оригинальные обработчики
        self.original_handlers = logger.handlers.copy()

        # Создаем обработчик для захвата
        self.capture_handler = logging.Handler()
        self.capture_handler.setLevel(self.level)

        # Добавляем обработчик
        logger.addHandler(self.capture_handler)

        # Настраиваем фильтр для захвата
        def capture_filter(record):
            self.captured_records.append({
                "level": record.levelname,
                "message": record.getMessage(),
                "timestamp": datetime.now().isoformat(),
                "module": record.module,
                "line": record.lineno
            })
            return False  # Не пропускаем дальше

        self.capture_handler.addFilter(capture_filter)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Завершение захвата логов"""
        logger = logging.getLogger(self.logger_name)

        # Удаляем обработчик захвата
        if self.capture_handler in logger.handlers:
            logger.removeHandler(self.capture_handler)

        # Восстанавливаем оригинальные обработчики
        for handler in self.original_handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)

    def get_records(self) -> list:
        """Получение захваченных записей"""
        return self.captured_records.copy()

    def clear(self):
        """Очистка захваченных записей"""
        self.captured_records.clear()


# Глобальный логгер по умолчанию
def get_default_logger() -> logging.Logger:
    """Получение логгера по умолчанию"""
    return setup_logger("zoom_agent")


# Утилиты для мониторинга логов
def get_log_stats(log_file: str, hours: int = 24) -> Dict[str, Any]:
    """Получение статистики по лог файлу"""
    stats = {
        "total_entries": 0,
        "by_level": {},
        "by_hour": {},
        "errors": 0,
        "warnings": 0
    }

    try:
        path = Path(log_file)
        if not path.exists():
            return stats

        cutoff_time = datetime.now().timestamp() - (hours * 3600)

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Пробуем парсить как JSON
                    log_entry = json.loads(line.strip())

                    # Проверяем время
                    timestamp = log_entry.get('timestamp', '')
                    if timestamp:
                        entry_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
                        if entry_time < cutoff_time:
                            continue

                    # Собираем статистику
                    stats["total_entries"] += 1

                    level = log_entry.get('level', 'UNKNOWN')
                    stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

                    if level == 'ERROR':
                        stats["errors"] += 1
                    elif level == 'WARNING':
                        stats["warnings"] += 1

                    # Статистика по часам
                    if timestamp:
                        hour = timestamp[11:13]  # Извлекаем час
                        stats["by_hour"][hour] = stats["by_hour"].get(hour, 0) + 1

                except (json.JSONDecodeError, ValueError):
                    # Не JSON формат, пропускаем
                    continue

        return stats

    except Exception as e:
        print(f"Error reading log stats: {e}")
        return stats


def cleanup_old_logs(log_dir: str = "logs", days_to_keep: int = 30):
    """Очистка старых лог файлов"""
    try:
        dir_path = Path(log_dir)
        if not dir_path.exists():
            return

        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)

        for log_file in dir_path.glob("*.log*"):
            if log_file.is_file():
                # Проверяем время последнего изменения
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    print(f"Deleted old log file: {log_file}")

    except Exception as e:
        print(f"Error cleaning up old logs: {e}")


# Декоратор для логирования выполнения функций
def logged(func):
    """Декоратор для логирования входа/выхода из функции"""
    logger = setup_logger(f"zoom_agent.{func.__module__}.{func.__name__}")

    if asyncio.iscoroutinefunction(func):
        async def async_wrapper(*args, **kwargs):
            logger.debug(f"Entering {func.__name__}")
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        return async_wrapper
    else:
        def sync_wrapper(*args, **kwargs):
            logger.debug(f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        return sync_wrapper
