import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Zoom настройки
    ZOOM_MEETING_URL: str = os.getenv("ZOOM_MEETING_URL", "")
    ZOOM_MEETING_ID: str = os.getenv("ZOOM_MEETING_ID", "")
    ZOOM_PASSWORD: str = os.getenv("ZOOM_PASSWORD", "")

    # Настройки бота
    BOT_NAME: str = os.getenv("BOT_NAME", "Meeting Assistant")
    BOT_EMAIL: str = os.getenv("BOT_EMAIL", "")
    UPDATE_INTERVAL_SECONDS: int = int(os.getenv("UPDATE_INTERVAL", "60"))  # <-- ДОБАВЛЕНО

    # LLM настройки
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4-1106-preview")

    # Поиск новостей
    SEARCH_API_KEY: str = os.getenv("SEARCH_API_KEY", "")
    SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "google")
    NEWS_FRESHNESS_DAYS: int = int(os.getenv("NEWS_FRESHNESS_DAYS", "30"))
    MAX_NEWS_PER_ENTITY: int = int(os.getenv("MAX_NEWS_PER_ENTITY", "5"))

    # Веб-сервер
    DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "localhost")
    DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8080"))
    TRANSCRIPT_WEBHOOK_PORT: int = int(os.getenv("TRANSCRIPT_WEBHOOK_PORT", "8081"))

    # Пути
    CHROME_DRIVER_PATH: str = os.getenv("CHROME_DRIVER_PATH", "")
    SCREENSHOT_DIR: str = os.getenv("SCREENSHOT_DIR", "./screenshots")

    # Флаги
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    USE_MOCK_TRANSCRIPT: bool = os.getenv("USE_MOCK_TRANSCRIPT", "False").lower() == "true"

    class Config:
        env_file = ".env"
        extra = "ignore"  # <-- ДОБАВЛЕНО: игнорировать лишние переменные


settings = Settings()