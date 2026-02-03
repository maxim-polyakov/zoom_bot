import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # ===== Zoom Desktop SDK Settings =====
    ZOOM_MEETING_URL: str = os.getenv("ZOOM_MEETING_URL", "")
    ZOOM_MEETING_ID: str = os.getenv("ZOOM_MEETING_ID", "")
    ZOOM_PASSWORD: str = os.getenv("ZOOM_PASSWORD", "")

    # Zoom OAuth credentials (для Desktop SDK)
    ZOOM_CLIENT_ID: str = os.getenv("ZOOM_CLIENT_ID", "Fl6nNGwIQXuobF_fdum0fQ")
    ZOOM_CLIENT_SECRET: str = os.getenv("ZOOM_CLIENT_SECRET", "LnfkomwDzEEqj44fCYqwIEaS6MYr51ei")
    ZOOM_ACCOUNT_ID: str = os.getenv("ZOOM_ACCOUNT_ID", "FJDdZxypQMS3vxQsqiVr5Q")

    # Zoom SDK behavior
    ZOOM_DISPLAY_NAME: str = os.getenv("ZOOM_DISPLAY_NAME", "AI Assistant")
    ZOOM_JOIN_WITHOUT_AUDIO: bool = os.getenv("ZOOM_JOIN_WITHOUT_AUDIO", "True").lower() == "true"
    ZOOM_JOIN_WITHOUT_VIDEO: bool = os.getenv("ZOOM_JOIN_WITHOUT_VIDEO", "True").lower() == "true"
    ZOOM_SCREEN_SHARE_MONITOR: int = int(os.getenv("ZOOM_SCREEN_SHARE_MONITOR", "0"))
    ZOOM_SKIP_AUTHENTICATION: bool = os.getenv("ZOOM_SKIP_AUTHENTICATION", "False").lower() == "true"
    ZOOM_AUTO_RECONNECT: bool = os.getenv("ZOOM_AUTO_RECONNECT", "True").lower() == "true"
    ZOOM_OAUTH_REDIRECT_URL: str = os.getenv("ZOOM_OAUTH_REDIRECT_URL", "http://localhost:3000")

    # ===== Bot Settings =====
    BOT_NAME: str = os.getenv("BOT_NAME", "Meeting Assistant")
    BOT_EMAIL: str = os.getenv("BOT_EMAIL", "")
    UPDATE_INTERVAL_SECONDS: int = int(os.getenv("UPDATE_INTERVAL", "60"))

    # ===== LLM Settings =====
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4-1106-preview")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # ===== News Search Settings =====
    SEARCH_API_KEY: str = os.getenv("SEARCH_API_KEY", "")
    SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "google")
    NEWS_FRESHNESS_DAYS: int = int(os.getenv("NEWS_FRESHNESS_DAYS", "30"))
    MAX_NEWS_PER_ENTITY: int = int(os.getenv("MAX_NEWS_PER_ENTITY", "5"))

    # ===== Web Server Settings =====
    DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8080"))
    TRANSCRIPT_WEBHOOK_PORT: int = int(os.getenv("TRANSCRIPT_WEBHOOK_PORT", "8081"))

    # ===== Paths =====
    CHROME_DRIVER_PATH: str = os.getenv("CHROME_DRIVER_PATH", "")
    SCREENSHOT_DIR: str = os.getenv("SCREENSHOT_DIR", "./screenshots")

    # ===== Transcription Settings =====
    TRANSCRIPTION_LANGUAGE: str = os.getenv("TRANSCRIPTION_LANGUAGE", "ru")
    TRANSCRIPTION_MODEL: str = os.getenv("TRANSCRIPTION_MODEL", "base")

    # ===== Feature Flags =====
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    USE_MOCK_TRANSCRIPT: bool = os.getenv("USE_MOCK_TRANSCRIPT", "False").lower() == "true"
    ZOOM_USE_DESKTOP_SDK: bool = os.getenv("ZOOM_USE_DESKTOP_SDK",
                                           "True").lower() == "true"  # Использовать Desktop SDK вместо Selenium

    # ===== Logging Settings =====
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "zoom_agent.log")
    ZOOM_SDK_DLL_PATH: str = os.getenv("ZOOM_SDK_DLL_PATH", "")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()