"""
Клиент для автоматического подключения к Zoom и управления встречей
Использует Selenium для автоматизации браузера Zoom
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse, parse_qs

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from config import settings
from utils.logger import setup_logger
from utils.helpers import retry_async, benchmark
from zoom_integration.screen_share import ScreenShareManager, get_screen_share_manager

logger = setup_logger(__name__)


class ZoomClientStatus(Enum):
    """Статусы Zoom клиента"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    IN_MEETING = "in_meeting"
    SCREEN_SHARING = "screen_sharing"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class ZoomMeetingInfo:
    """Информация о встрече Zoom"""
    meeting_id: str
    meeting_url: str
    meeting_topic: str = ""
    host_email: str = ""
    start_time: Optional[datetime] = None
    duration_minutes: int = 60
    password_required: bool = False
    waiting_room_enabled: bool = False
    screen_sharing_allowed: bool = True

    @classmethod
    def from_url(cls, url: str) -> 'ZoomMeetingInfo':
        """Создание из URL встречи"""
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)

        # Извлекаем ID встречи
        meeting_id = None
        path_parts = parsed.path.split('/')

        if 'j' in path_parts:
            idx = path_parts.index('j')
            if idx + 1 < len(path_parts):
                meeting_id = path_parts[idx + 1]

        # Ищем в query параметрах
        if not meeting_id:
            meeting_id = query_params.get('pwd', [None])[0]  # Иногда ID в параметре pwd

        return cls(
            meeting_id=meeting_id or "unknown",
            meeting_url=url
        )


class ZoomClient:
    """Клиент для работы с Zoom через Selenium"""

    def __init__(self, meeting_state=None):
        self.meeting_state = meeting_state
        self.driver: Optional[webdriver.Chrome] = None
        self.status = ZoomClientStatus.DISCONNECTED
        self.meeting_info: Optional[ZoomMeetingInfo] = None
        self.screen_share_manager: Optional[ScreenShareManager] = None

        # Время последней активности
        self.last_activity_time: Optional[float] = None

        # Статистика
        self.stats = {
            "connection_attempts": 0,
            "successful_connections": 0,
            "failed_connections": 0,
            "total_meeting_time": 0.0,
            "screen_share_starts": 0,
            "errors": 0
        }

        # Конфигурация
        self.config = {
            "timeout": 30,
            "poll_interval": 5,
            "max_reconnect_attempts": 3,
            "headless": not settings.DEBUG,
            "user_data_dir": None,  # Для сохранения сессии
            "chrome_options": self._get_chrome_options()
        }

        logger.info("ZoomClient initialized")

    def _get_chrome_options(self) -> Options:
        """Получение настроек Chrome"""
        options = Options()

        # Базовые настройки
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")

        # Для режима отладки
        if settings.DEBUG:
            options.add_argument("--auto-open-devtools-for-tabs")
            options.add_experimental_option("excludeSwitches", ["enable-logging"])
        else:
            options.add_argument("--headless")  # Без графического интерфейса

        # Настройки для автоматизации
        options.add_experimental_option(
            "excludeSwitches",
            ["enable-automation"]
        )
        options.add_experimental_option(
            'useAutomationExtension',
            False
        )

        # User-Agent
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36 ZoomBot/1.0"
        )

        # Настройки для сохранения сессии
        if hasattr(settings, 'CHROME_USER_DATA_DIR'):
            options.add_argument(f"user-data-dir={settings.CHROME_USER_DATA_DIR}")

        return options

    @retry_async(max_retries=3, delay=2.0, backoff_factor=2.0)
    async def connect(self, meeting_url: str = None) -> bool:
        """Подключение к встрече Zoom"""
        self.status = ZoomClientStatus.CONNECTING
        self.stats["connection_attempts"] += 1

        meeting_url = meeting_url or settings.ZOOM_MEETING_URL

        if not meeting_url:
            logger.error("No meeting URL provided")
            return False

        try:
            logger.info(f"Connecting to Zoom meeting: {meeting_url}")

            # Парсим информацию о встрече
            self.meeting_info = ZoomMeetingInfo.from_url(meeting_url)

            # Инициализируем драйвер
            await self._init_driver()

            # Переходим на страницу встречи
            self.driver.get(meeting_url)

            # Ожидаем загрузки
            await asyncio.sleep(5)

            # Обрабатываем различные сценарии входа
            await self._handle_login_scenarios()

            # Проверяем, что мы в встрече
            in_meeting = await self._verify_in_meeting()

            if in_meeting:
                self.status = ZoomClientStatus.IN_MEETING
                self.stats["successful_connections"] += 1
                self.last_activity_time = time.time()

                logger.info("Successfully connected to Zoom meeting")

                # Обновляем состояние встречи
                if self.meeting_state:
                    self.meeting_state.update_status("in_meeting")
                    self.meeting_state.meeting_id = self.meeting_info.meeting_id
                    self.meeting_state.meeting_url = meeting_url

                return True
            else:
                raise Exception("Could not verify meeting join")

        except Exception as e:
            self.status = ZoomClientStatus.ERROR
            self.stats["failed_connections"] += 1
            self.stats["errors"] += 1

            logger.error(f"Failed to connect to Zoom: {e}")

            # Закрываем драйвер
            await self.disconnect()

            return False

    async def _init_driver(self):
        """Инициализация Chrome драйвера"""
        try:
            logger.info("Initializing Chrome driver...")

            # Используем webdriver-manager для автоматической загрузки драйвера
            service = Service(ChromeDriverManager().install())

            # Создаем драйвер
            self.driver = webdriver.Chrome(
                service=service,
                options=self.config["chrome_options"]
            )

            # Настраиваем таймауты
            self.driver.set_page_load_timeout(self.config["timeout"])
            self.driver.implicitly_wait(10)

            # Устанавливаем размер окна
            self.driver.set_window_size(1400, 900)

            logger.info("Chrome driver initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {e}")
            raise

    async def _handle_login_scenarios(self):
        """Обработка различных сценариев входа в Zoom"""
        logger.info("Handling login scenarios...")

        # Сценарий 1: Вход через приложение Zoom
        try:
            open_zoom_btn = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(text(), 'Open Zoom Meetings')]")
                )
            )
            open_zoom_btn.click()
            logger.info("Clicked 'Open Zoom Meetings'")
            await asyncio.sleep(3)

        except TimeoutException:
            logger.debug("No 'Open Zoom Meetings' button found")

        # Сценарий 2: Ввод пароля
        if settings.ZOOM_PASSWORD:
            try:
                password_field = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.ID, "input-for-pwd"))
                )
                password_field.send_keys(settings.ZOOM_PASSWORD)

                join_btn = self.driver.find_element(By.ID, "joinBtn")
                join_btn.click()
                logger.info("Entered password and clicked join")
                await asyncio.sleep(3)

            except TimeoutException:
                logger.debug("No password field found")

        # Сценарий 3: Вход через браузер
        try:
            join_browser_btn = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//span[contains(text(), 'join from your browser')]")
                )
            )
            join_browser_btn.click()
            logger.info("Clicked 'join from your browser'")
            await asyncio.sleep(3)

        except TimeoutException:
            logger.debug("No 'join from your browser' option")

        # Сценарий 4: Ожидание загрузки интерфейса встречи
        try:
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located(
                    (By.CLASS_NAME, "footer-button")
                )
            )
            logger.info("Meeting interface loaded")

        except TimeoutException:
            logger.warning("Meeting interface not loaded within timeout")

        # Даем дополнительное время на загрузку
        await asyncio.sleep(5)

    async def _verify_in_meeting(self) -> bool:
        """Проверка, что мы успешно вошли в встречу"""
        try:
            # Проверяем наличие элементов интерфейса встречи
            meeting_controls = self.driver.find_elements(By.CLASS_NAME, "footer-button")

            if meeting_controls:
                logger.info(f"Found {len(meeting_controls)} meeting controls")
                return True

            # Альтернативная проверка
            if "zoom.us/wc/" in self.driver.current_url:
                logger.info("URL indicates we're in meeting")
                return True

            # Проверяем по заголовку окна
            page_title = self.driver.title.lower()
            if "zoom" in page_title and "meeting" in page_title:
                logger.info("Page title indicates we're in meeting")
                return True

            return False

        except Exception as e:
            logger.error(f"Error verifying meeting status: {e}")
            return False

    @benchmark
    async def start_screen_share(self) -> bool:
        """Запуск демонстрации экрана"""
        if self.status != ZoomClientStatus.IN_MEETING:
            logger.error("Not in meeting, cannot start screen share")
            return False

        try:
            logger.info("Starting screen share...")

            # Инициализируем менеджер демонстрации экрана
            if not self.screen_share_manager:
                self.screen_share_manager = get_screen_share_manager()

            # Запускаем демонстрацию
            success = await self.screen_share_manager.start_screen_share()

            if success:
                self.status = ZoomClientStatus.SCREEN_SHARING
                self.stats["screen_share_starts"] += 1

                logger.info("Screen share started successfully")

                # Обновляем состояние встречи
                if self.meeting_state:
                    self.meeting_state.update_status("screen_sharing")

                return True
            else:
                logger.error("Failed to start screen share")
                return False

        except Exception as e:
            logger.error(f"Error starting screen share: {e}")
            self.stats["errors"] += 1
            return False

    async def leave_meeting(self) -> bool:
        """Выход из встречи"""
        try:
            logger.info("Leaving Zoom meeting...")

            # Останавливаем демонстрацию экрана если активна
            if self.status == ZoomClientStatus.SCREEN_SHARING:
                await self.screen_share_manager.stop_screen_share()

            # Пытаемся найти кнопку выхода
            try:
                leave_btn = self.driver.find_element(
                    By.XPATH,
                    "//button[contains(@class, 'leave-meeting')]"
                )
                leave_btn.click()
                logger.info("Clicked leave button")

            except NoSuchElementException:
                # Если не нашли кнопку, просто закрываем браузер
                logger.warning("Leave button not found, closing browser")
                await self.disconnect()

            # Обновляем статус
            self.status = ZoomClientStatus.DISCONNECTED

            # Обновляем статистику
            if self.last_activity_time:
                meeting_time = time.time() - self.last_activity_time
                self.stats["total_meeting_time"] += meeting_time
                logger.info(f"Meeting duration: {meeting_time:.1f} seconds")

            # Обновляем состояние встречи
            if self.meeting_state:
                self.meeting_state.update_status("disconnected")

            logger.info("Successfully left the meeting")
            return True

        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")
            self.stats["errors"] += 1
            return False

    async def disconnect(self):
        """Отключение драйвера"""
        try:
            if self.driver:
                logger.info("Disconnecting Chrome driver...")
                self.driver.quit()
                self.driver = None
                logger.info("Chrome driver disconnected")

        except Exception as e:
            logger.error(f"Error disconnecting driver: {e}")

    async def reconnect(self) -> bool:
        """Переподключение к встрече"""
        if not self.meeting_info:
            logger.error("No meeting info for reconnection")
            return False

        logger.info(f"Attempting to reconnect to meeting {self.meeting_info.meeting_id}")

        self.status = ZoomClientStatus.RECONNECTING

        # Отключаемся
        await self.disconnect()

        # Пробуем подключиться заново
        return await self.connect(self.meeting_info.meeting_url)

    async def monitor_connection(self):
        """Мониторинг соединения с автоматическим восстановлением"""
        logger.info("Starting connection monitor...")

        while True:
            try:
                # Проверяем, что драйвер все еще активен
                if self.driver:
                    # Проверяем, что страница загружена
                    try:
                        current_url = self.driver.current_url

                        # Если URL изменился или страница не отвечает
                        if not current_url or "zoom.us" not in current_url:
                            logger.warning("Connection lost, attempting to reconnect...")
                            await self.reconnect()

                    except WebDriverException as e:
                        logger.warning(f"WebDriver exception: {e}, attempting to reconnect...")
                        await self.reconnect()

                # Проверяем активность демонстрации экрана
                if (self.status == ZoomClientStatus.SCREEN_SHARING and
                    self.screen_share_manager):

                    is_active = await self.screen_share_manager.verify_screen_share()
                    if not is_active:
                        logger.warning("Screen share inactive, attempting to restart...")
                        await self.start_screen_share()

                # Обновляем время последней активности
                self.last_activity_time = time.time()

                # Ждем перед следующей проверкой
                await asyncio.sleep(10)  # Проверяем каждые 10 секунд

            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
                await asyncio.sleep(5)

    def get_meeting_participants(self) -> List[Dict[str, Any]]:
        """Получение списка участников встречи (заглушка)"""
        # В реальной реализации здесь будет парсинг интерфейса Zoom
        # или использование Zoom API

        participants = []

        try:
            if self.driver:
                # Пробуем найти элементы участников
                participant_elements = self.driver.find_elements(
                    By.CLASS_NAME, "participant-item"
                )

                for element in participant_elements:
                    try:
                        name = element.text.strip()
                        if name:
                            participants.append({
                                "name": name,
                                "is_host": "host" in element.get_attribute("class", ""),
                                "is_muted": "muted" in element.get_attribute("class", "")
                            })
                    except:
                        continue

        except Exception as e:
            logger.debug(f"Could not get participants: {e}")

        return participants

    def take_screenshot(self, filename: str = None) -> Optional[str]:
        """Создание скриншота текущего экрана Zoom"""
        try:
            if not self.driver:
                logger.error("No driver available for screenshot")
                return None

            if not filename:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshots/zoom_{timestamp}.png"

            # Создаем директорию если не существует
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Делаем скриншот
            self.driver.save_screenshot(filename)

            logger.info(f"Zoom screenshot saved: {filename}")
            return filename

        except Exception as e:
            logger.error(f"Error taking Zoom screenshot: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса клиента"""
        meeting_duration = 0.0
        if self.last_activity_time and self.status in [
            ZoomClientStatus.IN_MEETING,
            ZoomClientStatus.SCREEN_SHARING
        ]:
            meeting_duration = time.time() - self.last_activity_time

        return {
            "status": self.status.value,
            "meeting_id": self.meeting_info.meeting_id if self.meeting_info else None,
            "meeting_url": self.meeting_info.meeting_url if self.meeting_info else None,
            "meeting_duration_seconds": meeting_duration,
            "stats": self.stats.copy(),
            "driver_active": self.driver is not None,
            "screen_share_active": self.status == ZoomClientStatus.SCREEN_SHARING,
            "current_url": self.driver.current_url if self.driver else None
        }

    async def send_chat_message(self, message: str) -> bool:
        """Отправка сообщения в чат Zoom (заглушка)"""
        # В реальной реализации здесь будет взаимодействие с интерфейсом чата
        logger.info(f"Would send chat message: {message[:50]}...")
        return False


# Фабрика для создания клиентов
class ZoomClientFactory:
    """Фабрика для создания Zoom клиентов"""

    @staticmethod
    def create_client(
        meeting_state=None,
        headless: bool = None,
        user_data_dir: str = None
    ) -> ZoomClient:
        """Создание нового Zoom клиента"""
        client = ZoomClient(meeting_state)

        if headless is not None:
            client.config["headless"] = headless

        if user_data_dir:
            client.config["user_data_dir"] = user_data_dir

        return client

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> ZoomClient:
        """Создание клиента из конфигурации"""
        client = ZoomClient()
        client.config.update(config)
        return client


# Глобальный экземпляр
zoom_client_instance: Optional[ZoomClient] = None


def get_zoom_client(meeting_state=None) -> ZoomClient:
    """Получение глобального экземпляра ZoomClient"""
    global zoom_client_instance

    if zoom_client_instance is None:
        zoom_client_instance = ZoomClientFactory.create_client(meeting_state)

    return zoom_client_instance