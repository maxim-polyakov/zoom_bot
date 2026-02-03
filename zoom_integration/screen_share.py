"""
Управление демонстрацией экрана в Zoom
Автоматический запуск и управление screen sharing
"""

import asyncio
import time
import subprocess
import platform
import pyautogui
import pygetwindow as gw
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from config import settings
from utils.logger import setup_logger
from utils.helpers import retry_async, benchmark

logger = setup_logger(__name__)


class ScreenShareStatus(Enum):
    """Статусы демонстрации экрана"""
    NOT_STARTED = "not_started"
    STARTING = "starting"
    SHARING = "sharing"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class ScreenShareConfig:
    """Конфигурация демонстрации экрана"""
    dashboard_url: str = f"http://{settings.DASHBOARD_HOST}:{settings.DASHBOARD_PORT}"
    dashboard_title: str = "Zoom Meeting Assistant Dashboard"
    zoom_window_title: str = "Zoom Meeting"
    share_button_timeout: int = 10  # секунды
    window_select_timeout: int = 5  # секунды
    retry_count: int = 3
    retry_delay: float = 2.0

    # Горячие клавиши (система/приложение)
    hotkey_share: Tuple[str, ...] = ("alt", "s")  # Alt+S для Zoom
    hotkey_select_window: Tuple[str, ...] = ("tab",)  # Tab для навигации
    hotkey_confirm: Tuple[str, ...] = ("enter",)  # Enter для подтверждения

    # Настройки для разных ОС
    if platform.system() == "Darwin":  # macOS
        hotkey_share = ("command", "shift", "s")
    elif platform.system() == "Linux":
        hotkey_share = ("ctrl", "shift", "s")


class ScreenShareManager:
    """Менеджер демонстрации экрана"""

    def __init__(self, config: ScreenShareConfig = None):
        self.config = config or ScreenShareConfig()
        self.status = ScreenShareStatus.NOT_STARTED
        self.zoom_window = None
        self.dashboard_window = None
        self.last_error: Optional[str] = None
        self.started_at: Optional[float] = None

        # Статистика
        self.stats = {
            "start_attempts": 0,
            "successful_starts": 0,
            "failed_starts": 0,
            "total_share_time": 0.0,
            "last_start_time": None
        }

        # Блокировка для предотвращения одновременного запуска
        self._lock = asyncio.Lock()

        logger.info(f"ScreenShareManager initialized for {platform.system()}")

    async def start_screen_share(self) -> bool:
        """Запуск демонстрации экрана"""
        async with self._lock:
            if self.status == ScreenShareStatus.SHARING:
                logger.warning("Screen share already active")
                return True

            self.status = ScreenShareStatus.STARTING
            self.stats["start_attempts"] += 1

            try:
                logger.info("Starting screen share process...")

                # 1. Открываем дашборд
                await self._open_dashboard()

                # 2. Находим окно Zoom
                await self._find_zoom_window()

                if not self.zoom_window:
                    raise Exception("Zoom window not found")

                # 3. Активируем окно Zoom
                await self._activate_zoom_window()

                # 4. Запускаем демонстрацию экрана
                await self._start_sharing()

                # 5. Выбираем окно дашборда
                await self._select_dashboard_window()

                # Успешный запуск
                self.status = ScreenShareStatus.SHARING
                self.started_at = time.time()
                self.stats["successful_starts"] += 1
                self.stats["last_start_time"] = time.time()
                self.last_error = None

                logger.info("Screen share started successfully")
                return True

            except Exception as e:
                self.status = ScreenShareStatus.ERROR
                self.last_error = str(e)
                self.stats["failed_starts"] += 1
                logger.error(f"Failed to start screen share: {e}")
                return False

    async def stop_screen_share(self) -> bool:
        """Остановка демонстрации экрана"""
        if self.status != ScreenShareStatus.SHARING:
            logger.warning("Screen share not active")
            return True

        try:
            logger.info("Stopping screen share...")

            # Используем горячие клавиши для остановки
            pyautogui.hotkey(*self.config.hotkey_share)
            await asyncio.sleep(1)

            # Обновляем статус и статистику
            if self.started_at:
                share_time = time.time() - self.started_at
                self.stats["total_share_time"] += share_time
                logger.info(f"Screen share stopped after {share_time:.1f} seconds")

            self.status = ScreenShareStatus.STOPPED
            self.started_at = None

            return True

        except Exception as e:
            logger.error(f"Error stopping screen share: {e}")
            return False

    async def restart_screen_share(self) -> bool:
        """Перезапуск демонстрации экрана"""
        logger.info("Restarting screen share...")

        # Сначала останавливаем
        await self.stop_screen_share()
        await asyncio.sleep(2)

        # Затем запускаем заново
        return await self.start_screen_share()

    @benchmark
    async def _open_dashboard(self):
        """Открытие дашборда в браузере"""
        logger.info(f"Opening dashboard: {self.config.dashboard_url}")

        try:
            # Используем системную команду для открытия URL
            system = platform.system()

            if system == "Darwin":  # macOS
                subprocess.run(["open", self.config.dashboard_url])
            elif system == "Windows":
                subprocess.run(["start", self.config.dashboard_url], shell=True)
            elif system == "Linux":
                subprocess.run(["xdg-open", self.config.dashboard_url])
            else:
                # Fallback: импортируем webbrowser
                import webbrowser
                webbrowser.open(self.config.dashboard_url)

            # Даем время на загрузку
            await asyncio.sleep(3)

            # Находим окно браузера
            await self._find_dashboard_window()

            logger.info("Dashboard opened successfully")

        except Exception as e:
            logger.error(f"Failed to open dashboard: {e}")
            raise

    async def _find_zoom_window(self, max_attempts: int = 5):
        """Поиск окна Zoom"""
        logger.info("Looking for Zoom window...")

        for attempt in range(max_attempts):
            try:
                # Ищем окна с Zoom в заголовке
                windows = gw.getWindowsWithTitle('Zoom')

                if windows:
                    self.zoom_window = windows[0]
                    logger.info(f"Found Zoom window: {self.zoom_window.title}")
                    return

                # Также пробуем другие варианты названий
                alternative_titles = ['Zoom Meeting', 'zoom', 'Meeting', 'Video Call']

                for title in alternative_titles:
                    windows = gw.getWindowsWithTitle(title)
                    if windows and 'zoom' in windows[0].title.lower():
                        self.zoom_window = windows[0]
                        logger.info(f"Found Zoom window (alternative): {self.zoom_window.title}")
                        return

                if attempt < max_attempts - 1:
                    logger.debug(f"Zoom window not found, attempt {attempt + 1}/{max_attempts}")
                    await asyncio.sleep(1)

            except Exception as e:
                logger.warning(f"Error finding Zoom window: {e}")
                await asyncio.sleep(1)

        raise Exception(f"Zoom window not found after {max_attempts} attempts")

    async def _find_dashboard_window(self, max_attempts: int = 5):
        """Поиск окна дашборда"""
        logger.info("Looking for dashboard window...")

        for attempt in range(max_attempts):
            try:
                # Ищем окна браузера с URL дашборда
                windows = gw.getAllWindows()

                for window in windows:
                    if window.title and (
                            self.config.dashboard_url in window.title or
                            self.config.dashboard_title in window.title or
                            'localhost' in window.title or
                            '127.0.0.1' in window.title
                    ):
                        self.dashboard_window = window
                        logger.info(f"Found dashboard window: {window.title}")
                        return

                if attempt < max_attempts - 1:
                    logger.debug(f"Dashboard window not found, attempt {attempt + 1}/{max_attempts}")
                    await asyncio.sleep(1)

            except Exception as e:
                logger.warning(f"Error finding dashboard window: {e}")
                await asyncio.sleep(1)

        logger.warning("Dashboard window not found, continuing anyway...")

    async def _activate_zoom_window(self):
        """Активация окна Zoom"""
        if not self.zoom_window:
            raise Exception("No Zoom window to activate")

        try:
            logger.info(f"Activating Zoom window: {self.zoom_window.title}")

            # Активируем окно
            self.zoom_window.activate()

            # Даем время на активацию
            await asyncio.sleep(1)

            # Проверяем, что окно активно
            if not self.zoom_window.isActive:
                logger.warning("Zoom window not active after activation attempt")
                # Пробуем еще раз
                self.zoom_window.activate()
                await asyncio.sleep(0.5)

            logger.info("Zoom window activated")

        except Exception as e:
            logger.error(f"Failed to activate Zoom window: {e}")
            raise

    async def _start_sharing(self):
        """Запуск демонстрации экрана в Zoom"""
        logger.info("Starting screen share in Zoom...")

        try:
            # Метод 1: Горячие клавиши
            logger.debug("Trying hotkey method...")
            pyautogui.hotkey(*self.config.hotkey_share)
            await asyncio.sleep(2)

            # Метод 2: Если не сработало, пробуем найти кнопку
            # (это требует кастомизации под конкретную версию Zoom)

            # Даем время на появление диалога выбора окна
            await asyncio.sleep(3)

            logger.info("Screen share dialog should be open now")

        except Exception as e:
            logger.error(f"Error starting screen share: {e}")
            raise

    async def _select_dashboard_window(self):
        """Выбор окна дашборда для демонстрации"""
        logger.info("Selecting dashboard window for sharing...")

        try:
            # Метод 1: Навигация с помощью Tab
            # Предполагаем, что фокус находится на диалоге выбора окна
            for i in range(5):  # Нажимаем Tab несколько раз
                pyautogui.press('tab')
                await asyncio.sleep(0.2)

            # Метод 2: Ищем окно по названию
            # В реальном проекте здесь нужна более сложная логика

            # Нажимаем Enter для выбора текущего варианта
            pyautogui.press('enter')
            await asyncio.sleep(1)

            logger.info("Dashboard window selected for sharing")

        except Exception as e:
            logger.error(f"Error selecting dashboard window: {e}")
            raise

    async def verify_screen_share(self) -> bool:
        """Проверка, что демонстрация экрана активна"""
        if self.status != ScreenShareStatus.SHARING:
            return False

        try:
            # Проверяем, что окно Zoom все еще активно
            if self.zoom_window and not self.zoom_window.isActive:
                logger.warning("Zoom window not active")
                return False

            # Можно добавить дополнительные проверки:
            # - Наличие индикатора демонстрации экрана
            # - Активность определенных элементов интерфейса

            return True

        except Exception as e:
            logger.error(f"Error verifying screen share: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса демонстрации экрана"""
        share_duration = 0.0
        if self.started_at and self.status == ScreenShareStatus.SHARING:
            share_duration = time.time() - self.started_at

        return {
            "status": self.status.value,
            "started_at": self.started_at,
            "share_duration_seconds": share_duration,
            "last_error": self.last_error,
            "stats": self.stats.copy(),
            "zoom_window": self.zoom_window.title if self.zoom_window else None,
            "dashboard_window": self.dashboard_window.title if self.dashboard_window else None
        }

    async def monitor_screen_share(self):
        """Мониторинг демонстрации экрана с автоматическим восстановлением"""
        logger.info("Starting screen share monitor...")

        while True:
            try:
                if self.status == ScreenShareStatus.SHARING:
                    # Проверяем, что демонстрация все еще активна
                    is_active = await self.verify_screen_share()

                    if not is_active:
                        logger.warning("Screen share appears to be inactive, attempting recovery...")
                        await self.restart_screen_share()

                # Ждем перед следующей проверкой
                await asyncio.sleep(30)  # Проверяем каждые 30 секунд

            except Exception as e:
                logger.error(f"Error in screen share monitor: {e}")
                await asyncio.sleep(10)

    async def take_screenshot(self, filename: str = None) -> Optional[str]:
        """Создание скриншота текущего экрана"""
        try:
            if not filename:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshots/screen_share_{timestamp}.png"

            # Создаем директорию если не существует
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Делаем скриншот
            screenshot = pyautogui.screenshot()
            screenshot.save(filename)

            logger.info(f"Screenshot saved: {filename}")
            return filename

        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return None


# Глобальный экземпляр менеджера
screen_share_manager_instance: Optional[ScreenShareManager] = None


def get_screen_share_manager() -> ScreenShareManager:
    """Получение глобального экземпляра ScreenShareManager"""
    global screen_share_manager_instance

    if screen_share_manager_instance is None:
        screen_share_manager_instance = ScreenShareManager()

    return screen_share_manager_instance


# Альтернативная реализация с использованием Selenium
class SeleniumScreenShareManager:
    """Менеджер демонстрации экрана с использованием Selenium"""

    def __init__(self, zoom_client):
        self.zoom_client = zoom_client
        self.driver = zoom_client.driver
        self.config = ScreenShareConfig()
        self.status = ScreenShareStatus.NOT_STARTED

    async def start_screen_share_selenium(self) -> bool:
        """Запуск демонстрации экрана через Selenium"""
        try:
            logger.info("Starting screen share via Selenium...")

            # 1. Находим кнопку демонстрации экрана
            share_button = await self._find_share_button()

            if not share_button:
                raise Exception("Share button not found")

            # 2. Нажимаем кнопку
            share_button.click()
            await asyncio.sleep(2)

            # 3. Выбираем окно дашборда
            await self._select_window_selenium()

            self.status = ScreenShareStatus.SHARING
            logger.info("Screen share started via Selenium")
            return True

        except Exception as e:
            logger.error(f"Selenium screen share failed: {e}")
            self.status = ScreenShareStatus.ERROR
            return False

    async def _find_share_button(self, max_attempts: int = 3):
        """Поиск кнопки демонстрации экрана"""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        selectors = [
            "button[aria-label*='share screen']",
            "button[aria-label*='поделиться экраном']",
            "//button[contains(@class, 'share-button')]",
            "//button[contains(text(), 'Share')]",
            "//button[contains(text(), 'Поделиться')]",
            "#shareScreenButton"
        ]

        for attempt in range(max_attempts):
            for selector in selectors:
                try:
                    if selector.startswith("//"):
                        element = WebDriverWait(self.driver, 2).until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                    else:
                        element = WebDriverWait(self.driver, 2).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )

                    logger.info(f"Found share button with selector: {selector}")
                    return element

                except:
                    continue

            if attempt < max_attempts - 1:
                logger.debug(f"Share button not found, attempt {attempt + 1}/{max_attempts}")
                await asyncio.sleep(1)

        return None

    async def _select_window_selenium(self):
        """Выбор окна для демонстрации через Selenium"""
        # После нажатия кнопки демонстрации появляется диалог выбора окна
        # В реальном проекте нужно адаптировать под конкретный интерфейс

        # Нажимаем Tab для навигации
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.common.action_chains import ActionChains

        actions = ActionChains(self.driver)

        # Нажимаем Tab несколько раз
        for _ in range(3):
            actions.send_keys(Keys.TAB)
            await asyncio.sleep(0.2)

        actions.perform()
        await asyncio.sleep(0.5)

        # Нажимаем Enter
        actions.send_keys(Keys.ENTER)
        actions.perform()

        logger.info("Window selected via Selenium")