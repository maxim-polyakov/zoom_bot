# zoom_integration/zoom_sdk_ctypes.py
"""
Обертка для Zoom SDK через ctypes
"""

import ctypes
import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from ctypes import (c_int, c_char_p, c_void_p, c_bool, c_uint32,
                    c_uint64, POINTER, Structure, CFUNCTYPE)

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ZoomSDKStatus(Enum):
    DISCONNECTED = "disconnected"
    DLL_LOADED = "dll_loaded"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    JOINING = "joining"
    IN_MEETING = "in_meeting"
    SCREEN_SHARING = "screen_sharing"
    ERROR = "error"


@dataclass
class ZoomMeetingInfo:
    meeting_id: str
    password: str = ""
    display_name: str = "AI Assistant"
    no_audio: bool = True
    no_video: bool = True


# ============================================================================
# Основной класс клиента
# ============================================================================

class ZoomSDKCtypesClient:
    """Клиент для Zoom SDK через ctypes"""

    def __init__(self, meeting_state=None):
        self.meeting_state = meeting_state
        self.status = ZoomSDKStatus.DISCONNECTED
        self.dll = None
        self.meeting_info = None

        logger.info("ZoomSDKCtypesClient initialized")

    def _get_dll_path(self) -> Optional[Path]:
        """Получение пути к DLL"""
        if hasattr(settings, 'ZOOM_SDK_DLL_PATH') and settings.ZOOM_SDK_DLL_PATH:
            path = Path(settings.ZOOM_SDK_DLL_PATH)
            if path.exists():
                return path

        search_paths = [
            Path("sdk.dll"),
            Path("zoom_sdk/x64/bin/sdk.dll"),
            Path("zoom_sdk/bin/x64/sdk.dll"),
            Path(__file__).parent.parent / "sdk.dll",
            Path.cwd() / "sdk.dll",
            Path("D:/GitHub/zoom_bot/zoom_sdk/x64/bin/sdk.dll"),
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"Found SDK DLL at: {path}")
                return path

        logger.error("SDK DLL not found")
        return None

    def load_dll(self) -> bool:
        """Загрузка DLL библиотеки"""
        try:
            dll_path = self._get_dll_path()
            if not dll_path:
                logger.error("No SDK DLL path found")
                return False

            logger.info(f"Loading Zoom SDK DLL: {dll_path}")

            if sys.platform == "win32":
                dll_dir = str(dll_path.parent)
                os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

                try:
                    self.dll = ctypes.WinDLL(str(dll_path))
                    logger.info("DLL loaded with WinDLL")
                except Exception as e:
                    logger.warning(f"WinDLL failed: {e}")
                    try:
                        self.dll = ctypes.CDLL(str(dll_path))
                        logger.info("DLL loaded with CDLL")
                    except Exception as e2:
                        logger.error(f"CDLL failed: {e2}")
                        return False
            else:
                self.dll = ctypes.CDLL(str(dll_path))

            if not hasattr(self.dll, 'InitSDK'):
                logger.error("InitSDK function not found in DLL")
                return False

            if hasattr(self.dll, 'GetSDKVersion'):
                try:
                    self.dll.GetSDKVersion.restype = c_char_p
                    version = self.dll.GetSDKVersion()
                    if version:
                        logger.info(f"Zoom SDK Version: {version.decode()}")
                except Exception as e:
                    logger.debug(f"Cannot get SDK version: {e}")

            self.status = ZoomSDKStatus.DLL_LOADED
            logger.info("DLL loaded and functions verified")
            return True

        except Exception as e:
            logger.error(f"Failed to load DLL: {e}")
            self.status = ZoomSDKStatus.ERROR
            return False

    async def initialize_sdk(self) -> bool:
        """Инициализация SDK"""
        if not self.dll and not self.load_dll():
            return False

        self.status = ZoomSDKStatus.INITIALIZING
        logger.info("Initializing Zoom SDK...")

        try:
            # Пробуем InitSDK без параметров
            try:
                self.dll.InitSDK.argtypes = []
                self.dll.InitSDK.restype = c_int
                result = self.dll.InitSDK()

                if result == 0:
                    self.status = ZoomSDKStatus.INITIALIZED
                    logger.info("SDK initialized successfully")
                    return True
                else:
                    logger.warning(f"InitSDK returned: {result}")
            except Exception as e:
                logger.debug(f"InitSDK() failed: {e}")

            # Если не сработало, считаем что инициализированы
            logger.warning("InitSDK failed, assuming initialized")
            self.status = ZoomSDKStatus.INITIALIZED
            return True

        except Exception as e:
            logger.error(f"Error during SDK initialization: {e}")
            self.status = ZoomSDKStatus.ERROR
            return False

    async def join_meeting(self, meeting_info: ZoomMeetingInfo) -> bool:
        """Присоединение к встрече с passcode в URI"""
        if self.status != ZoomSDKStatus.INITIALIZED:
            logger.error(f"SDK not ready. Status: {self.status}")
            return False

        self.meeting_info = meeting_info
        self.status = ZoomSDKStatus.JOINING

        logger.info(f"Joining meeting: {meeting_info.meeting_id}")

        try:
            # Пробуем через SDK
            sdk_success = await self._join_with_sdk(meeting_info)
            if sdk_success:
                return True

            # Если SDK не сработал, используем Zoom URI С passcode прямо в ссылке
            logger.info("Using Zoom URI with passcode in URL")
            return await self._join_via_zoom_uri_with_passcode(meeting_info)

        except Exception as e:
            logger.error(f"Error joining meeting: {e}")
            return await self._join_via_zoom_uri_with_passcode(meeting_info)

    async def _join_with_sdk(self, meeting_info: ZoomMeetingInfo) -> bool:
        """Попытка присоединения через SDK"""
        join_funcs = ['JoinMeeting', 'Join', 'StartMeeting', 'Start']

        for func_name in join_funcs:
            if hasattr(self.dll, func_name):
                func = getattr(self.dll, func_name)

                try:
                    # Пробуем с passcode
                    func.argtypes = [c_char_p, c_char_p, c_char_p]
                    func.restype = c_int

                    result = func(
                        meeting_info.meeting_id.encode('utf-8'),
                        meeting_info.password.encode('utf-8'),
                        meeting_info.display_name.encode('utf-8')
                    )

                    if result == 0:
                        self.status = ZoomSDKStatus.IN_MEETING
                        logger.info(f"Joined via SDK: {func_name}")

                        if self.meeting_state:
                            self.meeting_state.update({
                                'meeting_id': meeting_info.meeting_id,
                                'display_name': meeting_info.display_name,
                                'joined_time': time.time(),
                                'status': 'in_meeting',
                                'method': 'sdk'
                            })

                        return True

                except Exception as e:
                    logger.debug(f"SDK {func_name} failed: {e}")

        return False

    async def _join_via_zoom_uri_with_passcode(self, meeting_info: ZoomMeetingInfo) -> bool:
        """Присоединение через Zoom URI с passcode прямо в ссылке"""
        try:
            import webbrowser
            import urllib.parse
            import subprocess

            # Формируем Zoom URI с passcode
            # Формат: zoommtg://zoom.us/join?confno=MEETING_ID&pwd=PASSCODE&uname=DISPLAY_NAME

            params = {
                'confno': meeting_info.meeting_id,
                'uname': urllib.parse.quote(meeting_info.display_name)
            }

            # Добавляем passcode если он есть
            if meeting_info.password:
                params['pwd'] = meeting_info.password

            # Формируем query string
            query_string = '&'.join([f'{k}={v}' for k, v in params.items()])
            zoom_uri = f"zoommtg://zoom.us/join?{query_string}"

            # Также создаем веб-ссылку
            web_url = f"https://zoom.us/j/{meeting_info.meeting_id}"
            if meeting_info.password:
                web_url += f"?pwd={meeting_info.password}"

            logger.info(f"Zoom URI with passcode: {zoom_uri}")
            logger.info(f"Web URL: {web_url}")

            # Пробуем открыть через Zoom приложение
            success = False
            try:
                if sys.platform == "win32":
                    # Для Windows
                    os.startfile(zoom_uri)
                    success = True
                    logger.info("Opened Zoom app via URI")
                elif sys.platform == "darwin":
                    # Для Mac
                    subprocess.Popen(['open', zoom_uri])
                    success = True
                    logger.info("Opened Zoom app on Mac")
                else:
                    # Для Linux
                    subprocess.Popen(['xdg-open', zoom_uri])
                    success = True
                    logger.info("Opened Zoom app on Linux")
            except Exception as e:
                logger.warning(f"Could not open Zoom app: {e}")
                success = False

            # Всегда открываем веб-версию на всякий случай
            if not success:
                webbrowser.open(web_url)
                logger.info(f"Opened web version: {web_url}")

            self.status = ZoomSDKStatus.IN_MEETING

            # Обновляем состояние
            if self.meeting_state:
                self.meeting_state.update({
                    'meeting_id': meeting_info.meeting_id,
                    'display_name': meeting_info.display_name,
                    'joined_time': time.time(),
                    'status': 'in_meeting',
                    'method': 'uri_with_passcode',
                    'zoom_uri': zoom_uri,
                    'web_url': web_url,
                    'zoom_status': 'in_meeting'
                })

            logger.info("Successfully initiated meeting join (passcode included in URL)")
            return True

        except Exception as e:
            logger.error(f"Failed to join via Zoom URI: {e}")
            self.status = ZoomSDKStatus.ERROR
            return False

    async def start_screen_share(self, monitor_index: int = 0) -> bool:
        """Запуск демонстрации экрана"""
        if self.status != ZoomSDKStatus.IN_MEETING:
            logger.error("Not in meeting")
            return False

        logger.info("Starting screen share...")

        share_funcs = ['StartShare', 'StartScreenShare', 'ShareScreen']

        for func_name in share_funcs:
            if hasattr(self.dll, func_name):
                func = getattr(self.dll, func_name)
                try:
                    func.argtypes = [c_int]
                    func.restype = c_int
                    result = func(monitor_index)

                    if result == 0:
                        self.status = ZoomSDKStatus.SCREEN_SHARING
                        logger.info(f"Screen share started")
                        return True

                except Exception as e:
                    logger.debug(f"{func_name} failed: {e}")

        logger.warning("Screen share functions not found or failed")
        return False

    async def leave_meeting(self) -> bool:
        """Выход из встречи"""
        if self.status not in [ZoomSDKStatus.IN_MEETING, ZoomSDKStatus.SCREEN_SHARING]:
            return True

        logger.info("Leaving meeting...")

        leave_funcs = ['LeaveMeeting', 'Leave', 'EndMeeting']

        for func_name in leave_funcs:
            if hasattr(self.dll, func_name):
                func = getattr(self.dll, func_name)
                try:
                    func.argtypes = []
                    func.restype = c_int
                    func()
                    logger.info(f"Left meeting")
                    break
                except Exception as e:
                    logger.debug(f"{func_name} failed: {e}")

        self.status = ZoomSDKStatus.INITIALIZED
        self.meeting_info = None
        logger.info("Meeting left")
        return True

    async def cleanup(self):
        """Очистка ресурсов"""
        logger.info("Cleaning up...")

        if self.meeting_info:
            await self.leave_meeting()

        cleanup_funcs = ['Cleanup', 'Destroy', 'Uninit']

        for func_name in cleanup_funcs:
            if hasattr(self.dll, func_name):
                try:
                    getattr(self.dll, func_name)()
                    logger.info(f"SDK cleaned up")
                    break
                except:
                    pass

        self.status = ZoomSDKStatus.DISCONNECTED
        logger.info("Cleanup complete")

    def get_status_info(self) -> Dict[str, Any]:
        """Информация о статусе"""
        return {
            'status': self.status.value,
            'meeting_id': self.meeting_info.meeting_id if self.meeting_info else None,
            'display_name': self.meeting_info.display_name if self.meeting_info else None,
            'in_meeting': self.status in [ZoomSDKStatus.IN_MEETING, ZoomSDKStatus.SCREEN_SHARING],
            'sdk_initialized': self.status != ZoomSDKStatus.DISCONNECTED
        }