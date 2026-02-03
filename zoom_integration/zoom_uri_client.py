"""
Клиент для подключения к Zoom через URI протокол
"""

import os
import sys
import time
import asyncio
import webbrowser
import urllib.parse
import subprocess
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ZoomClientStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    IN_MEETING = "in_meeting"
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

class ZoomURIClient:
    """Клиент для подключения к Zoom через URI протокол"""

    def __init__(self, meeting_state=None):
        self.meeting_state = meeting_state
        self.status = ZoomClientStatus.DISCONNECTED
        self.meeting_info = None
        self.process = None

        logger.info("ZoomURIClient initialized")

    async def join_meeting(self, meeting_info: ZoomMeetingInfo) -> bool:
        """Присоединение к встрече через Zoom URI протокол"""
        self.meeting_info = meeting_info
        self.status = ZoomClientStatus.CONNECTING

        logger.info(f"Joining meeting via Zoom URI: {meeting_info.meeting_id}")

        try:
            success = await self._join_via_zoom_uri(meeting_info)

            if success:
                self.status = ZoomClientStatus.IN_MEETING

                # Обновляем состояние
                if self.meeting_state:
                    self.meeting_state.update({
                        'meeting_id': meeting_info.meeting_id,
                        'display_name': meeting_info.display_name,
                        'joined_time': time.time(),
                        'status': 'in_meeting',
                        'method': 'zoom_uri',
                        'zoom_status': 'in_meeting'
                    })

                logger.info("Successfully initiated meeting join via Zoom URI")
                return True
            else:
                self.status = ZoomClientStatus.ERROR
                return False

        except Exception as e:
            logger.error(f"Failed to join via Zoom URI: {e}")
            self.status = ZoomClientStatus.ERROR
            return False

    async def _join_via_zoom_uri(self, meeting_info: ZoomMeetingInfo) -> bool:
        """Присоединение через Zoom URI протокол"""
        try:
            # Формируем Zoom URI с параметрами
            params = {
                'confno': meeting_info.meeting_id,
                'uname': urllib.parse.quote(meeting_info.display_name)
            }

            # Добавляем passcode если он есть
            if meeting_info.password:
                params['pwd'] = meeting_info.password

            # Добавляем параметры аудио/видео если нужно
            if meeting_info.no_audio:
                params['audio'] = 'off'
            if meeting_info.no_video:
                params['video'] = 'off'

            # Формируем query string
            query_string = '&'.join([f'{k}={v}' for k, v in params.items()])
            zoom_uri = f"zoommtg://zoom.us/join?{query_string}"

            # Также создаем веб-ссылку на всякий случай
            web_url = f"https://zoom.us/j/{meeting_info.meeting_id}"
            if meeting_info.password:
                web_url += f"?pwd={meeting_info.password}"

            logger.info(f"Zoom URI: {zoom_uri}")
            logger.info(f"Web URL (fallback): {web_url}")

            # Сохраняем ссылки в состоянии через update, а не прямое присваивание
            if self.meeting_state and hasattr(self.meeting_state, 'update'):
                self.meeting_state.update({
                    'zoom_uri': zoom_uri,
                    'web_url': web_url
                })

            # Пробуем открыть через Zoom приложение
            app_opened = await self._open_zoom_app(zoom_uri)

            # Если не удалось открыть приложение, открываем веб-версию
            if not app_opened:
                logger.info("Opening web version as fallback...")
                webbrowser.open(web_url)

            return True

        except Exception as e:
            logger.error(f"Error creating Zoom URI: {e}")
            return False

    async def _open_zoom_app(self, zoom_uri: str) -> bool:
        """Открытие Zoom через приложение на разных ОС"""
        try:
            if sys.platform == "win32":
                # Для Windows
                try:
                    os.startfile(zoom_uri)
                    logger.info("Zoom app opened via URI on Windows")
                    return True
                except Exception as e:
                    logger.warning(f"os.startfile failed: {e}")

                    # Альтернативный способ для Windows
                    try:
                        subprocess.Popen(f'start "" "{zoom_uri}"', shell=True)
                        logger.info("Zoom app opened via subprocess on Windows")
                        return True
                    except Exception as e2:
                        logger.warning(f"subprocess failed: {e2}")
                        return False

            elif sys.platform == "darwin":
                # Для Mac
                try:
                    subprocess.Popen(['open', zoom_uri])
                    logger.info("Zoom app opened on Mac")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to open Zoom on Mac: {e}")
                    return False

            else:
                # Для Linux и других ОС
                try:
                    subprocess.Popen(['xdg-open', zoom_uri])
                    logger.info("Zoom app opened via xdg-open")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to open Zoom via xdg-open: {e}")

                    # Пробуем другие способы для Linux
                    try:
                        subprocess.Popen(['gio', 'open', zoom_uri])
                        logger.info("Zoom app opened via gio")
                        return True
                    except:
                        pass

                    try:
                        subprocess.Popen(['gnome-open', zoom_uri])
                        logger.info("Zoom app opened via gnome-open")
                        return True
                    except:
                        pass

                    return False

        except Exception as e:
            logger.error(f"Error opening Zoom app: {e}")
            return False

    async def start_screen_share(self, monitor_index: int = 0) -> bool:
        """Запуск демонстрации экрана (только информационное сообщение)"""
        if self.status != ZoomClientStatus.IN_MEETING:
            logger.error("Not in meeting")
            return False

        logger.info("Screen sharing must be initiated manually in Zoom client")
        logger.info("Please use the 'Share Screen' button in your Zoom meeting")

        # Можно добавить инструкции для пользователя
        instructions = """
        Чтобы начать демонстрацию экрана в Zoom:
        1. Нажмите кнопку "Показать" (Share) в панели управления Zoom
        2. Выберите экран или приложение для демонстрации
        3. Нажмите "Показать" (Share)
        """

        logger.info(instructions)

        if self.meeting_state:
            self.meeting_state['screen_share_instructions'] = instructions

        return True  # Возвращаем True, так как инструкции предоставлены

    async def leave_meeting(self) -> bool:
        """Выход из встречи (информационное сообщение)"""
        if self.status != ZoomClientStatus.IN_MEETING:
            return True

        logger.info("Please leave the meeting manually using the Zoom client")
        logger.info("Click the 'Leave' or 'End' button in your Zoom meeting")

        # Инструкции для выхода
        instructions = """
        Чтобы выйти из встречи в Zoom:
        1. Наведите курсор на панель управления Zoom
        2. Нажмите кнопку "Покинуть" (Leave) или "Завершить" (End)
        3. Подтвердите действие при необходимости
        """

        logger.info(instructions)

        self.status = ZoomClientStatus.DISCONNECTED
        self.meeting_info = None

        if self.meeting_state:
            self.meeting_state.update({
                'status': 'disconnected',
                'zoom_status': 'disconnected',
                'leave_instructions': instructions,
                'left_time': time.time()
            })

        logger.info("Meeting left instructions provided")
        return True

    async def cleanup(self):
        """Очистка ресурсов"""
        logger.info("Cleaning up...")

        if self.meeting_info:
            await self.leave_meeting()

        # Если есть запущенные процессы, закрываем их
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass

        self.status = ZoomClientStatus.DISCONNECTED
        logger.info("Cleanup complete")

    def get_status_info(self) -> Dict[str, Any]:
        """Информация о статусе"""
        return {
            'status': self.status.value,
            'meeting_id': self.meeting_info.meeting_id if self.meeting_info else None,
            'display_name': self.meeting_info.display_name if self.meeting_info else None,
            'in_meeting': self.status == ZoomClientStatus.IN_MEETING,
            'method': 'zoom_uri'
        }

    def get_join_links(self) -> Dict[str, str]:
        """Получение ссылок для присоединения к встрече"""
        if not self.meeting_info:
            return {}

        try:
            # Генерируем веб-ссылку
            web_url = f"https://zoom.us/j/{self.meeting_info.meeting_id}"
            if self.meeting_info.password:
                web_url += f"?pwd={self.meeting_info.password}"

            # Генерируем Zoom URI
            params = {
                'confno': self.meeting_info.meeting_id,
                'uname': urllib.parse.quote(self.meeting_info.display_name)
            }

            if self.meeting_info.password:
                params['pwd'] = self.meeting_info.password

            if self.meeting_info.no_audio:
                params['audio'] = 'off'
            if self.meeting_info.no_video:
                params['video'] = 'off'

            query_string = '&'.join([f'{k}={v}' for k, v in params.items()])
            zoom_uri = f"zoommtg://zoom.us/join?{query_string}"

            return {
                'zoom_uri': zoom_uri,
                'web_url': web_url,
                'meeting_id': self.meeting_info.meeting_id,
                'passcode': self.meeting_info.password
            }

        except Exception as e:
            logger.error(f"Error generating join links: {e}")
            return {
                'zoom_uri': None,
                'web_url': None,
                'meeting_id': self.meeting_info.meeting_id,
                'passcode': self.meeting_info.password
            }