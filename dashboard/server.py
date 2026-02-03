"""
Веб-сервер для дашборда
"""

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DashboardServer:
    def __init__(self, state_manager):
        self.state = state_manager
        self.app = FastAPI(title="Zoom Agent Dashboard")
        self.websocket_clients = []

        # Настройка статических файлов и шаблонов
        self.templates_dir = Path(__file__).parent / "templates"
        self.static_dir = Path(__file__).parent / "static"

        self.templates = Jinja2Templates(directory=str(self.templates_dir))

        # Монтируем статические файлы
        self.app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")

        # Регистрируем роуты
        self._setup_routes()
        self._setup_websocket()

    def _setup_routes(self):
        """Настройка маршрутов"""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Главная страница дашборда"""
            return self.templates.TemplateResponse(
                "index.html",
                {"request": request}
            )

        @self.app.get("/api/state")
        async def get_state():
            """API для получения текущего состояния"""
            return self.state.get_dashboard_data()

        @self.app.get("/api/transcript")
        async def get_transcript():
            """API для получения транскрипта"""
            return {
                "transcript": self.state.get_full_transcript(),
                "last_updated": self.state.last_transcript_update
            }

        @self.app.get("/api/news")
        async def get_news():
            """API для получения новостей"""
            return {
                "news": self.state.get_recent_news(),
                "entities": self.state.get_active_entities()
            }

    def _setup_websocket(self):
        """Настройка WebSocket для реального обновления"""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_clients.append(websocket)

            try:
                # Отправляем начальное состояние
                initial_state = self.state.get_dashboard_data()
                await websocket.send_json(initial_state)

                # Ждем сообщений (можно использовать для управления)
                while True:
                    data = await websocket.receive_text()
                    # Обработка сообщений от клиента

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_clients.remove(websocket)

    async def broadcast_update(self):
        """Отправка обновления всем подключенным клиентам"""
        if not self.websocket_clients:
            return

        update_data = self.state.get_dashboard_data()

        for client in self.websocket_clients:
            try:
                await client.send_json(update_data)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")

    async def start(self):
        """Запуск сервера"""
        config = uvicorn.Config(
            self.app,
            host=settings.DASHBOARD_HOST,
            port=settings.DASHBOARD_PORT,
            log_level="info" if settings.DEBUG else "warning"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def stop(self):
        """Остановка сервера"""
        # Закрываем все WebSocket соединения
        for client in self.websocket_clients:
            await client.close()
        self.websocket_clients.clear()