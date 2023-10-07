from typing import Dict

from fastapi import WebSocket

from backend.schemas import DeviceInDB, UserInDB


class SocketManager:
    def __init__(self):
        self.devices: Dict[str, DeviceInDB] = {}
        self.sockets: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        # cookies = websocket.cookies
        query_params = websocket.query_params
        user_id = query_params["user"]
        device_id = query_params["device_id"]
        if not device_id in self.sockets:
            self.sockets[device_id] = websocket
            user = UserInDB(id=user_id)
            device = DeviceInDB(id=1, user=user)
            self.devices[device_id] = device

    def disconnect(self, websocket: WebSocket):
        query_params = websocket.query_params
        device_id = query_params["device_id"]
        if device_id in self.sockets:
            self.sockets.pop(device_id)
            self.devices.pop(device_id)
