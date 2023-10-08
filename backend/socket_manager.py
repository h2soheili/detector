import json
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

    def on_message(self, msg):
        data_loaded = json.loads(msg)
        # data_loaded['Data'] = {k[0].lower() + k[1:]: v for k, v in data_loaded['Data'].items()}
        # if data_loaded['Data'].get('objects', 0):
        #     data_loaded['Data']['objects'] = [{k[0].lower() + k[1:]: v for k, v in obj.items()} for obj in
        #                                       data_loaded['Data']['objects']]
        #
        # if data_loaded['Data'].get('details', 0):
        #     data_loaded['Data']['details'] = [{k[0].lower() + k[1:]: v for k, v in obj.items()} for obj in
        #                                       data_loaded['Data']['details']]
        #
        # if data_loaded['key'] == 'Token':
        #     self.action_handler_thread.login_to_system(data_loaded['Data'])
        #
        # elif data_loaded['key'] == 'get_stream':
        #     Thread(target=self.action_handler_thread.send_camera_stream, args=(data_loaded['Data'],)).start()
        #
        # elif data_loaded['key'] == 'CameraCreated':
        #     self.action_handler_thread.camera_added(data_loaded['Data'])
        #
        # elif data_loaded['key'] == 'Stop':
        #     self.action_handler_thread.stop_runing_active_cameras(data_loaded)
        #
        # elif data_loaded['key'] == 'CameraUpdated':
        #     self.action_handler_thread.update_camera(data_loaded['Data'])
        #
        # elif data_loaded['key'] == 'CameraRemoved':
        #     self.action_handler_thread.remove_camera(data_loaded['Data'])
        #
        # elif data_loaded['key'] == 'SendToServer':
        #     self.action_handler_thread.send_data_to_server(data_loaded['Data'])

