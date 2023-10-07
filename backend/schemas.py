from typing import Optional, Tuple, List

from pydantic import BaseModel


class StreamCreateDTO(BaseModel):
    id: int = 1
    name: Optional[str] = None
    source: Optional[str] = None
    boundary: Optional[List[List[Tuple[int, int]]]] = None  # array of some polygon points(x,y)
    img_size: Tuple[int, int] = (640, 640)
    stride: int = 32
    auto: bool = True
    vid_stride: int = 1
    classes: Optional[List[int]] = None  # only detect certain classes


class UserCreateDTO(BaseModel):
    username: str
    password: str


class UserUpdateDTO(BaseModel):
    username: str
    password: str


class UserInDB(UserCreateDTO):
    id: int


class StreamInDB(StreamCreateDTO):
    id: int
    user: UserInDB


class DeviceCreateDTO(BaseModel):
    id: int = 1
    name: Optional[str] = None
    ip: Optional[str] = None


class DeviceInDB(DeviceCreateDTO):
    id: int
    user: UserInDB
