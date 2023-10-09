from typing import Optional, Tuple, List, Any

from pydantic import BaseModel


class UserCreateDTO(BaseModel):
    username: str
    password: str


class UserUpdateDTO(BaseModel):
    password: str


class UserInDB(UserCreateDTO):
    id: int


class AfterDetectTriggerCreateDTO(BaseModel):
    type: str
    config: Any
    target_classes: Optional[List[int]] = None


class AfterDetectTriggerInDB(AfterDetectTriggerCreateDTO):
    id: int
    user: UserInDB


class DetectorConfigCreateDTO(BaseModel):
    include_classes: Optional[List[int]] = None  # only detect certain classes
    exclude_classes: Optional[List[int]] = None  # not detect certain classes
    after_detect_triggers: Optional[List[AfterDetectTriggerInDB]] = None
    boundary: Optional[List[Any]] = None  # array of some polygon points(x,y)


class DetectorConfigInDB(DetectorConfigCreateDTO):
    id: int
    user: UserInDB


class DetectionPolicyCreateDTO(BaseModel):
    type: str
    config: Any


class DetectionPolicyInDB(DetectionPolicyCreateDTO):
    id: int
    user: UserInDB


class StreamCreateDTO(BaseModel):
    name: Optional[str] = None
    source: Optional[str] = None
    img_size: Tuple[int, int] = (640, 640)
    stride: int = 32
    auto: bool = True
    vid_stride: int = 1
    confidence_threshold: float = 0.25  # confidence threshold
    iou_threshold: float = 0.45  # NMS IOU threshold
    configs: Optional[List[DetectorConfigInDB]] = None
    detection_policy: Optional[DetectionPolicyInDB] = None


class StreamInDB(StreamCreateDTO):
    id: int
    user: UserInDB


class DeviceCreateDTO(BaseModel):
    name: Optional[str] = None
    ip: Optional[str] = None


class DeviceInDB(DeviceCreateDTO):
    id: int
    user: UserInDB
