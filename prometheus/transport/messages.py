"""Transport-neutral records; no ROS message runtime or pickle on the wire."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, ClassVar

import msgpack
import numpy as np


@dataclass
class Stamp:
    sec: int = 0
    nanosec: int = 0


@dataclass
class Header:
    stamp: Stamp = field(default_factory=Stamp)
    frame_id: str = ""


@dataclass
class String:
    data: str = ""


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Pose:
    position: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class PoseStamped:
    header: Header = field(default_factory=Header)
    pose: Pose = field(default_factory=Pose)


@dataclass
class JointState:
    header: Header = field(default_factory=Header)
    name: list = field(default_factory=list)
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)
    effort: list = field(default_factory=list)


@dataclass
class Image:
    header: Header = field(default_factory=Header)
    height: int = 0
    width: int = 0
    encoding: str = ""
    is_bigendian: int = 0
    step: int = 0
    data: bytes = b""


@dataclass
class CameraInfo:
    header: Header = field(default_factory=Header)
    height: int = 0
    width: int = 0
    distortion_model: str = ""
    d: list = field(default_factory=list)
    k: list = field(default_factory=list)
    r: list = field(default_factory=list)
    p: list = field(default_factory=list)
    binning_x: int = 0
    binning_y: int = 0


@dataclass
class PointField:
    FLOAT32: ClassVar[int] = 7
    name: str = ""
    offset: int = 0
    datatype: int = 7
    count: int = 1


@dataclass
class PointCloud2:
    header: Header = field(default_factory=Header)
    height: int = 1
    width: int = 0
    fields: list = field(default_factory=list)
    is_bigendian: bool = False
    point_step: int = 0
    row_step: int = 0
    data: bytes = b""
    is_dense: bool = True


class ImageBridge:
    def cv2_to_imgmsg(self, image: Any, encoding: str) -> Image:
        array = np.ascontiguousarray(image)
        return Image(height=array.shape[0], width=array.shape[1], encoding=encoding,
                     step=array.strides[0], data=array.tobytes())


def create_cloud(header: Header, point_fields: list, points: Any) -> PointCloud2:
    array = np.ascontiguousarray(points, dtype="<f4")
    if array.ndim != 2 or array.shape[1] != len(point_fields):
        raise ValueError("point cloud shape does not match fields")
    return PointCloud2(header=header, width=len(array), fields=point_fields,
                       point_step=4 * array.shape[1], row_step=array.nbytes, data=array.tobytes())


def read_points(msg: Any, *, field_names: tuple, skip_nans: bool = False) -> np.ndarray:
    # Also reads recorded PointCloud2-shaped objects without sensor_msgs_py.
    mapping = {item.name: item for item in msg.fields}
    dtype = np.dtype({"names": list(field_names),
                      "formats": [(">f4" if msg.is_bigendian else "<f4")] * len(field_names),
                      "offsets": [mapping[name].offset for name in field_names],
                      "itemsize": int(msg.point_step)})
    if any(mapping[name].datatype != 7 or mapping[name].count != 1 for name in field_names):
        raise ValueError("only scalar float32 point fields are supported")
    rows = np.ndarray((int(msg.height), int(msg.width)), dtype=dtype, buffer=bytes(msg.data),
                      strides=(int(msg.row_step), int(msg.point_step)))
    array = np.column_stack([rows[name].reshape(-1) for name in field_names]).astype(np.float32)
    return array[np.isfinite(array).all(axis=1)] if skip_nans else array


_TYPES = {cls.__name__: cls for cls in (Stamp, Header, String, JointState, Image, CameraInfo,
                                      PointField, PointCloud2, Vector3, Quaternion, Pose, PoseStamped)}


def _encode(value: Any) -> Any:
    if is_dataclass(value) and type(value).__name__ in _TYPES:
        return {"__record__": type(value).__name__,
                "values": {item.name: getattr(value, item.name) for item in fields(value)}}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported wire value: {type(value).__name__}")


def _decode(value: dict) -> Any:
    if "__record__" in value:
        cls = _TYPES.get(value["__record__"])
        if cls is None:
            raise ValueError("unknown wire record")
        return cls(**value["values"])
    return value


def pack(message: Any) -> bytes:
    return msgpack.packb({"version": 1, "message": message}, default=_encode, use_bin_type=True)


def unpack(payload: bytes) -> Any:
    value = msgpack.unpackb(payload, object_hook=_decode, raw=False)
    if value.get("version") != 1:
        raise ValueError("unsupported ZMQ wire version")
    return value["message"]
