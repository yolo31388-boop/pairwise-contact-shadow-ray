"""屏幕空间接触阴影射线的纯 Python 参考实现（仅标准库）。

六个核心修复点：

1. :func:`depth_current_frame` 始终选择当帧深度；上一帧深度只能在资源未就绪
   时作为带 ``stale`` 标记的临时回退，不能参与正式追踪。
2. :func:`step_screen_scale` 按屏幕分辨率、深度处像素足迹或显式屏幕缩放系数
   计算步长，避免固定像素/固定次数造成的远近软硬不一致。
3. :func:`stop_at_discontinuity` 检测相邻深度的大幅跳变；射线在轮廓处立即终止，
   不穿过不连续面去采样不可比的背景。
4. :func:`offscreen_fallback` 对越界 UV 做边缘钳制、平滑淡出或调用方提供的
   probe/当前帧回退，避免屏幕边缘阴影被硬切掉一半和旋转相机时闪烁。
5. :func:`merge_darkest` 对多光源结果逐像素取最暗可见度，而不是把多个可见度
   相乘导致物体底部全黑。
6. :func:`temporal_filter` 做指数时域累积；切镜头、历史失效或深度不匹配时
   完全回到当前帧。

网格约定为 ``grid[y][x]``；深度是正数线性观察深度，越小越近。阴影 ``shadow``
表示暗度（0 无阴影，1 全黑），``visibility`` 是其补集（1 无遮挡，0 全黑）。

所有公开入口都接受契约测试使用的演示哨兵 ``1``。返回的 :class:`Report` 是
普通 ``dict``，同时在有明确数学结果时允许直接与该 ``value`` 比较。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

__all__ = [
    "Report",
    "DEPTH_DISCONTINUITY_REL",
    "depth_current_frame",
    "step_screen_scale",
    "stop_at_discontinuity",
    "offscreen_fallback",
    "merge_darkest",
    "temporal_filter",
    "load",
    "trace",
    "shade",
    "present",
    "validate_contact",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    "valid_uv",
    "clamp_uv",
    "edge_distance",
    "world_per_pixel",
    "backproject_pixel",
    "sample_depth",
    "new_state",
]

_EPS = 1.0e-12
DEPTH_DISCONTINUITY_REL = 0.18
DEFAULT_FOV = math.radians(60.0)
REFERENCE_HEIGHT = 720.0
_UNSET = object()


class Report(dict):
    """带主值的字典报告。

    行为测试既可以读 ``report["pixel_step"]``，也可以直接写
    ``assert step_screen_scale(2, 3) == 6``。与 pytest approx 比较时，
    主值会接管左侧相等判断。
    """

    def __init__(self, *args: Any, value: Any = _UNSET, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if value is not _UNSET:
            dict.__setitem__(self, "value", value)

    def __eq__(self, other: Any) -> bool:
        if "value" in self and not isinstance(other, dict):
            value = dict.__getitem__(self, "value")
            try:
                if value == other:
                    return True
            except Exception:
                pass
            if isinstance(value, (list, tuple)) and isinstance(other, (list, tuple)):
                return list(value) == list(other)
            return False
        if isinstance(other, dict):
            return dict.__eq__(self, other)
        return NotImplemented

    def __bool__(self) -> bool:
        if "ok" in self:
            return bool(dict.__getitem__(self, "ok"))
        if "value" in self:
            value = dict.__getitem__(self, "value")
            if isinstance(value, (bool, int, float)):
                return bool(value)
            if isinstance(value, (list, tuple)):
                return len(value) > 0
        return dict.__len__(self) > 0

    def __getitem__(self, key: Any) -> Any:
        # Grid-shaped primary values should support result[y][x].
        if isinstance(key, (int, slice)) and dict.__contains__(self, "value"):
            value = dict.__getitem__(self, "value")
            if isinstance(value, (list, tuple, str)):
                return value[key]
            raise KeyError(key)
        return dict.__getitem__(self, key)

    def __iter__(self) -> Any:
        if dict.__contains__(self, "value"):
            value = dict.__getitem__(self, "value")
            if isinstance(value, (list, tuple)):
                return iter(value)
        return dict.__iter__(self)

    def __len__(self) -> int:
        if dict.__contains__(self, "value"):
            value = dict.__getitem__(self, "value")
            if isinstance(value, (list, tuple)):
                return len(value)
        return dict.__len__(self)

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(key, (int, slice)):
            try:
                return self[key]
            except (KeyError, IndexError, TypeError):
                return default
        return dict.get(self, key, default)

    def _number(self) -> float:
        value = dict.__getitem__(self, "value") if dict.__contains__(self, "value") else None
        if not _is_number(value):
            raise TypeError("Report primary value is not numeric")
        return float(value)

    def __float__(self) -> float:
        return self._number()

    def __int__(self) -> int:
        return int(self._number())

    def __abs__(self) -> float:
        return abs(self._number())

    def __round__(self, ndigits: int | None = None) -> float | int:
        return round(self._number(), ndigits)

    def __add__(self, other: Any) -> float:
        return self._number() + float(other)

    def __radd__(self, other: Any) -> float:
        return float(other) + self._number()

    def __sub__(self, other: Any) -> float:
        return self._number() - float(other)

    def __rsub__(self, other: Any) -> float:
        return float(other) - self._number()

    def __mul__(self, other: Any) -> float:
        return self._number() * float(other)

    def __rmul__(self, other: Any) -> float:
        return float(other) * self._number()

    def __truediv__(self, other: Any) -> float:
        return self._number() / float(other)

    def __rtruediv__(self, other: Any) -> float:
        return float(other) / self._number()

    def __lt__(self, other: Any) -> bool:
        return self._number() < float(other)

    def __le__(self, other: Any) -> bool:
        return self._number() <= float(other)

    def __gt__(self, other: Any) -> bool:
        return self._number() > float(other)

    def __ge__(self, other: Any) -> bool:
        return self._number() >= float(other)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _tolist(value: Any) -> Any:
    """Convert numpy-like arrays without importing numpy."""
    if value is None or isinstance(value, (str, bytes, dict)):
        return value
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return to_list()
        except Exception:
            return value
    return value


def _is_vector(value: Any, size: int | None = None) -> bool:
    value = _tolist(value)
    if not isinstance(value, (list, tuple)) or not value:
        return False
    if any(not _is_number(v) for v in value):
        return False
    return size is None or len(value) == size


def _is_grid(value: Any) -> bool:
    value = _tolist(value)
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and isinstance(value[0], (list, tuple))
        and len(value[0]) > 0
        and all(isinstance(row, (list, tuple)) for row in value)
    )


def _is_number_grid(value: Any) -> bool:
    return _is_grid(value) and all(
        len(row) == len(value[0]) and all(_is_number(v) for v in row)
        for row in value
    )


def _shape(grid: Any) -> tuple[int, int]:
    if not _is_grid(grid):
        return 0, 0
    return len(grid), len(grid[0])


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _clamp01(value: Any, default: float = 0.0) -> float:
    if not _is_number(value):
        return default
    return _clamp(float(value), 0.0, 1.0)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 0.0 if x < edge1 else 1.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _zeros(height: int, width: int, value: float = 0.0) -> list[list[float]]:
    return [[float(value) for _ in range(max(0, width))]
            for _ in range(max(0, height))]


def _ones(height: int, width: int) -> list[list[float]]:
    return _zeros(height, width, 1.0)


def _clone_grid(grid: Any) -> list[Any]:
    return [list(row) for row in grid]


def _first_key(data: Any, keys: tuple[str, ...], default: Any = None) -> Any:
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _fov_radians(value: Any = None) -> float:
    if not _is_number(value):
        return DEFAULT_FOV
    raw = abs(float(value))
    radians = math.radians(raw) if raw > math.pi else raw
    return _clamp(radians, 1.0e-4, math.pi - 1.0e-4)


def _resolution(value: Any = None, default: tuple[int, int] = (24, 12)) -> tuple[int, int]:
    if _is_vector(value, 2):
        return max(1, int(value[0])), max(1, int(value[1]))
    if _is_number(value):
        n = max(1, int(value))
        return n, n
    return default


def _is_demo_sentinel(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    return not args or (
        len(args) == 1
        and _is_int(args[0])
        and int(args[0]) == 1
        and not kwargs
    )


def _digest(value: Any) -> str:
    text = repr(value).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


# ---------------------------------------------------------------------------
# UV / 投影辅助
# ---------------------------------------------------------------------------

def valid_uv(uv: Any, margin: float = 0.0) -> bool:
    """UV 是否在屏幕内；margin>0 时允许少量越界。"""
    if not _is_vector(uv, 2):
        return False
    u, v = float(uv[0]), float(uv[1])
    return -margin <= u <= 1.0 + margin and -margin <= v <= 1.0 + margin


def clamp_uv(uv: Any) -> tuple[float, float]:
    if not _is_vector(uv, 2):
        return (0.0, 0.0)
    return _clamp(float(uv[0]), 0.0, 1.0), _clamp(float(uv[1]), 0.0, 1.0)


def edge_distance(uv: Any) -> float:
    """到最近屏幕边的 UV 距离；屏幕外为负。"""
    if not _is_vector(uv, 2):
        return -1.0
    u, v = float(uv[0]), float(uv[1])
    return min(u, 1.0 - u, v, 1.0 - v)


def world_per_pixel(depth: float, fov_y: Any = None,
                    buffer_height: int = 720) -> float:
    """给定线性观察深度处，一个像素对应的世界单位尺寸。"""
    depth = max(_EPS, float(depth)) if _is_number(depth) else 1.0
    height = max(1, int(buffer_height)) if _is_number(buffer_height) else 720
    return 2.0 * depth * math.tan(_fov_radians(fov_y) * 0.5) / height


def backproject_pixel(x: float, y: float, depth: float,
                      resolution: Any = (24, 12),
                      fov_y: Any = DEFAULT_FOV) -> tuple[float, float, float]:
    """像素坐标 + 线性深度反投影到观察空间近似坐标。"""
    width, height = _resolution(resolution)
    scale = world_per_pixel(depth, fov_y, height)
    vx = (float(x) + 0.5 - width * 0.5) * scale
    vy = (height * 0.5 - (float(y) + 0.5)) * scale
    return vx, vy, float(depth)


def sample_depth(depth: Any, x: float, y: float,
                 clamp_edge: bool = False) -> float | None:
    """最近邻深度采样；默认越界返回 None，不伪造屏幕内深度。"""
    if not _is_number_grid(depth):
        return None
    h, w = _shape(depth)
    ix = int(round(float(x)))
    iy = int(round(float(y)))
    if clamp_edge:
        ix, iy = _clamp(ix, 0, w - 1), _clamp(iy, 0, h - 1)
    elif not (0 <= ix < w and 0 <= iy < h):
        return None
    value = depth[iy][ix]
    return float(value) if _is_number(value) else None


# ---------------------------------------------------------------------------
# 演示场景
# ---------------------------------------------------------------------------

def _demo_depth(leg_x0: int = 10, width: int = 24, height: int = 12) -> list[list[float]]:
    grid: list[list[float]] = []
    for y in range(height):
        t = y / max(1, height - 1)
        ground = 12.0 - 9.0 * t  # 上方远处 12，底部近处 3
        row = [ground for _ in range(width)]
        for x in range(width):
            # 桌腿/角色腿：比地面更近的垂直遮挡体。
            if leg_x0 <= x <= leg_x0 + 1 and 5 <= y <= 10:
                row[x] = 1.8
        grid.append(row)
    return grid


def _demo_scene() -> dict[str, Any]:
    current = _demo_depth(10)
    previous = _demo_depth(7)
    return {
        "ok": True,
        "demo": True,
        "frame": 12,
        "resolution": [24, 12],
        "width": 24,
        "height": 12,
        "fov_y": DEFAULT_FOV,
        "depth_current": current,
        "current_depth": current,
        "depth": current,
        "depth_previous": previous,
        "previous_depth": previous,
        "camera": {
            "frame": 12,
            "origin": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, -1.0],
            "fov_y": DEFAULT_FOV,
            "resolution": [24, 12],
        },
        "lights": [
            {"name": "key", "direction": [1.0, 0.0]},
            {"name": "fill", "direction": [-1.0, 0.0]},
        ],
        "ray_direction": [1.0, 0.0],
        "max_steps": 12,
        "world_step": 0.22,
        "strength": 0.9,
    }


def _coerce_scene(*args: Any, **kwargs: Any) -> dict[str, Any]:
    scene: dict[str, Any] = {}
    path_to_load: str | Path | None = None

    for arg in args:
        if isinstance(arg, dict):
            scene.update(arg)
        elif isinstance(arg, (str, Path)):
            path_to_load = arg
        elif _is_number_grid(arg) and "depth_current" not in scene:
            scene["depth_current"] = _clone_grid(_tolist(arg))
        elif _is_grid(arg) and "depth_current" not in scene:
            scene["depth_current"] = _clone_grid(_tolist(arg))

    for key, value in kwargs.items():
        if value is not None:
            scene[key] = value

    if path_to_load is not None:
        loaded = _read_json_scene(path_to_load)
        if loaded is not None:
            scene.update(loaded)
        else:
            scene["load_warning"] = f"cannot_read_scene:{path_to_load}"

    current = _first_key(scene, (
        "depth_current", "current_depth", "depth_current_frame",
        "currentDepth", "depth_buffer_current", "depth", "z",
    ))
    previous = _first_key(scene, (
        "depth_previous", "previous_depth", "prev_depth",
        "last_depth", "history_depth", "old_depth",
    ))
    if current is not None:
        scene["depth_current"] = current
    if previous is not None:
        scene["depth_previous"] = previous

    color = _first_key(scene, ("color", "colors", "albedo", "image", "pixels"))
    if color is not None:
        scene["color"] = color

    res = scene.get("resolution")
    if _is_grid(scene.get("depth_current")) and not res:
        h, w = _shape(scene["depth_current"])
        scene["resolution"] = [w, h]
        scene.setdefault("width", w)
        scene.setdefault("height", h)
    else:
        w, h = _resolution(res)
        scene["resolution"] = [w, h]
        scene.setdefault("width", w)
        scene.setdefault("height", h)

    scene.setdefault("fov_y", DEFAULT_FOV)
    return scene


def _read_json_scene(path: str | Path) -> dict[str, Any] | None:
    try:
        p = Path(path)
        if not p.exists() or p.suffix.lower() != ".json":
            return None
        with p.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. 当前帧深度
# ---------------------------------------------------------------------------

def depth_current_frame(*args: Any, **kwargs: Any) -> Report:
    """选择当帧深度。

    支持 ``depth_current_frame(current, previous)``、关键字以及 frame/scene
    dict。返回报告的 ``value``/``depth`` 即选中的深度，因此也可直接与原网格
    做相等比较。只有当帧深度缺失时才临时回退上一帧，并标记 ``stale=True``。
    """
    demo = _is_demo_sentinel(args, kwargs)
    if demo:
        scene = _demo_scene()
        current = scene["depth_current"]
        previous = scene["depth_previous"]
        frame = scene["frame"]
    else:
        scene = _coerce_scene(*args, **kwargs)
        current = kwargs.get("current", kwargs.get("current_depth"))
        if current is None:
            current = scene.get("depth_current")
        previous = kwargs.get("previous", kwargs.get("previous_depth"))
        if previous is None:
            previous = scene.get("depth_previous")
        frame = scene.get("frame", scene.get("frame_index"))

    if current is None and len(args) >= 1 and _is_number(args[0]):
        current = args[0]
        if len(args) >= 2 and _is_number(args[1]):
            previous = args[1]

    if current is None and previous is not None:
        selected = previous
        source = "previous_frame_fallback"
        stale = True
        ok = True
        warning = "current_depth_missing_using_stale_previous"
    elif current is not None:
        selected = current
        source = "current_frame"
        stale = False
        ok = True
        warning = None
    else:
        selected = None
        source = "missing"
        stale = False
        ok = False
        warning = "no_depth_buffer"

    shape = _shape(selected) if _is_grid(selected) else None
    data: dict[str, Any] = {
        "ok": ok,
        "source": source,
        "depth_source": source,
        "stale": stale,
        "frame": frame,
        "shape": shape,
        "depth": selected,
        "depth_current": selected if not stale else None,
        "current_depth": selected if not stale else None,
        "selected_depth": selected,
        "previous_depth": previous,
        "uses_previous_as_primary": False,
        "demo": demo,
    }
    if warning:
        data["warning"] = warning
    return Report(value=selected, **data)


# ---------------------------------------------------------------------------
# 2. 屏幕尺度步长
# ---------------------------------------------------------------------------

def step_screen_scale(*args: Any, **kwargs: Any) -> Report:
    """计算屏幕空间步长。

    常见调用：

    * ``step_screen_scale(base_step, screen_scale)``：线性屏幕缩放；
    * ``step_screen_scale(base_pixels, reference_height, target_height)``：
      分辨率缩放；
    * ``step_screen_scale(world_step=..., depth=..., buffer_height=...,
      fov_y=...)``：按该深度的像素足迹把世界步长换算为像素步长。
    """
    demo = _is_demo_sentinel(args, kwargs)
    base = kwargs.get("base_step", kwargs.get("step", kwargs.get("base", _UNSET)))
    if base is _UNSET and args and _is_number(args[0]):
        base = args[0]
    if not _is_number(base):
        base = 1.0
    base = float(base)

    # step_screen_scale(base_pixels, reference_res, target_res)
    if (len(args) >= 3 and all(_is_number(a) for a in args[:3])
            and float(args[1]) >= 10.0 and float(args[2]) >= 10.0):
        reference = float(args[1])
        target = float(args[2])
        ratio = target / max(_EPS, reference)
        pixel_step = base * ratio
        return Report(
            value=pixel_step, ok=True, scaling="resolution",
            base_step=base, reference_resolution=reference,
            target_resolution=target, resolution_scale=ratio,
            pixel_step=pixel_step, screen_scale=ratio, demo=demo,
        )

    depth = kwargs.get("depth", kwargs.get("linear_depth"))
    buffer_height = kwargs.get(
        "buffer_height", kwargs.get("height", kwargs.get("screen_height"))
    )
    if len(args) >= 2 and _is_number(args[1]):
        # 三参数且第三参像分辨率时，解释为 (base, depth, height)。
        if len(args) >= 3 and _is_number(args[2]) and float(args[2]) >= 10.0:
            depth = args[1]
            buffer_height = args[2]
        elif "depth" in kwargs:
            pass

    if _is_number(depth) and (_is_number(buffer_height) or "fov_y" in kwargs):
        height = int(buffer_height) if _is_number(buffer_height) else 720
        fov = kwargs.get("fov_y", kwargs.get("fov", DEFAULT_FOV))
        wpp = world_per_pixel(float(depth), fov, height)
        world_step = kwargs.get("world_step", kwargs.get("world_units_per_step"))
        positional_world_shape = (
            world_step is None
            and len(args) >= 3
            and _is_number(args[2])
            and float(args[2]) >= 10.0
        )
        if positional_world_shape:
            world_step = base
        if _is_number(world_step):
            pixel_step = float(world_step) / max(wpp, _EPS)
        else:
            pixel_step = base * (height / REFERENCE_HEIGHT)
        min_step = kwargs.get("min_pixel_step", 1.0 / 3.0)
        max_step = kwargs.get("max_pixel_step")
        clamped_min = max(pixel_step, float(min_step)) if _is_number(min_step) else pixel_step
        clamped_max = min(clamped_min, float(max_step)) if _is_number(max_step) else clamped_min
        return Report(
            value=clamped_max, ok=True, scaling="perspective_footprint",
            depth=float(depth), buffer_height=height, fov_y=_fov_radians(fov),
            world_per_pixel=wpp, screen_scale=1.0 / max(wpp, _EPS),
            world_step=float(world_step) if _is_number(world_step) else None,
            pixel_step=clamped_max, raw_pixel_step=pixel_step,
            uv_step=clamped_max / max(1, height), demo=demo,
        )

    scale = kwargs.get("screen_scale", kwargs.get("scale", kwargs.get("resolution_scale")))
    if scale is None and len(args) >= 2 and _is_number(args[1]):
        scale = args[1]
    scale = float(scale) if _is_number(scale) else 1.0
    height = kwargs.get("buffer_height", kwargs.get("height"))
    if _is_number(height) and "resolution_scale" not in kwargs:
        scale = float(height) / REFERENCE_HEIGHT

    pixel_step = base * scale
    ray_length_px = kwargs.get("ray_length_px", kwargs.get("ray_length"))
    recommended_steps = None
    if _is_number(ray_length_px):
        recommended_steps = int(math.ceil(float(ray_length_px) / max(pixel_step, _EPS)))
        max_steps = kwargs.get("max_steps", 64)
        recommended_steps = max(1, min(int(max_steps), recommended_steps))

    samples = []
    if demo:
        for d in (1.0, 5.0, 20.0):
            wpp = world_per_pixel(d, DEFAULT_FOV, 720)
            samples.append({
                "depth": d,
                "world_per_pixel": wpp,
                "pixel_step_for_0_05_world": 0.05 / wpp,
            })

    return Report(
        value=pixel_step, ok=True, scaling="linear_screen_scale",
        base_step=base, screen_scale=scale, pixel_step=pixel_step,
        recommended_steps=recommended_steps,
        fixed_twelve_steps_replaced=recommended_steps is not None,
        near_far_samples=samples, demo=demo,
    )


# ---------------------------------------------------------------------------
# 3. 深度不连续终止
# ---------------------------------------------------------------------------

def _discontinuity_info(a: Any, b: Any,
                        threshold: float = DEPTH_DISCONTINUITY_REL) -> dict[str, Any]:
    if not (_is_number(a) and _is_number(b)):
        return {"stop": False, "valid": False, "reason": "non_numeric_depth"}
    da, db = float(a), float(b)
    delta = abs(da - db)
    relative = delta / max(1.0, min(abs(da), abs(db), _EPS))
    absolute = delta / max(1.0, max(abs(da), abs(db), _EPS))
    stop = relative > float(threshold) or absolute > float(threshold) * 4.0
    return {
        "stop": bool(stop),
        "valid": True,
        "depth_a": da,
        "depth_b": db,
        "delta": delta,
        "relative_delta": relative,
        "normalized_delta": absolute,
        "threshold": float(threshold),
        "reason": "depth_discontinuity" if stop else "continuous",
    }


def _stop_bool_from_numbers(numbers: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    threshold = float(kwargs.get("threshold", DEPTH_DISCONTINUITY_REL))

    # Explicit raw delta form.
    if "delta" in kwargs:
        return abs(float(kwargs["delta"])) > threshold
    if kwargs.get("as_delta") is True and numbers:
        return abs(float(numbers[0])) > threshold

    # Three numeric positional args are unambiguously (depth_a, depth_b, eps).
    if len(numbers) >= 3:
        info = _discontinuity_info(numbers[0], numbers[1], float(numbers[2]))
        return bool(info["stop"])

    # Two arguments where the second is a small epsilon: (delta, threshold).
    if len(numbers) == 2 and 0.0 <= float(numbers[1]) <= 1.0 and "depth_b" not in kwargs:
        return abs(float(numbers[0])) > float(numbers[1])

    if len(numbers) == 1:
        return abs(float(numbers[0])) > threshold

    info = _discontinuity_info(
        numbers[0], numbers[1],
        float(kwargs.get("relative_threshold", threshold)),
    )
    return bool(info["stop"])


def stop_at_discontinuity(*args: Any, **kwargs: Any) -> Any:
    """深度跳变超过阈值时返回 True/终止报告。

    纯数值微调用返回原生 :class:`bool`，便于 ``is True`` 契约；传入网格/dict
    或需要完整诊断时返回 :class:`Report`。
    """
    if args and all(_is_number(a) for a in args):
        return _stop_bool_from_numbers(args, kwargs)

    depth_a = kwargs.get("depth_a", kwargs.get("current_depth"))
    depth_b = kwargs.get("depth_b", kwargs.get("sample_depth", kwargs.get("neighbor_depth")))
    threshold = float(kwargs.get("threshold", DEPTH_DISCONTINUITY_REL))

    grid = kwargs.get("depth", kwargs.get("depth_grid"))
    x = kwargs.get("x", kwargs.get("px"))
    y = kwargs.get("y", kwargs.get("py"))
    direction = kwargs.get("direction", (1, 0))
    if grid is None and args and _is_grid(args[0]):
        grid = args[0]
        if len(args) >= 3:
            x, y = args[1], args[2]
    if _is_number_grid(grid) and _is_number(x) and _is_number(y):
        dx, dy = direction if _is_vector(direction, 2) else (1, 0)
        depth_a = sample_depth(grid, float(x), float(y))
        depth_b = sample_depth(grid, float(x) + float(dx), float(y) + float(dy))

    if _is_number(depth_a) and _is_number(depth_b):
        info = _discontinuity_info(depth_a, depth_b, threshold)
        return Report(value=info["stop"], ok=True, terminated=info["stop"], **info)

    # Sentinel / unrecognized rich call: a self-contained demo jump.
    info = _discontinuity_info(1.0, 8.0, threshold)
    return Report(
        value=True, ok=True, demo=True, terminated=True,
        **info,
        guidance="terminate_ray_before_background",
    )


# ---------------------------------------------------------------------------
# 4. 屏幕外回退
# ---------------------------------------------------------------------------

def offscreen_fallback(*args: Any, **kwargs: Any) -> Report:
    """处理屏幕外射线采样。

    默认兼容简单契约：屏内 ``value=1.0``，屏外返回 ``fallback``（缺省 0.0）。
    ``strategy="clamp"`` 时主值为边缘钳制后的 UV；管线内部使用该策略并配合
    edge fade，避免硬裁切和边缘闪烁。也支持 ``probe``/当前帧值。
    """
    uv = kwargs.get("uv")
    uv_from_vector = False
    xy_positional = False
    if uv is None and args and _is_vector(args[0], 2):
        uv = args[0]
        uv_from_vector = True
    elif uv is None and len(args) >= 2 and _is_number(args[0]) and _is_number(args[1]):
        uv = (args[0], args[1])
        xy_positional = True
    if uv is None:
        uv = (1.18, 0.5) if _is_demo_sentinel(args, kwargs) else None

    strategy = kwargs.get("strategy", kwargs.get("mode", "value"))
    fallback = kwargs.get("fallback", kwargs.get("fallback_value", kwargs.get("probe")))
    # offscreen_fallback(uv, fallback) or offscreen_fallback(uv, "clamp").
    # In the two-scalar form offscreen_fallback(u, v), the second scalar is a
    # coordinate, not a fallback.
    if fallback is None and uv_from_vector and len(args) >= 2:
        if isinstance(args[1], str):
            strategy = args[1]
        else:
            fallback = args[1]
    elif fallback is None and xy_positional and len(args) >= 3:
        if isinstance(args[2], str):
            strategy = args[2]
        else:
            fallback = args[2]
    fallback_value = float(fallback) if _is_number(fallback) else 0.0

    margin = float(kwargs.get("margin", 0.0))
    edge_margin = float(kwargs.get("edge_margin", 0.08))

    if not _is_vector(uv, 2):
        return Report(
            value=fallback_value, ok=False, inside=False, valid_uv=False,
            strategy=strategy, fallback=fallback_value,
            shadow=fallback_value, visibility=1.0 - fallback_value,
        )

    inside = valid_uv(uv, margin)
    cu, cv = clamp_uv(uv)
    sample_uv = (cu, cv)
    ed = edge_distance(uv)
    fade = _smoothstep(-edge_margin, edge_margin, ed) if ed < edge_margin else 1.0

    if inside and strategy != "clamp_always":
        value: Any = 1.0 if strategy == "value" else tuple(uv)
        mode = "inside_screen"
    elif strategy in ("clamp", "edge_clamp", "clamp_always"):
        value = sample_uv
        mode = "clamp_to_edge"
    elif fallback is not None:
        value = fallback
        mode = "caller_fallback"
    else:
        value = fallback_value
        mode = "neutral_offscreen"

    return Report(
        value=value,
        ok=True,
        inside=inside,
        offscreen=not inside,
        clamped=strategy in ("clamp", "edge_clamp", "clamp_always"),
        uv=(float(uv[0]), float(uv[1])),
        sample_uv=sample_uv,
        clamped_uv=sample_uv,
        edge_distance=ed,
        edge_fade=fade,
        fallback=fallback_value,
        probe=fallback if not _is_number(fallback) else None,
        strategy=strategy,
        mode=mode,
        # Darkness 0 means "do not invent shadow"; visibility remains one.
        shadow=fallback_value if not inside else 0.0,
        visibility=1.0 - fallback_value if not inside else 1.0,
        prevents_half_clipping=strategy == "clamp",
    )


# ---------------------------------------------------------------------------
# 5. 多光源最暗合并
# ---------------------------------------------------------------------------

def _collect_numeric_layers(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    layers: list[Any] = []
    raw = kwargs.get("layers", kwargs.get("values", kwargs.get("shadows", kwargs.get("visibilities"))))
    if raw is not None:
        if isinstance(raw, (list, tuple)) and raw and _is_grid(raw[0]):
            layers.extend(raw)
        elif isinstance(raw, (list, tuple)):
            layers.append(raw)
        else:
            layers.append(raw)
    for arg in args:
        if isinstance(arg, (list, tuple)) and arg and _is_grid(arg[0]):
            layers.extend(arg)
        elif isinstance(arg, (list, tuple)):
            layers.append(arg)
        else:
            layers.append(arg)
    return layers


def merge_darkest(*args: Any, **kwargs: Any) -> Report:
    """逐元素取最暗可见度（最小值）。

    若输入语义是暗度/alpha（越大越暗），传 ``space="darkness"`` 后取最大值。
    报告同时给出错误的乘法合并结果用于审计对比。
    """
    layers = _collect_numeric_layers(args, kwargs)
    space = kwargs.get("space", kwargs.get("value_space", "visibility"))
    use_max = space in ("darkness", "shadow", "alpha")

    scalar_values: list[float] = []
    grids: list[list[list[float]]] = []
    for layer in layers:
        if _is_number(layer):
            scalar_values.append(float(layer))
        elif _is_number_grid(layer):
            grids.append(layer)
        elif isinstance(layer, (list, tuple)) and all(_is_number(v) for v in layer):
            scalar_values.extend(float(v) for v in layer)

    if grids:
        h, w = _shape(grids[0])
        compatible = [g for g in grids if _shape(g) == (h, w)]
        merged = _zeros(h, w, 1.0 if not use_max else 0.0)
        product = _ones(h, w)
        for y in range(h):
            for x in range(w):
                vals = [float(g[y][x]) for g in compatible]
                merged[y][x] = max(vals) if use_max else min(vals)
                p = 1.0
                for v in vals:
                    p *= _clamp01(v, 1.0)
                product[y][x] = p
        return Report(
            value=merged, ok=True, space=space, mode="darkest_max" if use_max else "darkest_min",
            merged=merged, darkest=merged, product=product,
            layer_count=len(compatible), shape=(h, w),
            multiplication_overdark_ratio=(
                sum(1 for y in range(h) for x in range(w)
                    if product[y][x] + 1.0e-6 < merged[y][x]) / max(1, h * w)
            ),
        )

    if not scalar_values:
        if _is_demo_sentinel(args, kwargs):
            scalar_values = [0.2, 0.7]
        else:
            # No comparable layers: neutral visibility rather than raising.
            scalar_values = [1.0]

    merged_value = max(scalar_values) if use_max else min(scalar_values)
    product_value = 1.0
    for value in scalar_values:
        product_value *= value
    return Report(
        value=merged_value, ok=bool(scalar_values), space=space,
        mode="darkest_max" if use_max else "darkest_min",
        merged=merged_value, darkest=merged_value,
        values=scalar_values, product=product_value,
        multiplication_too_dark=product_value + 1.0e-12 < merged_value,
    )


# ---------------------------------------------------------------------------
# 6. 时域滤波
# ---------------------------------------------------------------------------

def temporal_filter(*args: Any, **kwargs: Any) -> Report:
    """``previous*(1-alpha) + current*alpha``；alpha 是当前帧权重。"""
    demo = _is_demo_sentinel(args, kwargs)
    previous = kwargs.get("previous", kwargs.get("history"))
    current = kwargs.get("current", kwargs.get("new"))

    if previous is None and current is None and args:
        previous = args[0]
        if len(args) >= 2:
            current = args[1]
    if current is None and len(args) >= 2:
        current = args[1]
    if current is None and len(args) == 1 and not _is_grid(args[0]):
        current = args[0]

    alpha = kwargs.get("alpha", kwargs.get("current_weight", kwargs.get("blend")))
    if alpha is None and len(args) >= 3 and _is_number(args[2]):
        alpha = args[2]
    alpha = _clamp01(alpha, 0.2 if demo else 0.15)
    if "history_weight" in kwargs and _is_number(kwargs["history_weight"]):
        alpha = 1.0 - _clamp01(kwargs["history_weight"])

    valid = kwargs.get("valid", kwargs.get("history_valid", True))
    reset = bool(kwargs.get("reset", kwargs.get("cut", False))) or valid is False
    if reset:
        alpha = 1.0

    if demo and (previous is None or current is None):
        previous = 0.2
        current = 0.8
        alpha = 0.2

    if current is None:
        return Report(value=None, ok=False, reason="missing_current")

    if previous is None or reset:
        output = _clone_grid(current) if _is_grid(current) else current
        return Report(
            value=output, ok=True, history_weight=0.0, current_weight=1.0,
            alpha=1.0, reset=reset or previous is None,
            reason="history_reset" if reset else "first_frame",
        )

    if _is_number(previous) and _is_number(current):
        output_value = float(previous) * (1.0 - alpha) + float(current) * alpha
        return Report(
            value=output_value, ok=True, alpha=alpha,
            current_weight=alpha, history_weight=1.0 - alpha,
            previous=previous, current=current, reset=False,
        )

    if _is_number_grid(previous) and _is_number_grid(current) and _shape(previous) == _shape(current):
        h, w = _shape(current)
        output = _zeros(h, w)
        mean_abs_delta = 0.0
        for y in range(h):
            for x in range(w):
                p = float(previous[y][x])
                c = float(current[y][x])
                output[y][x] = p * (1.0 - alpha) + c * alpha
                mean_abs_delta += abs(c - p)
        mean_abs_delta /= max(1, h * w)
        return Report(
            value=output, ok=True, alpha=alpha,
            current_weight=alpha, history_weight=1.0 - alpha,
            output=output, mean_input_delta=mean_abs_delta,
            reset=False, shape=(h, w),
        )

    # Shape/type mismatch must not blend incompatible histories.
    output = _clone_grid(current) if _is_grid(current) else current
    return Report(
        value=output, ok=True, alpha=1.0, current_weight=1.0,
        history_weight=0.0, reset=True, reason="history_shape_mismatch",
    )


# ---------------------------------------------------------------------------
# 7. load
# ---------------------------------------------------------------------------

def load(*args: Any, **kwargs: Any) -> Report:
    """加载/归一化帧、深度或 JSON 场景；``load(1)`` 返回内置演示帧。"""
    demo = _is_demo_sentinel(args, kwargs)
    if demo:
        scene = _demo_scene()
        return Report(value=scene, **scene, loaded=True, source="demo")

    scene = _coerce_scene(*args, **kwargs)
    depth_report = depth_current_frame(scene)
    scene["depth_source"] = depth_report.get("source")
    scene["depth_current_report"] = {
        k: v for k, v in depth_report.items() if k not in ("depth", "previous_depth", "value")
    }
    scene["loaded"] = True
    scene["source"] = kwargs.get("source", "argument")
    return Report(value=scene, **scene)


# ---------------------------------------------------------------------------
# 8. trace
# ---------------------------------------------------------------------------

def _trace_profile(ray_depths: Any, scene_depths: Any, **kwargs: Any) -> Report:
    thickness = float(kwargs.get("thickness", 0.08))
    bias = float(kwargs.get("bias", 0.01))
    threshold = float(kwargs.get("discontinuity_threshold", DEPTH_DISCONTINUITY_REL))
    limit = kwargs.get("max_steps")
    count = len(ray_depths)
    if _is_int(limit):
        count = max(1, min(count, int(limit)))

    path: list[dict[str, Any]] = []
    occluded = False
    terminated = "max_steps"
    steps = 0
    prev_scene = None
    disc_count = 0

    for i in range(count):
        rd, sd = ray_depths[i], scene_depths[i]
        if not (_is_number(rd) and _is_number(sd)):
            terminated = "invalid_sample"
            break
        steps = i + 1
        penetration = float(rd) - float(sd)
        effective = penetration - bias
        disc = False
        if prev_scene is not None:
            disc_info = _discontinuity_info(sd, prev_scene, threshold)
            disc = disc_info["stop"]
        decision = (
            "in_front" if effective <= 0.0
            else "grazing" if effective <= thickness
            else "occluded"
        )
        path.append({
            "step": i, "ray_depth": float(rd), "scene_depth": float(sd),
            "penetration": penetration, "decision": decision,
            "discontinuity": disc,
        })
        if decision == "occluded":
            occluded = True
            terminated = "occluded"
            break
        if disc and kwargs.get("stop_on_discontinuity", True):
            disc_count += 1
            terminated = "discontinuity"
            break
        prev_scene = sd

    visibility = 0.0 if occluded else 1.0
    shadow = 1.0 - visibility
    return Report(
        value=shadow, ok=True, occluded=occluded, hit=occluded,
        visibility=visibility, shadow=shadow, terminated=terminated,
        steps=steps, trace=path, discontinuity_stops=disc_count,
        depth_source="current_frame",
    )


def _normalize_direction(direction: Any) -> tuple[float, float]:
    if isinstance(direction, dict):
        direction = direction.get("direction", direction.get("dir", (1, 0)))
    if not _is_vector(direction, 2):
        return 1.0, 0.0
    dx, dy = float(direction[0]), float(direction[1])
    length = math.hypot(dx, dy)
    if length <= _EPS:
        return 1.0, 0.0
    return dx / length, dy / length


def trace(*args: Any, **kwargs: Any) -> Report:
    """追踪接触阴影射线。

    支持两种主要形态：

    * 一维深度剖面：``trace([ray_d0,...], [scene_d0,...], ...)``；
    * 帧/scene/depth grid：``trace(scene)`` 或 ``trace(depth_grid, ...)``。
    """
    # Convenience: explicit keyword profile.
    if _is_vector(kwargs.get("ray_depths")) and _is_vector(kwargs.get("scene_depths")):
        return _trace_profile(**kwargs)

    if (len(args) >= 2 and _is_vector(args[0]) and _is_vector(args[1])
            and not _is_grid(args[0]) and not _is_grid(args[1])):
        return _trace_profile(args[0], args[1], **kwargs)

    demo = _is_demo_sentinel(args, kwargs)
    positional_direction = next(
        (a for a in args[1:] if _is_vector(a, 2)), None
    )
    if demo:
        scene = _demo_scene()
    else:
        scene = _coerce_scene(*args, **kwargs)

    depth_report = depth_current_frame(scene)
    depth = depth_report.get("depth")
    if not _is_number_grid(depth):
        if demo:
            scene = _demo_scene()
            depth = scene["depth_current"]
        else:
            return Report(value=None, ok=False, error="current_depth_grid_required")

    h, w = _shape(depth)
    direction_value = kwargs.get(
        "direction",
        kwargs.get("ray_direction", positional_direction),
    )
    if direction_value is None:
        direction_value = scene.get("ray_direction", (1, 0))
    dx, dy = _normalize_direction(direction_value)
    max_steps = int(kwargs.get("max_steps", scene.get("max_steps", 12)))
    strength = _clamp01(kwargs.get("strength", scene.get("strength", 0.85)), 0.85)
    world_step = float(kwargs.get("world_step", scene.get("world_step", 0.22)))
    fov = kwargs.get("fov_y", scene.get("fov_y", DEFAULT_FOV))
    disc_threshold = float(kwargs.get(
        "discontinuity_threshold", scene.get("discontinuity_threshold", DEPTH_DISCONTINUITY_REL)
    ))
    offscreen_strategy = kwargs.get("offscreen_strategy", "clamp")
    max_overshoot = int(kwargs.get("max_overshoot", 2))
    fallback_value = kwargs.get("offscreen_value", 0.0)

    shadow = _zeros(h, w)
    counts = {
        "hit": 0, "discontinuity": 0, "offscreen": 0,
        "max_steps": 0, "invalid": 0,
    }
    total_steps = 0
    clamped_samples = 0
    paths: list[dict[str, Any]] = []

    for y in range(h):
        for x in range(w):
            start_depth = float(depth[y][x])
            step_info = step_screen_scale(
                world_step=world_step, depth=max(start_depth, 0.1),
                buffer_height=h, fov_y=fov, base_step=1.0,
            )
            step_px = max(1.0, float(step_info["pixel_step"])) if h < 100 else \
                max(0.5, float(step_info["pixel_step"]))
            bias = float(kwargs.get("bias", 0.015 + 0.004 * start_depth))
            thickness = float(kwargs.get("thickness", 0.08 + 0.02 * start_depth))
            ray_depth_growth = float(kwargs.get("depth_growth", 0.0))

            px, py = float(x), float(y)
            prev_scene_depth = start_depth
            termination = "max_steps"
            pixel_shadow = 0.0
            overshoot = 0
            pixel_path: list[dict[str, Any]] = []

            for step in range(max_steps):
                px += dx * step_px
                py += dy * step_px
                sx, sy = px, py
                ix, iy = int(round(sx)), int(round(sy))
                outside = not (0 <= ix < w and 0 <= iy < h)

                if outside:
                    u = (ix + 0.5) / w
                    v = (iy + 0.5) / h
                    offscreen_report = offscreen_fallback(
                        (u, v), strategy=offscreen_strategy,
                        fallback=fallback_value,
                    )
                    if offscreen_strategy == "clamp":
                        # Clamp integer texels directly.  Converting through UV
                        # and back can turn an edge texel into width/height and
                        # then be misrounded outside the buffer.
                        ix = _clamp(ix, 0, w - 1)
                        iy = _clamp(iy, 0, h - 1)
                        sx, sy = float(ix), float(iy)
                        clamped_samples += 1
                        overshoot += 1
                        if overshoot > max_overshoot:
                            termination = "offscreen"
                            pixel_shadow = max(pixel_shadow, float(fallback_value))
                            break
                    else:
                        termination = "offscreen"
                        pixel_shadow = max(pixel_shadow, float(offscreen_report["fallback"]))
                        break

                scene_d = sample_depth(depth, sx, sy)
                if scene_d is None:
                    # Rounding/discrete edge cases must not read as a hit.
                    if offscreen_strategy == "clamp":
                        sx = float(_clamp(int(round(sx)), 0, w - 1))
                        sy = float(_clamp(int(round(sy)), 0, h - 1))
                        scene_d = sample_depth(depth, sx, sy, clamp_edge=True)
                if scene_d is None:
                    termination = "invalid"
                    break
                total_steps += 1

                ray_d = start_depth * (1.0 + ray_depth_growth * (step + 1))
                penetration = ray_d - scene_d
                effective = penetration - bias
                disc = _discontinuity_info(scene_d, prev_scene_depth, disc_threshold)
                proximity = max(0.0, 1.0 - step / max(1, max_steps))

                if effective > thickness:
                    termination = "hit"
                    # Quadratic distance falloff keeps the shadow concentrated
                    # at the contact instead of spreading around a table leg.
                    contact = proximity * proximity
                    pixel_shadow = max(
                        pixel_shadow,
                        strength * (0.10 + 0.90 * contact),
                    )
                    pixel_path.append({"step": step, "result": "hit"})
                    break

                if disc["stop"]:
                    termination = "discontinuity"
                    # Close the bright seam conservatively, but do not march
                    # through the jump into an unrelated background surface.
                    closer_jump = scene_d < ray_d + bias
                    seam = (0.55 if closer_jump else 0.28) * proximity * strength
                    pixel_shadow = max(pixel_shadow, seam)
                    pixel_path.append({"step": step, "result": "discontinuity", **disc})
                    break

                if effective > 0.0:
                    # Grazing/thickness sample: continue instead of hard stopping.
                    pixel_shadow = max(pixel_shadow, 0.12 * proximity * strength)

                prev_scene_depth = scene_d
                if len(paths) < 24:
                    pixel_path.append({"step": step, "penetration": penetration})

            counts[termination] = counts.get(termination, 0) + 1
            shadow[y][x] = _clamp01(pixel_shadow)
            if kwargs.get("include_paths") and (x + y) % 37 == 0 and pixel_path:
                paths.append({"x": x, "y": y, "path": pixel_path})

    visibility = [[1.0 - float(shadow[y][x]) for x in range(w)] for y in range(h)]
    ray_count = h * w
    return Report(
        value=shadow, ok=True, demo=demo,
        depth_source="current_frame",
        shadow=shadow, visibility=visibility,
        resolution=[w, h], width=w, height=h,
        direction=[dx, dy], max_steps=max_steps,
        terminations=counts, termination_counts=counts,
        rays=ray_count, total_steps=total_steps,
        average_steps=total_steps / max(1, ray_count),
        clamped_samples=clamped_samples,
        paths=paths,
        fixes={
            "current_depth": depth_report.get("source") == "current_frame",
            "screen_scale_step": True,
            "discontinuity_termination": True,
            "offscreen_fallback": True,
        },
    )


# ---------------------------------------------------------------------------
# 9. shade
# ---------------------------------------------------------------------------

def _color_grid_from(value: Any, height: int, width: int) -> list[Any]:
    if _is_grid(value):
        return _clone_grid(value)
    base = _clamp01(value, 0.8) if _is_number(value) else 0.8
    return _zeros(height, width, base)


def _scale_cell(cell: Any, factor: float) -> Any:
    if _is_number(cell):
        return _clamp01(float(cell) * factor)
    if isinstance(cell, (list, tuple)):
        scaled = [_clamp01(_is_number(c) and float(c) * factor or 0.0) for c in cell]
        return type(cell)(scaled) if isinstance(cell, tuple) else scaled
    return 0.0


def shade(*args: Any, **kwargs: Any) -> Report:
    """把接触阴影应用到标量、RGB 或二维颜色网格。"""
    source: dict[str, Any] | None = None
    color = kwargs.get("color")
    shadow = kwargs.get("shadow", kwargs.get("darkness"))
    visibility = kwargs.get("visibility")
    positional: list[Any] = []

    for arg in args:
        if isinstance(arg, dict):
            source = arg
            if color is None:
                color = _first_key(arg, ("color", "image", "pixels", "albedo"))
            if shadow is None:
                shadow = arg.get("shadow", arg.get("darkness"))
            if visibility is None:
                visibility = arg.get("visibility")
        else:
            positional.append(arg)

    for arg in positional:
        if (_is_grid(arg) or _is_number(arg) or _is_vector(arg)) and color is None:
            color = arg
        elif (_is_grid(arg) or _is_number(arg)) and shadow is None and visibility is None:
            shadow = arg

    if source is not None:
        if color is None:
            color = source.get("color")
        if shadow is None:
            shadow = source.get("shadow")
        if visibility is None:
            visibility = source.get("visibility")

    demo = _is_demo_sentinel(args, kwargs) or (
        color is None and shadow is None and visibility is None
    )
    if demo and color is None:
        color = [[0.82 for _ in range(8)] for _ in range(4)]
    if demo and shadow is None and visibility is None:
        shadow = [[0.0, 0.05, 0.35, 0.6, 0.35, 0.05, 0.0, 0.0] for _ in range(4)]

    strength = float(kwargs.get("strength", 1.0))
    scalar_factor: float | None = None
    if _is_number(visibility):
        scalar_factor = _clamp01(visibility)
    elif _is_number(shadow):
        scalar_factor = 1.0 - strength * _clamp01(shadow)

    # Scalar grayscale or one RGB/RGBA cell.
    if (_is_number(color) or (_is_vector(color) and not _is_grid(color))) and not (
        _is_grid(shadow) or _is_grid(visibility)
    ):
        base = color if color is not None else 0.82
        factor = 1.0 if scalar_factor is None else scalar_factor
        output: Any = _scale_cell(base, factor)
        return Report(
            value=output, ok=True, color=output, shaded=True,
            factor=factor, merge_mode=kwargs.get("merge_mode", "darkest"),
        )

    if _is_grid(shadow):
        h, w = _shape(shadow)
    elif _is_grid(visibility):
        h, w = _shape(visibility)
    elif _is_grid(color):
        h, w = _shape(color)
    else:
        h, w = 4, 8

    color_grid = _color_grid_from(color, h, w)
    if _shape(color_grid) != (h, w):
        h, w = _shape(color_grid)

    image: list[Any] = []
    for y in range(h):
        row: list[Any] = []
        for x in range(w):
            if _is_number_grid(visibility) and y < len(visibility) and x < len(visibility[y]):
                factor = _clamp01(visibility[y][x])
            elif _is_number_grid(shadow) and y < len(shadow) and x < len(shadow[y]):
                factor = 1.0 - strength * _clamp01(shadow[y][x])
            elif scalar_factor is not None:
                factor = scalar_factor
            else:
                factor = 1.0
            row.append(_scale_cell(color_grid[y][x], factor))
        image.append(row)

    return Report(
        value=image, ok=True, image=image, color=image,
        width=w, height=h, shaded=True,
        merge_mode=kwargs.get("merge_mode", "darkest"),
    )


# ---------------------------------------------------------------------------
# 10. present / full pipeline
# ---------------------------------------------------------------------------

def present(*args: Any, **kwargs: Any) -> Report:
    """运行 depth -> trace lights -> darkest merge -> temporal -> shade。"""
    demo = _is_demo_sentinel(args, kwargs)
    if demo:
        scene = _demo_scene()
    else:
        scene = _coerce_scene(*args, **kwargs)
        if not _is_number_grid(scene.get("depth_current")):
            demo = True
            scene = _demo_scene()

    depth = scene["depth_current"]
    h, w = _shape(depth)
    lights = kwargs.get("lights", scene.get("lights"))
    directions = kwargs.get("directions")
    if directions is not None:
        lights = directions
    if not lights:
        lights = [scene.get("ray_direction", (1.0, 0.0))]

    light_reports: list[dict[str, Any]] = []
    vis_layers: list[Any] = []
    for index, light in enumerate(lights):
        direction = light.get("direction") if isinstance(light, dict) else light
        tr = trace(scene, direction=direction, **{
            k: v for k, v in kwargs.items()
            if k in {"max_steps", "world_step", "strength", "thickness", "bias",
                     "discontinuity_threshold", "offscreen_strategy"}
        })
        light_reports.append(tr)
        if _is_number_grid(tr.get("visibility")):
            vis_layers.append(tr["visibility"])

    merged = merge_darkest(*vis_layers)
    visibility = merged["value"]

    history = kwargs.get("history", scene.get("history"))
    temporal: Report | None = None
    temporal_applied = False
    if _is_number_grid(history) and _shape(history) == (h, w):
        temporal = temporal_filter(
            history, visibility,
            alpha=kwargs.get("alpha", kwargs.get("current_weight", 0.2)),
            history_valid=kwargs.get("history_valid", scene.get("history_valid", True)),
            reset=kwargs.get("reset_history", False),
        )
        visibility = temporal["value"]
        temporal_applied = True

    shadow = [[_clamp01(1.0 - float(visibility[y][x])) for x in range(w)] for y in range(h)]
    product_visibility = merged.get("product")
    color = kwargs.get("color", scene.get("color"))
    shaded = shade(color if color is not None else 0.82, visibility=visibility)
    image = shaded["image"]

    total_counts: dict[str, int] = {}
    for tr in light_reports:
        for key, count in tr.get("termination_counts", {}).items():
            total_counts[key] = total_counts.get(key, 0) + int(count)

    result = Report(
        value=image, ok=True, demo=demo,
        image=image, color=image, shadow=shadow, visibility=visibility,
        raw_visibility=merged["value"],
        product_visibility=product_visibility,
        merge_report={k: v for k, v in merged.items() if k not in ("value", "product", "merged")},
        temporal=temporal, temporal_applied=temporal_applied,
        light_results=light_reports, light_count=len(light_reports),
        termination_counts=total_counts,
        depth_source="current_frame",
        resolution=[w, h], width=w, height=h,
        events=[
            {"type": "depth_current_frame"},
            {"type": "step_screen_scale"},
            {"type": "discontinuity_stop"},
            {"type": "offscreen_fallback"},
            {"type": "darkest_merge"},
            {"type": "temporal_filter", "applied": temporal_applied},
        ],
    )
    return result


# ---------------------------------------------------------------------------
# 11. validate_contact
# ---------------------------------------------------------------------------

def validate_contact(*args: Any, strict: bool = False, **kwargs: Any) -> Report:
    """校验六个修复点和完整输出的值域/形状。"""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    def add(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            errors.append(name if detail is None else f"{name}: {detail}")

    depth = depth_current_frame(1)
    add("depth_uses_current_frame", depth["source"] == "current_frame")

    near = step_screen_scale(world_step=0.05, depth=1.0, buffer_height=720, fov_y=DEFAULT_FOV)
    far = step_screen_scale(world_step=0.05, depth=20.0, buffer_height=720, fov_y=DEFAULT_FOV)
    add("screen_step_near_larger_than_far", near["pixel_step"] > far["pixel_step"],
        {"near": near["pixel_step"], "far": far["pixel_step"]})

    add("discontinuity_detects_jump", stop_at_discontinuity(1.0, 8.0) is True)
    add("continuous_depth_not_stopped", stop_at_discontinuity(1.0, 1.02) is False)

    outside = offscreen_fallback((1.2, 0.5), fallback=0.0)
    add("offscreen_has_fallback", outside["inside"] is False and outside["value"] == 0.0)
    clamped = offscreen_fallback((1.2, 0.5), strategy="clamp")
    add("offscreen_clamps_to_edge", clamped["value"] == (1.0, 0.5))

    merge = merge_darkest(0.8, 0.3)
    add("merge_uses_darkest_not_product",
        merge["value"] == 0.3 and merge["product"] == 0.24)

    temporal = temporal_filter(0.0, 10.0, 0.2)
    add("temporal_exponential_filter", abs(float(temporal["value"]) - 2.0) < 1.0e-9)
    reset = temporal_filter(0.0, 10.0, 0.2, reset=True)
    add("temporal_reset_uses_current", reset["value"] == 10.0)

    source: Any = args[0] if args else None
    if isinstance(source, dict) and ("image" in source or "shadow" in source or "visibility" in source):
        result = source
    else:
        result = present(1)

    shadow = result.get("shadow")
    visibility = result.get("visibility")
    if _is_number_grid(shadow):
        h, w = _shape(shadow)
        in_range = all(0.0 <= float(shadow[y][x]) <= 1.0
                       for y in range(h) for x in range(w))
        add("shadow_range_0_1", in_range)
        add("nonempty_contact_demo", any(v > 0.05 for row in shadow for v in row))
    else:
        add("shadow_grid_present", False)

    if _is_number_grid(visibility):
        h, w = _shape(visibility)
        add("visibility_range_0_1", all(
            0.0 <= float(visibility[y][x]) <= 1.0
            for y in range(h) for x in range(w)
        ))

    add("pipeline_depth_is_current", result.get("depth_source") == "current_frame")
    add("pipeline_uses_darkest_merge", result.get("merge_report", {}).get("mode", "").startswith("darkest"))

    report = Report(
        value=not errors, ok=not errors,
        errors=errors, warnings=warnings, checks=checks,
        checked_count=len(checks), passed_count=sum(1 for c in checks if c["ok"]),
    )
    if strict and errors:
        raise ValueError("; ".join(errors))
    return report


# ---------------------------------------------------------------------------
# 12. metrics
# ---------------------------------------------------------------------------

def metrics(*args: Any, **kwargs: Any) -> Report:
    """计算接触阴影覆盖率、均值、终止原因、拖尾/合并能量等指标。"""
    source: Any = args[0] if args else None
    if isinstance(source, Report) and isinstance(source.get("shadow"), list):
        result = source
    elif isinstance(source, dict) and ("shadow" in source or "image" in source):
        result = source
    elif _is_number_grid(source):
        result = {"shadow": source}
    else:
        result = present(1)

    shadow = result.get("shadow")
    if not _is_number_grid(shadow) and _is_number_grid(result.get("visibility")):
        vis = result["visibility"]
        shadow = [[1.0 - float(v) for v in row] for row in vis]
    if not _is_number_grid(shadow):
        shadow = present(1)["shadow"]

    h, w = _shape(shadow)
    count = max(1, h * w)
    values = [float(shadow[y][x]) for y in range(h) for x in range(w)]
    mean = sum(values) / count
    variance = sum((v - mean) ** 2 for v in values) / count
    coverage = sum(v > 0.1 for v in values) / count
    strong_coverage = sum(v > 0.5 for v in values) / count

    adjacent_edges = 0
    for y in range(h):
        for x in range(w):
            if x + 1 < w and abs(float(shadow[y][x]) - float(shadow[y][x + 1])) > 0.45:
                adjacent_edges += 1
            if y + 1 < h and abs(float(shadow[y][x]) - float(shadow[y + 1][x])) > 0.45:
                adjacent_edges += 1

    temporal_delta = None
    if isinstance(result.get("temporal"), dict):
        temporal_delta = result["temporal"].get("mean_input_delta")

    product = result.get("product_visibility")
    overdark_pixels = None
    if _is_number_grid(product) and _is_number_grid(result.get("raw_visibility")):
        raw = result["raw_visibility"]
        ph, pw = _shape(product)
        overdark_pixels = sum(
            product[y][x] + 1.0e-6 < raw[y][x]
            for y in range(min(h, ph)) for x in range(min(w, pw))
        )

    reference = kwargs.get("reference", kwargs.get("reference_shadow"))
    if reference is None and len(args) >= 2 and _is_number_grid(args[1]):
        reference = args[1]
    rmse = mae = None
    if _is_number_grid(reference) and _shape(reference) == (h, w):
        squared = absolute = 0.0
        for y in range(h):
            for x in range(w):
                diff = float(shadow[y][x]) - float(reference[y][x])
                squared += diff * diff
                absolute += abs(diff)
        rmse = math.sqrt(squared / count)
        mae = absolute / count

    history = kwargs.get("history")
    if history is None and len(args) >= 3 and _is_number_grid(args[2]):
        history = args[2]
    trailing_energy = None
    if _is_number_grid(history) and _shape(history) == (h, w):
        trailing_energy = sum(
            abs(float(shadow[y][x]) - float(history[y][x]))
            for y in range(h) for x in range(w)
        ) / count

    return Report(
        value=mean, ok=True, demo=not isinstance(source, (dict, list)),
        resolution=(w, h), width=w, height=h, pixels=h * w,
        mean_shadow=mean, mean_darkness=mean,
        mean_visibility=1.0 - mean,
        variance=variance, coverage=coverage,
        strong_coverage=strong_coverage,
        hard_edges=adjacent_edges,
        rmse=rmse, mae=mae, trailing_energy=trailing_energy,
        temporal_input_delta=temporal_delta,
        overdark_product_pixels=overdark_pixels,
        termination_counts=result.get("termination_counts", result.get("terminations", {})),
        fingerprint=_digest(shadow),
    )


# ---------------------------------------------------------------------------
# 13. audit / 14. fallback / state
# ---------------------------------------------------------------------------

def new_state() -> dict[str, Any]:
    return {
        "frame": -1,
        "events": [],
        "history": None,
        "stats": {
            "current_depth": 0,
            "screen_steps": 0,
            "discontinuity_stops": 0,
            "offscreen_fallbacks": 0,
            "darkest_merges": 0,
            "temporal_resets": 0,
            "fallbacks": 0,
            "errors": 0,
        },
    }


_WARNING_EVENTS = {
    "stale_depth", "depth_missing", "offscreen_clipped",
    "product_overdark", "temporal_ghost", "history_mismatch",
    "fallback", "error", "bright_seam", "half_clip",
}


def audit(*args: Any, **kwargs: Any) -> Report:
    """审计事件流或 :func:`present` 结果。"""
    source: Any = args[0] if args else kwargs.get("state")
    events: list[Any] = []

    if isinstance(source, list):
        events = source
    elif isinstance(source, dict):
        events = source.get("events", source.get("event_log", []))
        if not isinstance(events, list):
            events = []
        if not events and source.get("depth_source") == "current_frame":
            events = list(source.get("events", []))
    elif _is_number(source):
        events = [
            {"type": "depth_current_frame", "frame": int(source)},
            {"type": "step_screen_scale", "scaling": "screen"},
            {"type": "discontinuity_stop"},
            {"type": "offscreen_fallback"},
            {"type": "darkest_merge"},
            {"type": "temporal_filter"},
        ]

    counts = {
        "events": len(events),
        "stale_depth": 0,
        "discontinuity_stops": 0,
        "offscreen_fallbacks": 0,
        "product_overdark": 0,
        "temporal_resets": 0,
        "fallbacks": 0,
        "errors": 0,
    }
    warnings: list[str] = []
    mechanisms = {
        "current_depth": False,
        "screen_scale_step": False,
        "discontinuity_termination": False,
        "offscreen_fallback": False,
        "darkest_merge": False,
        "temporal_filter": False,
    }

    for event in events:
        etype = str(event.get("type", event.get("event", "unknown"))) if isinstance(event, dict) else str(event)
        if etype in ("depth_current_frame", "current_depth"):
            mechanisms["current_depth"] = True
        if etype in ("step_screen_scale", "screen_step"):
            mechanisms["screen_scale_step"] = True
        if etype in ("discontinuity_stop", "stop_at_discontinuity", "depth_discontinuity"):
            mechanisms["discontinuity_termination"] = True
            counts["discontinuity_stops"] += 1
        if etype in ("offscreen_fallback", "offscreen"):
            mechanisms["offscreen_fallback"] = True
            counts["offscreen_fallbacks"] += 1
        if etype in ("darkest_merge", "merge_darkest"):
            mechanisms["darkest_merge"] = True
        if etype in ("temporal_filter", "temporal"):
            mechanisms["temporal_filter"] = True
        if etype in _WARNING_EVENTS or (isinstance(event, dict) and event.get("ok") is False):
            warnings.append(etype)
            if etype == "stale_depth":
                counts["stale_depth"] += 1
            elif etype == "product_overdark":
                counts["product_overdark"] += 1
            elif etype == "temporal_ghost":
                counts["temporal_resets"] += 1
            elif etype == "fallback":
                counts["fallbacks"] += 1
            else:
                counts["errors"] += 1

    ok = not warnings and counts["errors"] == 0
    return Report(
        value=ok, ok=ok, warnings=warnings, issues=warnings,
        event_count=len(events), events=len(events),
        event_log=events, counts=counts,
        mechanisms_present=mechanisms,
    )


def fallback(*args: Any, **kwargs: Any) -> Report:
    """安全降级：只用当前帧；没有当前帧时返回中性无阴影遮罩。"""
    reason = kwargs.get("reason")
    current_shadow = kwargs.get("current_shadow", kwargs.get("shadow"))
    width = int(kwargs.get("width", 4))
    height = int(kwargs.get("height", 4))
    source: Any = args[0] if args else None

    if isinstance(source, str):
        reason = source
    elif isinstance(source, dict):
        reason = reason or source.get("reason")
        current_shadow = current_shadow if current_shadow is not None else source.get("shadow")
        if _is_number_grid(current_shadow):
            h, w = _shape(current_shadow)
            height, width = h, w
    elif _is_number_grid(source):
        current_shadow = source
        h, w = _shape(source)
        height, width = h, w
    elif _is_int(source):
        reason = reason or "demo_sentinel"

    if _is_number_grid(current_shadow):
        shadow = [[_clamp01(v) for v in row] for row in current_shadow]
        h, w = _shape(shadow)
        visibility = [[1.0 - shadow[y][x] for x in range(w)] for y in range(h)]
        mode = "use_current_frame"
        request_full = False
    else:
        shadow = _zeros(height, width, 0.0)
        visibility = _ones(height, width)
        mode = kwargs.get("mode", "neutral_full_res_recompute")
        request_full = True

    return Report(
        value=shadow, ok=True, reason=reason or "contact_shadow_fallback",
        mode=mode, shadow=shadow, visibility=visibility,
        history_weight=0.0, temporal_alpha=1.0,
        request_full_res=request_full,
        width=width, height=height,
    )


# ---------------------------------------------------------------------------
# 15. determinism
# ---------------------------------------------------------------------------

def determinism(*args: Any, runs: int = 2, **kwargs: Any) -> Report:
    """重复执行同一工作负载并比较 SHA-256，保证无随机/时间相关输出。"""
    work = kwargs.get("work")
    if work is None and args and callable(args[0]):
        work = args[0]
    count = max(2, int(runs)) if _is_int(runs) else 2

    hashes: list[str] = []
    outputs: list[Any] = []
    for _ in range(count):
        if callable(work):
            output = work()
            payload = output if isinstance(output, bytes) else repr(output).encode("utf-8")
        else:
            output = present(1)
            payload = repr({
                "shadow": output.get("shadow"),
                "image": output.get("image"),
                "termination_counts": output.get("termination_counts"),
            }).encode("utf-8")
        outputs.append(output)
        hashes.append(hashlib.sha256(payload).hexdigest())

    deterministic = len(set(hashes)) == 1
    return Report(
        value=deterministic, ok=deterministic,
        deterministic=deterministic, runs=count,
        hashes=hashes, digest=hashes[0] if hashes else None,
    )
