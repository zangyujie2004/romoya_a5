from __future__ import annotations
from typing import Any, Literal

# camera: ir + rgb
CAMERA_BUNDLES = ("base_0", "left_wrist_0", "right_wrist_0")
CAMERA_STREAMS: dict[str, dict[str, str]] = {
    "rgb": {"suffix": "_rgb.mp4", "kind": "rgb"},
    "left_fish_eye": {"suffix": "_ir_left.mkv", "kind": "ir"},
    "right_fish_eye": {"suffix": "_ir_right.mkv", "kind": "ir"},
}
# Remove-hand is hard-coded to base_0 only (see resolve_impainting_mode).
RGB_REMOVE_HAND_BUNDLE = "base_0"
RGB_REMOVE_HAND_SUFFIX = "_rgb_remove_hand.mp4"
RGB_REMOVE_HAND_STREAM = "rgb_remove_hand"
# tactile: difference
TACTILE_BUNDLES = ("left_wrist_0", "left_wrist_1", "right_wrist_0", "right_wrist_1")
TACTILE_STREAMS: dict[str, dict[str, str]] = {
    "difference": {"suffix": "_difference.mkv", "kind": "tactile"},
    "pointcloud": {"suffix": "", "kind": "pkl"},
}
# robot: joint + vel + effort + eef
ROBOT_BUNDLES = ("left", "right")
ROBOT_STREAMS = ("joint", "vel", "effort", "eef")

ImpaintingMode = Literal["off", "replace", "dual"]
IMPAINTING_MODES: tuple[ImpaintingMode, ...] = ("off", "replace", "dual")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def resolve_impainting_mode(bundle_cfg: dict[str, Any] | None = None) -> ImpaintingMode:
    """Parse streams.camera.<bundle>.impainting → off | replace | dual.

    Bool compat: false → off, true → replace (legacy single-source remove).
    Aliases: none → off.
    """
    raw = (bundle_cfg or {}).get("impainting", "off")
    if isinstance(raw, bool):
        return "replace" if raw else "off"
    if raw is None:
        return "off"
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return "replace"
    if s in {"0", "false", "no", "off", "none"}:
        return "off"
    if s in IMPAINTING_MODES:
        return s  # type: ignore[return-value]
    raise ValueError(
        f"invalid impainting={raw!r}; expected one of {IMPAINTING_MODES} "
        f"(bool false→off, true→replace)"
    )


def camera_stream_suffix(
    stream_name: str,
    default_suffix: str,
    bundle_cfg: dict[str, Any] | None = None,
) -> str:
    """Resolve camera file suffix for the primary stream load.

    Primary rgb always uses the default suffix (``_rgb.mp4``). Remove-hand is
    loaded separately in the loader based on ``resolve_impainting_mode``.
    ``bundle_cfg`` is kept for call-site compatibility.
    """
    del stream_name, bundle_cfg  # primary path never swaps to remove-hand
    return default_suffix


def selected_streams(
    stream_config: dict[str, Any] | None,
    registry: dict[str, dict[str, str]],
    bundles: tuple[str, ...],
) -> dict[str, list[str]]:
    stream_config = stream_config or {}
    result: dict[str, list[str]] = {}
    for bundle in bundles:
        bundle_cfg = stream_config.get(bundle, {})
        enabled = [
            name
            for name in registry
            if _coerce_bool(bundle_cfg.get(name, False))
        ]
        if enabled:
            result[bundle] = enabled
    return result
