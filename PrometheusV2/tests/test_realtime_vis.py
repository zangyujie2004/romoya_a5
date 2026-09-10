from __future__ import annotations

from prometheus.dagger.controller import DAggerController
from prometheus.dagger.phase import DAggerPhase
from prometheus.visualization.preview import (
    build_dagger_preview_detail,
    realtime_vis_enabled,
    resolve_realtime_vis_cfg,
    start_collection_preview,
    start_realtime_vis,
)


class _Recorder:
    def details(self) -> dict:
        return {"episode_dir": "/tmp/out/corrections/0003"}


class _DataSession:
    recorder = _Recorder()


class _Controller:
    event_level = 1
    phase = DAggerPhase.CORRECTING


def test_realtime_vis_enabled():
    assert not realtime_vis_enabled({})
    assert not realtime_vis_enabled({"realtime_vis": False})
    assert realtime_vis_enabled({"realtime_vis": True})


def test_resolve_realtime_vis_cfg_merges_defaults():
    cfg = resolve_realtime_vis_cfg({"realtime_vis_cfg": {"fps": 8}})
    assert cfg["fps"] == 8.0
    assert cfg["window"] == "DAgger Preview"
    assert cfg["max_poll_ms"] == 2.0


def test_build_dagger_preview_detail():
    detail = build_dagger_preview_detail(_Controller(), _DataSession())
    assert "ep=0003" in detail
    assert "event=1" in detail
    assert "B:teach" in detail


def test_start_realtime_vis_disabled_without_flag():
    assert start_realtime_vis(_DataSession(), {"realtime_vis": False}) is None


def test_collection_preview_passes_tactile_flag(monkeypatch):
    from prometheus.visualization import local_preview

    created = {}

    class FakePreview:
        url = "local-window:test"

        def __init__(self, data, **kwargs):
            created["data"] = data
            created["kwargs"] = kwargs

        def start(self):
            created["started"] = True

    monkeypatch.setattr(local_preview, "LocalDataPreview", FakePreview)
    viewer = start_collection_preview(
        _DataSession(),
        {
            "preview": {
                "enabled": True,
                "mode": "local",
                "show_tactile": True,
                "tactile_noise_floor_m": 0.0005,
                "tactile_max_deformation_m": 0.01,
                "tactile_median_kernel": 3,
            }
        },
    )

    assert viewer is not None
    assert created["started"] is True
    assert created["kwargs"]["show_tactile"] is True
    assert created["kwargs"]["tactile_noise_floor_m"] == 0.0005
    assert created["kwargs"]["tactile_max_deformation_m"] == 0.01
    assert created["kwargs"]["tactile_median_kernel"] == 3
