from __future__ import annotations

import numpy as np

from prometheus.utils.remote import pack_msg, unpack_msg


def test_remote_msgpack_roundtrips_numpy_payloads():
    payload = {
        "image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "matrix": np.eye(4, dtype=np.float64),
        "nested": {"value": np.float32(1.5)},
    }

    restored = unpack_msg(pack_msg(payload))

    np.testing.assert_array_equal(restored["image"], payload["image"])
    np.testing.assert_array_equal(restored["matrix"], payload["matrix"])
    assert restored["nested"]["value"] == np.float32(1.5).item()
