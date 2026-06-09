"""
Tests for publish_mode in stream_processor.

All tests are self-contained — no ROS2 init required. They verify:
  1. is_complete only requires cam0+cam1 when _required_keys is so restricted.
  2. is_complete requires all 5 keys in normal mode.
  3. The uint16 conversion applied before publishing produces valid mono16 data.
  4. vio_cal_launch.py exists and references only the expected nodes.
"""

import os

import numpy as np


# ---------------------------------------------------------------------------
# Helpers that replicate stream_processor logic without spinning a ROS node
# ---------------------------------------------------------------------------

def _make_job(cam0=None, cam1=None, pose=None, spec=None, radalt=None):
    """Build a minimal job dict as pps_cb creates it."""
    return {
        "data": {
            "cam0": cam0,
            "cam1": cam1,
            "pose": pose,
            "spec": spec,
            "radalt": radalt,
        }
    }


def _is_complete(job, required_keys):
    """Replicate SyncNode.is_complete with a given required_keys set."""
    return all(job["data"].get(k) is not None for k in required_keys)


# ---------------------------------------------------------------------------
# 1. is_complete — publish_mode (cam0 + cam1 only)
# ---------------------------------------------------------------------------

def test_is_complete_publish_mode_cams_only():
    """Only cam0 and cam1 filled: complete in publish_mode."""
    job = _make_job(cam0="img", cam1="img")
    assert _is_complete(job, {"cam0", "cam1"})


def test_is_complete_publish_mode_missing_cam0():
    """cam0 None: incomplete even in publish_mode."""
    job = _make_job(cam0=None, cam1="img")
    assert not _is_complete(job, {"cam0", "cam1"})


def test_is_complete_publish_mode_missing_cam1():
    """cam1 None: incomplete even in publish_mode."""
    job = _make_job(cam0="img", cam1=None)
    assert not _is_complete(job, {"cam0", "cam1"})


def test_is_complete_publish_mode_pose_radalt_ignored():
    """pose/spec/radalt None: still complete in publish_mode."""
    job = _make_job(cam0="img", cam1="img", pose=None, spec=None, radalt=None)
    assert _is_complete(job, {"cam0", "cam1"})


# ---------------------------------------------------------------------------
# 2. is_complete — normal mode (all 5 keys)
# ---------------------------------------------------------------------------

_ALL_KEYS = {"cam0", "cam1", "pose", "spec", "radalt"}


def test_is_complete_normal_mode_all_filled():
    job = _make_job(cam0="img", cam1="img", pose="p", spec="s", radalt="r")
    assert _is_complete(job, _ALL_KEYS)


def test_is_complete_normal_mode_missing_pose():
    job = _make_job(cam0="img", cam1="img", pose=None, spec="s", radalt="r")
    assert not _is_complete(job, _ALL_KEYS)


def test_is_complete_normal_mode_missing_radalt():
    job = _make_job(cam0="img", cam1="img", pose="p", spec="s", radalt=None)
    assert not _is_complete(job, _ALL_KEYS)


def test_is_complete_normal_mode_partial_complete_in_publish():
    """A job incomplete in normal mode may still be complete in publish_mode."""
    job = _make_job(cam0="img", cam1="img", pose=None)
    assert not _is_complete(job, _ALL_KEYS)
    assert _is_complete(job, {"cam0", "cam1"})


# ---------------------------------------------------------------------------
# 3. uint16 conversion for cam0 bands before publishing
# ---------------------------------------------------------------------------

def test_band_u16_zero():
    """All-zero float32 band → all-zero uint16."""
    band = np.zeros((800, 1280), dtype=np.float32)
    out = np.clip(band * 65535.0, 0, 65535).astype(np.uint16)
    assert out.dtype == np.uint16
    assert out.max() == 0


def test_band_u16_ones():
    """All-ones float32 band (reflectance = 1.0) → all 65535 uint16."""
    band = np.ones((800, 1280), dtype=np.float32)
    out = np.clip(band * 65535.0, 0, 65535).astype(np.uint16)
    assert out.dtype == np.uint16
    assert out.min() == 65535


def test_band_u16_midrange():
    """0.5 reflectance → 32767 (within ±1 of exact)."""
    band = np.full((800, 1280), 0.5, dtype=np.float32)
    out = np.clip(band * 65535.0, 0, 65535).astype(np.uint16)
    assert abs(int(out[0, 0]) - 32767) <= 1


def test_band_u16_clamps_above_one():
    """Values > 1.0 are clamped to 65535, not wrapped."""
    band = np.full((10, 10), 2.0, dtype=np.float32)
    out = np.clip(band * 65535.0, 0, 65535).astype(np.uint16)
    assert out.max() == 65535


def test_band_u16_clamps_negative():
    """Negative values (shouldn't occur, but guard against) are clamped to 0."""
    band = np.full((10, 10), -0.5, dtype=np.float32)
    out = np.clip(band * 65535.0, 0, 65535).astype(np.uint16)
    assert out.min() == 0


def test_band_u16_shape_preserved():
    """Output shape equals input shape."""
    band = np.random.rand(800, 1280).astype(np.float32)
    out = np.clip(band * 65535.0, 0, 65535).astype(np.uint16)
    assert out.shape == (800, 1280)


# ---------------------------------------------------------------------------
# 4. vio_cal_launch.py content checks (no ROS2 runtime needed)
# ---------------------------------------------------------------------------

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# Walk up: test/ -> stream_processor/ -> src/
_SRC_DIR = os.path.abspath(os.path.join(_TEST_DIR, "..", ".."))
_VIO_LAUNCH = os.path.join(
    _SRC_DIR, "frc_payload_launcher", "launch", "vio_cal_launch.py"
)


def test_vio_cal_launch_file_exists():
    assert os.path.isfile(_VIO_LAUNCH), (
        f"vio_cal_launch.py not found at {_VIO_LAUNCH}"
    )


def test_vio_cal_launch_includes_expected_nodes():
    with open(_VIO_LAUNCH) as f:
        src = f.read()
    for token in ("pps_time_pub", "auto_cal", "publish_mode", "force_cal"):
        assert token in src, f"Expected '{token}' in vio_cal_launch.py"


def test_vio_cal_launch_excludes_unwanted_nodes():
    """Check that excluded ROS package names don't appear in Node() declarations."""
    with open(_VIO_LAUNCH) as f:
        src = f.read()
    # Use package/executable names specific to Node() args — these don't appear
    # in comments or docstrings in any legitimate form.
    for token in ("ros2_radalt", "inertial_sense_ros2", "as7265x_at"):
        assert token not in src, (
            f"ROS package '{token}' should not appear in vio_cal_launch.py"
        )
    # panel_scan is both a comment word and executable — check executable context
    assert 'executable="panel_scan"' not in src, (
        'panel_scan Node should not appear in vio_cal_launch.py'
    )


def test_vio_cal_launch_force_cal_always_true():
    """auto_cal must always start with force_cal=True in VIO_CAL mode."""
    with open(_VIO_LAUNCH) as f:
        src = f.read()
    assert '"force_cal": True' in src, (
        "auto_cal must have force_cal=True in vio_cal_launch.py"
    )


def test_vio_cal_launch_publish_mode_true():
    """sync_node must have publish_mode=True."""
    with open(_VIO_LAUNCH) as f:
        src = f.read()
    assert '"publish_mode": True' in src, (
        "sync_node must have publish_mode=True in vio_cal_launch.py"
    )
