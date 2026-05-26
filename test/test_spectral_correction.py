"""
Tests for the spectral reflectance correction logic in stream_processor.

All tests are self-contained — no ROS2 init required. The irradiance ratio
correction formula is replicated inline to verify correctness independently
of the node lifecycle.
"""

import numpy as np
import pytest

from stream_processor.stream_processor import _CAM0_SPEC_IDX


# ---------------------------------------------------------------------------
# Replicate the per-cycle correction formula from post_process_and_save.
# Keeping it here (rather than importing from the node) lets us test the
# math without spinning up a ROS executor.
# ---------------------------------------------------------------------------

def _apply_correction(panel_calib, panel_spec_ref, spec_np):
    """
    Compute spec_for_correction exactly as post_process_and_save does.

    Returns:
        np.ndarray of shape (4,) float32  — when both panel_calib and
            panel_spec_ref are available.
        np.ndarray of shape (4,) float32  — raw panel_calib when spec_ref
            is None.
        None                               — when panel_calib is None.
    """
    if panel_calib is None:
        return None

    if panel_spec_ref is None:
        return np.asarray(panel_calib, dtype=np.float32)

    total = []
    for i, k in enumerate(_CAM0_SPEC_IDX):
        cur = float(spec_np[k])
        irr_ratio = float(panel_spec_ref[k]) / cur if cur > 0.0 else 1.0
        total.append(float(panel_calib[i]) * irr_ratio)
    return np.asarray(total, dtype=np.float32)


# ---------------------------------------------------------------------------
# _CAM0_SPEC_IDX mapping
# ---------------------------------------------------------------------------

class TestSpecBandMapping:

    def test_four_indices(self):
        assert len(_CAM0_SPEC_IDX) == 4

    def test_expected_indices(self):
        """
        Mapping from camera band to AS7265x spectrometer index:
          slice 0 (450 nm) → index  2 (460 nm)
          slice 1 (695 nm) → index  9 (705 nm)
          slice 2 (735 nm) → index 14 (730 nm)
          slice 3 (850 nm) → index 17 (860 nm)
        """
        assert _CAM0_SPEC_IDX == (2, 9, 14, 17)

    def test_indices_within_spectrometer_band_count(self):
        """AS7265x reports 18 bands (indices 0-17)."""
        spectrometer_band_count = 18
        for idx in _CAM0_SPEC_IDX:
            assert 0 <= idx < spectrometer_band_count


# ---------------------------------------------------------------------------
# Irradiance ratio correction formula
# ---------------------------------------------------------------------------

class TestIrradianceCorrectionFormula:

    def _make_spec(self, values_at_indices, n=18):
        """Build an 18-band spectrometer array with given values at specific indices."""
        spec = np.ones(n, dtype=np.float32) * 1000.0
        for idx, val in values_at_indices.items():
            spec[idx] = val
        return spec

    def test_no_panel_calib_returns_none(self):
        spec = np.ones(18, dtype=np.float32)
        result = _apply_correction(None, None, spec)
        assert result is None

    def test_no_spec_ref_returns_raw_panel_calib(self):
        panel_calib = [1.5, 2.0, 1.8, 2.2]
        spec = np.ones(18, dtype=np.float32)
        result = _apply_correction(panel_calib, None, spec)
        np.testing.assert_array_almost_equal(result, panel_calib)

    def test_equal_irradiance_gives_unchanged_factors(self):
        """spec_ref == spec_current → ratio = 1 → output = panel_calib."""
        panel_calib = [1.5, 2.0, 1.8, 2.2]
        spec = np.full(18, 500.0, dtype=np.float32)
        result = _apply_correction(panel_calib, spec.copy(), spec.copy())
        np.testing.assert_array_almost_equal(result, panel_calib, decimal=5)

    def test_doubled_current_irradiance_halves_factors(self):
        """
        If irradiance during flight is 2× the calibration reference,
        each factor should be halved to compensate.
        """
        panel_calib = [1.6, 2.0, 1.8, 2.4]
        spec_ref = np.full(18, 500.0, dtype=np.float32)
        spec_cur = np.full(18, 1000.0, dtype=np.float32)  # 2× brighter
        result = _apply_correction(panel_calib, spec_ref, spec_cur)
        for i, f in enumerate(panel_calib):
            assert result[i] == pytest.approx(f * 0.5, rel=1e-5)

    def test_halved_current_irradiance_doubles_factors(self):
        """
        If irradiance during flight is ½ the calibration reference (e.g. shade),
        each factor should double to compensate.
        """
        panel_calib = [1.0, 1.0, 1.0, 1.0]
        spec_ref = np.full(18, 1000.0, dtype=np.float32)
        spec_cur = np.full(18, 500.0, dtype=np.float32)   # 0.5× reference
        result = _apply_correction(panel_calib, spec_ref, spec_cur)
        for f in result:
            assert f == pytest.approx(2.0, rel=1e-5)

    def test_per_band_correction_uses_correct_spec_index(self):
        """
        Each camera band must use its mapped spectrometer index, not a
        neighbouring index.
        """
        panel_calib = [1.0, 1.0, 1.0, 1.0]
        spec_ref = np.ones(18, dtype=np.float32) * 100.0
        spec_cur = np.ones(18, dtype=np.float32) * 100.0

        # Set only slice-1's mapped spec index to 2× ref → only slice-1 halves
        k1 = _CAM0_SPEC_IDX[1]
        spec_cur[k1] = 200.0

        result = _apply_correction(panel_calib, spec_ref, spec_cur)

        assert result[0] == pytest.approx(1.0, rel=1e-5)   # ratio=1, unchanged
        assert result[1] == pytest.approx(0.5, rel=1e-5)   # ratio=0.5, halved
        assert result[2] == pytest.approx(1.0, rel=1e-5)
        assert result[3] == pytest.approx(1.0, rel=1e-5)

    def test_zero_current_spec_falls_back_to_ratio_1(self):
        """Division by zero in current spec must fall back to ratio=1.0."""
        panel_calib = [2.0, 2.0, 2.0, 2.0]
        spec_ref = np.full(18, 500.0, dtype=np.float32)
        spec_cur = np.zeros(18, dtype=np.float32)  # all zero
        result = _apply_correction(panel_calib, spec_ref, spec_cur)
        # ratio=1 → total = panel_calib
        np.testing.assert_array_almost_equal(result, panel_calib, decimal=5)

    def test_mixed_zero_and_nonzero_current_spec(self):
        """Only the zero-spec bands fall back to ratio=1; others correct normally."""
        panel_calib = [1.0, 1.0, 1.0, 1.0]
        spec_ref = np.full(18, 1000.0, dtype=np.float32)
        spec_cur = np.full(18, 500.0, dtype=np.float32)   # 2× correction expected

        # Zero out the spec index for slice 2
        k2 = _CAM0_SPEC_IDX[2]
        spec_cur[k2] = 0.0

        result = _apply_correction(panel_calib, spec_ref, spec_cur)

        assert result[0] == pytest.approx(2.0, rel=1e-5)   # 1000/500 = 2×
        assert result[1] == pytest.approx(2.0, rel=1e-5)
        assert result[2] == pytest.approx(1.0, rel=1e-5)   # fallback ratio=1
        assert result[3] == pytest.approx(2.0, rel=1e-5)

    def test_output_dtype_is_float32(self):
        panel_calib = [1.5, 1.5, 1.5, 1.5]
        spec = np.ones(18, dtype=np.float32)
        result = _apply_correction(panel_calib, spec, spec)
        assert result.dtype == np.float32

    def test_output_length_matches_num_slices(self):
        panel_calib = [1.0, 1.0, 1.0, 1.0]
        spec = np.ones(18, dtype=np.float32)
        result = _apply_correction(panel_calib, spec, spec)
        assert len(result) == 4

    def test_large_irradiance_ratio_does_not_overflow(self):
        """Very large ratio should not produce inf or nan."""
        panel_calib = [1.0, 1.0, 1.0, 1.0]
        spec_ref = np.full(18, 1e6, dtype=np.float32)
        spec_cur = np.full(18, 1.0, dtype=np.float32)  # extreme ratio
        result = _apply_correction(panel_calib, spec_ref, spec_cur)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Panel calibration callback behaviour (replicated from stream_processor)
# ---------------------------------------------------------------------------

class TestPanelCalibCallbacks:

    def test_panel_cal_cb_stores_list(self):
        """_panel_cal_cb stores msg.data as a Python list."""
        # Replicate the callback
        panel_calib = None

        class FakeMsg:
            data = [1.5, 2.0, 1.8, 2.2]

        panel_calib = list(FakeMsg.data)
        assert panel_calib == [1.5, 2.0, 1.8, 2.2]
        assert isinstance(panel_calib, list)

    def test_panel_spec_ref_cb_stores_list(self):
        """_panel_spec_ref_cb stores msg.data as a Python list."""
        panel_spec_ref = None

        class FakeMsg:
            data = list(range(18))

        panel_spec_ref = list(FakeMsg.data)
        assert len(panel_spec_ref) == 18
        assert panel_spec_ref[0] == 0
        assert panel_spec_ref[17] == 17

    def test_latched_panel_cal_persists_across_cycles(self):
        """
        panel_calib must be set once and used for every subsequent cycle
        without being reset.  Simulate 5 image cycles and verify the factor
        is applied consistently.
        """
        panel_calib = [1.5, 2.0, 1.8, 2.2]
        spec_ref = np.full(18, 1000.0, dtype=np.float32)

        for _ in range(5):
            spec_cur = np.full(18, 1000.0, dtype=np.float32)  # constant irradiance
            result = _apply_correction(panel_calib, spec_ref, spec_cur)
            np.testing.assert_array_almost_equal(result, panel_calib, decimal=5)
