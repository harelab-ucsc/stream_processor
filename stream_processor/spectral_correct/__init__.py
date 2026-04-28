"""
Single-pass split + spectral correction (cam0) / debayer (cam1).

Wraps the pybind11 extension `_core` built by CMake. See process.cpp for the
per-pixel logic and CAM0_ALIGNMENT mapping (slice -> AS7265x channel).
"""
from ._core import process_cam0, process_cam1

__all__ = ["process_cam0", "process_cam1"]
