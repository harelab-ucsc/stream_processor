"""
Single-pass split + spectral correction (cam0) / debayer (cam1).

Wraps the pybind11 extension `_core` built by CMake. See process.cpp for the
per-pixel logic and calibration factor application.
"""
from ._core import process_cam0, process_cam1, check_slice_health

__all__ = ["process_cam0", "process_cam1", "check_slice_health"]
