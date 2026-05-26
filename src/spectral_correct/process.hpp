#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace stream_processor {
namespace spectral_correct {

namespace py = pybind11;

// cam0 spectral filter → AS7265x band mapping (used by MicaCRPCal to look up
// per-band panel albedo values from the CSV).  Not used inside process_cam0
// anymore — cal_factors arrive pre-indexed as a 4-element array.
//   slice 0: BP340_UV    → 410 nm
//   slice 1: BP450_Blue  → 460 nm
//   slice 2: BP695_Red   → 705 nm
//   slice 3: BP735_Edge  → 730 nm
constexpr int CAM0_ALIGNMENT[4] = {0, 2, 9, 14};

// Split a full cam0 mono image into num_slices reflectance-corrected float32
// sub-images.  cal_factors is a 4-element float32 array [F_0…F_3] produced by
// MicaCRPCal where F_i = panel_albedo_i / (mean_panel_DN_i / dtype_max).
// Pass None for uncalibrated passthrough (output = raw_DN / dtype_max).
std::vector<py::array_t<float>> process_cam0(
    py::array input,
    py::object cal_factors,
    int num_slices);

std::vector<py::array> process_cam1(
    py::array input,
    int num_slices);

// Returns a (possibly empty) list of human-readable issue strings.
// An empty list means the band is healthy.
std::vector<std::string> check_slice_health(py::array slice);

}  // namespace spectral_correct
}  // namespace stream_processor
