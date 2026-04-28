#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace stream_processor {
namespace spectral_correct {

namespace py = pybind11;

// AS7265x channel index per cam0 sub-image. Filter -> matching irradiance band.
//   slice 0: BP340_UV    -> spec[0]  (410 nm)
//   slice 1: BP450_Blue  -> spec[2]  (460 nm)
//   slice 2: BP695_Red   -> spec[9]  (705 nm)
//   slice 3: BP735_Edge  -> spec[14] (730 nm)
constexpr int CAM0_ALIGNMENT[4] = {0, 2, 9, 14};

std::vector<py::array_t<float>> process_cam0(
    py::array input,
    py::object spec_vals,
    int num_slices);

std::vector<py::array> process_cam1(
    py::array input,
    int num_slices);

}  // namespace spectral_correct
}  // namespace stream_processor
