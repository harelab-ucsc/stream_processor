#include "process.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using stream_processor::spectral_correct::process_cam0;
using stream_processor::spectral_correct::process_cam1;
using stream_processor::spectral_correct::check_slice_health;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Single-pass split + reflectance correction (cam0) / debayer (cam1).";

    m.def("process_cam0", &process_cam0,
          py::arg("input"),
          py::arg("cal_factors") = py::none(),
          py::arg("num_slices") = 4,
          R"doc(
Split a full cam0 mono image into N reflectance-corrected float32 sub-images.

Per slice i, applies: output = (raw_DN / dtype_max) * cal_factors[i]
where cal_factors[i] = panel_albedo_i / (mean_panel_DN_i / dtype_max),
computed by MicaCRPCal at boot from a camera image of the white panel.

When cal_factors is None, factor=1.0 is used and output = raw_DN / dtype_max.

Args:
    input: (H, W) uint8 or uint16 ndarray, C-contiguous.
    cal_factors: 4-element float32 array [F_0, F_1, F_2, F_3], or None.
    num_slices: must be 4.

Returns:
    List of 4 (H, W/4) float32 ndarrays of true reflectance in [0, 1].
)doc");

    m.def("process_cam1", &process_cam1,
          py::arg("input"),
          py::arg("num_slices") = 4,
          R"doc(
Split a full cam1 BG-Bayer image into N RGB sub-images via cv::cvtColor.

Slice width must be even to preserve Bayer phase across the cut.

Args:
    input: (H, W) uint8 or uint16 ndarray, C-contiguous.
    num_slices: integer.

Returns:
    List of (H, W/num_slices, 3) ndarrays, dtype matching the input.
)doc");

    m.def("check_slice_health", &check_slice_health,
          py::arg("slice"),
          R"doc(
Check a single post-split image band for quality issues.

Returns a (possibly empty) list of human-readable issue strings.
An empty list means the band is healthy.

Checks:
  - All-zero pixels  — USB/driver failure or dead camera
  - Near-black mean (<1% of dtype max) — lens cap or dead camera

Args:
    slice: numpy array, any rank. Supported dtypes: uint8, uint16, float32.

Returns:
    List[str] of issue descriptions; empty => healthy.
)doc");
}
