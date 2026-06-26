#include "process.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace stream_processor {
namespace spectral_correct {

namespace {

// Resolve the per-slice calibration factor.
// cal_data is a 4-element array [F_0 … F_3] produced by MicaCRPCal.
// Returns 1.0f when missing/invalid — output becomes raw_DN / dtype_max ∈ [0,1].
float resolve_factor(const float * cal_data, py::ssize_t cal_len, int slice_idx) {
    if (!cal_data || slice_idx >= cal_len) {
        std::fprintf(stderr,
            "[spectral_correct] ERROR: no calibration factor for slice %d "
            "(cal_len=%zd) — falling back to factor=1.0\n",
            slice_idx, static_cast<ssize_t>(cal_len));
        return 1.0f;
    }
    const float v = cal_data[slice_idx];
    if (!std::isfinite(v) || v <= 0.0f) {
        std::fprintf(stderr,
            "[spectral_correct] ERROR: calibration factor for slice %d "
            "is invalid (%.6g) — falling back to factor=1.0\n",
            slice_idx, static_cast<double>(v));
        return 1.0f;
    }
    return v;
}

template <typename T>
void apply_factor(const py::array & in, py::array_t<float> & out,
                  py::ssize_t slice_w, py::ssize_t slice_offset, float factor,
                  float dtype_max) {
    const py::ssize_t h = in.shape(0);
    const py::ssize_t in_stride_row = in.strides(0) / sizeof(T);
    const T * in_data = static_cast<const T *>(in.data());
    float * out_data = out.mutable_data();
    // output = (raw_DN / dtype_max) * factor → true reflectance ∈ [0, 1].
    // factor = albedo / (mean_panel_DN / dtype_max), computed by MicaCRPCal.
    // When factor = 1.0 (no calibration) output is simply raw_DN / dtype_max.
    const float scale = factor / dtype_max;
    for (py::ssize_t y = 0; y < h; ++y) {
        const T * src_row = in_data + y * in_stride_row + slice_offset;
        float * dst_row = out_data + y * slice_w;
        for (py::ssize_t x = 0; x < slice_w; ++x) {
            dst_row[x] = static_cast<float>(src_row[x]) * scale;
        }
    }
}

}  // namespace

std::vector<py::array_t<float>> process_cam0(
    py::array input,
    py::object cal_factors,
    int num_slices) {
    if (input.ndim() != 2) {
        throw std::invalid_argument(
            "process_cam0: expected 2-D mono image, got ndim=" +
            std::to_string(input.ndim()));
    }
    if (num_slices != 4) {
        throw std::invalid_argument(
            "process_cam0: only num_slices=4 supported");
    }

    const py::ssize_t h = input.shape(0);
    const py::ssize_t w = input.shape(1);
    const py::ssize_t slice_w = w / num_slices;
    if (slice_w == 0) {
        throw std::invalid_argument(
            "process_cam0: image width " + std::to_string(w) +
            " too small for " + std::to_string(num_slices) + " slices");
    }

    // Parse cal_factors — expected to be a 4-element float32 array from MicaCRPCal,
    // or None for uncalibrated passthrough (factor = 1.0 per slice).
    py::array_t<float> cal_arr;
    const float * cal_data = nullptr;
    py::ssize_t cal_len = 0;
    if (!cal_factors.is_none()) {
        cal_arr = py::array_t<float, py::array::c_style | py::array::forcecast>(cal_factors);
        if (cal_arr.ndim() == 1) {
            cal_data = cal_arr.data();
            cal_len  = cal_arr.shape(0);
        }
    }

    float factors[4];
    for (int i = 0; i < num_slices; ++i) {
        factors[i] = resolve_factor(cal_data, cal_len, i);
    }

    std::vector<py::array_t<float>> outputs;
    outputs.reserve(num_slices);
    for (int i = 0; i < num_slices; ++i) {
        outputs.emplace_back(py::array_t<float>({h, slice_w}));
    }

    const auto dt = input.dtype();
    const auto itemsize = dt.itemsize();
    const char kind = dt.kind();
    const bool is_int_kind = (kind == 'u' || kind == 'i');
    const bool is_8bit  = is_int_kind && (itemsize == 1);
    const bool is_16bit = is_int_kind && (itemsize == 2);
    if (!is_8bit && !is_16bit) {
        throw std::invalid_argument(
            std::string("process_cam0: input dtype must be 1- or 2-byte integer; got ") +
            py::str(static_cast<py::object>(dt)).cast<std::string>());
    }

    const float dtype_max = is_8bit ? 255.0f : 65535.0f;

    {
        py::gil_scoped_release release;
        for (int i = 0; i < num_slices; ++i) {
            const py::ssize_t offset = i * slice_w;
            if (is_8bit) {
                apply_factor<uint8_t>(input, outputs[i], slice_w, offset, factors[i], dtype_max);
            } else {
                apply_factor<uint16_t>(input, outputs[i], slice_w, offset, factors[i], dtype_max);
            }
        }
    }

    return outputs;
}

std::vector<py::array> process_cam1(py::array input, int num_slices) {
    if (input.ndim() != 2) {
        throw std::invalid_argument(
            "process_cam1: expected 2-D Bayer image, got ndim=" +
            std::to_string(input.ndim()));
    }
    if (num_slices <= 0) {
        throw std::invalid_argument("process_cam1: num_slices must be > 0");
    }

    const py::ssize_t h = input.shape(0);
    const py::ssize_t w = input.shape(1);
    const py::ssize_t slice_w = w / num_slices;
    if (slice_w == 0) {
        throw std::invalid_argument(
            "process_cam1: image width " + std::to_string(w) +
            " too small for " + std::to_string(num_slices) + " slices");
    }
    if (slice_w % 2 != 0) {
        throw std::invalid_argument(
            "process_cam1: slice width " + std::to_string(slice_w) +
            " must be even to preserve Bayer phase");
    }

    const auto dt = input.dtype();
    const auto itemsize = dt.itemsize();
    const char kind = dt.kind();
    const bool is_int_kind = (kind == 'u' || kind == 'i');
    const bool is_8bit  = is_int_kind && (itemsize == 1);
    const bool is_16bit = is_int_kind && (itemsize == 2);
    if (!is_8bit && !is_16bit) {
        throw std::invalid_argument(
            std::string("process_cam1: input dtype must be 1- or 2-byte integer; got ") +
            py::str(static_cast<py::object>(dt)).cast<std::string>());
    }

    const int cv_src_type = is_8bit ? CV_8UC1 : CV_16UC1;
    const int cv_dst_type = is_8bit ? CV_8UC3 : CV_16UC3;
    const py::ssize_t in_step = input.strides(0);
    const py::ssize_t bpp = is_8bit ? 1 : 2;

    std::vector<py::array> outputs;
    outputs.reserve(num_slices);
    for (int i = 0; i < num_slices; ++i) {
        py::array out = is_8bit
            ? py::array(py::dtype::of<uint8_t>(),  {h, slice_w, py::ssize_t(3)})
            : py::array(py::dtype::of<uint16_t>(), {h, slice_w, py::ssize_t(3)});
        outputs.push_back(std::move(out));
    }

    {
        py::gil_scoped_release release;
        const std::uint8_t * in_base = static_cast<const std::uint8_t *>(input.data());

        for (int i = 0; i < num_slices; ++i) {
            const py::ssize_t offset_bytes = i * slice_w * bpp;
            cv::Mat src(static_cast<int>(h), static_cast<int>(slice_w), cv_src_type,
                        const_cast<std::uint8_t *>(in_base + offset_bytes),
                        static_cast<size_t>(in_step));

            std::uint8_t * out_data = static_cast<std::uint8_t *>(outputs[i].mutable_data());
            const py::ssize_t out_step = outputs[i].strides(0);
            cv::Mat dst(static_cast<int>(h), static_cast<int>(slice_w), cv_dst_type,
                        out_data, static_cast<size_t>(out_step));

            cv::cvtColor(src, dst, cv::COLOR_BayerBG2RGB);
        }
    }

    return outputs;
}

std::vector<std::string> check_slice_health(py::array slice) {
    std::vector<std::string> issues;

    if (slice.size() == 0) {
        issues.push_back("empty array (0 elements)");
        return issues;
    }

    const auto dt = slice.dtype();
    const char kind = dt.kind();
    const auto itemsize = dt.itemsize();
    const py::ssize_t n = slice.size();

    bool all_zero = true;
    double sum = 0.0;
    double dtype_max = 1.0;

    if (kind == 'u' && itemsize == 1) {
        dtype_max = 255.0;
        auto arr = py::array_t<uint8_t,
            py::array::c_style | py::array::forcecast>(slice);
        const uint8_t * data = arr.data();
        for (py::ssize_t i = 0; i < n; ++i) {
            if (data[i] != 0) all_zero = false;
            sum += data[i];
        }
    } else if ((kind == 'u' || kind == 'i') && itemsize == 2) {
        dtype_max = 65535.0;
        auto arr = py::array_t<uint16_t,
            py::array::c_style | py::array::forcecast>(slice);
        const uint16_t * data = arr.data();
        for (py::ssize_t i = 0; i < n; ++i) {
            if (data[i] != 0) all_zero = false;
            sum += data[i];
        }
    } else if (kind == 'f' && itemsize == 4) {
        dtype_max = 1.0;
        auto arr = py::array_t<float,
            py::array::c_style | py::array::forcecast>(slice);
        const float * data = arr.data();
        for (py::ssize_t i = 0; i < n; ++i) {
            if (data[i] != 0.0f) all_zero = false;
            sum += static_cast<double>(data[i]);
        }
    } else {
        return issues;
    }

    if (all_zero) {
        issues.push_back(
            "all-zero pixels — possible USB/driver failure or dead camera");
        return issues;
    }

    const double mean_norm = (sum / static_cast<double>(n)) / dtype_max;
    if (mean_norm < 0.01) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "near-black image (normalised mean=%.4f)"
            " — possible dead camera or lens cap",
            mean_norm);
        issues.push_back(std::string(buf));
    }

    return issues;
}

}  // namespace spectral_correct
}  // namespace stream_processor
