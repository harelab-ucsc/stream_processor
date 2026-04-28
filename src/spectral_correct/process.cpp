#include "process.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace stream_processor {
namespace spectral_correct {

namespace {

// Resolve per-slice irradiance divisor. Returns 1.0f when missing/non-finite/
// non-positive — matches existing Python tolerance (NaN > 0 is false, so the
// original code already passes through identity for NaN).
float resolve_irr(const float * spec_data, py::ssize_t spec_len, int slice_idx) {
    const int spec_idx = CAM0_ALIGNMENT[slice_idx];
    if (!spec_data || spec_idx >= spec_len) {
        return 1.0f;
    }
    const float v = spec_data[spec_idx];
    if (!std::isfinite(v) || v <= 0.0f) {
        return 1.0f;
    }
    return v;
}

template <typename T>
void divide_slice(const py::array & in, py::array_t<float> & out,
                  py::ssize_t slice_w, py::ssize_t slice_offset, float irr) {
    const py::ssize_t h = in.shape(0);
    const py::ssize_t in_stride_row = in.strides(0) / sizeof(T);
    const T * in_data = static_cast<const T *>(in.data());
    float * out_data = out.mutable_data();
    const float inv = 1.0f / irr;

    for (py::ssize_t y = 0; y < h; ++y) {
        const T * src_row = in_data + y * in_stride_row + slice_offset;
        float * dst_row = out_data + y * slice_w;
        for (py::ssize_t x = 0; x < slice_w; ++x) {
            dst_row[x] = static_cast<float>(src_row[x]) * inv;
        }
    }
}

}  // namespace

std::vector<py::array_t<float>> process_cam0(
    py::array input,
    py::object spec_vals,
    int num_slices) {
    if (input.ndim() != 2) {
        throw std::invalid_argument(
            "process_cam0: expected 2-D mono image, got ndim=" +
            std::to_string(input.ndim()));
    }
    if (num_slices != 4) {
        throw std::invalid_argument(
            "process_cam0: only num_slices=4 supported (CAM0_ALIGNMENT has 4 entries)");
    }

    const py::ssize_t h = input.shape(0);
    const py::ssize_t w = input.shape(1);
    const py::ssize_t slice_w = w / num_slices;
    if (slice_w == 0) {
        throw std::invalid_argument(
            "process_cam0: image width " + std::to_string(w) +
            " too small for " + std::to_string(num_slices) + " slices");
    }

    py::array_t<float> spec_arr;
    const float * spec_data = nullptr;
    py::ssize_t spec_len = 0;
    if (!spec_vals.is_none()) {
        spec_arr = py::array_t<float, py::array::c_style | py::array::forcecast>(spec_vals);
        if (spec_arr.ndim() == 1) {
            spec_data = spec_arr.data();
            spec_len = spec_arr.shape(0);
        }
    }

    float irr[4];
    for (int i = 0; i < num_slices; ++i) {
        irr[i] = resolve_irr(spec_data, spec_len, i);
    }

    std::vector<py::array_t<float>> outputs;
    outputs.reserve(num_slices);
    for (int i = 0; i < num_slices; ++i) {
        outputs.emplace_back(py::array_t<float>({h, slice_w}));
    }

    // Accept any 1- or 2-byte integer dtype (signed or unsigned). cv_bridge's
    // passthrough sometimes returns a non-canonical dtype that doesn't pass
    // py::dtype::of<uint16_t>() identity comparison even though the bytes
    // are uint16. Treat by itemsize; the divide-and-promote-to-float math
    // is identical for signed and unsigned at our pixel-value ranges.
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

    {
        py::gil_scoped_release release;
        for (int i = 0; i < num_slices; ++i) {
            const py::ssize_t offset = i * slice_w;
            if (is_8bit) {
                divide_slice<uint8_t>(input, outputs[i], slice_w, offset, irr[i]);
            } else {
                divide_slice<uint16_t>(input, outputs[i], slice_w, offset, irr[i]);
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
        // Bayer phase is preserved only on even-column splits. An odd split
        // flips BG/GR alignment and produces wrong colors.
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

}  // namespace spectral_correct
}  // namespace stream_processor
