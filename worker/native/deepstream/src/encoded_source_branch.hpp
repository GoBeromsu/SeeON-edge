#pragma once

#include "source_runtime.hpp"

#ifdef SEEON_HAS_GSTREAMER
#include <gst/gst.h>

#include <string>

namespace seeon {
[[nodiscard]] GstElement* build_encoded_rtsp_pipeline(const std::string& camera,
                                                       const std::string& uri,
                                                       const FrameCallback& frames,
                                                       const FailureCallback& failures,
                                                       const AccessUnitCallback& access_units,
                                                       const PreviewCallback& previews,
                                                       const PipelineBindingPtr& binding,
                                                       std::string* error_code);
[[nodiscard]] bool set_encoded_preview_viewers(GstElement* pipeline, std::uint32_t viewers);
[[nodiscard]] std::optional<PreviewStatus> encoded_preview_status(GstElement* pipeline);
[[nodiscard]] bool wait_encoded_preview(GstElement* pipeline, std::uint64_t target);
void quiesce_encoded_pipeline(GstElement* pipeline);
}  // namespace seeon
#endif
