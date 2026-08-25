#pragma once

#include "source_runtime.hpp"

#ifdef SEEON_HAS_GSTREAMER
#include <gst/gst.h>
namespace seeon {
void attach_source_bus_handler(GstElement* pipeline, const std::string& camera,
                               const FailureCallback& failures);
}  // namespace seeon
#endif
