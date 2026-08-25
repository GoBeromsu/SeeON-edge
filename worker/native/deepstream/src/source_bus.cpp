#include "source_bus.hpp"

#ifdef SEEON_HAS_GSTREAMER
namespace seeon {
namespace {
struct BusContext { FailureCallback failures; std::string camera; };
std::string factory_name(const GstMessage* message) {
  if (!GST_IS_ELEMENT(GST_MESSAGE_SRC(message))) return {};
  GstElementFactory* factory = gst_element_get_factory(GST_ELEMENT(GST_MESSAGE_SRC(message)));
  return factory == nullptr ? std::string{} : std::string{gst_plugin_feature_get_name(factory)};
}
GstBusSyncReply on_bus(GstBus*, GstMessage* message, gpointer raw) {
  auto* context = static_cast<BusContext*>(raw);
  if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
    context->failures(classify_bus_failure(true, 0, 0, {}, context->camera));
    return GST_BUS_DROP;
  }
  if (GST_MESSAGE_TYPE(message) != GST_MESSAGE_ERROR) return GST_BUS_PASS;
  GError* error = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(message, &error, &debug);
  context->failures(classify_bus_failure(
      false, error == nullptr ? 0U : error->domain, error == nullptr ? 0 : error->code,
      factory_name(message), context->camera));
  g_clear_error(&error);
  g_free(debug);
  return GST_BUS_DROP;
}
}  // namespace

NativeFailure classify_bus_failure(bool eos, unsigned int error_domain, int error_code,
                                   const std::string& element_factory,
                                   const std::string& camera) {
  if (eos) return {camera, "eos", FailureScope::kSourceLocal};
  const bool shared_factory = element_factory == "seeonperceptiontransform" ||
                              element_factory.starts_with("nv");
  const bool fatal_domain = error_domain == GST_LIBRARY_ERROR ||
      (error_domain == GST_CORE_ERROR && error_code != GST_CORE_ERROR_NEGOTIATION);
  const bool fatal = shared_factory || fatal_domain;
  return {camera, fatal ? "shared_pipeline" : "decoder_source",
          fatal ? FailureScope::kFatal : FailureScope::kSourceLocal};
}

void attach_source_bus_handler(GstElement* pipeline, const std::string& camera,
                               const FailureCallback& failures) {
  GstBus* bus = gst_element_get_bus(pipeline);
  gst_bus_set_sync_handler(
      bus, on_bus, new BusContext{failures, camera},
      [](gpointer raw) { delete static_cast<BusContext*>(raw); });
  gst_object_unref(bus);
}
}  // namespace seeon
#endif
