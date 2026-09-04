#include "nvdsinfer_custom_impl.h"

#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
constexpr int kRows = 300;
constexpr int kStride = 57;
constexpr float kScoreThreshold = 0.05F;
constexpr char kPoseLayerName[] = "output0";

bool shape_is_pose(NvDsInferLayerInfo const& layer) {
  const auto& dims = layer.inferDims;
  return (dims.numDims == 3 && dims.d[1] == kRows && dims.d[2] == kStride) ||
         (dims.numDims == 2 && dims.d[0] == kRows && dims.d[1] == kStride);
}

// The engine exposes more than one output binding, and their order is a
// property of how the engine was built rather than of the model: a TensorRT
// engine built by `trtexec` and one built by nvinfer itself do not agree on it.
// Selecting index 0 therefore silently parsed the wrong buffer and dropped
// every detection, so bind the pose tensor by the name the nvinfer config
// declares in `output-blob-names` and only then fall back to shape.
NvDsInferLayerInfo const* pose_layer(std::vector<NvDsInferLayerInfo> const& layers) {
  for (auto const& layer : layers) {
    if (layer.buffer != nullptr && layer.layerName != nullptr &&
        std::strcmp(layer.layerName, kPoseLayerName) == 0 && shape_is_pose(layer)) {
      return &layer;
    }
  }
  for (auto const& layer : layers) {
    if (layer.buffer != nullptr && shape_is_pose(layer)) return &layer;
  }
  return nullptr;
}
}  // namespace

extern "C" bool NvDsInferParseCustomYolo26Pose(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const&, 
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  NvDsInferLayerInfo const* const layer = pose_layer(outputLayersInfo);
  if (layer == nullptr) {
    // Refusing silently is what hid this failure for a whole bring-up: the
    // pipeline ran at full frame rate and produced no objects at all. Say so
    // once, naming what was actually offered.
    static bool reported = false;
    if (!reported) {
      reported = true;
      std::cerr << "yolo26-pose parser: no output layer named '" << kPoseLayerName
                << "' with shape [" << kRows << "," << kStride << "]; offered:";
      for (auto const& candidate : outputLayersInfo) {
        std::cerr << ' ' << (candidate.layerName != nullptr ? candidate.layerName : "<unnamed>");
      }
      std::cerr << std::endl;
    }
    return false;
  }
  const float* rows = static_cast<const float*>(layer->buffer);
  const float width = static_cast<float>(networkInfo.width);
  const float height = static_cast<float>(networkInfo.height);
  for (int index = 0; index < kRows; ++index) {
    const float* row = rows + static_cast<std::size_t>(index) * kStride;
    if (!(row[4] > kScoreThreshold)) continue;
    const float x1 = row[0] < 0.0F ? 0.0F : (row[0] > width ? width : row[0]);
    const float y1 = row[1] < 0.0F ? 0.0F : (row[1] > height ? height : row[1]);
    const float x2 = row[2] < 0.0F ? 0.0F : (row[2] > width ? width : row[2]);
    const float y2 = row[3] < 0.0F ? 0.0F : (row[3] > height ? height : row[3]);
    NvDsInferObjectDetectionInfo object{};
    object.classId = static_cast<unsigned int>(row[5] < 0.0F ? 0.0F : row[5]);
    object.detectionConfidence = row[4];
    object.left = x1;
    object.top = y1;
    object.width = x2 - x1;
    object.height = y2 - y1;
    objectList.push_back(object);
  }
  return true;
}
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolo26Pose);
