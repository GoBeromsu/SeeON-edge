#include "nvdsinfer_custom_impl.h"

#include <cstddef>
#include <vector>

// YOLO26 pose head, end-to-end: rows are x1,y1,x2,y2,score,class in tensor
// space followed by 17x3 keypoints (57 floats). The head already resolved
// duplicates, so this deliberately performs no second NMS and preserves source
namespace {

constexpr int kRows = 300;
constexpr int kStride = 57;
constexpr float kScoreThreshold = 0.05F;

bool shape_is_pose(NvDsInferLayerInfo const& layer) {
  const auto& dims = layer.inferDims;
  if (dims.numDims == 3) {
    return dims.d[1] == kRows && dims.d[2] == kStride;
  }
  if (dims.numDims == 2) {
    return dims.d[0] == kRows && dims.d[1] == kStride;
  }
  return false;
}

}  // namespace

extern "C" bool NvDsInferParseCustomYolo26Pose(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& /*detectionParams*/,
    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  if (outputLayersInfo.empty() || outputLayersInfo[0].buffer == nullptr) {
    return false;
  }
  if (!shape_is_pose(outputLayersInfo[0])) {
    return false;
  }
  const float* rows = static_cast<const float*>(outputLayersInfo[0].buffer);
  const float width = static_cast<float>(networkInfo.width);
  const float height = static_cast<float>(networkInfo.height);
  for (int index = 0; index < kRows; ++index) {
    const float* row = rows + static_cast<std::size_t>(index) * kStride;
    const float score = row[4];
    if (!(score > kScoreThreshold)) {
      continue;
    }
    const float x1 = row[0] < 0.0F ? 0.0F : (row[0] > width ? width : row[0]);
    const float y1 = row[1] < 0.0F ? 0.0F : (row[1] > height ? height : row[1]);
    const float x2 = row[2] < 0.0F ? 0.0F : (row[2] > width ? width : row[2]);
    const float y2 = row[3] < 0.0F ? 0.0F : (row[3] > height ? height : row[3]);
    if (!(x2 > x1) || !(y2 > y1)) {
      continue;
    }
    NvDsInferObjectDetectionInfo object{};
    object.classId = static_cast<unsigned int>(row[5] < 0.0F ? 0.0F : row[5]);
    object.detectionConfidence = score;
    object.left = x1;
    object.top = y1;
    object.width = x2 - x1;
    object.height = y2 - y1;
    objectList.push_back(object);
  }
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolo26Pose);
