#include <nvdsinfer_custom_impl.h>

#include <algorithm>
#include <cstring>
#include <vector>

extern "C" bool NvDsInferParseCustomYolo26Pose(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const&, std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  if (outputLayersInfo.size() != 1 || outputLayersInfo[0].inferDims.numDims != 3 ||
      outputLayersInfo[0].inferDims.d[1] != 300 || outputLayersInfo[0].inferDims.d[2] != 57 ||
      outputLayersInfo[0].buffer == nullptr) {
    return false;
  }
  const float* rows = static_cast<const float*>(outputLayersInfo[0].buffer);
  constexpr int kRows = 300;
  constexpr int kStride = 57;
  constexpr float kScoreThreshold = 0.05F;
  for (int index = 0; index < kRows; ++index) {
    const float* row = rows + index * kStride;
    if (!(row[4] > kScoreThreshold)) continue;
    NvDsInferObjectDetectionInfo object{};
    object.classId = static_cast<unsigned int>(row[5]);
    object.detectionConfidence = row[4];
    object.left = std::clamp(row[0], 0.0F, static_cast<float>(networkInfo.width));
    object.top = std::clamp(row[1], 0.0F, static_cast<float>(networkInfo.height));
    object.width = std::max(0.0F, std::min(row[2], static_cast<float>(networkInfo.width)) - object.left);
    object.height = std::max(0.0F, std::min(row[3], static_cast<float>(networkInfo.height)) - object.top);
    objectList.push_back(object);
  }
  return true;
}
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolo26Pose);
