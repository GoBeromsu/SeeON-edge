#include "source_runtime.hpp"

#include <algorithm>
#include <string_view>

namespace seeon {
bool valid_source_uri(const std::string& uri) {
  constexpr std::size_t kMaximumUriBytes = 4096;
  if (uri.empty() || uri.size() > kMaximumUriBytes ||
      !std::ranges::none_of(uri, [](unsigned char character) {
        return character < 32 || character == 127;
      })) {
    return false;
  }
  constexpr std::string_view rtsp_prefix = "rtsp://";
  constexpr std::string_view loopback_prefix = "loopback://";
  if (uri.starts_with(loopback_prefix)) {
    return uri.size() > loopback_prefix.size();
  }
  if (!uri.starts_with(rtsp_prefix)) {
    return false;
  }
  const std::string_view authority{uri.data() + rtsp_prefix.size(),
                                   uri.size() - rtsp_prefix.size()};
  const auto path = authority.find_first_of("/?#");
  const std::string_view credentials_and_host = authority.substr(0, path);
  const auto at = credentials_and_host.rfind('@');
  const std::string_view host_port = credentials_and_host.substr(
      at == std::string_view::npos ? 0 : at + 1);
  return !host_port.empty() && host_port.front() != ':';
}
}  // namespace seeon
