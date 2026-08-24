cmake_minimum_required(VERSION 3.22)
project(seeon_deepstream_preflight VERSION 1.0.0 LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0)

add_executable(seeon-deepstream-preflight src/main.cpp)
target_compile_features(seeon-deepstream-preflight PRIVATE cxx_std_20)
target_compile_options(seeon-deepstream-preflight PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(seeon-deepstream-preflight PRIVATE PkgConfig::GSTREAMER)

install(TARGETS seeon-deepstream-preflight RUNTIME DESTINATION bin)
