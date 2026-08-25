cmake_minimum_required(VERSION 3.22)
project(seeon_deepstream_preflight VERSION 1.0.0 LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(
  GSTREAMER REQUIRED IMPORTED_TARGET
  gstreamer-1.0
  gstreamer-app-1.0
  gstreamer-base-1.0
  gstreamer-video-1.0
)

set(SEEON_WARNINGS -Wall -Wextra -Wpedantic -Werror)
# Bit-stable float behavior for the parity-pinned perception math.
set(SEEON_FP_FLAGS -ffp-contract=off)

# TensorRT + CUDA runtime (present in the pinned DeepStream image).
find_library(SEEON_NVINFER nvinfer REQUIRED)
find_library(
  SEEON_CUDART cudart
  PATHS /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib
  REQUIRED
)
find_path(
  SEEON_TRT_INCLUDE NvInfer.h
  PATHS /usr/include/x86_64-linux-gnu /usr/include
  REQUIRED
)
find_path(
  SEEON_CUDA_INCLUDE cuda_runtime_api.h
  PATHS /usr/local/cuda/include /usr/local/cuda/targets/x86_64-linux/include
  REQUIRED
)

add_executable(
  seeon-deepstream-preflight
  src/main.cpp
  src/native_perception.cpp
  src/trt_perception.cpp
)
target_compile_features(seeon-deepstream-preflight PRIVATE cxx_std_20)
target_compile_options(seeon-deepstream-preflight PRIVATE ${SEEON_WARNINGS} ${SEEON_FP_FLAGS})
target_include_directories(
  seeon-deepstream-preflight PRIVATE ${SEEON_TRT_INCLUDE} ${SEEON_CUDA_INCLUDE}
)
target_link_libraries(
  seeon-deepstream-preflight PRIVATE PkgConfig::GSTREAMER ${SEEON_NVINFER} ${SEEON_CUDART}
)

add_executable(
  seeon-deepstream-child
  src/au_parser.cpp
  src/au_transport.cpp
  src/child.cpp
  src/child_commands.cpp
  src/child_command_support.cpp
  src/child_failures.cpp
  src/child_server.cpp
  src/encoded_source_bus.cpp
  src/encoded_source_branch.cpp
  src/encoded_source_sample.cpp
  src/native_perception.cpp
  src/perception_wire.cpp
  src/preview_transport.cpp
  src/source_bus.cpp
  src/source_runtime.cpp
  src/source_uri.cpp
  src/trt_perception.cpp
)
target_compile_features(seeon-deepstream-child PRIVATE cxx_std_20)
target_compile_definitions(seeon-deepstream-child PRIVATE SEEON_HAS_GSTREAMER=1)
target_compile_options(seeon-deepstream-child PRIVATE ${SEEON_WARNINGS} ${SEEON_FP_FLAGS})
target_include_directories(
  seeon-deepstream-child PRIVATE ${SEEON_TRT_INCLUDE} ${SEEON_CUDA_INCLUDE}
)
target_link_libraries(
  seeon-deepstream-child PRIVATE PkgConfig::GSTREAMER ${SEEON_NVINFER} ${SEEON_CUDART}
)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(
    seeon-deepstream-ipc-test
    src/ipc_protocol_test.cpp
    src/perception_wire.cpp
  )
  target_compile_features(seeon-deepstream-ipc-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-ipc-test PRIVATE ${SEEON_WARNINGS})
  add_test(NAME seeon-deepstream-ipc-test COMMAND seeon-deepstream-ipc-test)
  add_test(
    NAME seeon-deepstream-wire-cross-language-test
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/perception_wire_cross_language_test.py
      $<TARGET_FILE:seeon-deepstream-ipc-test>
  )
  add_executable(
    seeon-deepstream-source-runtime-test
    src/source_runtime_test.cpp
    src/au_transport.cpp
    src/source_runtime.cpp
    src/source_uri.cpp
  )
  target_compile_features(seeon-deepstream-source-runtime-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-source-runtime-test PRIVATE ${SEEON_WARNINGS})
  add_test(NAME seeon-deepstream-source-runtime-test COMMAND seeon-deepstream-source-runtime-test)
  add_executable(
    seeon-deepstream-source-runtime-gstreamer-test
    src/source_runtime_gstreamer_test.cpp
    src/au_parser.cpp
    src/au_transport.cpp
    src/encoded_source_bus.cpp
    src/encoded_source_branch.cpp
    src/encoded_source_sample.cpp
    src/source_bus.cpp
    src/source_runtime.cpp
    src/source_uri.cpp
  )
  add_executable(
    seeon-deepstream-native-perception-driver
    src/native_perception_driver.cpp
    src/native_perception.cpp
  )
  target_compile_features(seeon-deepstream-native-perception-driver PRIVATE cxx_std_20)
  target_compile_options(
    seeon-deepstream-native-perception-driver PRIVATE ${SEEON_WARNINGS} ${SEEON_FP_FLAGS}
  )
  target_compile_features(seeon-deepstream-source-runtime-gstreamer-test PRIVATE cxx_std_20)
  target_compile_definitions(
    seeon-deepstream-source-runtime-gstreamer-test PRIVATE SEEON_HAS_GSTREAMER=1
  )
  target_compile_options(
    seeon-deepstream-source-runtime-gstreamer-test PRIVATE ${SEEON_WARNINGS}
  )
  target_link_libraries(
    seeon-deepstream-source-runtime-gstreamer-test PRIVATE PkgConfig::GSTREAMER
  )
  add_test(
    NAME seeon-deepstream-source-runtime-gstreamer-test
    COMMAND seeon-deepstream-source-runtime-gstreamer-test
  )
  add_executable(
    seeon-deepstream-child-commands-test
    src/child_commands_test.cpp
    src/au_transport.cpp
    src/child_commands.cpp
    src/child_command_support.cpp
    src/child_failures.cpp
    src/native_perception.cpp
    src/perception_wire.cpp
    src/preview_transport.cpp
    src/source_runtime.cpp
    src/source_uri.cpp
    src/trt_perception_stub.cpp
  )
  target_compile_features(seeon-deepstream-child-commands-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-child-commands-test PRIVATE ${SEEON_WARNINGS})
  add_test(NAME seeon-deepstream-child-commands-test COMMAND seeon-deepstream-child-commands-test)
  add_executable(seeon-deepstream-au-parser-test src/au_parser_test.cpp src/au_parser.cpp)
  target_compile_features(seeon-deepstream-au-parser-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-au-parser-test PRIVATE ${SEEON_WARNINGS})
  add_test(NAME seeon-deepstream-au-parser-test COMMAND seeon-deepstream-au-parser-test)
  add_executable(seeon-deepstream-au-transport-test src/au_transport_test.cpp src/au_transport.cpp)
  target_compile_features(seeon-deepstream-au-transport-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-au-transport-test PRIVATE ${SEEON_WARNINGS})
  add_test(NAME seeon-deepstream-au-transport-test COMMAND seeon-deepstream-au-transport-test)
endif()

install(TARGETS seeon-deepstream-preflight seeon-deepstream-child RUNTIME DESTINATION bin)
