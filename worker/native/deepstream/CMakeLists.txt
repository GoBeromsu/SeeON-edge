cmake_minimum_required(VERSION 3.22)
project(seeon_deepstream_preflight VERSION 1.0.0 LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-base-1.0)

set(SEEON_WARNINGS -Wall -Wextra -Wpedantic -Werror)

add_executable(seeon-deepstream-preflight src/main.cpp)
target_compile_features(seeon-deepstream-preflight PRIVATE cxx_std_20)
target_compile_options(seeon-deepstream-preflight PRIVATE ${SEEON_WARNINGS})
target_link_libraries(seeon-deepstream-preflight PRIVATE PkgConfig::GSTREAMER)

add_executable(
  seeon-deepstream-child
  src/child.cpp
  src/child_commands.cpp
  src/child_command_support.cpp
  src/child_failures.cpp
  src/child_server.cpp
  src/perception_wire.cpp
  src/source_runtime.cpp
  src/source_uri.cpp
)
target_compile_features(seeon-deepstream-child PRIVATE cxx_std_20)
target_compile_definitions(seeon-deepstream-child PRIVATE SEEON_HAS_GSTREAMER=1)
target_compile_options(seeon-deepstream-child PRIVATE ${SEEON_WARNINGS})
target_link_libraries(seeon-deepstream-child PRIVATE PkgConfig::GSTREAMER)

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
    src/source_runtime.cpp
    src/source_uri.cpp
  )
  target_compile_features(seeon-deepstream-source-runtime-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-source-runtime-test PRIVATE ${SEEON_WARNINGS})
  add_test(NAME seeon-deepstream-source-runtime-test COMMAND seeon-deepstream-source-runtime-test)
  add_executable(
    seeon-deepstream-source-runtime-gstreamer-test
    src/source_runtime_gstreamer_test.cpp
    src/source_runtime.cpp
    src/source_uri.cpp
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
    src/child_commands.cpp
    src/child_command_support.cpp
    src/child_failures.cpp
    src/perception_wire.cpp
    src/source_runtime.cpp
    src/source_uri.cpp
  )
  target_compile_features(seeon-deepstream-child-commands-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-child-commands-test PRIVATE ${SEEON_WARNINGS})
  add_test(NAME seeon-deepstream-child-commands-test COMMAND seeon-deepstream-child-commands-test)
endif()

install(TARGETS seeon-deepstream-preflight seeon-deepstream-child RUNTIME DESTINATION bin)
