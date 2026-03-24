cmake_minimum_required(VERSION 3.0.2)
project(test_dfmatch)

# Set the C and C++ compilers
# set(CMAKE_CXX_COMPILER /root/dockershare/1_RosCode/work_humble_ws_x5/compiler/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)

set(CMAKE_CXX_STANDARD 17)

# Set default build type (can be overridden by external command)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(PLATFORM_X5)
  add_definitions(-DPLATFORM_X5)
elseif(PLATFORM_S100)
  add_definitions(-DPLATFORM_S100)
endif()

# Compiler optimization settings
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=armv8-a -funroll-loops -ffast-math -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# ===================================================== include ====================================================
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lib_opencv4.5.4/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

include_directories(
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen3
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/magic_enum/include
)
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn/hobot/lib
)

# ===================================================== executable ==================================================

add_executable(dfmatch_infer
  main.cpp
  src/DFeatInfer.cpp
  src/LightGlueInfer.cpp
  src/timer_utils.cpp
)

target_link_libraries(dfmatch_infer
  dnn cnn_intf hbmem hbrt_bayes_aarch64 alog
  ${OpenCV_LIBS}
)