cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2_cpf)

SET(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
#add_subdirectory(./backward-cpp)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
set(CXX_COMPILER_FLAGS -03 -ffast-math -fomit-frame-pointer -funroll-loops -fopenmp=libomp -msse2)

# Check C++11 or C++0x support
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
#if(COMPILER_SUPPORTS_CXX11)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#   add_definitions(-DCOMPILEDWITHC11)
#   message(STATUS "Using flag -std=c++11.")
#elseif(COMPILER_SUPPORTS_CXX0X)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#   add_definitions(-DCOMPILEDWITHC0X)
#   message(STATUS "Using flag -std=c++0x.")
#else()
#   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
#endif()

SET(CMAKE_C_COMPILER "/usr/bin/gcc-5")

SET(CMAKE_CXX_COMPILER "/usr/bin/g++-5")

add_definitions(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
#list(APPEND CMAKE_C_FLAGS "-std=c++11 -O3 -ffast-math -Wno-unused-result")
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_PREFIX_PATH  /home/shuke/libtorch/libtorch)

find_package (Boost REQUIRED)
find_package(Torch REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
# adding for point cloud viewer and mapper
find_package(PCL 1.8 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
#find_package(CUDA  10.1 REQUIRED)
find_package(OpenMP)

set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.8)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
ADD_DEFINITIONS(-DBoost_USE_STATIC_LIBS=ON)
set(Boost_INCLUDE_DIR /usr/include/)
set(Boost_LIBRARY_DIR /usr/lib/x86_64-linux-gnu/)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${OpenCV_INCLUDE_DIRS}
# openni
${OPENNI_INCLUDE_DIR}
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR}/gco-v3.0
)


add_definitions(${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS} )

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS
    include/gco-v3.0/graph.h
    include/gco-v3.0/block.h
    include/gco-v3.0/energy.h
    include/gco-v3.0/GCoptimization.h
    include/gco-v3.0/LinkedBlockList.h
    include/config.h
    include/lsa_tr.h
    include/segmentation.h
    include/segmentation_helpers.h
    )


add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/pointcloudmapping.cc
src/segmentation.cc
src/Darknet.cpp

gco-v3.0/maxflow.cpp gco-v3.0/maxflow.cpp gco-v3.0/LinkedBlockList.cpp gco-v3.0/GCoptimization.cpp gco-v3.0/graph.cpp ${HEADERS})

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PCL_LIBRARIES}
${TORCH_LIBRARIES}
#openni
/usr/lib/libOpenNI2.so
m
dl
pthread
stdc++
-lboost_system
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)

#CONFIGURE_FILE(./src/segmentation.pc.in segmentation.pc @ONLY)
# Build examples

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#add_executable(rgbd_tum Examples/RGB-D/rgbd_tum.cc ${BACKWARD_ENABLE})
add_executable(rgbd_tum Examples/RGB-D/rgbd_tum.cc )
#add_backward(rgbd_tum)
target_link_libraries(rgbd_tum ${PROJECT_NAME} )
set_property(TARGET rgbd_tum PROPERTY CXX_STANDARD 11)

# Build tools
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools)
#add_executable(bin_vocabulary
#tools/bin_vocabulary.cc)
#target_link_libraries(bin_vocabulary ${PROJECT_NAME})
