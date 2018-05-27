#
#  CMakeList.txt.pl
#  YOLOv3_SpringEdition
#
#  Created by kimbomm on 2018. 04. 01...
#  Copyright 2018 Sogang CVIP. All rights reserved.
#
cmake_minimum_required(VERSION 3.2)
project(YOLOv3_SpringEdition)

find_package(CUDA QUIET REQUIRED)
include(FindCUDA)

set(CMAKE_C_STANDARD 11)
list(APPEND CMAKE_C_FLAGS "-std=c11 -O3 -ffast-math -Wno-unused-result")
list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)    #for maxwell
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)    #for maxwell
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)    #for pascal
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)    #for pascal
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)    #for jetson tx2
#https://developer.nvidia.com/cuda-gpus

add_definitions(-DGPU)
add_definitions(-DCUDNN)
add_definitions(-D_TIMESPEC_DEFINED)
add_definitions(-D_TIMESPEC_DEFINED)

set(SOURCE_FILES
        src/detector.c
        src/darknet.h
        src/activation_layer.c
        src/activation_layer.h
        src/activations.c
        src/activations.h
        src/avgpool_layer.c
        src/avgpool_layer.h
        src/batchnorm_layer.c
        src/batchnorm_layer.h
        src/blas.c
        src/blas.h
        src/box.c
        src/box.h
        src/classifier.h
        src/col2im.c
        src/col2im.h
        src/connected_layer.c
        src/connected_layer.h
        src/convolutional_layer.c
        src/convolutional_layer.h
        src/cost_layer.c
        src/cost_layer.h
        src/crnn_layer.c
        src/crnn_layer.h
        src/crop_layer.c
        src/crop_layer.h
        src/cuda.c
        src/cuda.h
        src/data.c
        src/data.h
        src/deconvolutional_layer.c
        src/deconvolutional_layer.h
        src/demo.c
        src/demo.h
        src/detection_layer.c
        src/detection_layer.h
        src/dropout_layer.c
        src/dropout_layer.h
        src/gemm.c
        src/gemm.h
        src/gru_layer.c
        src/gru_layer.h
        src/im2col.c
        src/im2col.h
        src/image.c
        src/image.h
        src/l2norm_layer.c
        src/l2norm_layer.h
        src/layer.c
        src/layer.h
        src/list.c
        src/list.h
        src/local_layer.c
        src/local_layer.h
        src/logistic_layer.c
        src/logistic_layer.h
        src/lstm_layer.c
        src/lstm_layer.h
        src/matrix.c
        src/matrix.h
        src/maxpool_layer.c
        src/maxpool_layer.h
        src/network.c
        src/network.h
        src/normalization_layer.c
        src/normalization_layer.h
        src/option_list.c
        src/option_list.h
        src/parser.c
        src/parser.h
        src/region_layer.c
        src/region_layer.h
        src/reorg_layer.c
        src/reorg_layer.h
        src/rnn_layer.c
        src/rnn_layer.h
        src/route_layer.c
        src/route_layer.h
        src/shortcut_layer.c
        src/shortcut_layer.h
        src/softmax_layer.c
        src/softmax_layer.h
        src/stb_image.h
        src/stb_image_write.h
        src/tree.c
        src/tree.h
        src/upsample_layer.c
        src/upsample_layer.h
        src/utils.c
        src/utils.h
        src/yolo_layer.c
        src/yolo_layer.h
        src/activation_kernels.cu
        src/convolutional_kernels.cu
        src/im2col_kernels.cu
        src/avgpool_layer_kernels.cu
        src/crop_layer_kernels.cu
        src/maxpool_layer_kernels.cu
        src/blas_kernels.cu
        src/deconvolutional_kernels.cu
        src/col2im_kernels.cu
        src/dropout_layer_kernels.cu
        )
add_library(cuda INTERFACE)
set_target_properties(cuda PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
    INTERFACE_LINK_LIBRARIES "${CUDA_LIBRARIES};${CUDA_curand_LIBRARY};${CUDA_CUFFT_LIBRARIES};${CUDA_CUBLAS_LIBRARIES}"
)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
if(UNIX)
	include_directories(/usr/local/cuda/include)
	cuda_add_executable(YOLOv3SE_Train ${SOURCE_FILES} src/main.c)   #Make train executable
	target_link_libraries(YOLOv3SE_Train cuda pthread m
			/usr/local/cuda/lib64/libcudnn.so
	)
	cuda_add_library(YOLOv3SE SHARED ${SOURCE_FILES} src/main.c src/yolo_v3.c)       #make so file
	SET_TARGET_PROPERTIES(YOLOv3SE PROPERTIES COMPILE_FLAGS "-D_WINDLL")
	target_link_libraries(YOLOv3SE cuda pthread m
			/usr/local/cuda/lib64/libcudnn.so
			#/usr/lib/aarch64-linux-gnu/libcudnn.so	# for Jetson Tx2
	)
	target_include_directories(YOLOv3SE PUBLIC ${PROJECT_SOURCE_DIR}/include)
endif(UNIX)
if(MSVC)
	add_definitions(-D_CRT_RAND_S)
	include_directories(${PROJECT_SOURCE_DIR}/3rdparty/include)
	set(PTHREAD_SRC ${PROJECT_SOURCE_DIR}/3rdparty/src/pthread/pthread.c)
	cuda_add_library(libYOLOv3SE SHARED ${SOURCE_FILES} ${PTHREAD_SRC} src/yolo_v3.c)       #Make dll file
	cuda_add_executable(YOLOv3SE_Train ${SOURCE_FILES} ${PTHREAD_SRC} src/main.c)   		#Make train executable
	target_link_libraries(YOLOv3SE_Train ${PROJECT_SOURCE_DIR}/3rdparty/lib/cudnn.lib cuda)
	target_link_libraries(libYOLOv3SE ${PROJECT_SOURCE_DIR}/3rdparty/lib/cudnn.lib cuda)
endif(MSVC)
