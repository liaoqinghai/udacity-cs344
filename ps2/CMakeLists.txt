cmake_minimum_required (VERSION 2.8)

# projectname is the same as the main-executable
project(PS2)

add_definitions('-g')
add_definitions('-Wall')
add_definitions('-std=c++11')

# CUDA PACKAGE
find_package(CUDA REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
find_package(OpenCV REQUIRED)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	)

list( APPEND CUDA_NVCC_FLAGS "-O3 -std=c++11")

cuda_add_executable(
	blur
	gaussian_blur.cu
	)
target_link_libraries(
	blur
	${OpenCV_LIBS} 
	)

