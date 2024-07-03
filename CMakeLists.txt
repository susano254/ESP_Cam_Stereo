cmake_minimum_required(VERSION 3.5)
project(ESP_Stereo_OpenCV)

set(CMAKE_BUILD_TYPE "Debug")
# Add c++11 standard support
set(CMAKE_CXX_STANDARD 11)


# Eigen
include_directories("/usr/include/eigen3")

# Find OpenCV library
find_package(OpenCV REQUIRED)
#Add header file
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)

find_package(OpenAL REQUIRED)
# # Find OpenAL manually
# find_library(OPENAL_LIBRARY NAMES openal OpenAL libopenal PATHS /usr/local/lib /usr/lib /lib)
add_library(OpenAL SHARED IMPORTED)
set_target_properties(OpenAL PROPERTIES IMPORTED_LOCATION /usr/local/lib/libopenal.so)

add_executable(ESP_StereoVision ESP_StereoVision.cpp Stereo_Glasses.cpp)
target_link_libraries(ESP_StereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES} opencv_ximgproc OpenAL)