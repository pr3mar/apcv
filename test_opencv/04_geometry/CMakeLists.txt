PROJECT(cpp)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)

FIND_PACKAGE(OpenCV REQUIRED)
LINK_LIBRARIES(${OpenCV_LIBS})
INCLUDE_DIRECTORIES(AFTER ${CMAKE_CURRENT_BINARY_DIR} ${OpenCV_INCLUDE_DIR})

ADD_EXECUTABLE(calibration calibration.cpp)
ADD_EXECUTABLE(rectify rectify.cpp)
ADD_EXECUTABLE(pose pose.cpp features.cpp)
ADD_EXECUTABLE(keypoints keypoints.cpp features.cpp)
ADD_EXECUTABLE(matching matching.cpp features.cpp)
ADD_EXECUTABLE(homography homography.cpp features.cpp)

#FILE(COPY cathedral.jpg cathedral1.jpg cathedral2.jpg cathedral3.jpg cathedral4.jpg marker.png DESTINATION ${BUILD_ROOT})
