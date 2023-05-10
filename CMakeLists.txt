CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(PhotometricStereo)

# required packages
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(VTK REQUIRED)

#required macros and variables
INCLUDE(${VTK_USE_FILE})
ADD_DEFINITIONS(-std=c++17)

# add all sources automagically
FILE(GLOB SOURCES *.cpp *.h)

ADD_EXECUTABLE(PhotometricStereo ${SOURCES})
TARGET_LINK_LIBRARIES(PhotometricStereo ${OpenCV_LIBS} ${VTK_LIBRARIES})
