CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(PhotometricStereo)

# required packages
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(VTK REQUIRED)
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)

#required macros and variables
INCLUDE(${VTK_USE_FILE})

set(CMAKE_CXX_STANDARD 17)

add_compile_options(
  -mfpmath=sse
  -fstack-protector-all
  -W
  -Wall
  -Wextra
  -Werror
  -Wunused
  -Wcast-align
  -pedantic
  -pedantic-errors
  -Wfloat-equal
  -Wpointer-arith
  -Wmissing-format-attribute
  -Wformat=2
  -Wwrite-strings
  -Wcast-align
  -Wno-long-long
  -Wcast-qual
  -Wpedantic
  -Werror=switch
  -Wswitch
)

# add all sources automagically
FILE(GLOB SOURCES *.cpp *.h)

ADD_EXECUTABLE(PhotometricStereo ${SOURCES})
TARGET_LINK_LIBRARIES(PhotometricStereo ${OpenCV_LIBS} ${VTK_LIBRARIES} ${Boost_LIBRARIES})
