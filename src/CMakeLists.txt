cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(PCD)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable (PCD "main.cpp" "kinect2_grabber.h" "conversions.h" "definitions.h" "visualization.h" "ocv_PlaneDetect.h" "ocv_SkinDetect.h" "pcl_HelperFunctions.h" "pcl_RegionEstimation.h" "pcl_RegionGrowing.h")
find_package(PCL 1.7 REQUIRED)
find_package(KinectSDK2 REQUIRED)
find_package(CUDA)

# OpenCV Dependencies
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( PCD ${OpenCV_LIBS})

if( PCL_FOUND AND KinectSDK2_FOUND )
  # Additional Include Directories
  include_directories( ${PCL_INCLUDE_DIRS} )
  include_directories( ${KinectSDK2_INCLUDE_DIRS} )

  find_package(FLANN)

  # Preprocessor Definitions
  add_definitions( ${PCL_DEFINITIONS} )

  # Additional Library Directories
  link_directories( ${PCL_LIBRARY_DIRS} )
  link_directories( ${KinectSDK2_LIBRARY_DIRS} )

  # Additional Dependencies
  target_link_libraries( PCD ${PCL_LIBRARIES} )
  target_link_libraries( PCD ${KinectSDK2_LIBRARIES} )

endif()


# Enable testing
enable_testing()

# Add Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add the test executable
add_executable(runTests "UnitTests.cpp" "kinect2_grabber.h" "conversions.h" "definitions.h" "visualization.h" "ocv_PlaneDetect.h" "ocv_SkinDetect.h" "pcl_HelperFunctions.h" "pcl_RegionEstimation.h" "pcl_RegionGrowing.h")
target_compile_definitions(runTests PRIVATE UNIT_TEST) # Define UNIT_TEST for the test executable
target_link_libraries(runTests gtest_main ${PCL_LIBRARIES} ${OpenCV_LIBS} ${KinectSDK2_LIBRARIES})

# Enable CTest
include(GoogleTest)
gtest_discover_tests(runTests)