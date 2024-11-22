cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(cloud_viewer)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
add_executable (cloud_viewer cloud_viewer.cpp "kinect2_grabber.h")
find_package(PCL 1.7 REQUIRED)
find_package(KinectSDK2 REQUIRED)
find_package(CUDA)

#find_package(OpenCV REQUIRED)

if( PCL_FOUND AND KinectSDK2_FOUND )
  # Additional Include Directories
  include_directories( ${PCL_INCLUDE_DIRS} )
  include_directories( ${KinectSDK2_INCLUDE_DIRS} )
  #include_directories(${OpenCV_INCLUDE_DIRS})

  find_package(FLANN)

  # Preprocessor Definitions
  add_definitions( ${PCL_DEFINITIONS} )

  # Additional Library Directories
  link_directories( ${PCL_LIBRARY_DIRS} )
  link_directories( ${KinectSDK2_LIBRARY_DIRS} )

  # Additional Dependencies
  target_link_libraries( cloud_viewer ${PCL_LIBRARIES} )
  target_link_libraries( cloud_viewer ${KinectSDK2_LIBRARIES} )
  #target_link_libraries(my_app ${OpenCV_LIBS})

endif()


