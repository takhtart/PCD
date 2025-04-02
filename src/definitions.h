#ifndef DEFINITIONS
#define DEFINITIONS

// Include Libraries
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>

using namespace std::chrono_literals;
using namespace cv;
using namespace std;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

bool debug = false;

std::vector<std::string> debug_strings;

#endif
