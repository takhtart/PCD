#ifndef CONVERSIONS
#define CONVERSIONS

// Include Libraries
#include "definitions.h"

void convertPCLtoOpenCV(const PointCloudT::Ptr& cloud, cv::Mat& rgb,
                        cv::Mat& depth) {
  int width = cloud->width;
  int height = cloud->height;

  rgb = cv::Mat(height, width, CV_8UC3);
  depth = cv::Mat(height, width, CV_32FC1);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int mirrored_x = width - 1 - x;  // Fix horizontal mirroring
      const pcl::PointXYZRGBA& point = cloud->at(mirrored_x, y);

      // Correct color channel order (RGB)
      rgb.at<cv::Vec3b>(y, x) = cv::Vec3b(point.b, point.g, point.r);

      depth.at<float>(y, x) = point.z;
    }
  }
}

#endif
