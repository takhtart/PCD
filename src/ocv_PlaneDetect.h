#ifndef OCVPLANES
#define OCVPLANES

// Include Libraries
#include "definitions.h"

bool hasDepthDiscontinuity(const cv::Mat& depthFloat, int y, int x,
                           float baseThreshold = 0.02f) {
  float centerDepth = depthFloat.at<float>(y, x);

  // Ignore invalid depth values
  if (centerDepth <= 0 || centerDepth > 10000) return true;

  int discontinuityCount = 0;

  // Adaptive threshold based on distance (closer = stricter)
  float adaptiveThreshold = baseThreshold * (1.0f + (centerDepth / 4.5f));

  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) continue;

      float neighborDepth = depthFloat.at<float>(y + dy, x + dx);

      // Ignore invalid neighbors
      if (neighborDepth <= 0 || neighborDepth > 10000) {
        discontinuityCount++;
        continue;
      }

      // Normalize difference to avoid distance bias
      float relativeDifference =
          std::abs(neighborDepth - centerDepth) / centerDepth;
      if (relativeDifference > adaptiveThreshold) {
        discontinuityCount++;
      }
    }
  }

  return discontinuityCount >= 1;  // More than 2 depth jumps = edge
}

cv::Mat computeStrictDepthGradients(const cv::Mat& depth) {
  cv::Mat gradX, gradY, gradientMagnitude;

  // Apply Sobel filter
  cv::Sobel(depth, gradX, CV_32F, 1, 0, 3);
  cv::Sobel(depth, gradY, CV_32F, 0, 1, 3);

  // Compute magnitude of gradients
  cv::magnitude(gradX, gradY, gradientMagnitude);

  // Normalize by depth to remove distance bias
  cv::Mat normalizedGradient =
      gradientMagnitude / (depth + 1e-6);  // Avoid division by zero

  return normalizedGradient;
}

bool hasDepthVariance(const cv::Mat& depth, int y, int x, int windowSize = 5,
                      float varianceThreshold = 0.0005f) {
  cv::Rect roi(x - windowSize / 2, y - windowSize / 2, windowSize, windowSize);

  // Ensure ROI is inside depth image bounds
  roi &= cv::Rect(0, 0, depth.cols, depth.rows);

  // Extract local depth region
  cv::Mat localRegion = depth(roi);

  // Compute mean and standard deviation
  cv::Scalar mean, stddev;
  cv::meanStdDev(localRegion, mean, stddev);

  return stddev[0] > varianceThreshold;  // Only accept regions with noticeable
                                         // depth variation
}

bool isStrictlyPlanar(const cv::Mat& normals, int y, int x,
                      float maxAngleThreshold = 5.0f) {
  if (y < 1 || y >= normals.rows - 1 || x < 1 || x >= normals.cols - 1) {
    return false;  // Prevent out-of-bounds access
  }

  cv::Vec3f centerNormal = normals.at<cv::Vec3f>(y, x);
  float maxAngleDiff = 0.0f;
  int validCount = 0;

  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      if (dx == 0 && dy == 0) continue;

      int ny = y + dy;
      int nx = x + dx;

      if (ny < 0 || ny >= normals.rows || nx < 0 || nx >= normals.cols)
        continue;  // Boundary check

      cv::Vec3f neighborNormal = normals.at<cv::Vec3f>(ny, nx);
      float dotProduct = centerNormal.dot(neighborNormal);
      float angleDiff = acos(dotProduct) * 180.0f / CV_PI;

      maxAngleDiff = std::max(maxAngleDiff, angleDiff);
      validCount++;
    }
  }

  return (validCount > 0) && (maxAngleDiff < maxAngleThreshold);
}

void detectAndShowPlanes(const cv::Mat& rgb, const cv::Mat& depth,
                         cv::Mat& planeMask) {
  cv::Mat display = rgb.clone();

  // Convert depth to float
  cv::Mat depthFloat;
  depth.convertTo(depthFloat, CV_32F);

  // Apply median blur to smooth noise
  cv::medianBlur(depthFloat, depthFloat, 5);

  // Compute strict depth gradients
  cv::Mat normalizedGradient = computeStrictDepthGradients(depthFloat);

  // Compute surface normals (approximated using Sobel)
  cv::Mat normalX, normalY;
  cv::Sobel(depthFloat, normalX, CV_32F, 1, 0, 3);
  cv::Sobel(depthFloat, normalY, CV_32F, 0, 1, 3);
  cv::Mat normals;
  cv::merge(std::vector<cv::Mat>{normalX, normalY,
                                 cv::Mat::ones(depth.size(), CV_32F)},
            normals);

  // Normalize normal vectors
  for (int y = 0; y < depth.rows; y++) {
    for (int x = 0; x < depth.cols; x++) {
      cv::Vec3f& n = normals.at<cv::Vec3f>(y, x);
      float norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      if (norm > 1e-6) {
        n /= norm;
      }
    }
  }

  // Apply strict depth & normal filtering with variance check
  planeMask = cv::Mat::zeros(depth.size(), CV_8UC1);
  const float gradientThreshold = 0.005f;

  for (int y = 1; y < depth.rows - 1; y++) {
    for (int x = 1; x < depth.cols - 1; x++) {
      if (normalizedGradient.at<float>(y, x) > gradientThreshold) continue;
      if (!isStrictlyPlanar(normals, y, x)) continue;
      if (!hasDepthVariance(depthFloat, y, x))
        continue;  // Reject low-variance areas (e.g., clothes)
      if (hasDepthDiscontinuity(depthFloat, y, x))
        continue;  // Reject edges with sharp depth jumps

      planeMask.at<uchar>(y, x) = 255;
    }
  }

  // Morphological operations to refine results
  cv::morphologyEx(planeMask, planeMask, cv::MORPH_CLOSE,
                   cv::Mat::ones(3, 3, CV_8U));
  cv::morphologyEx(planeMask, planeMask, cv::MORPH_OPEN,
                   cv::Mat::ones(3, 3, CV_8U));

  // Find connected components (planes)
  cv::Mat labels, stats, centroids;
  int numLabels =
      cv::connectedComponentsWithStats(planeMask, labels, stats, centroids);

  // Color different planes
  std::vector<cv::Scalar> colors = {
      cv::Scalar(255, 0, 0),    // Blue
      cv::Scalar(0, 255, 0),    // Green
      cv::Scalar(0, 0, 255),    // Red
      cv::Scalar(255, 255, 0),  // Cyan
      cv::Scalar(255, 0, 255)   // Magenta
  };

  // Draw detected planes
  for (int label = 1; label < numLabels; label++) {
    if (stats.at<int>(label, cv::CC_STAT_AREA) < 1000) continue;

    cv::Mat currentPlane = (labels == label);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(currentPlane, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
      cv::Scalar color = colors[(label - 1) % colors.size()];
      cv::drawContours(display, contours, -1, color, 2);
      cv::putText(display, "Plane " + std::to_string(label),
                  cv::Point(centroids.at<double>(label, 0),
                            centroids.at<double>(label, 1)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
  }

  cv::imshow("Detected Planes", display);
  cv::waitKey(1);
}

#endif
