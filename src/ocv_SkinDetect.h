#ifndef OCVSKINDETECT
#define OCVSKINDETECT

// Include Libraries
#include "definitions.h"

std::vector<pcl::PointXYZ> detectSkinAndConvertToPCL(
    const cv::Mat& frame, const cv::Mat& depthframe,
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud,
    const cv::Mat& planeMask) {
  Mat hsv, mask_hsv, mask_red, lower_red, upper_red, combined_mask, blurred,
      thresholded, inv_red;
  std::vector<pcl::PointXYZ> detectedPoints;

  // Convert frame to HSV and YCrCb
  cvtColor(frame, hsv, COLOR_BGR2HSV);

  // Skin detection range in HSV
  Scalar hsv_lower(0, 40, 40);
  Scalar hsv_upper(20, 255, 255);
  inRange(hsv, hsv_lower, hsv_upper, mask_hsv);

  // Remove reds
  Scalar hsv_red_lower1(0, 90, 90);
  Scalar hsv_red_lower2(9, 255, 255);

  Scalar hsv_red_upper1(160, 70, 50);
  Scalar hsv_red_upper2(180, 255, 255);
  inRange(hsv, hsv_red_lower1, hsv_red_lower2, lower_red);
  inRange(hsv, hsv_red_upper1, hsv_red_upper2, upper_red);

  bitwise_or(lower_red, upper_red, mask_red);

  Mat kernel1 = getStructuringElement(MORPH_RECT, Size(7, 7));

  morphologyEx(mask_red, mask_red, MORPH_CLOSE, kernel1, Point(-1, -1), 20);

  // Invert to create a filter for reds
  bitwise_not(mask_red, inv_red);

  bitwise_and(inv_red, mask_hsv, combined_mask);

  // Apply plane exclusion
  bitwise_and(combined_mask, ~planeMask, combined_mask);

  // Display masks for debugging
  imshow("Initial Skin Mask", mask_hsv);
  imshow("Red Mask (Lower)", lower_red);
  imshow("Red Mask (Upper)", upper_red);
  imshow("Final Red Mask", mask_red);
  imshow("Filtered Skin Mask", combined_mask);

  // Local intensity analysis
  Mat localIntensity;
  Mat kernel = Mat::ones(9, 9, CV_32F) / (9 * 9);
  filter2D(combined_mask, localIntensity, -1, kernel);

  // Threshold high-confidence areas
  Mat highConfidenceMask;
  threshold(localIntensity, highConfidenceMask, 100, 255, THRESH_BINARY);

  // Clean up high-confidence mask
  Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
  morphologyEx(highConfidenceMask, highConfidenceMask, MORPH_CLOSE, element2);

  // Combine high-confidence mask with final mask
  Mat finalMask;
  bitwise_and(combined_mask, highConfidenceMask, finalMask);

  // Display debugging masks
  imshow("High-Confidence Mask", highConfidenceMask);
  imshow("Final Skin Regions Mask", finalMask);

  // Find contours
  vector<vector<Point>> contours;
  findContours(finalMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  for (size_t i = 0; i < contours.size(); i++) {
    double area = contourArea(contours[i]);

    // Approximate contour
    vector<Point> approx;
    approxPolyDP(contours[i], approx, 10, true);

    // Bounding box and aspect ratio check
    Rect boundingBox = boundingRect(approx);
    float aspectRatio = (float)boundingBox.width / (float)boundingBox.height;

    // Convex hull check
    vector<Point> hullPoints;
    convexHull(contours[i], hullPoints, false);
    double hullArea = contourArea(hullPoints);

    // Get centroid
    Moments M = moments(contours[i]);
    int cx = static_cast<int>(M.m10 / M.m00);
    int cy = static_cast<int>(M.m01 / M.m00);

    // Draw detected regions
    drawContours(frame, contours, static_cast<int>(i), Scalar(0, 255, 0), 2);
    circle(frame, Point(cx, cy), 5, Scalar(255, 0, 0), -1);

    // Convert OpenCV pixel position to PCL world coordinates
    if (cx >= 0 && cy >= 0 && cx < cloud->width && cy < cloud->height) {
      pcl::PointXYZRGBA pclPoint = cloud->at(cx, cy);
      detectedPoints.push_back(
          pcl::PointXYZ(pclPoint.x, pclPoint.y, pclPoint.z));
    }
  }

  // Imshow Waiting
  waitKey(1);

  return detectedPoints;
}

#endif
