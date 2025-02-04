// Point Types
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Visualizer
#include <pcl/visualization/pcl_visualizer.h>

// Threading
#include <mutex>
#include <thread>

// Load PCD File
#include <pcl/io/pcd_io.h>

// Voxel Grid Downsampling
#include <pcl/filters/voxel_grid.h>

// Noise Outlier Removal
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

// Plane Segmentation
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

// Euclidean Cluster Extraction
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

// Bounding Box Calculations
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>

// Grabber for KinectV2
#include "kinect2_grabber.h"

// OpenCV
#include <Kinect.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <chrono>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>

// Define the point cloud type
using namespace std::chrono_literals;
using namespace cv;
using namespace std;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// Initialize cloud mutex
std::mutex cloud_mutex;

// Convert PCL PointCloud to OpenCV RGB and Depth Mat
void convertPCLtoOpenCV(const PointCloudT::Ptr& cloud, cv::Mat& rgb,
                        cv::Mat& depth) {
  int width = cloud->width;
  int height = cloud->height;

  rgb = cv::Mat(height, width, CV_8UC3);
  depth = cv::Mat(height, width, CV_32FC1);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const pcl::PointXYZRGBA& point = cloud->at(x, y);
      rgb.at<cv::Vec3b>(y, x) =
          cv::Vec3b(point.b, point.g, point.r);  // Convert to OpenCV format
      depth.at<float>(y, x) = point.z;           // Store depth
    }
  }
}

// Detect ORB keypoints in the RGB image
void detectKeypoints(cv::Mat& rgb) {
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  orb->detectAndCompute(rgb, cv::noArray(), keypoints, descriptors);
  cv::drawKeypoints(rgb, keypoints, rgb,
                    cv::Scalar(0, 255, 0));  // Draw keypoints
}

// Depth-based segmentation (Thresholding)
cv::Mat segmentDepth(const cv::Mat& depth, float minDepth, float maxDepth) {
  cv::Mat mask;
  cv::inRange(depth, minDepth, maxDepth, mask);
  return mask;
}

// Display OpenCV results
void displayOpenCVResults(cv::Mat& rgb, cv::Mat& depth_mask) {
  cv::imshow("RGB Skin Detect", rgb);
  // cv::imshow("Depth Segmentation", depth_mask);
  cv::waitKey(1);
}

std::vector<pcl::PointXYZ> detectSkinAndConvertToPCL(
    const cv::Mat& frame, const cv::Mat& depthframe,
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud) {
  Mat hsv, ycrcb, mask_hsv, mask_ycrcb, mask_rgb, combined_mask, blurred,
      thresholded;
  std::vector<pcl::PointXYZ> detectedPoints;

  // Convert frame to HSV and YCrCb
  cvtColor(frame, hsv, COLOR_BGR2HSV);
  cvtColor(frame, ycrcb, COLOR_BGR2YCrCb);

  // Refined HSV range based on the article
  Scalar hsv_lower(0, 48, 80);     // Lower bound for skin detection
  Scalar hsv_upper(20, 255, 255);  // Upper bound for skin detection
  inRange(hsv, hsv_lower, hsv_upper, mask_hsv);

  // Refined YCrCb range based on the article
  Scalar ycrcb_lower(50, 137, 85);
  Scalar ycrcb_upper(200, 177, 135);
  inRange(ycrcb, ycrcb_lower, ycrcb_upper, mask_ycrcb);

  // Add RGB thresholds
  // Scalar rgb_lower(95, 40, 20);
  // Scalar rgb_upper(255, 255, 255);
  // inRange(frame, rgb_lower, rgb_upper, mask_rgb);

  // Modify the weighted mask merging to include RGB
  addWeighted(mask_hsv, 0.6, mask_ycrcb, 0.4, 0, combined_mask);
  // addWeighted(combined_mask, 1.0, mask_rgb, 0.2, 0, combined_mask);

  // Additional morphological operations to clean up noise
  Mat element = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
  erode(combined_mask, combined_mask, element);
  dilate(combined_mask, combined_mask, element);

  // Noise reduction
  GaussianBlur(combined_mask, blurred, Size(5, 5), 0);
  threshold(blurred, thresholded, 100, 255, THRESH_BINARY);

  // **Display masks for debugging**
  imshow("HSV Mask", mask_hsv);
  imshow("YCrCb Mask", mask_ycrcb);
  // imshow("RGB Mask", mask_rgb);
  imshow("Merged Mask", combined_mask);

  // Analyze local intensity of the merged mask
  Mat localIntensity;
  Mat kernel =
      Mat::ones(9, 9, CV_32F) / (9 * 9);  // Smaller kernel for local averaging
  filter2D(combined_mask, localIntensity, -1,
           kernel);  // Compute local intensity

  // Threshold to highlight areas with high concentration of white pixels (less
  // strict)
  Mat highConfidenceMask;
  threshold(localIntensity, highConfidenceMask, 150, 255,
            THRESH_BINARY);  // Lower threshold

  // Clean up the high-confidence mask (less strict)
  Mat element2 = getStructuringElement(
      MORPH_ELLIPSE, Size(5, 5));  // Smaller structuring element
  morphologyEx(highConfidenceMask, highConfidenceMask, MORPH_OPEN,
               element2);  // Remove small noise
  morphologyEx(highConfidenceMask, highConfidenceMask, MORPH_CLOSE,
               element2);  // Fill small holes

  // Combine high-confidence mask with the original merged mask
  Mat finalMask;
  bitwise_and(combined_mask, highConfidenceMask,
              finalMask);  // Retain only overlapping regions

  // Display the high-confidence mask and final mask
  imshow("High-Confidence Mask", highConfidenceMask);
  imshow("Final Mask", finalMask);

  // Find all contours in the final mask
  vector<vector<Point>> contours;
  findContours(finalMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  for (size_t i = 0; i < contours.size(); i++) {
    double area = contourArea(contours[i]);
    // if (area < 150)
    // continue;

    // Approximate contour to remove noise
    vector<Point> approx;
    approxPolyDP(contours[i], approx, 10, true);

    // Aspect Ratio Check: Ensure the contour is not too wide (avoids detecting
    // boxes)
    Rect boundingBox = boundingRect(approx);
    float aspectRatio = (float)boundingBox.width / (float)boundingBox.height;
    // if (aspectRatio < 0.3 || aspectRatio > 1.5) // Typical hand aspect ratio
    // continue;

    // Convex Hull Filtering (Fix: Get actual hull points, not indices)
    vector<Point> hullPoints;
    convexHull(contours[i], hullPoints, false);
    double hullArea =
        contourArea(hullPoints);  // Now correctly calculates hull area
    // if (hullArea / area < 1.0) // Ensures the shape is convex like a hand
    // continue;

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

  // **Wait for keypress to refresh debugging windows**
  waitKey(1);

  return detectedPoints;
}

// Visualizer thread function
void run_visualizer(PointCloudT::Ptr cloud, bool* viewer_running,
                    std::mutex* cloud_mutex) {
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");
  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);

  {
    std::lock_guard<std::mutex> lock(*cloud_mutex);
    viewer.addPointCloud<PointT>(cloud, rgb, "input_cloud");
    viewer.setCameraPosition(0, 0, -2, 0, 1, 0, 0);
  }

  while (!viewer.wasStopped() && *viewer_running) {
    {
      std::lock_guard<std::mutex> lock(*cloud_mutex);
      viewer.updatePointCloud<PointT>(cloud, rgb, "input_cloud");
    }
    viewer.spinOnce();
  }

  *viewer_running = false;
}

void body_part_segmentation(PointCloudT::Ptr& cloud_filtered) {
  if (cloud_filtered->points.empty()) {
    std::cerr << "Point cloud is empty!" << std::endl;
    return;
  }

  // Step 1: Compute Torso Centroid (Approximate Density-based)
  Eigen::Vector3f torso_centroid(0, 0, 0);
  float min_z = std::numeric_limits<float>::max();
  float max_z = std::numeric_limits<float>::lowest();

  for (const auto& point : cloud_filtered->points) {
    torso_centroid += Eigen::Vector3f(point.x, point.y, point.z);
    min_z = std::min(min_z, point.z);
    max_z = std::max(max_z, point.z);
  }

  torso_centroid /= cloud_filtered->points.size();

  // Step 2: Segment Body Parts
  std::vector<int> head_indices, torso_indices, arm_indices, leg_indices;

  for (size_t i = 0; i < cloud_filtered->points.size(); ++i) {
    const auto& point = cloud_filtered->points[i];

    float dx = point.x - torso_centroid[0];
    float dy = point.y - torso_centroid[1];
    float dz = point.z - torso_centroid[2];

    // Heuristic-based classification
    if (dz > 0.25f) {  // Head is above torso
      head_indices.push_back(i);
    } else if (dz < -0.2f) {  // Legs are below torso
      leg_indices.push_back(i);
    } else if (std::abs(dx) > 0.2f) {  // Arms extend outward
      arm_indices.push_back(i);
    } else {  // Torso region
      torso_indices.push_back(i);
    }
  }

  // Step 3: Assign Colors
  for (size_t i = 0; i < cloud_filtered->points.size(); ++i) {
    auto& point = cloud_filtered->points[i];
    if (std::find(head_indices.begin(), head_indices.end(), i) !=
        head_indices.end()) {
      point.r = 255;
      point.g = 255;
      point.b = 0;  // Yellow (Head)
    } else if (std::find(leg_indices.begin(), leg_indices.end(), i) !=
               leg_indices.end()) {
      point.r = 0;
      point.g = 255;
      point.b = 0;  // Green (Legs)
    } else if (std::find(arm_indices.begin(), arm_indices.end(), i) !=
               arm_indices.end()) {
      point.r = 0;
      point.g = 0;
      point.b = 255;  // Blue (Arms)
    } else {
      point.r = 255;
      point.g = 0;
      point.b = 0;  // Red (Torso)
    }
  }
}

void recoverSurroundingPoints(PointCloudT::Ptr& original_cloud,
                              PointCloudT::Ptr& extracted_region,
                              PointCloudT::Ptr& result_cloud,
                              float surrounding_radius = 0.05f) {
  // 1. Voxel Downsampling with larger leaf size for performance
  pcl::VoxelGrid<PointT> voxel_filter;
  voxel_filter.setInputCloud(original_cloud);
  voxel_filter.setLeafSize(0.017f, 0.017f, 0.017f);  // Larger leaf size
  voxel_filter.filter(*original_cloud);

  // Create a KD-Tree for the extracted region
  pcl::search::KdTree<PointT>::Ptr extracted_tree(
      new pcl::search::KdTree<PointT>);
  extracted_tree->setInputCloud(extracted_region);

  // Create a KD-Tree for the original cloud
  pcl::search::KdTree<PointT>::Ptr original_tree(
      new pcl::search::KdTree<PointT>);
  original_tree->setInputCloud(original_cloud);

  // Create a set of existing points to avoid duplicates
  std::set<std::tuple<float, float, float>> existing_points;
  for (const auto& point : extracted_region->points) {
    existing_points.insert({point.x, point.y, point.z});
  }

  // Temporary cloud to store recovered points
  PointCloudT::Ptr recovered_points(new PointCloudT);

  // Search for points in the extracted region that are close to original cloud
  for (const auto& extracted_point : extracted_region->points) {
    // Find nearest neighbors in the original cloud
    std::vector<int> neighbor_indices;
    std::vector<float> neighbor_distances;

    // Radius search around the extracted point in the original cloud
    original_tree->radiusSearch(extracted_point, surrounding_radius,
                                neighbor_indices, neighbor_distances);

    // If close neighbors exist in the original cloud, add them
    for (size_t i = 0; i < neighbor_indices.size(); ++i) {
      const auto& original_point = original_cloud->points[neighbor_indices[i]];

      // Skip if point is already in the extracted region
      if (existing_points.find({original_point.x, original_point.y,
                                original_point.z}) != existing_points.end()) {
        continue;
      }

      // Color consistency check
      float r_ratio =
          std::abs((original_point.r + 1.0f) / (extracted_point.r + 1.0f));
      float g_ratio =
          std::abs((original_point.g + 1.0f) / (extracted_point.g + 1.0f));
      float b_ratio =
          std::abs((original_point.b + 1.0f) / (extracted_point.b + 1.0f));

      // Consistency thresholds
      const float color_ratio_threshold = 1.5f;

      // Check color consistency
      if (r_ratio < color_ratio_threshold && g_ratio < color_ratio_threshold) {
        recovered_points->points.push_back(original_point);
      }
    }
  }

  // Combine extracted region with recovered points
  *result_cloud = *extracted_region;
  result_cloud->points.insert(result_cloud->points.end(),
                              recovered_points->points.begin(),
                              recovered_points->points.end());

  // Update cloud metadata
  result_cloud->width = result_cloud->points.size();
  result_cloud->height = 1;
  result_cloud->is_dense = true;
}

void process_cloud(PointCloudT::Ptr& cloud, PointCloudT::Ptr& cloud_filtered,
                   std::vector<pcl::PointXYZ> skinPoints) {
  // Store original cloud before filtering
  PointCloudT::Ptr original_cloud(new PointCloudT(*cloud));

  // 1. Voxel Downsampling with larger leaf size for performance
  pcl::VoxelGrid<PointT> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(0.018f, 0.018f, 0.018f);  // Larger leaf size
  voxel_filter.filter(*cloud_filtered);

  // 2. Remove Planes
  pcl::SACSegmentation<PointT> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(250);
  seg.setDistanceThreshold(0.002);

  seg.setInputCloud(cloud_filtered);

  int iterations = 0;
  while (iterations < 6) {
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
      break;
    }

    // Extract the normal vector of the plane
    float nx = coefficients->values[0];
    float ny = coefficients->values[1];
    float nz = coefficients->values[2];

    // Remove Horizontal planes: i.e. floor, ceiling or tables
    if (std::abs(nz) > 0.9f) {
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud(cloud_filtered);
      extract.setIndices(inliers);
      extract.setNegative(true);
      extract.filter(*cloud_filtered);
    }
    // Remove Vertical planes: i.e. walls
    else if (std::abs(nx) < 0.1f && std::abs(ny) < 0.1f) {
      float plane_size = 0.0f;
      for (const auto& idx : inliers->indices) {
        plane_size += cloud_filtered->points[idx].x;
      }
      // Remove large planes
      if (plane_size > 180.0f) {
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_filtered);
      }
    }

    iterations++;
  }

  // 3. Noise Removal
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(cloud_filtered);
  sor.setMeanK(25);
  sor.setStddevMulThresh(0.018);
  sor.filter(*cloud_filtered);

  pcl::RadiusOutlierRemoval<PointT> ror;
  ror.setInputCloud(cloud_filtered);
  ror.setRadiusSearch(0.018);
  ror.setMinNeighborsInRadius(4);
  ror.filter(*cloud_filtered);

  // 4. Multi-Stage Clustering
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(cloud_filtered);

  // Compute normals for region growing
  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  ne.setInputCloud(cloud_filtered);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(0.07f);
  ne.compute(*normals);

  // Prepare for region growing
  std::vector<bool> processed(cloud_filtered->points.size(), false);
  PointCloudT::Ptr result_cloud(new PointCloudT);
  std::vector<PointCloudT::Ptr> initial_clusters;

  // Multi-scale region growing for each skin point
  for (const auto& skin_point : skinPoints) {
    PointCloudT::Ptr current_cluster(new PointCloudT);
    std::queue<size_t> seeds;
    std::vector<float> search_radii = {0.05f, 0.1f, 0.15f};

    for (float radius : search_radii) {
      PointT searchPoint;
      searchPoint.x = skin_point.x;
      searchPoint.y = skin_point.y;
      searchPoint.z = skin_point.z;

      std::vector<int> neighborIndices;
      std::vector<float> neighborDistances;

      // Search for points near skin point
      if (tree->radiusSearch(searchPoint, radius, neighborIndices,
                             neighborDistances) > 0) {
        // Sort and select initial seeds
        std::vector<std::pair<float, int>> sortedNeighbors;
        for (size_t i = 0; i < neighborIndices.size(); ++i) {
          sortedNeighbors.push_back({neighborDistances[i], neighborIndices[i]});
        }
        std::sort(sortedNeighbors.begin(), sortedNeighbors.end());

        // Select initial seeds
        const int maxInitialSeeds = 40;
        for (int i = 0;
             i < std::min(maxInitialSeeds, (int)sortedNeighbors.size()); ++i) {
          int idx = sortedNeighbors[i].second;
          if (!processed[idx]) {
            seeds.push(idx);
            processed[idx] = true;
            current_cluster->points.push_back(cloud_filtered->points[idx]);
          }
        }
      }
    }

    // Region growing for current cluster
    const float growthRadius = 0.07f;
    const float normalThreshold = 0.4f;
    const float maxDistance = 0.4f;
    const int minClusterSize = 40;

    while (!seeds.empty()) {
      size_t currentIdx = seeds.front();
      seeds.pop();

      const auto& seedPoint = cloud_filtered->points[currentIdx];

      std::vector<int> neighborIndices;
      std::vector<float> neighborDistances;
      tree->radiusSearch(cloud_filtered->points[currentIdx], growthRadius,
                         neighborIndices, neighborDistances);

      for (size_t i = 0; i < neighborIndices.size(); ++i) {
        int neighborIdx = neighborIndices[i];
        if (processed[neighborIdx]) continue;

        const auto& currentPoint = cloud_filtered->points[currentIdx];
        const auto& neighborPoint = cloud_filtered->points[neighborIdx];

        // Distance check
        float distToSeed =
            std::sqrt(std::pow(neighborPoint.x - seedPoint.x, 2) +
                      std::pow(neighborPoint.y - seedPoint.y, 2) +
                      std::pow(neighborPoint.z - seedPoint.z, 2));
        if (distToSeed > maxDistance) continue;

        // Color similarity check
        float currentIntensity =
            (currentPoint.r + currentPoint.g + currentPoint.b) / 3.0f;
        float neighborIntensity =
            (neighborPoint.r + neighborPoint.g + neighborPoint.b) / 3.0f;

        bool colorSimilar = true;
        if (currentIntensity > 0 && neighborIntensity > 0) {
          float rRatio = (currentPoint.r + 1.0f) / (neighborPoint.r + 1.0f);
          float gRatio = (currentPoint.g + 1.0f) / (neighborPoint.g + 1.0f);
          float bRatio = (currentPoint.b + 1.0f) / (neighborPoint.b + 1.0f);

          const float ratioThreshold = 1.4f;
          colorSimilar =
              (rRatio < ratioThreshold && rRatio > 1.0f / ratioThreshold) &&
              (gRatio < ratioThreshold && gRatio > 1.0f / ratioThreshold) &&
              (bRatio < ratioThreshold && bRatio > 1.0f / ratioThreshold);
        }

        // Normal coherence check
        float normalDot = std::abs(normals->points[currentIdx].normal_x *
                                       normals->points[neighborIdx].normal_x +
                                   normals->points[currentIdx].normal_y *
                                       normals->points[neighborIdx].normal_y +
                                   normals->points[currentIdx].normal_z *
                                       normals->points[neighborIdx].normal_z);

        if (colorSimilar && normalDot > normalThreshold) {
          seeds.push(neighborIdx);
          processed[neighborIdx] = true;
          current_cluster->points.push_back(
              cloud_filtered->points[neighborIdx]);
        }
      }
    }

    // Add substantive clusters
    if (current_cluster->points.size() > minClusterSize) {
      initial_clusters.push_back(current_cluster);
    }
  }

  std::vector<PointCloudT::Ptr> final_clusters;

  // Store all clusters above a minimum size threshold
  const int minValidClusterSize = 150;
  for (const auto& cluster : initial_clusters) {
    if (cluster->points.size() >= minValidClusterSize) {
      final_clusters.push_back(cluster);
    }
  }

  // If no valid clusters, fallback to using the original filtered cloud
  if (final_clusters.empty()) {
    final_clusters.push_back(cloud_filtered);
  }

  PointCloudT::Ptr final_result_cloud(new PointCloudT);

  for (auto& person_cluster : final_clusters) {
    PointCloudT::Ptr recovered_cloud(new PointCloudT);

    recoverSurroundingPoints(
        original_cloud,   // Original cloud
        person_cluster,   // Each detected cluster separately
        recovered_cloud,  // Output for each person
        0.05f             // Adjust recovery radius
    );

    // Create a KD-Tree for segmentation
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(recovered_cloud);

    //// Compute centroid of the cluster
    // Eigen::Vector4f centroid;
    // pcl::compute3DCentroid(*recovered_cloud, centroid);
    // PointT cluster_center;
    // cluster_center.x = centroid[0];
    // cluster_center.y = centroid[1];
    // cluster_center.z = centroid[2];

    // Apply body part segmentation
    // body_part_segmentation(recovered_cloud);

    // Merge recovered cloud into the final result
    *final_result_cloud += *recovered_cloud;
  }

  // Update the final output cloud
  *cloud_filtered = *final_result_cloud;
}

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

void detectAndShowPlanes(const cv::Mat& rgb, const cv::Mat& depth) {
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
  cv::Mat planeMask = cv::Mat::zeros(depth.size(), CV_8UC1);
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

// Process cloud and display OpenCV output
void process_cloudOCV(PointCloudT::Ptr& cloud, PointCloudT::Ptr& cloud_filtered,
                      CameraIntrinsics& intrinsics) {
  if (cloud->empty()) return;

  // Convert to OpenCV
  cv::Mat rgb, depth;
  convertPCLtoOpenCV(cloud, rgb, depth);

  // Detect and visualize planes using only OpenCV
  detectAndShowPlanes(rgb, depth);

  // Detect all skin regions and get corresponding PCL points
  std::vector<pcl::PointXYZ> skinPoints =
      detectSkinAndConvertToPCL(rgb, depth, cloud);

  // Debug: Print detected points
  // for (const auto& skin : skinPoints) {
  //   std::cout << "Detected Skin Position (PCL): " << skin.x << ", " << skin.y
  //   << ", " << skin.z << std::endl;
  //}

  // Display results
  displayOpenCVResults(rgb, depth);

  // Copy processed cloud
  *cloud_filtered = *cloud;

  // Further processing (if needed)
  process_cloud(cloud, cloud_filtered, skinPoints);
}

// Function to retrieve camera intrinsics for the Kinect V2 depth camera
bool GetKinectV2DepthCameraIntrinsics(CameraIntrinsics& intrinsics) {
  // Initialize Kinect sensor
  IKinectSensor* pSensor = nullptr;
  HRESULT hr = GetDefaultKinectSensor(&pSensor);
  if (FAILED(hr) || !pSensor) {
    std::cerr << "Failed to get Kinect sensor!" << std::endl;
    return false;
  }

  // Get the coordinate mapper
  ICoordinateMapper* pCoordinateMapper = nullptr;
  hr = pSensor->get_CoordinateMapper(&pCoordinateMapper);

  // Retrieve camera intrinsics
  hr = pCoordinateMapper->GetDepthCameraIntrinsics(&intrinsics);

  return true;
}

void live_stream() {
  PointCloudT::Ptr cloud(new PointCloudT);
  PointCloudT::Ptr cloud_filtered(new PointCloudT);  // For downsampled cloud
  bool new_cloud_available_flag = false;
  bool viewer_running = true;

  std::thread visualizer_thread(run_visualizer, cloud_filtered, &viewer_running,
                                &cloud_mutex);

  std::thread visualizer_thread2(run_visualizer, cloud, &viewer_running,
                                 &cloud_mutex);

  // Kinect2Grabber Stream
  boost::shared_ptr<pcl::Grabber> grabber =
      boost::make_shared<pcl::Kinect2Grabber>();

  std::function<void(const PointCloudT::ConstPtr&)> f =
      [&cloud,
       &new_cloud_available_flag](const PointCloudT::ConstPtr& input_cloud) {
        std::lock_guard<std::mutex> lock(cloud_mutex);
        *cloud = *input_cloud;
        new_cloud_available_flag = true;
      };

  boost::signals2::connection connection = grabber->registerCallback(f);

  grabber->start();

  while (viewer_running) {
    if (new_cloud_available_flag) {
      std::lock_guard<std::mutex> lock(cloud_mutex);

      CameraIntrinsics intrinsics;

      if (GetKinectV2DepthCameraIntrinsics(intrinsics)) {
      } else {
        std::cerr << "Failed to retrieve camera intrinsics!" << std::endl;
      }

      // process_cloud(cloud, cloud_filtered);
      process_cloudOCV(cloud, cloud_filtered, intrinsics);

      new_cloud_available_flag = false;  // Reset flag after processing
    }

    std::this_thread::sleep_for(10ms);
  }

  grabber->stop();
  visualizer_thread.join();
  visualizer_thread2.join();
}

void offline_view() {
  PointCloudT::Ptr cloud(new PointCloudT);
  PointCloudT::Ptr cloud_filtered(new PointCloudT);  // For downsampled cloud

  std::string file;
  std::cout << "Enter the path to the PCD file: ";
  std::cin >> file;

  if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1) {
    PCL_ERROR("Couldn't read file\n");
    return;
  }

  bool viewer_running = true;

  std::thread visualizer_thread(run_visualizer, cloud, &viewer_running,
                                &cloud_mutex);

  while (viewer_running) {
    // process_cloud(cloud, cloud_filtered);
    std::this_thread::sleep_for(10ms);
  }

  visualizer_thread.join();
}

int main() {
  while (true) {
    std::cout << "Choose mode: (1) Live (2) Offline (3) Exit: ";
    int choice;
    std::cin >> choice;

    if (choice == 1) {
      live_stream();
    } else if (choice == 2) {
      offline_view();
    } else if (choice == 3) {
      std::cout << "Exiting." << std::endl;
      break;
    } else {
      std::cout << "Invalid choice. Please try again." << std::endl;
    }
  }

  return 0;
}
