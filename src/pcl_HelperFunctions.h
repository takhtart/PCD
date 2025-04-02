#ifndef PCLHELPERFUNCTIONS
#define PCLHELPERFUNCTIONS

// Include Libraries
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "definitions.h"

// Function to draw a 3D line in the point cloud
void drawLine(PointCloudT::Ptr cloud, const Eigen::Vector3f& start,
              const Eigen::Vector3f& end, int steps = 10, uint8_t r = 0,
              uint8_t g = 0, uint8_t b = 255) {
  Eigen::Vector3f step = (end - start) / steps;
  for (int i = 0; i <= steps; ++i) {
    Eigen::Vector3f point = start + step * i;
    pcl::PointXYZRGBA p;
    p.x = point.x();
    p.y = point.y();
    p.z = point.z();
    p.r = r;
    p.g = g;
    p.b = b;
    p.a = 255;
    cloud->points.push_back(p);
  }
}

// Function to draw a 3D orange box for a standing person
void drawVerticalOrangeBox(PointCloudT::Ptr cloud,
                           const Eigen::Vector3f& center, float width,
                           float height, float depth) {
  // Calculate the half dimensions
  float half_width = width / 2.0f;
  float half_height = height / 2.0f;
  float half_depth = depth / 2.0f;

  // Define the 8 corners of the box
  std::vector<Eigen::Vector3f> corners = {
      {center.x() - half_width, center.y() - half_height,
       center.z() - half_depth},
      {center.x() + half_width, center.y() - half_height,
       center.z() - half_depth},
      {center.x() + half_width, center.y() + half_height,
       center.z() - half_depth},
      {center.x() - half_width, center.y() + half_height,
       center.z() - half_depth},
      {center.x() - half_width, center.y() - half_height,
       center.z() + half_depth},
      {center.x() + half_width, center.y() - half_height,
       center.z() + half_depth},
      {center.x() + half_width, center.y() + half_height,
       center.z() + half_depth},
      {center.x() - half_width, center.y() + half_height,
       center.z() + half_depth}};

  // Define the 12 edges of the box (vertex pairs)
  const std::vector<std::pair<int, int>> edges = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face
      {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face
      {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
  };

  // Draw each edge with points
  const float step = 0.01f;  // 1cm resolution for box edges
  for (const auto& edge : edges) {
    const auto& p1 = corners[edge.first];
    const auto& p2 = corners[edge.second];

    // Calculate direction vector and distance between points
    Eigen::Vector3f dir = p2 - p1;
    float length = dir.norm();
    dir.normalize();

    // Add points along the edge
    for (float t = 0; t <= length; t += step) {
      PointT point;
      point.x = p1.x() + dir.x() * t;
      point.y = p1.y() + dir.y() * t;
      point.z = p1.z() + dir.z() * t;
      point.r = 255;  // Orange color
      point.g = 165;
      point.b = 0;
      point.a = 255;
      cloud->points.push_back(point);
    }
  }
}

bool valid_cluster(PointCloudT::Ptr& cluster) {
  // Store all clusters above minimum size and dimensional constraints
  const int minValidClusterSize = 35;
  const float maxDimension = 2.14f;  // 2.14 meters max height,width
  const float maxArea = 3.0f;        // 3.0 m^2 area

  // Calculate bounding box dimensions
  pcl::PointXYZRGBA minPt, maxPt;
  pcl::getMinMax3D(*cluster, minPt, maxPt);

  const float x_length = maxPt.x - minPt.x;
  const float y_width = maxPt.y - minPt.y;
  const float area = x_length * y_width;

  // Debug output
  debug_strings.push_back("Cluster analysis:");
  debug_strings.push_back("- Points: " + std::to_string(cluster->size()));
  debug_strings.push_back("- Dimensions: " + std::to_string(x_length) + "m x " +
                          std::to_string(y_width) + "m");
  debug_strings.push_back("- Area: " + std::to_string(area) + "m^2");

  // Combined validity checks
  if (x_length <= maxDimension && y_width <= maxDimension && area <= maxArea) {
    debug_strings.push_back("Cluster ACCEPTED");
    return true;
  } else {
    debug_strings.push_back("Cluster REJECTED - Exceeds size constraints");
    return false;
  }
}

// Reduce Number of Points to be processed
void voxel_downsample(PointCloudT::Ptr& cloud, PointCloudT::Ptr& cloud_filtered,
                      float leaf_size = 0.018f) {
  pcl::VoxelGrid<PointT> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel_filter.filter(*cloud_filtered);
}

// Remove Noise
void removeNoise(PointCloudT::Ptr& cloud) {
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(25);
  sor.setStddevMulThresh(0.018);
  sor.filter(*cloud);

  pcl::RadiusOutlierRemoval<PointT> ror;
  ror.setInputCloud(cloud);
  ror.setRadiusSearch(0.018);
  ror.setMinNeighborsInRadius(4);
  ror.filter(*cloud);
}

// RANSAC Based Plane Segmentation and Removal
void remove_planes(PointCloudT::Ptr& cloud_filtered) {
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
}

// Enhance Resolution of Extracted Human Clusters (Usage Dependant on Compute
// Power)
void recoverSurroundingPoints(PointCloudT::Ptr& original_cloud,
                              PointCloudT::Ptr& extracted_region,
                              PointCloudT::Ptr& result_cloud,
                              float surrounding_radius = 0.05f) {
  // 1. Voxel Downsampling with larger leaf size for performance
  voxel_downsample(original_cloud, original_cloud, 0.02f);

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

#endif
