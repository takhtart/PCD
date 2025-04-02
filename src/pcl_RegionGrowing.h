#ifndef PCLREGIONGROWING
#define PCLREGIONGROWING

// Include Required Libraries
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>

#include "definitions.h"
#include "pcl_HelperFunctions.h"

std::tuple<std::vector<PointCloudT::Ptr>, PointCloudT::Ptr, PointCloudT::Ptr>
region_grow(const PointCloudT::Ptr& cloud,
            const PointCloudT::Ptr& cloud_filtered,
            const std::vector<pcl::PointXYZ>& skinPoints) {
  // Multi-Stage Clustering
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
  std::vector<bool> processed_neighbours(cloud_filtered->points.size(), false);
  PointCloudT::Ptr result_cloud(new PointCloudT);
  std::vector<PointCloudT::Ptr> initial_clusters;
  PointCloudT::Ptr centroid_cloud(new PointCloudT);

  // Use Centroid Approximate Skinpoints to find Actual Skinpoints
  for (const auto& skin_point : skinPoints) {
    PointCloudT::Ptr current_cluster(new PointCloudT);
    std::queue<size_t> seeds;
    std::vector<float> search_radii = {0.01f, 0.015f, 0.01f};

    std::vector<int> neighborIndices;
    std::vector<float> neighborDistances;

    for (float radius : search_radii) {
      PointT searchPoint;
      searchPoint.x = skin_point.x;
      searchPoint.y = skin_point.y;
      searchPoint.z = skin_point.z;

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
    const float initialGrowthRadius = 0.09f;
    const float initialNormalThreshold = 0.2f;
    const float initialMaxDistance = 0.5f;
    const int minClusterSize = 35;
    const float clusterTolerance = 0.1f;
    const float directionBlendFactor = 0.7f;
    const float horizontalOffsetFactor = 0.3f;

    std::vector<Eigen::Vector3f> growthDirections;
    std::vector<Eigen::Vector3f> neighborhoodCentroids;
    std::vector<Eigen::Vector3f> seedPositions;

    while (!seeds.empty()) {
      size_t currentIdx = seeds.front();
      seeds.pop();
      const auto& seedPoint = cloud_filtered->points[currentIdx];
      std::vector<int> neighborIndices;
      std::vector<float> neighborDistances;
      float growthRadius = initialGrowthRadius;
      tree->radiusSearch(cloud_filtered->points[currentIdx], growthRadius,
                         neighborIndices, neighborDistances);

      for (size_t i = 0; i < neighborIndices.size(); ++i) {
        int neighborIdx = neighborIndices[i];
        if (processed_neighbours[neighborIdx]) continue;
        const auto& currentPoint = cloud_filtered->points[currentIdx];
        const auto& neighborPoint = cloud_filtered->points[neighborIdx];

        // Distance check
        float distToSeed =
            std::sqrt(std::pow(neighborPoint.x - seedPoint.x, 2) +
                      std::pow(neighborPoint.y - seedPoint.y, 2) +
                      std::pow(neighborPoint.z - seedPoint.z, 2));
        if (distToSeed > initialMaxDistance) continue;

        // Normal coherence check
        float normalDot = std::abs(normals->points[currentIdx].normal_x *
                                       normals->points[neighborIdx].normal_x +
                                   normals->points[currentIdx].normal_y *
                                       normals->points[neighborIdx].normal_y +
                                   normals->points[currentIdx].normal_z *
                                       normals->points[neighborIdx].normal_z);
        if (normalDot > initialNormalThreshold) {
          seeds.push(neighborIdx);
          processed_neighbours[neighborIdx] = true;
          current_cluster->points.push_back(
              cloud_filtered->points[neighborIdx]);

          // Track growth direction using neighborhood centroid
          Eigen::Vector3f neighborCentroid(0, 0, 0);
          for (const auto& idx : neighborIndices) {
            const auto& pt = cloud_filtered->points[idx];
            neighborCentroid += Eigen::Vector3f(pt.x, pt.y, pt.z);
          }
          neighborCentroid /= neighborIndices.size();
          Eigen::Vector3f direction =
              neighborCentroid -
              Eigen::Vector3f(seedPoint.x, seedPoint.y, seedPoint.z);
          growthDirections.push_back(direction);
          neighborhoodCentroids.push_back(neighborCentroid);
          seedPositions.emplace_back(seedPoint.x, seedPoint.y, seedPoint.z);
        }
      }
    }

    if (current_cluster->points.size() > minClusterSize &&
        valid_cluster(current_cluster)) {
      initial_clusters.push_back(current_cluster);

      if (!growthDirections.empty()) {
        // Manual accumulation for average direction
        Eigen::Vector3f averageDirection = Eigen::Vector3f::Zero();
        for (const auto& dir : growthDirections) {
          averageDirection += dir;
        }
        if (!growthDirections.empty()) {
          averageDirection /= static_cast<float>(growthDirections.size());
        }

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*current_cluster, centroid);
        Eigen::Vector3f clusterCentroid = centroid.head<3>();

        // Process and cluster centroids
        std::vector<Eigen::Vector3f> simplifiedCentroids;
        if (!neighborhoodCentroids.empty()) {
          const float clusterTolerance = 0.1f;

          struct GridCell {
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            int count = 0;
          };

          std::unordered_map<std::string, GridCell> gridMap;

          // Original spatial hashing implementation
          for (const auto& centroid : neighborhoodCentroids) {
            const int xCell =
                static_cast<int>(std::floor(centroid.x() / clusterTolerance));
            const int yCell =
                static_cast<int>(std::floor(centroid.y() / clusterTolerance));
            const int zCell =
                static_cast<int>(std::floor(centroid.z() / clusterTolerance));

            const std::string cellKey = std::to_string(xCell) + "|" +
                                        std::to_string(yCell) + "|" +
                                        std::to_string(zCell);

            gridMap[cellKey].sum += centroid;
            gridMap[cellKey].count++;
          }

          // Generate simplified centroids using original method
          for (const auto& [key, cell] : gridMap) {
            const Eigen::Vector3f simplifiedCentroid = cell.sum / cell.count;

            simplifiedCentroids.push_back(simplifiedCentroid);

            // Original visualization code
            pcl::PointXYZRGBA p;
            p.x = simplifiedCentroid.x();
            p.y = simplifiedCentroid.y();
            p.z = simplifiedCentroid.z();
            p.r = 255;
            p.g = 0;
            p.b = 0;
            p.a = 255;
            centroid_cloud->points.push_back(p);
          }
        }

        // Calculate combined growth direction
        Eigen::Vector3f principalDirection =
            computePrincipalDirection(simplifiedCentroids);
        Eigen::Vector3f combinedDirection =
            principalDirection * directionBlendFactor +
            averageDirection.normalized() * (1.0f - directionBlendFactor);

        // Determine placement direction
        std::string direction;
        if (std::abs(combinedDirection.x()) > std::abs(combinedDirection.y())) {
          direction = combinedDirection.x() > 0 ? "right" : "left";
        } else {
          direction = combinedDirection.y() > 0 ? "up" : "down";
        }

        // Connect centroids with lines
        for (size_t i = 1; i < simplifiedCentroids.size(); ++i) {
          drawLine(result_cloud, simplifiedCentroids[i - 1],
                   simplifiedCentroids[i]);
        }

        pcl::PointXYZRGBA minPt, maxPt;
        pcl::getMinMax3D(*current_cluster, minPt, maxPt);

        // Call Region Estimation
        estimateRegion(cloud, result_cloud, clusterCentroid,
                       simplifiedCentroids, growthDirections, minPt, maxPt,
                       seedPositions, current_cluster);
      }
    }
  }

  return {initial_clusters, result_cloud, centroid_cloud};
}

#endif
