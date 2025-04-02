#ifndef PCLREGIONESTIMATION
#define PCLREGIONESTIMATION

// Include Required Libraries
#include <pcl/common/eigen.h>
#include <pcl/common/pca.h>

#include "definitions.h"
#include "pcl_HelperFunctions.h"

// Add these helper functions
struct LineFitResult {
  float slope;
  float intercept;
  float rSquared;
};

// Modified line fitting calculation with relaxed criteria
LineFitResult calculateLineOfBestFit(
    const std::vector<Eigen::Vector3f>& points) {
  LineFitResult result{0, 0, 0};
  const size_t n = points.size();
  if (n < 2) return result;

  // Calculate using robust coordinates (x,z for depth alignment)
  float sumX = 0, sumZ = 0, sumXZ = 0, sumX2 = 0;
  for (const auto& p : points) {
    sumX += p.x();
    sumZ += p.z();  // Using depth (z) instead of height (y)
    sumXZ += p.x() * p.z();
    sumX2 += p.x() * p.x();
  }

  // Calculate slope (x-z plane) and intercept
  result.slope = (n * sumXZ - sumX * sumZ) / (n * sumX2 - sumX * sumX);
  result.intercept = (sumZ - result.slope * sumX) / n;

  // Calculate modified R-squared with noise tolerance
  float ssTot = 0, ssRes = 0;
  const float meanZ = sumZ / n;
  for (const auto& p : points) {
    const float z = p.z();
    const float f = result.slope * p.x() + result.intercept;
    ssTot += (z - meanZ) * (z - meanZ);
    ssRes += (z - f) * (z - f);
  }

  if (ssTot > 0) {
    result.rSquared = 1.0f - (ssRes / ssTot);
    // Apply leniency for small sample sizes
    if (n < 5) result.rSquared *= 1.2f;
  }

  return result;
}

Eigen::Vector3f computePrincipalDirection(
    const std::vector<Eigen::Vector3f>& points) {
  if (points.empty()) return Eigen::Vector3f::Zero();

  // Manual accumulation for Eigen vectors
  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  for (const auto& p : points) {
    centroid += p;
  }
  centroid /= static_cast<float>(points.size());

  // Compute covariance matrix
  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
  for (const auto& p : points) {
    Eigen::Vector3f adjusted = p - centroid;
    covariance += adjusted * adjusted.transpose();
  }

  // Perform PCA
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(covariance);
  return eigenSolver.eigenvectors().col(2).normalized();
}

void estimateRegion(const PointCloudT::Ptr& cloud,
                    const PointCloudT::Ptr& result_cloud,
                    const Eigen::Vector3f& clusterCentroid,
                    const std::vector<Eigen::Vector3f>& centroids,
                    const std::vector<Eigen::Vector3f>& growthDirections,
                    const pcl::PointXYZRGBA& minPt,
                    const pcl::PointXYZRGBA& maxPt,
                    const std::vector<Eigen::Vector3f>& seedPositions,
                    PointCloudT::Ptr cluster) {
  const float horizontalOffsetFactor = 0.3f;
  const float rsqThreshold = 0.9f;
  const float horizontalAngleThreshold = 75.0f;  // Degrees from x-axis
  const float verticalAngleThreshold = 30.0f;
  const float growthDirectionThreshold = 0.2f;

  const float minBoxWidth = 0.5f;
  float clusterWidth = maxPt.x - minPt.x;
  float boxWidth = std::max(minBoxWidth, clusterWidth);

  debug_strings.push_back("Region Estimation Results:");
  debug_strings.push_back("Cluster width: " + std::to_string(clusterWidth));

  // 1. Calculate line of best fit
  LineFitResult fit = calculateLineOfBestFit(centroids);
  bool validLine = !centroids.empty() && (centroids.size() >= 1) &&
                   (fit.rSquared >= rsqThreshold);

  if (!validLine) {
    drawVerticalOrangeBox(result_cloud, clusterCentroid, boxWidth, 1.83f, 0.1f);
    drawVerticalOrangeBox(cloud, clusterCentroid, boxWidth, 1.83f, 0.1f);
    debug_strings.push_back("Central placement: Scattered centroids (R^2= " +
                            std::to_string(fit.rSquared));
    return;
  }

  // Calculate raw angle in degrees (-90° to 90° range)
  float rawAngle = std::atan(fit.slope) * (180.0f / M_PI);

  // Convert to 0-360° range
  float fullAngle = rawAngle < 0 ? 180.0f + rawAngle : rawAngle;

  // Determine horizontal status using cosine similarity
  float angleFromHorizontal = std::min(fullAngle, 180.0f - fullAngle);
  bool isHorizontal = angleFromHorizontal <= horizontalAngleThreshold;

  // 3. Calculate growth direction components
  Eigen::Vector3f avgGrowthDir = Eigen::Vector3f::Zero();
  for (const auto& dir : growthDirections) {
    avgGrowthDir += dir;
  }
  if (!growthDirections.empty()) {
    avgGrowthDir.normalize();
  }

  // 4. Determine placement
  const float xSpread = maxPt.x - minPt.x;
  Eigen::Vector3f boxCenter = clusterCentroid;
  std::string placementReason;

  if (isHorizontal) {
    // 1. Furthest seed-centroid pair analysis (2D distance)
    float maxDistanceSq = 0.0f;
    float decisiveDeltaX = 0.0f;
    bool rightFromFurthest = false;
    Eigen::Vector3f furthestSeed, furthestCentroid;

    for (const auto& seed : seedPositions) {
      for (const auto& centroid : centroids) {
        const float dx = centroid.x() - seed.x();
        const float dy = centroid.y() - seed.y();
        const float distSq = dx * dx + dy * dy;

        if (distSq > maxDistanceSq) {
          maxDistanceSq = distSq;
          decisiveDeltaX = dx;
          furthestSeed = seed;
          furthestCentroid = centroid;

          // Determine relative position
          rightFromFurthest = (centroid.x() > seed.x());
        }
      }
    }

    // 2. Average growth direction analysis
    Eigen::Vector3f avgGrowthDir = Eigen::Vector3f::Zero();
    for (const auto& dir : growthDirections) avgGrowthDir += dir;
    if (!growthDirections.empty()) avgGrowthDir.normalize();
    const bool rightFromAvg = avgGrowthDir.x() > growthDirectionThreshold;

    // 3. Line slope analysis (only if valid)
    bool rightFromSlope = false;
    if (validLine) {
      rightFromSlope = fit.slope > 0;
    }

    // Weighted decision matrix (sums to 1.0)
    constexpr float furthestWeight =
        0.4f;  // Most reliable - direct spatial observation
    constexpr float avgWeight = 0.4f;    // Reliable - actual growth vectors
    constexpr float slopeWeight = 0.1f;  // Contextual - depends on line quality

    float totalScore = 0.0f;
    totalScore += rightFromFurthest ? furthestWeight : -furthestWeight;
    totalScore += rightFromAvg ? avgWeight : -avgWeight;
    if (validLine) {
      totalScore += rightFromSlope ? slopeWeight : -slopeWeight;
    }

    // Final determination
    const bool finalRight = totalScore > 0.0f;

    // Apply offset with combined logic
    const float xSpread = maxPt.x - minPt.x;
    boxCenter.x() = finalRight ? maxPt.x + xSpread * horizontalOffsetFactor
                               : minPt.x - xSpread * horizontalOffsetFactor;

    // Diagnostic reporting
    std::stringstream reason;
    reason << "Combined decision - ";
    reason << "Furthest: " << (rightFromFurthest ? "R" : "L")
           << " (del_X=" << decisiveDeltaX << ") ";
    reason << "AvgGrowth: " << (rightFromAvg ? "R" : "L")
           << " (X=" << avgGrowthDir.x() << ") ";
    reason << "| Score: " << totalScore;
    reason << " (R^2 = " << fit.rSquared << ")";
    placementReason = reason.str();

    // Visual debugging
    drawLine(result_cloud, furthestSeed, furthestCentroid, 255, 0,
             255);  // Purple for furthest pair
  } else {
    // Vertical line - offset based on growth direction
    if (std::abs(avgGrowthDir.x()) > growthDirectionThreshold) {
      const bool rightOffset = avgGrowthDir.x() > 0;
      boxCenter.x() = clusterCentroid.x();

      placementReason = "Vertical line: Centrally Placed";
    } else {
      drawVerticalOrangeBox(result_cloud, clusterCentroid, boxWidth, 1.83f,
                            0.1f);
      drawVerticalOrangeBox(cloud, clusterCentroid, boxWidth, 1.83f, 0.1f);
      placementReason = "Central placement: Ambiguous vertical growth";
      return;
    }
  }

  drawVerticalOrangeBox(result_cloud, boxCenter, boxWidth, 1.83f, 0.1f);
  drawVerticalOrangeBox(cloud, boxCenter, boxWidth, 1.83f, 0.1f);
  debug_strings.push_back("Offset placement: " + placementReason);
}

#endif
