
// Include Libraries and Header Files
#include <gtest/gtest.h>
#define UNIT_TEST  // Define UNIT_TEST to exclude the main function
#include "main.cpp"

// Test for convertPCLtoOpenCV
TEST(ConversionsTest, ConvertPCLtoOpenCV) {
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGBA>);
  cloud->width = 2;
  cloud->height = 2;
  cloud->points.resize(cloud->width * cloud->height);
  for (size_t i = 0; i < cloud->points.size(); ++i) {
    cloud->points[i].r = 255;
    cloud->points[i].g = 0;
    cloud->points[i].b = 0;
    cloud->points[i].a = 255;
    cloud->points[i].z = static_cast<float>(i);
  }

  cv::Mat rgb, depth;
  convertPCLtoOpenCV(cloud, rgb, depth);

  ASSERT_EQ(rgb.rows, cloud->height);
  ASSERT_EQ(rgb.cols, cloud->width);
  ASSERT_EQ(depth.rows, cloud->height);
  ASSERT_EQ(depth.cols, cloud->width);

  ASSERT_EQ(rgb.at<cv::Vec3b>(0, 1), cv::Vec3b(0, 0, 255));
  ASSERT_EQ(depth.at<float>(0, 1), 0.0f);
}

// Test for valid_cluster
TEST(PCLHelperFunctionsTest, ValidClusterTest) {
  PointCloudT::Ptr cluster(new PointCloudT);
  cluster->width = 10;
  cluster->height = 1;
  cluster->points.resize(cluster->width * cluster->height);

  for (size_t i = 0; i < cluster->points.size(); ++i) {
    cluster->points[i].x = static_cast<float>(i) * 0.1f;
    cluster->points[i].y = 0.0f;
    cluster->points[i].z = 0.0f;
  }

  ASSERT_TRUE(valid_cluster(cluster));
}

TEST(PCLHelperFunctionsTest, InvalidClusterTest) {
  PointCloudT::Ptr cluster(new PointCloudT);
  cluster->width = 10;
  cluster->height = 1;
  cluster->points.resize(cluster->width * cluster->height);

  for (size_t i = 0; i < cluster->points.size(); ++i) {
    cluster->points[i].x = static_cast<float>(i) * 10.0f;
    cluster->points[i].y = 0.0f;
    cluster->points[i].z = 0.0f;
  }

  ASSERT_FALSE(valid_cluster(cluster));
}

// Test for voxel_downsample
TEST(PCLHelperFunctionsTest, VoxelDownsampleTest) {
  PointCloudT::Ptr cloud(new PointCloudT);
  PointCloudT::Ptr cloud_filtered(new PointCloudT);
  cloud->width = 100;
  cloud->height = 1;
  cloud->points.resize(cloud->width * cloud->height);

  for (size_t i = 0; i < cloud->points.size(); ++i) {
    cloud->points[i].x = static_cast<float>(i) * 0.01f;
    cloud->points[i].y = 0.0f;
    cloud->points[i].z = 0.0f;
  }

  voxel_downsample(cloud, cloud_filtered, 0.05f);

  ASSERT_LT(cloud_filtered->points.size(), cloud->points.size());
}

// Test for removeNoise
TEST(PCLHelperFunctionsTest, RemoveNoiseTest) {
  PointCloudT::Ptr cloud(new PointCloudT);
  cloud->width = 100;
  cloud->height = 1;
  cloud->points.resize(cloud->width * cloud->height);

  for (size_t i = 0; i < cloud->points.size(); ++i) {
    cloud->points[i].x = static_cast<float>(i) * 0.01f;
    cloud->points[i].y = 0.0f;
    cloud->points[i].z = 0.0f;
  }

  // Add some noise
  cloud->points[50].x = 100.0f;
  cloud->points[50].y = 100.0f;
  cloud->points[50].z = 100.0f;

  removeNoise(cloud);

  ASSERT_LT(cloud->points.size(), 100);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
