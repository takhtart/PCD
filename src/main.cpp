// Include Libraries
#include <conio.h>
#include <pcl/io/pcd_io.h>

#include <filesystem>
#include <iostream>
#include <mutex>
#include <thread>

// Import Modules
#include "conversions.h"
#include "definitions.h"
#include "kinect2_grabber.h"
#include "ocv_PlaneDetect.h"
#include "ocv_SkinDetect.h"
#include "pcl_HelperFunctions.h"
#include "pcl_RegionEstimation.h"
#include "pcl_RegionGrowing.h"
#include "visualization.h"

// OpenCV Detected Planes Mask
cv::Mat planeMask;

// Initialize cloud mutex
std::mutex cloud_mutex;

void process_cloud(PointCloudT::Ptr& cloud, PointCloudT::Ptr& cloud_filtered,
                   std::vector<pcl::PointXYZ> skinPoints) {
  // Store original cloud before filtering
  PointCloudT::Ptr original_cloud(new PointCloudT(*cloud));
  PointCloudT::Ptr blank_cloud(new PointCloudT);

  // 1. Voxel Downsampling with larger leaf size for performance
  voxel_downsample(cloud, cloud_filtered, 0.018f);

  // 2. Remove Planes
  remove_planes(cloud_filtered);

  // 3. Noise Removal
  removeNoise(cloud_filtered);

  // 4. Region Growing
  auto [initial_clusters, result_cloud, centroid_cloud] =
      region_grow(cloud, cloud_filtered, skinPoints);

  // If no valid clusters, fallback to using the original filtered cloud
  if (!initial_clusters.empty()) {
    PointCloudT::Ptr final_result_cloud(new PointCloudT);
    for (auto& person_cluster : initial_clusters) {
      PointCloudT::Ptr recovered_cloud(new PointCloudT);
      recoverSurroundingPoints(
          original_cloud,   // Original cloud
          person_cluster,   // Each detected cluster separately
          recovered_cloud,  // Output for each person
          0.03f             // Adjust recovery radius
      );

      // Create a KD-Tree for segmentation
      pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
      tree->setInputCloud(recovered_cloud);

      // Merge recovered cloud into the final result
      *final_result_cloud += *person_cluster;
    }

    for (size_t i = 0; i < final_result_cloud->points.size(); ++i) {
      auto& point = final_result_cloud->points[i];
      point.r = 0;
      point.g = 255;
      point.b = 0;
    }

    // Update the final output cloud
    // removeNoise(final_result_cloud);

    // Add the cutoff points after coloring the clusters green
    for (const auto& point : result_cloud->points) {
      final_result_cloud->points.push_back(point);
    }
    for (const auto& point : centroid_cloud->points) {
      final_result_cloud->points.push_back(point);
    }
    *cloud_filtered = *final_result_cloud;

  } else {
    *cloud_filtered = *blank_cloud;
  }
}

void process_cloudOCV(PointCloudT::Ptr& cloud,
                      PointCloudT::Ptr& cloud_filtered) {
  PointCloudT::Ptr blank_cloud(new PointCloudT);

  if (cloud->empty()) return;

  // Convert to OpenCV
  cv::Mat rgb, depth, planeMask;
  convertPCLtoOpenCV(cloud, rgb, depth);

  // Detect and visualize planes using only OpenCV
  detectAndShowPlanes(rgb, depth, planeMask);

  // Detect all skin regions and get corresponding PCL points
  std::vector<pcl::PointXYZ> skinPoints =
      detectSkinAndConvertToPCL(rgb, depth, cloud, planeMask);

  // Display results
  imshow("OpenCV Skin Detection", rgb);

  if (!skinPoints.empty()) {
    // Copy processed cloud
    *cloud_filtered = *cloud;

    // Process Output
    process_cloud(cloud, cloud_filtered, skinPoints);
  } else {
    *cloud_filtered = *blank_cloud;
  }
}

// Function to get current date and time
std::string get_current_datetime() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
  ss << '_' << std::setw(3) << std::setfill('0') << ms.count();
  return ss.str();
}

void live_stream() {
  PointCloudT::Ptr cloud(new PointCloudT);
  PointCloudT::Ptr cloud_filtered(new PointCloudT);
  bool new_cloud_available_flag = false;
  bool viewer_running = true;

  std::thread visualizer_thread(run_visualizer, cloud_filtered, &viewer_running,
                                &cloud_mutex, "Human Clusters");

  std::thread visualizer_thread2(run_visualizer, cloud, &viewer_running,
                                 &cloud_mutex, "Original View");

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
    if (_kbhit()) {  // Check if a key is pressed
      char key = _getch();
      if (key == 's') {
        std::lock_guard<std::mutex> lock(cloud_mutex);
        std::filesystem::create_directories("saved_pcd_files");
        std::string filename =
            "saved_pcd_files/" + get_current_datetime() + ".pcd";
        pcl::io::savePCDFileASCII(filename, *cloud);
        std::cout << "Saved " << filename << std::endl;
      } else if (key == 'd') {
        debug = !debug;
        std::cout << "Debug mode: " << (debug ? "ON" : "OFF") << std::endl;
      }
    }

    if (new_cloud_available_flag) {
      std::lock_guard<std::mutex> lock(cloud_mutex);

      process_cloudOCV(cloud, cloud_filtered);

      if (debug && !debug_strings.empty()) {
        std::cout << "==========================================" << std::endl;
        for (const auto& debug_string : debug_strings) {
          std::cout << debug_string << std::endl;
        }

        std::cout << "==========================================" << std::endl;
      }
      debug_strings.clear();

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
  PointCloudT::Ptr cloud_filtered(new PointCloudT);

  std::string file;
  std::cout
      << "Enter the full path to the PCD file (eg. C:\\PCD Files\\file1.pcd): ";
  std::getline(std::cin, file);

  if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1) {
    PCL_ERROR("Couldn't read file\n");
    return;
  }

  bool viewer_running = true;
  bool new_cloud_available_flag = true;

  std::thread visualizer_thread(run_visualizer, cloud_filtered, &viewer_running,
                                &cloud_mutex, "Human Clusters");

  std::thread visualizer_thread2(run_visualizer, cloud, &viewer_running,
                                 &cloud_mutex, "Original View");

  while (viewer_running) {
    if (new_cloud_available_flag) {
      std::lock_guard<std::mutex> lock(cloud_mutex);

      process_cloudOCV(cloud, cloud_filtered);

      std::cout << "==========================================" << std::endl;

      for (const auto& debug_string : debug_strings) {
        std::cout << debug_string << std::endl;
      }
      debug_strings.clear();

      std::cout << "==========================================" << std::endl;

      new_cloud_available_flag = false;
    }
    waitKey(1);
    std::this_thread::sleep_for(10ms);
  }

  visualizer_thread.join();
  visualizer_thread2.join();
}

#ifndef UNIT_TEST
int main() {
  while (true) {
    debug = false;
    std::cout << "Choose mode: (1) Live (2) Offline (3) Exit: ";
    std::string input;
    std::getline(std::cin, input);

    // Remove spaces from the input string
    input.erase(std::remove(input.begin(), input.end(), ' '), input.end());

    if (input == "1") {
      std::cout << "Press 's' to save the current frame." << std::endl;
      std::cout << "Press 'd' to toggle debug mode." << std::endl;
      live_stream();
    } else if (input == "2") {
      offline_view();
    } else if (input == "3") {
      std::cout << "Exiting." << std::endl;
      break;
    } else {
      std::cout << "Invalid choice. Please try again." << std::endl;
    }
  }

  return 0;
}
#endif
