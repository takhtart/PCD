#ifndef VISUALIZATION
#define VISUALIZATION

// Include Required Libraries
#include <pcl/visualization/pcl_visualizer.h>

#include "definitions.h"

// PCL Threaded Visualizer Handler
void run_visualizer(PointCloudT::Ptr cloud, bool* viewer_running,
                    std::mutex* cloud_mutex, const std::string& window_title) {
  pcl::visualization::PCLVisualizer viewer(window_title);
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
    viewer.spinOnce(10);
  }

  *viewer_running = false;
}

#endif
