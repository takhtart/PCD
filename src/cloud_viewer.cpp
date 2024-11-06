#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "kinect2_grabber.h"
#include <mutex>
#include <thread>
#include <pcl/io/pcd_io.h>

using namespace std::chrono_literals;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;


std::mutex cloud_mutex;

void cloud_cb_(const PointCloudT::ConstPtr& callback_cloud, PointCloudT::Ptr& cloud, bool* new_cloud_available_flag)
{
    std::lock_guard<std::mutex> lock(cloud_mutex);
    *cloud = *callback_cloud;
    *new_cloud_available_flag = true;
}

void live_stream()
{   
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    PointCloudT::Ptr cloud(new PointCloudT);
    bool new_cloud_available_flag = false;
    

    // Kinect2Grabber
    boost::shared_ptr<pcl::Grabber> grabber = boost::make_shared<pcl::Kinect2Grabber>();

    std::function<void(const PointCloudT::ConstPtr&)> f =
        [&cloud, &new_cloud_available_flag](const PointCloudT::ConstPtr& input_cloud)
        {
            cloud_cb_(input_cloud, cloud, &new_cloud_available_flag);  // Pass by reference to cloud
        };

    // Register Callback Function with std::function
    boost::signals2::connection connection = grabber->registerCallback(f);

    // Start Grabber
    grabber->start();

    while (!new_cloud_available_flag)
        std::this_thread::sleep_for(1ms);
    new_cloud_available_flag = false;

    {
        std::lock_guard<std::mutex> lock(cloud_mutex);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
        viewer.addPointCloud<PointT>(cloud, rgb, "input_cloud");
        viewer.setCameraPosition(0, 0, -2, 0, 1, 0, 0);
    }

    while (!viewer.wasStopped())
    {
        if (new_cloud_available_flag && cloud_mutex.try_lock())
        {
            new_cloud_available_flag = false;
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
            viewer.updatePointCloud<PointT>(cloud, rgb, "input_cloud");
            cloud_mutex.unlock();
        }
        viewer.spinOnce();
    }

    grabber->stop();
}

void offline_view()
{
    PointCloudT::Ptr cloud(new PointCloudT);
    std::string file;
    std::cout << "Enter the path to the PCD file: ";
    std::cin >> file;

    if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file\n");
        return;
    }

    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
    viewer.addPointCloud<PointT>(cloud, rgb, "input_cloud");
    viewer.setCameraPosition(0, 0, -2, 0, -1, 0, 0);

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
}

int main()
{
    PointCloudT::Ptr cloud(new PointCloudT);
    bool new_cloud_available_flag = false;

    while (true)
    {
        std::cout << "Choose mode: (1) Live (2) Offline (3) Exit: ";
        int choice;
        std::cin >> choice;

        if (choice == 1)
        {
            live_stream();
        }
        else if (choice == 2)
        {
            offline_view();
        }
        else if (choice == 3)
        {
            std::cout << "Exiting." << std::endl;
            break;
        }
        else
        {
            std::cout << "Invalid choice. Please try again." << std::endl;
        }
    }

    return 0;
}