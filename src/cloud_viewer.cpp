// Point Types
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
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>

// Plane Segmentation
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

// Euclidean Cluster Extraction
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

// Bounding Box Calculations
#include <pcl/common/common.h>        
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>

// Grabber for KinectV2
#include "kinect2_grabber.h"


// Define the point cloud type
using namespace std::chrono_literals;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// Initialize cloud mutex
std::mutex cloud_mutex;


// Function to add bounding box outline lines to the cloud
void bounding(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr human_cloud, pcl::PointXYZRGBA min_pt, pcl::PointXYZRGBA max_pt, float thickness = 0.02f, float scale_factor = 0.95f) {
    // Define the eight corners of the bounding box
    float x_center = (min_pt.x + max_pt.x) / 2;
    float z_center = (min_pt.z + max_pt.z) / 2;
    min_pt.x = x_center + (min_pt.x - x_center) * scale_factor;
    max_pt.x = x_center + (max_pt.x - x_center) * scale_factor;
    min_pt.z = z_center + (min_pt.z - z_center) * scale_factor;
    max_pt.z = z_center + (max_pt.z - z_center) * scale_factor;

    std::vector<pcl::PointXYZRGBA> bounding_box_corners(8);
    bounding_box_corners[0] = pcl::PointXYZRGBA(min_pt.x, min_pt.y, min_pt.z);
    bounding_box_corners[1] = pcl::PointXYZRGBA(max_pt.x, min_pt.y, min_pt.z);
    bounding_box_corners[2] = pcl::PointXYZRGBA(max_pt.x, max_pt.y, min_pt.z);
    bounding_box_corners[3] = pcl::PointXYZRGBA(min_pt.x, max_pt.y, min_pt.z);
    bounding_box_corners[4] = pcl::PointXYZRGBA(min_pt.x, min_pt.y, max_pt.z);
    bounding_box_corners[5] = pcl::PointXYZRGBA(max_pt.x, min_pt.y, max_pt.z);
    bounding_box_corners[6] = pcl::PointXYZRGBA(max_pt.x, max_pt.y, max_pt.z);
    bounding_box_corners[7] = pcl::PointXYZRGBA(min_pt.x, max_pt.y, max_pt.z);

	// Set color to red
    for (auto& pt : bounding_box_corners) {
        pt.r = 255;
        pt.g = 0;
        pt.b = 0;
        pt.a = 255;
    }

	// Add intermediate points between two points to thicken bounding box lines
    auto addThickLine = [&human_cloud, thickness](pcl::PointXYZRGBA start, pcl::PointXYZRGBA end) {
        Eigen::Vector3f start_vec(start.x, start.y, start.z);
        Eigen::Vector3f end_vec(end.x, end.y, end.z);
        Eigen::Vector3f direction = (end_vec - start_vec).normalized();
        Eigen::Vector3f perpendicular = direction.cross(Eigen::Vector3f::UnitX()).normalized() * thickness;

        for (float t = 0.0; t <= 1.0; t += 0.01) {
            Eigen::Vector3f point_vec = start_vec + t * (end_vec - start_vec);
            pcl::PointXYZRGBA point;
            point.x = point_vec.x();
            point.y = point_vec.y();
            point.z = point_vec.z();
            point.r = 255;
            point.g = 0;
            point.b = 0;
            point.a = 255;

            // Add multiple points around the line to make it thicker
            for (float s = -1.0; s <= 1.0; s += 0.1) {
                pcl::PointXYZRGBA thick_point = point;
                Eigen::Vector3f offset = s * perpendicular;
                thick_point.x += offset.x();
                thick_point.y += offset.y();
                thick_point.z += offset.z();
                human_cloud->points.push_back(thick_point);
            }
        }
        };


	// Form Bounding Box
    addThickLine(bounding_box_corners[0], bounding_box_corners[1]); addThickLine(bounding_box_corners[1], bounding_box_corners[2]); addThickLine(bounding_box_corners[2], bounding_box_corners[3]); addThickLine(bounding_box_corners[3], bounding_box_corners[0]); // Bottom face
    addThickLine(bounding_box_corners[4], bounding_box_corners[5]); addThickLine(bounding_box_corners[5], bounding_box_corners[6]); addThickLine(bounding_box_corners[6], bounding_box_corners[7]); addThickLine(bounding_box_corners[7], bounding_box_corners[4]); // Top face
    addThickLine(bounding_box_corners[0], bounding_box_corners[4]); addThickLine(bounding_box_corners[1], bounding_box_corners[5]); addThickLine(bounding_box_corners[2], bounding_box_corners[6]); addThickLine(bounding_box_corners[3], bounding_box_corners[7]); // Vertical edges
}

// Visualizer thread function
void run_visualizer(PointCloudT::Ptr cloud, bool* viewer_running, std::mutex* cloud_mutex)
{
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);

    {
        std::lock_guard<std::mutex> lock(*cloud_mutex);
        viewer.addPointCloud<PointT>(cloud, rgb, "input_cloud");
        viewer.setCameraPosition(0, 0, -2, 0, 1, 0, 0);
    }

    while (!viewer.wasStopped() && *viewer_running)
    {
        {
            std::lock_guard<std::mutex> lock(*cloud_mutex);
            viewer.updatePointCloud<PointT>(cloud, rgb, "input_cloud");
        }
        viewer.spinOnce();
    }

    *viewer_running = false;
}

void process_cloud(PointCloudT::Ptr& cloud, PointCloudT::Ptr& cloud_filtered) {

    // 1. Voxel Downsampling
    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(0.015f, 0.015f, 0.015f);
    voxel_filter.filter(*cloud_filtered);


    // 2. Remove Planes 
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(200);
    seg.setDistanceThreshold(0.002);

    seg.setInputCloud(cloud_filtered);

    int iterations = 0;
    while (iterations < 5) {
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            std::cerr << "No more planes detected in the cloud.\n";
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
            std::cout << "Detected a vertical plane (likely a wall)\n";
            float plane_size = 0.0f;
            for (const auto& idx : inliers->indices) {
                plane_size += cloud_filtered->points[idx].x; 
            }
			// Remove large planes
			if (plane_size > 200.0f) { //size threshold
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
    sor.setMeanK(20); 
    sor.setStddevMulThresh(0.02); 
    sor.filter(*cloud_filtered);


    pcl::RadiusOutlierRemoval<PointT> ror;
    ror.setInputCloud(cloud_filtered);
    ror.setRadiusSearch(0.02); 
    ror.setMinNeighborsInRadius(5); 
    ror.filter(*cloud_filtered);

	// 4. Euclidean Cluster Extraction (Extract the people from the scene)
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.05); 
    ec.setMinClusterSize(2000);   
    ec.setMaxClusterSize(7500);    
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);

	// Identify person clusters
    PointCloudT::Ptr person_cloud(new PointCloudT);

    if (!cluster_indices.empty())
    {
        // Iterate over all clusters
        for (const auto& cluster : cluster_indices)
        {
            PointCloudT::Ptr cluster_cloud(new PointCloudT);

            for (const auto& idx : cluster.indices)
            {
                const auto& point = cloud_filtered->points[idx];

                cluster_cloud->points.push_back(point);

            }

            cluster_cloud->width = cluster_cloud->points.size();
            cluster_cloud->height = 1;
            cluster_cloud->is_dense = true;

            pcl::PointXYZRGBA min_pt, max_pt;

            *person_cloud += *cluster_cloud;

            pcl::getMinMax3D(*cluster_cloud, min_pt, max_pt);

            // Print bounding box coordinates
            std::cout << "Bounding box for cluster:" << std::endl;
            std::cout << "Min: (" << min_pt.x << ", " << min_pt.y << ", " << min_pt.z << ")" << std::endl;
            std::cout << "Max: (" << max_pt.x << ", " << max_pt.y << ", " << max_pt.z << ")" << std::endl;
            


            // Add bounding box outline points to the current cluster
            bounding(cloud, min_pt, max_pt);
        }

        // Update filtered cloud
        *cloud_filtered = *person_cloud;
    }


}



void live_stream()
{
    PointCloudT::Ptr cloud(new PointCloudT);
    // For downsampled cloud
    PointCloudT::Ptr cloud_filtered(new PointCloudT); 
    bool new_cloud_available_flag = false;
    bool viewer_running = true;

    std::thread visualizer_thread2(run_visualizer, cloud, &viewer_running, &cloud_mutex);

    std::thread visualizer_thread(run_visualizer, cloud_filtered, &viewer_running, &cloud_mutex);

	// Kinect2Grabber Stream
    boost::shared_ptr<pcl::Grabber> grabber = boost::make_shared<pcl::Kinect2Grabber>();

    std::function<void(const PointCloudT::ConstPtr&)> f =
        [&cloud, &new_cloud_available_flag](const PointCloudT::ConstPtr& input_cloud)
        {
            std::lock_guard<std::mutex> lock(cloud_mutex);
            *cloud = *input_cloud;
            new_cloud_available_flag = true;
        };

    boost::signals2::connection connection = grabber->registerCallback(f);

    grabber->start();

    while (viewer_running)
    {
        if (new_cloud_available_flag)
        {
            std::lock_guard<std::mutex> lock(cloud_mutex);

            process_cloud(cloud, cloud_filtered);

            // Reset flag after processing
            new_cloud_available_flag = false; 
        }

        std::this_thread::sleep_for(10ms);
    }

    grabber->stop();
    visualizer_thread.join();
}


void offline_view()
{
    PointCloudT::Ptr cloud(new PointCloudT);
    // For downsampled cloud
    PointCloudT::Ptr cloud_filtered(new PointCloudT); 

    std::string file;
    std::cout << "Enter the path to the PCD file: ";
    std::cin >> file;

    if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file\n");
        return;
    }

    bool viewer_running = true;

    std::thread visualizer_thread(run_visualizer, cloud, &viewer_running, &cloud_mutex);

    while (viewer_running)
    {
        process_cloud(cloud, cloud_filtered);
        std::this_thread::sleep_for(10ms);
    }

    visualizer_thread.join();
}

int main()
{
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