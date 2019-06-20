#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Header.h>

#include <pcl/point_cloud.h> // pcl::PointCloud
#include <pcl/point_types.h> // pcl::PointXYZ
#include <pcl_conversions/pcl_conversions.h>  // pcl::fromROSMsg and pcl::toROSMsg
#include <pcl/filters/extract_indices.h>  // pcl::ExtractIndices
#include <pcl/io/io.h>                    // pcl::copyPointCloud

// To save time about all the namespace crap:
typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef PointCloud::Ptr PointCloudPtr;
typedef PointCloud::ConstPtr PointCloudConstPtr;

typedef pcl::PointXYZI PointI;
typedef pcl::PointCloud<PointI> PointICloud;
typedef PointICloud::Ptr PointICloudPtr;
typedef PointICloud::ConstPtr PointICloudConstPtr;


// ROS Subscriber
ros::Subscriber pointcloud_sub;
// ROS Publisher
ros::Publisher filtered_pub;
// frame id
std::string frame_id = "os1_lidar";

// This is the callback function: subscribe a topic, do whatever you
// want, then generate a new sensor_msgs/PointCloud2 message
void OnPointCloud(const sensor_msgs::PointCloud2ConstPtr& ros_pc2) {


    // Transfer from ROS sensor_msg to PCL cloud msg
	PointICloudPtr cloud(new PointICloud);
    pcl::fromROSMsg(*ros_pc2, *cloud);
    //ROS_INFO_STREAM(" Cloud inputs: #" << cloud->size() << " Points");


	// Read the header for later generating new pc back
    std_msgs::Header header = ros_pc2->header;
    header.frame_id = frame_id;
    header.stamp = ros::Time::now();


	// ------Do whatever you want for the PCL cloud:------
	PointICloudPtr cloud_filtered(new PointICloud);
	cloud_filtered->clear();

	const PointICloud& cloud_in = *cloud;

	pcl::PointIndices::Ptr filtered_indices(new pcl::PointIndices);
	filtered_indices->indices.clear();

	for (size_t pt = 0u; pt < cloud_in.points.size(); ++pt) {
        if (cloud_in.points[pt].x < 0){
			filtered_indices->indices.push_back(pt);
		}
    }

	pcl::copyPointCloud(cloud_in, *filtered_indices, *cloud_filtered);
	// ----------------------Finish----------------------


	//From PCL cloud msg to ROS sensor_msg
	sensor_msgs::PointCloud2 msg_cloud;
	pcl::toROSMsg(*cloud_filtered, msg_cloud);
	msg_cloud.header = header;
	filtered_pub.publish(msg_cloud);
}


int main (int argc, char** argv)
{
	// Initialize ROS
	ros::init (argc, argv, "demo_pc_cpp");
	ros::NodeHandle nh;

	// Define topic and queue size
	std::string pub_pc_topic = "/filtered";
	std::string sub_pc_topic = "/os1_cloud_node/points";
	int sub_pc_queue_size = 1;
	int pub_pc_queue_size = 1;

	// Subscribe and callback
	pointcloud_sub = nh.subscribe<sensor_msgs::PointCloud2>(sub_pc_topic, sub_pc_queue_size, OnPointCloud);

	// ROS publisher for the output point cloud
	filtered_pub = nh.advertise<sensor_msgs::PointCloud2>(pub_pc_topic, pub_pc_queue_size);

	// Spin
	ros::spin ();
}
