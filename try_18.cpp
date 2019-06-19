#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Header.h>

#include <pcl/point_cloud.h> /* pcl::PointCloud */
#include <pcl/point_types.h> /* pcl::PointXYZ */
#include <pcl_conversions/pcl_conversions.h>  // pcl::fromROSMsg
#include <pcl/filters/extract_indices.h>  // pcl::ExtractIndices
#include <pcl/io/io.h>                    // pcl::copyPointCloud
#include <pcl/common/projection_matrix.h>

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef PointCloud::Ptr PointCloudPtr;
typedef PointCloud::ConstPtr PointCloudConstPtr;

typedef pcl::PointXYZI PointI;
typedef pcl::PointCloud<PointI> PointICloud;
typedef PointICloud::Ptr PointICloudPtr;
typedef PointICloud::ConstPtr PointICloudConstPtr;

std::string frame_id_ = "os1_lidar";

// ROS Subscriber
ros::Subscriber pointcloud_sub_;
// ROS Publisher
ros::Publisher ground_pub_;

/* 
static void publishCloud(const ros::Publisher& publisher,
                         const std_msgs::Header& header,
                         const typename pcl::PointCloud<PointT>& cloud) {
    if (cloud.size()) {
        sensor_msgs::PointCloud2 msg_cloud;
        pcl::toROSMsg(cloud, msg_cloud);
        msg_cloud.header = header;
        publisher.publish(msg_cloud);
    }
}
*/



void mainLoop(const PointICloud& cloud_in, PointICloudPtr cloud_gnds){
	cloud_gnds->clear();

	pcl::PointIndices::Ptr gnds_indices(new pcl::PointIndices);
	gnds_indices->indices.clear();

    
    for (size_t pt = 0u; pt < cloud_in.points.size(); ++pt) {
        if (cloud_in.points[pt].x < 0){
			gnds_indices->indices.push_back(pt);
			//ROS_INFO_STREAM("x:" << cloud_in.points[pt].x << " y:" << cloud_in.points[pt].y << " z:" << cloud_in.points[pt].z);

		}
    }
	
	pcl::copyPointCloud(cloud_in, *gnds_indices, *cloud_gnds);
	//ROS_INFO_STREAM(" Cloud inputs: #" << cloud_gnds->size() << " Points");

}


void filter(const PointICloud& cloud_in, PointICloudPtr cloud_ground){
	PointICloudPtr cloud_ground2(new PointICloud);
	mainLoop(cloud_in, cloud_ground2);
	ROS_INFO_STREAM(" filter step: #" << cloud_ground2->size() << " Points");
}

void OnPointCloud(const sensor_msgs::PointCloud2ConstPtr& ros_pc2) {

	PointICloudPtr cloud(new PointICloud);

    //transfer from ROS sensor_msg to PCL cloud msg
    pcl::fromROSMsg(*ros_pc2, *cloud);
    ROS_INFO_STREAM(" Cloud inputs: #" << cloud->size() << " Points");

	//Read the header for later generating new pc back
    std_msgs::Header header = ros_pc2->header;
    header.frame_id = frame_id_;
    header.stamp = ros::Time::now();

	std::vector<PointICloudPtr> cloud_clusters;
	PointICloudPtr cloud_ground(new PointICloud);

	filter(*cloud, cloud_ground);
	//ROS_INFO_STREAM(" after filter step: #" << cloud_ground->size() << " Points");

	//publishCloud(ground_pub_, header, *cloud_ground);

	//ROS_INFO_STREAM(" Cloud inputs: #" << cloud_ground->size() << " Points");
	sensor_msgs::PointCloud2 msg_cloud;
	pcl::toROSMsg(*cloud_ground, msg_cloud);
	msg_cloud.header = header;
	ground_pub_.publish(msg_cloud);
	

	

	ROS_INFO_STREAM("Done");
}


int main (int argc, char** argv)
{
	// Initialize ROS
	ros::init (argc, argv, "try_18");
	ros::NodeHandle nh;

	std::string pub_pc_ground_topic = "/output";
	std::string sub_pc_topic = "/os1_cloud_node/points";
	int sub_pc_queue_size = 1;


	pointcloud_sub_ = nh.subscribe<sensor_msgs::PointCloud2>(
						sub_pc_topic, sub_pc_queue_size, OnPointCloud);

		// Create a ROS publisher for the output point cloud
	ground_pub_ = nh.advertise<sensor_msgs::PointCloud2>(pub_pc_ground_topic, 1);

	// Spin
	ros::spin ();
}
