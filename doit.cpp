#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

#include <boost/foreach.hpp>

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& msg){
    
    printf("Cloud: width = %d, height = %d\n", msg->width, msg->height);
    
    // Container for data
    pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2; 
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);

    // Convert to PCL data type
    pcl_conversions::toPCL(*msg, *cloud);

    BOOST_FOREACH(const pcl::PointXYZ& pt, cloud->points)
        printf("\t(%f, %f, %f)\n", pt.x, pt.y, pt.z);
}

int main(int argc, char** argv){
    // Initialize ROS
    ros::init(argc, argv, "doit");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2> ("/velodyne_points", 100, cloud_cb);

    // pub = nh.advertise<sensor_msgs::PointCloud2> ("/output", 1);

    ros::spin();
}