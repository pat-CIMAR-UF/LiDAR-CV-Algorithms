#!/usr/bin/env python
# license removed for brevity
import rospy
import math
import numpy as np
import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2

#I want to publish a topic named '/filter':
pub = rospy.Publisher('/filter', sensor_msgs.msg.PointCloud2, queue_size=10)


def callback(msg):
    new_cloud = []

    header = msg.header
    fields = msg.fields

    #The point cloud sensor message of Ouster LiDAR could be like this:
    #sensor_msgs.point_cloud2 = {float x, float y, float z, float intensity, uint32_t, uint16_t, uint8_t, uint16_t noise, uint32_t range}
    #float x, y, z, float intensity, uinit16_t noise and uint32_t range is the 6 parameters returned from the LiDAR
    #x, y, z and intensity is known for sure as each pixel's coordinate and reflectivity, where I am still investigating the meaning of noise and range.

    
    for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans = True):
        if point[0] > 0 : 
            new_cloud.append([point[0], point[1], point[2], point[3], 0, 0, 0, 0, point[4]])
            #Different from Velodyne, the intensity(point[3]) of Ouster is range from 0-1023, where Velodyne 0-255
            #For point[4]:range(guess): order is 10e7, this could be wrong
    
    #I append my qualified points, and use create_cloud to generate a new msg to ROS
    new_msg = sensor_msgs.point_cloud2.create_cloud(header, fields, new_cloud)
    pub.publish(new_msg)

def main():
    rospy.init_node('pc_read')
    #Subscribe from the original topic of points published from Ouster, call the callback function to do operations:
    rospy.Subscriber("/os1_cloud_node/points", PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
	main()