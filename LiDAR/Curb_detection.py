#!/usr/bin/env python

import rospy
import math
import numpy as np
import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2

pub = rospy.Publisher('/velodyne_curb', sensor_msgs.msg.PointCloud2, queue_size=10)

p = math.pi
angel_degree = 10
ci = math.cos(angel_degree * p / 180)
si = math.sin(angel_degree * p / 180)

detect_width = 6.82
tire_width = 0.11
threshold = 0.04

n = int(2*detect_width/tire_width)
s = int(n)
arr = np.linspace(-detect_width, detect_width, s)
index = list(range(s-1))

def grouping(point,lists):
	for index_num in index:
    		if point[1] >= arr[index_num] and point[1] < arr[(index_num + 1)]:
			lists[index_num].append(point[2])
	return lists
	#grouping tends to return 62 sets of points for the ring_0, with an increment of 0.22mm	

def callback(msg):
	new_cloud = []
	final = []
	header = msg.header
	fields = msg.fields
	grouped = []
	v = []
	max_list_right = []
	lists = [[] for _ in range(s)] # generate 62 lists

	for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans = True):
		point_0 = ci * point[0] + si * point[2]
		point_2 = -si * point[0] + ci * point[2]
		if point[4] == 0 and point_0 > 0: 
			new_cloud.append([point_0, point[1], point_2, point[3], point[4]])
	new_msg = sensor_msgs.point_cloud2.create_cloud(header, fields, new_cloud)
	# Now the data would be splitted as many small groups, along y axis.
    	for point in sensor_msgs.point_cloud2.read_points(new_msg, skip_nans=True):
        	grouped = grouping(point,lists)	
	
	# rospy.loginfo(len(grouped))
	for i in range(len(grouped)):
		if len(grouped[i]) == 0:
			max_list_right.append(-100)
		else:
			max_list_right.append(max(grouped[i]))

	v = abs(np.diff(max_list_right)).tolist()
	'''
	rospy.loginfo('here is what you need:')
	rospy.loginfo(max_list_right)
	rospy.loginfo(len(max_list_right))
	rospy.loginfo(v)
	rospy.loginfo(len(v))
	'''
	#Now need to divide them with the number of 61
	
	i = 61
	while ((v[i] > 0.55 or v[i] <= threshold) and i >= 0 ):
		i = i - 1

	j = 61
	while (j <= 121 and (v[j] > 0.55 or v[j] <= threshold)):
		j = j + 1
	# now the i which is breaking the law was found out:
	# rospy.loginfo(i)
	# rospy.loginfo(arr[i])
	# rospy.loginfo('*******************************')


    	for point in sensor_msgs.point_cloud2.read_points(new_msg, skip_nans=True):
		if (abs(arr[i]) >= 0.1 and point[1] <= (arr[i] + 0.1) and point[1] >= (arr[i] - 0.1)) or (point[1] <= (arr[j] + 0.1) and point[1] >= (arr[j] - 0.1)):
			final.append([point[0], point[1], point[2], point[3], point[4]])
	final_msg = sensor_msgs.point_cloud2.create_cloud(header, fields, final)
	pub.publish(final_msg)
	
def main():
	rospy.init_node('pc_read')
	rospy.Subscriber("/velodyne_points", PointCloud2, callback)
	rospy.spin()


if __name__ == '__main__':
	main()
