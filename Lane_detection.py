#!/usr/bin/env python

#This code is for curb detection. Didn't do the filtering of moving car yet, this is the part need more time later on.

import rospy
import math
import numpy as np
import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2

pub = rospy.Publisher('/lane', sensor_msgs.msg.PointCloud2, queue_size=10)

p = math.pi
angel_degree = 10
ci = math.cos(angel_degree * p / 180)
si = math.sin(angel_degree * p / 180)

def sortY(val):
	return val[1]

def callback(msg):
	new_cloud = []
	header = msg.header
	fields = msg.fields
	rl = []
	lanecount = 0
	j = 0
	k = j + 1
	max_list = []
	lists = [[] for _ in range(12)]
	index = 0
	for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans = True):
		point_0 = ci * point[0] + si * point[2]
		point_2 = -si * point[0] + ci * point[2]
		if point_0 > 0 and point[4] <= 4 and point[3] >= 70 and point_2 <= -1.8 and point[1] > -6 and point[1] < 6: 
			new_cloud.append((point_0, point[1], point_2, point[3], point[4]))
	new_msg = sensor_msgs.point_cloud2.create_cloud(header, fields, new_cloud)
	pub.publish(new_msg)

	new_cloud.sort(key = sortY)
	# rospy.loginfo(new_cloud)
	while( k <= (len(new_cloud)-2)):
		while(((new_cloud[k][1] - new_cloud[j][1]) < 0.035 or abs(new_cloud[j][1]>=5)) and (k <= len(new_cloud) - 2)):
			lists[index].append(new_cloud[j][1])
			j = j + 1
			k = j + 1
		if len(lists[index]) >= 5:
			index = index + 1
		j = j + 1
		k = j + 1

	list2 = list(filter(None, lists))
	# rospy.loginfo('the filtered lane:')
	# rospy.loginfo(list2)
	
	
	for i in list2:
		max_list.append(max(i))
	rospy.loginfo('max_list:')
	rospy.loginfo(max_list)

	di = np.diff(max_list).tolist()
	# rospy.loginfo('difference')
	# rospy.loginfo(di)

	for check in list(range(len(di))):
		if di[check] <= 0.35:
			rospy.loginfo("Double Solid Lines detected, at car's:")
			rospy.loginfo(min(list2[check]))
			break
	lanecount = len(list2)
	rospy.loginfo('lane count:')
	rospy.loginfo(lanecount)

	for j in max_list:
		if j < 0:
			rl.append(j)
	if len(rl) != 0:	
		right_lane = max(rl)
		rospy.loginfo('distance from right lane:')
		rospy.loginfo(right_lane)			

	

	rospy.loginfo('================end=================')
def main():
	rospy.init_node('pc_read')
	rospy.Subscriber("/velodyne_points", PointCloud2, callback)
	rospy.spin()


if __name__ == '__main__':
	main()
