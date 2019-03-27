#!/usr/bin/env python

import rospy
import math
import numpy as np
import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2


pub_curb = rospy.Publisher('/curb', sensor_msgs.msg.PointCloud2, queue_size=10)
pub_lane = rospy.Publisher('/lane', sensor_msgs.msg.PointCloud2, queue_size=10)
pub_midPoint = rospy.Publisher('/midPoint', sensor_msgs.msg.PointCloud2, queue_size=10)

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

def forfindmid(new_list3, value):
	for ind1 in list(range(len(new_list3))):
		for ind2 in list(range(len(new_list3[ind1]))):
			if new_list3[ind1][ind2][1] == value:
				l1 = ind1
				l2 = ind2
	return l1,l2


def sortY(val):
	return val[1]

def checkCurb(left_set_curb, right_set_curb):
	if len(left_set_curb) == 0 or len(right_set_curb) == 0:
		rospy.loginfo("Some curbs missing, no further work")
		return 0,0,0
	else:
		left_distance_curb = np.mean(left_set_curb)
		right_distance_curb = np.mean(right_set_curb)
		rospy.loginfo("Curb on the left: %f meters", left_distance_curb)	
		rospy.loginfo("Curb on the right: %f meters", right_distance_curb)
		lane_on_curb = int((abs(left_distance_curb) + abs(right_distance_curb))/3.7)
		rospy.loginfo("Based on curb, maybe how many lanes: %d or %d", lane_on_curb-1, lane_on_curb)
		return left_distance_curb,right_distance_curb,lane_on_curb


def grouping(point, lists):
	for index_num in index:
		if point[1] >= arr[index_num] and point[1] <arr[index_num +1]:
			lists[index_num].append(point[2])
	return lists



def callback(msg):
	curb_temp = [] # container for potential curb info
	lane_temp = [] # container for potential lane info
	header = msg.header
	fields = msg.fields

	rl = [] # container for lane info on car's right side
	curb_detect = []
	grouped = []
	v = []
	showc = []
	mll = [] # max value list for lane
	mlc = [] # max value list for curb
	right_set_curb = []
	left_set_curb = []

	linecount = 0
	jlane = 0
	klane = jlane + 1
	ll = [ [ ] for _ in range(12) ] # lists for lane detection
	lists = [ [ ] for _ in range(s) ] # lists for curb detection

	il = 0 # index for lane detection

	for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans= True ):
		point_0 = ci * point[0] +si * point[2]
		point_2 = -si * point[0] + ci *point[2]
		if point_0 > 0 and point[4] <= 4:
			if point[4] == 0:
				curb_temp.append([point_0, point[1], point_2, point[3], point[4]])
			elif point[3] >= 70 and point_2 <= -1.8 and point[1] > -6 and point[1] < 6:
				lane_temp.append((point_0, point[1], point_2, point[3], point[4]))
				# append curb and lane info by one shot
		
	curb_msg = sensor_msgs.point_cloud2.create_cloud(header, fields, curb_temp)
	lane_msg = sensor_msgs.point_cloud2.create_cloud(header, fields, lane_temp)
	# ===============================================================================
	# Now do the curb:
	rospy.loginfo("---------Based on the curb detection: ------------")
	for point in sensor_msgs.point_cloud2.read_points(curb_msg):
		grouped = grouping(point, lists)

	for i in range(len(grouped)):
		if len(grouped[i]) == 0:
			mlc.append(-100)
		else:
			mlc.append(max(grouped[i]))
	
	v = abs(np.diff(mlc)).tolist()

	i = int(detect_width / tire_width - 1)
	while (i >= 0 and (v[i] > 0.55 or v[i] <= threshold)):
		i = i - 1

	j = int(detect_width / tire_width - 1)
	while (j <= 121 and (v[j] > 0.55 or v[j] <= threshold)):
		j = j + 1

	for point in sensor_msgs.point_cloud2.read_points(curb_msg, skip_nans=True):
		if (abs(arr[i]) >= 0.1 and point[1] <= (arr[i] + 0.1) and point[1] >= (arr[i] - 0.1)) or (point[1] <= (arr[j] + 0.1) and point[1] >= (arr[j] - 0.1)):
			curb_detect.append((point[0], point[1], point[2], point[3], point[4]))
			showc.append(point[1])

	for entry in showc:
		if entry > 0:
			left_set_curb.append(entry)
		elif entry < 0:
			right_set_curb.append(entry)

	left_distance_curb,right_distance_curb,lane_on_curb = checkCurb(left_set_curb, right_set_curb)

	curb_detect_msg = sensor_msgs.point_cloud2.create_cloud(header, fields, curb_detect)	
	pub_curb.publish(curb_detect_msg)

	# ===============================================================================
	# Now do the lane:
	# Since just filtered the reflection, can already publish it 
	rospy.loginfo("---------Based on the lane detection: ------------")
	pub_lane.publish(lane_msg)

	lane_temp.sort(key = sortY)
	
	new_list = [ [ ] for _ in range(12) ]

	while( klane <= (len(lane_temp) - 2 )):
		while(((lane_temp[klane][1] - lane_temp[jlane][1]) < 0.035 or abs(lane_temp[jlane][1]>=5)) and (klane <= len(lane_temp) - 2)):
			ll[il].append(lane_temp[jlane][1])
			new_list[il].append((lane_temp[jlane][0],lane_temp[jlane][1]))
			jlane = jlane + 1
			klane = jlane + 1
		if len(ll[il]) >= 5:
			il = il + 1
		jlane = jlane + 1
		klane = jlane + 1

	new_list2 = list(filter(None, new_list))
	list2 = list(filter(None, ll))
	# rospy.loginfo(list2)

	for element in list2:
		mll.append(max(element))
	
	di = np.diff(mll).tolist()
	
	y_double_solid = 0

	double_cross = 0
	for check in list(range(len(di))):
		if di[check] <= 0.35:
			y_double_solid = min(list2[check])
			double_cross = 1
			rospy.loginfo("Doube Solid Lines detected, at car's: %f meters", y_double_solid)
	
	lane_right = 0

	for j in mll:
		if j <0: 
			rl.append(j)
	if len(rl) != 0:
		lane_right = max(rl)
		rospy.loginfo('distance from right lane: %f meters', lane_right)

	linecount = len(list2)
	rospy.loginfo("painted lines found: %d", linecount)

	if double_cross == 1:
		if linecount <= 4:
			lane_count = 2
		else:
			lane_count = linecount - 2
	else:
		lane_count = 1

	rospy.loginfo("lane_count: %d", lane_count)

	if lane_count == lane_on_curb - 1 or lane_count == lane_on_curb:
		match = 1
		rospy.loginfo("lane and curb prediction match")
	else:
		match = 0
		rospy.loginfo("lane and curb prediction don't match")
		rospy.loginfo("FURTHER ANALYSIS REQUIRED")

	
	l1_left,l2_left = forfindmid(new_list2, y_double_solid)

	l1_right,l2_right = forfindmid(new_list2, lane_right)

	midPoint_x = (new_list[l1_left][l2_left][0] + new_list[l1_right][l2_right][0])/2
	midPoint_y = (new_list[l1_left][l2_left][1] + new_list[l1_right][l2_right][1])/2

	midPointPublishingList = [[midPoint_x, midPoint_y, 0, 150, 4]]

	rospy.loginfo("Next mid line coordinate (Not considering z value): (x,y) = (%f,%f)", midPoint_x, midPoint_y)
	midPoint_msg = sensor_msgs.point_cloud2.create_cloud(header, fields, midPointPublishingList)
	pub_midPoint.publish(midPoint_msg)
	rospy.loginfo("======================End========================")


def main():
	rospy.init_node('pc_read')
	rospy.Subscriber("/velodyne_points", PointCloud2, callback)
	rospy.spin()

if __name__ == '__main__':
	main()