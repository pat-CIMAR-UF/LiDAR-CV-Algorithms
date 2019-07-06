#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import roslib
roslib.load_manifest('beginner_tutorials')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def canny(image):
    copy_image = np.copy(image)
    gray_image = cv2.cvtColor(copy_image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    canny_image = cv2.Canny(blur_image, 10,30)
    return canny_image

def region_interest(image):
    polygons = np.array([
            [(550,435), (220,740), (1075,740),(730, 435)]
            ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def check_slope(lines):
    new_lines = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,y1), (x2,y2), 1)
            slope = parameters[0]
        # intercept = parameters[1]
            if abs(slope) >= 0.3 and abs(slope) <= 1.12:
                #print('slope is %f' %(slope))
                new_lines.append([x1,y1,x2,y2])
    return np.asarray(new_lines)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        cv2.line(line_image, (x1,y1), (x2,y2), (255, 0,0), 4)
    return line_image

class image_converter:
  def __init__(self):
    self.image_pub = rospy.Publisher("/camera/image_edit",Image, queue_size=10)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/image_color",Image,self.callback) #2.Subscribe the topic from ROS bag file, start the callback

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") #convert the ROS image msg to cv2 format image message, as "bgr8" format
    except CvBridgeError as e:
      print(e)
    
    (rows,cols,channels) = cv_image.shape
    if cols > 60 and rows > 60 :
      cv2.circle(cv_image, (50,50), 10, 255)

    canny_image = canny(cv_image)
    cropped_image = region_interest(canny_image)
    hough_lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    lines = check_slope(hough_lines)

    if lines is not None:
        line_image = display_lines(cv_image, lines)
        combo_image = cv2.addWeighted(cv_image, 1, line_image, 1, 1)
        cv_image = combo_image
    
    #else:
    #    cv_image = combo_image
    #cv_image = cropped_image

    cv2.imshow("Canny Image", canny_image)
    cv2.waitKey(3)
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8")) #bgr8 #mono8
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin() # you start the program here
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
