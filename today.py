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
'''
def select_rgb_white_yellow(image):
  #white color mask
  lower = np.uint8([120, 120, 120])
  upper = np.uint8([255,255, 255])
  white_mask = cv2.inRange(image, lower, upper)
  #yellow color mask
  lower = np.uint8([170, 170, 0])
  upper = np.uint8([255, 255, 255])
  yellow_mask = cv2.inRange(image, lower, upper)
  mask = cv2.bitwise_or(white_mask, yellow_mask)
  masked = cv2.bitwise_and(image, image, mask = mask)
  return masked
'''

def convert_hsv(image):
  return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def convert_hls(image):
  return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def select_white_yellow(image):
  converted = convert_hsv(image)
  #white color mast
  lower = np.uint8([0, 0, 78])
  upper = np.uint8([0, 0, 100])
  white_mask = cv2.inRange(converted, lower, upper)
  #yellow color mask
  lower = np.uint8([25, 50, 50])
  upper = np.uint8([32, 255, 255])
  yellow_mask = cv2.inRange(converted, lower, upper)
  # combine the mask
  mask = cv2.bitwise_or(white_mask, yellow_mask)
  return cv2.bitwise_and(image, image, mask = mask)
  

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
            intercept = parameters[1]
            if abs(slope) >= 0.3 and abs(slope) <= 1.12:
                #print('slope is %f' %(slope))
                new_lines.append([x1,y1,x2,y2])
                print("slope is %f" %(slope))
                print("intercept is %f" %(intercept))
    return np.asarray(new_lines)

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    if len(left_fit) == 0 and len(right_fit) != 0:
      right_fit_average = np.average(right_fit, axis=0)
      right_line = make_coordinates(image, right_fit_average)
      return np.array([right_line])

    if len(right_fit) == 0 and len(left_fit) != 0:
      left_fit_average = np.average(left_fit, axis = 0)
      left_line = make_coordinates(image, left_fit_average)
      return np.array([left_line])

    if left_fit and right_fit:
      left_fit_average = np.average(left_fit, axis = 0)
      right_fit_average = np.average(right_fit, axis = 0)
      left_line = make_coordinates(image, left_fit_average)
      right_line = make_coordinates(image, right_fit_average)    
      return np.array([left_line, right_line])

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        cv2.line(line_image, (x1,y1), (x2,y2), (255, 0,0), 6)
    return line_image

class image_converter:
  def __init__(self):
    self.image_pub = rospy.Publisher("/camera/image_edit",Image, queue_size=10)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/image_color",Image, self.callback) #2.Subscribe the topic from ROS bag file, start the callback

  def callback(self,data):
    rospy.loginfo("-------------------------------------------")
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") #convert the ROS image msg to cv2 format image message, as "bgr8" format
    except CvBridgeError as e:
      print(e)
    '''
    (rows,cols,channels) = cv_image.shape
    if cols > 60 and rows > 60 :
      cv2.circle(cv_image, (50,50), 10, 255)
    '''
    lane_image = np.copy(cv_image)
    hsv_white_yellow_image = select_white_yellow(lane_image)
    #hsv_white_yellow_image = convert_hsv(lane_image)
    cv2.imshow("hsv_masked_y_w", hsv_white_yellow_image)
    cv2.waitKey(3)

    canny_image = canny(cv_image)
    cropped_image = region_interest(canny_image)
    hough_lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 50, maxLineGap = 5)
    lines = check_slope(hough_lines)


    if lines is None:
      cv_image = cv_image
    else:
      try:
        averaged_lines = average_slope_intercept(lane_image, lines)
        line_image = display_lines(cv_image, averaged_lines)
        #cv2.imshow("result", line_image)
        #cv2.waitKey(3)
        combo_image = cv2.addWeighted(cv_image, 1, line_image, 1, 1)
        cv_image = combo_image
        print(averaged_lines)
      except:
        cv_image = cv_image
  
        
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
