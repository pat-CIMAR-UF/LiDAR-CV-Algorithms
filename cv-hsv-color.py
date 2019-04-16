#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('beginner_tutorials')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def region_interest_hsv(image):
    polygons = np.array([
            [(550,435), (220,740), (640,740),(640, 435)]
            ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def region_interest_bgr(image):
    polygons = np.array([
            [(640,435), (640,740), (1060,740),(730, 435)]
            ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

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

def display_lines_yellow(image, lines):
    line_image = np.zeros_like(image)
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        cv2.line(line_image, (x1,y1), (x2,y2), (0, 255, 255), 4)
    return line_image

def display_lines_white(image, lines):
    line_image = np.zeros_like(image)
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        cv2.line(line_image, (x1,y1), (x2,y2), (255, 255, 255), 4)
    return line_image

def canny_hsv(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    canny_image = cv2.Canny(blur_image, 100, 110)
    return canny_image

def canny_bgr(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    canny_image = cv2.Canny(blur_image, 30, 60)
    return canny_image

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

class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("/camera/image_edit", Image, queue_size = 10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_color", Image, self.callback)

    def callback(self,data):
        rospy.loginfo("------------------------------------------------------------")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        copy_image = np.copy(cv_image)
        line_image_bgr = np.zeros_like(copy_image)
        line_image_bgr_left = np.zeros_like(copy_image)
        line_image_hsv = np.zeros_like(copy_image)

        canny_image_bgr = canny_bgr(copy_image)
        cropped_image_bgr = region_interest_bgr(canny_image_bgr)
        hough_lines_bgr = cv2.HoughLinesP(cropped_image_bgr, 2, np.pi/180, 100, np.array([]), minLineLength = 50, maxLineGap = 5)
        if hough_lines_bgr is not None:
            lines_bgr = check_slope(hough_lines_bgr)
            averaged_lines_bgr = average_slope_intercept(copy_image, lines_bgr)
            if averaged_lines_bgr is not None:
                line_image_bgr = display_lines_white(cv_image, averaged_lines_bgr)

        image_hsv = convert_hsv(copy_image)
        canny_image_hsv = canny_hsv(image_hsv)
        cropped_image_hsv = region_interest_hsv(canny_image_hsv)
        hough_lines_hsv = cv2.HoughLinesP(cropped_image_hsv, 2, np.pi/180, 100, np.array([]), minLineLength = 50, maxLineGap = 5)
        if hough_lines_hsv is not None:
            lines_hsv = check_slope(hough_lines_hsv) #now you received hsv yellow lines coordinate
            averaged_lines_hsv = average_slope_intercept(copy_image, lines_hsv)
            if averaged_lines_hsv is not None:
                line_image_hsv = display_lines_yellow(cv_image, averaged_lines_hsv)
        else:
            cropped_image_bgr_left = region_interest_hsv(canny_image_bgr)
            hough_lines_bgr_left = cv2.HoughLinesP(cropped_image_bgr_left, 2, np.pi/180, 100, np.array([]), minLineLength = 50, maxLineGap = 5)
            if hough_lines_bgr_left is not None:
                lines_bgr_left = check_slope(hough_lines_bgr_left)
                averaged_lines_bgr_left = average_slope_intercept(copy_image, lines_bgr_left)
                if averaged_lines_bgr_left is not None:
                    line_image_bgr_left = display_lines_white(cv_image, averaged_lines_bgr_left)
        

        temp_image = cv2.bitwise_or(line_image_bgr,line_image_bgr_left)
        line_image = cv2.bitwise_or(temp_image, line_image_hsv)

        if line_image is not None:
            combo_image = cv2.addWeighted(cv_image, 1, line_image, 0.6, 1)
            cv_image = combo_image
        else:
            cv_image = cv_image
        
        cv2.imshow("resutl", cv_image)
        cv2.waitKey(3)

def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous = True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)


    

