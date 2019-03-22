# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:47:09 2019

@author: Yiqun
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
image = cv2.imread('test.jpg')
lane_image = np.copy(image)
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray)
'''

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
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
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    # Convert RGB to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Reduce noise with Gaussian Filter
    #5x5 Gaussian Kernal, deviation as 0, 'gray' as the blur target
    blur = cv2.GaussianBlur(gray, (5,5), 0) 

    # Canny edge detection,which takes gradient
    # 50 and 150 are the low threshold and high threshold
    # find the sharpest change in intensity
    canny = cv2.Canny(blur, 30, 150)
    return gray
    
    
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255, 0, 0), 10)
    return line_image
    
def region_of_interest(image):
    # height = image.shape[0]
    polygons = np.array([
    [(450, 1100), (1800, 1100), (1270, 720)]
    ])
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

    
image = cv2.imread('china.jpg')
lane_image = np.copy(image)
canny_image = canny(lane_image)
#cropped_image = region_of_interest(canny_image)
#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap = 5)
#
#averaged_lines = average_slope_intercept(lane_image, lines)
#
#line_image = display_lines(lane_image, averaged_lines)
#combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow("result", canny_image)
cv2.waitKey(0)