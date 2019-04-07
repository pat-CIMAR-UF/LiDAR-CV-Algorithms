# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:13:07 2019

@author: Yiqun
"""

import cv2
import numpy as np
import time

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    print('slope type:')
    print(type(slope))
    print('intercept type:')
    print(type(intercept))
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope) 
    if x1 > 2000 or x2 > 2000:
        x1 = 1894
        x2 = 1323
    print(np.array([x1, y1, x2, y2]))
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        print("slope: %f, inter: %f" % (slope, intercept))
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if len(left_fit) == 0:
        left_fit.append((-9.85185185*0.1, 1.04825185*1000))
    if len(right_fit) == 0:
        right_fit.append((1.39272446,  -523.417127))  
    left_fit_average = np.average(left_fit, axis = 0)
    print("left_fit")
    print(left_fit_average)
    right_fit_average = np.average(right_fit, axis = 0)
    print("right_fit")
    print(right_fit_average)
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
    return canny
    
    
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            print("x1: %d, y1: %d, x2: %d, y2: %d" %(x1,y1,x2,y2))
            if x1 < 0 or x2 < 0:
                x1 = 85
                x2 = 477
            cv2.line(line_image, (x1,y1), (x2,y2), (255, 0, 0), 6)
    return line_image
    
def region_of_interest(image):
    # height = image.shape[0]
    polygons = np.array([
    [(550, 435),(220, 740),(1075, 740), (730,435)]
    ])
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

  
#image = cv2.imread('test_image.jpg')
#lane_image = np.copy(image)
#canny_image = canny(lane_image)
#cropped_image = region_of_interest(canny_image)
#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap = 5)
#averaged_lines = average_slope_intercept(lane_image, lines)
#line_image = display_lines(lane_image, averaged_lines)
#combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#cv2.imshow("result", combo_image)
#cv2.waitKey(0)


cap = cv2.VideoCapture("record1.mp4")
while(cap.isOpened()):
    print('******************************************')
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
#    cv2.imshow("result", canny_image)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
    lines = cv2.HoughLinesP(cropped_image, 3, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap = 5)
    print("lines:")
    print(lines)
    if lines is not None:
        averaged_lines = average_slope_intercept(frame, lines)
        print("averaged_lines:")
        print(averaged_lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        print('finish')
        cv2.imshow("result", combo_image)
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.imshow("result", frame)
        time.sleep(0.1)
    

cap.release()
cv2.destroyAllWindows()
    
