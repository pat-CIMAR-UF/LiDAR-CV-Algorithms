# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:22:24 2019

@author: yiqun
"""

import cv2
import numpy as np
import time

def canny(image):
    copy_image = np.copy(image)
    gray_image = cv2.cvtColor(copy_image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    canny_image = cv2.Canny(blur_image, 25,100)
    return canny_image

def region_interest(image):
    polygons = np.array([
            [(550,435), (220,740), (1075,740),(730, 435)]
            ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        cv2.line(line_image, (x1,y1), (x2,y2), (255, 0,0), 4)
    return line_image

def check_slope(lines):
    new_lines = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,y1), (x2,y2), 1)
            slope = parameters[0]
        # intercept = parameters[1]
            if abs(slope) >= 0.3 and abs(slope) <= 1.12:
                print('slope is %f' %(slope))
                new_lines.append([x1,y1,x2,y2])
    return np.asarray(new_lines)



cap = cv2.VideoCapture("output.mp4")
while(cap.isOpened()):
    print('----------------------------')
    _, frame = cap.read()
    time.sleep(0.1)
    canny_image = canny(frame)
    cropped_image = region_interest(canny_image)
    
    hough_lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
#    print('hough_line length:')
#    print(len(hough_lines))
    lines = check_slope(hough_lines)
    if lines is not None:
#        print('lines are: ')
#        print(lines)
        line_image = display_lines(frame, lines)
        combo_image = cv2.addWeighted(frame, 1, line_image, 1, 1)
        cv2.imshow('result', combo_image)
    else:
        cv2.imshow("result", frame)
        
    if cv2.waitKey(1) & 0xFf == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
# just adding comment    
    
