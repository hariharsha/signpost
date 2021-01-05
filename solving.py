# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:18:22 2020
Z
@author: hariharsha
"""
from bisect import bisect
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pyautogui as pyauto
import pytesseract as pytess
from time import time
pytess.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

pyauto.click(694,1055)
pyauto.sleep(0.3)
for i in range(15):
    pyauto.screenshot('test1.png',region=(146,160,679,667))
    pyauto.sleep(0.2)
    start_time = time()
    im = cv.imread('test1.png')
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 1, 255, cv.THRESH_TOZERO)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    contourarea = [cv.contourArea(i) for i in contours]
    contours = np.array(contours)[np.array(contourarea)>100]
    contoursarea = [cv.contourArea(i) for i in contours]
    
    
    def detect_arrows(arr):
        def slp_len(p1,p2):
            slope = 1000 if p2[0]==p1[0] else (p2[1]-p1[1])/(p2[0]-p1[0])
            length = ((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)**.5
            return [slope,length]
        def midpt(p1,p2):
            return [(p1[0]+p2[0])/2,(p2[1]+p1[1])/2]
        if (np.size(arr)>35 or np.size(arr)<10):
            return 0
        arr = np.concatenate((arr,np.array([arr[0]])))
        slp_lens = np.array([slp_len(arr[i][0], arr[i+1][0]) for i in range(len(arr)-1)])
        lens_4 = sorted(slp_lens[:,1])[-4]
        boolean_select = slp_lens[:,1]>=lens_4
        slopes_we_like = slp_lens[boolean_select][:,0]
        pts_4 = [midpt(arr[i][0],arr[i+1][0]) for i in np.where(boolean_select)[0]]
        repeated_slope = [i for i in slopes_we_like if list(slopes_we_like).count(i)==2][0]
        tail = [pts_4[i] for i in range(4) if slopes_we_like[i]==repeated_slope]
        tail_mid = midpt(tail[0],tail[1])
        head = [pts_4[i] for i in range(4) if slopes_we_like[i]!=repeated_slope]
        head_mid = midpt(head[0],head[1])
        arrow_mid_pt = midpt(head_mid,tail_mid)
        if (repeated_slope == 0 and 1 in slopes_we_like and -1 in slopes_we_like):
            direction = "left" if head_mid[0]<tail_mid[0] else "right"
        elif (repeated_slope == 1000 and 1 in slopes_we_like and -1 in slopes_we_like):
            direction = "up" if head_mid[1]<tail_mid[1] else "down"
        elif (repeated_slope == -1 and 0 in slopes_we_like and 1000 in slopes_we_like):
            direction = "right up" if head_mid[1]<tail_mid[1] else "left down"
        elif (repeated_slope ==1 and 0 in slopes_we_like and 1000 in slopes_we_like):
            direction = "left up" if head_mid[0]<tail_mid[0] else "right down"
        else:
            return 0
        # plt.plot(arr[:,0,0],arr[:,0,1])
        # plt.plot([head_mid[0],tail_mid[0]],[head_mid[1],tail_mid[1]])
        # plt.title(direction)
        # plt.scatter(arrow_mid_pt[0],arrow_mid_pt[1])
        #plt.text(arrow_mid_pt[0],arrow_mid_pt[1],direction)
        # plt.gca().invert_yaxis()
        #plt.show()
        return [direction, arrow_mid_pt]
    
    all_arrows = [detect_arrows(i) for i in contours]
    
    #x_dim length
    x_dim,x_prev = 0,0
    x_axes = []
    for i in sorted([i[1][0] for i in all_arrows if i!=0]):
        if ((i-x_prev)>10):
            x_prev = i
            x_dim=x_dim+1
            x_axes.append(i)
    y_dim,y_prev = 0,0
    y_axes =[]
    for i in sorted([i[1][1] for i in all_arrows if i!=0]):
        if ((i-y_prev)>10):
            y_prev = i
            y_dim=y_dim+1
            y_axes.append(i)
    if (y_dim==0 or x_dim==0):
        continue
    all_arrow_matrix = np.zeros((y_dim,x_dim), dtype = object)
    all_arrow_destinations = np.zeros((y_dim,x_dim), dtype = object)
    for i in all_arrows:
        if i!=0:
            all_arrow_matrix[bisect(y_axes,i[1][1]-5)][bisect(x_axes,i[1][0]-5)] = i[0]
    for i in range(y_dim):
        for j in range(x_dim):
            direction = all_arrow_matrix[i][j]
            if direction==0: continue
            x_incr = -1 if "left" in direction else 1 if "right" in direction else 0
            y_incr = -1 if "up" in direction else 1 if "down" in direction else 0
            next_xs = [j] if x_incr==0 else [p for p in range(x_dim)][j+1 if x_incr==1 else 0: j if x_incr==-1 else 100]
            next_ys = [i] if y_incr==0 else [p for p in range(y_dim)][i+1 if y_incr==1 else 0: i if y_incr==-1 else 100]
            
            if x_incr*y_incr==0:
                all_arrow_destinations[i][j]=[(p,q) for p in next_ys for q in next_xs]
            else:
                all_arrow_destinations[i][j]=[(i+y_incr*p,j+x_incr*p) for p in range(1,1+min(len(next_xs),len(next_ys)))]
            #print(i,j,all_arrow_destinations[i][j],next_xs,next_ys,x_incr,y_incr)    
        
    lower_blue = np.array([0, 0, 0]) 
    upper_blue = np.array([255, 10, 10])
    mask = cv.inRange(im, lower_blue, upper_blue) 
    num_only = cv.bitwise_and(im, im, mask = mask)   
    num_only[np.where((num_only==[0,0,0]).all(axis=2))] = [255,255,255]
    # cv.imshow('only numbers', num_only) 
    # cv.waitKey(0) 
    
    
    num_data = pytess.image_to_data(num_only, config='--psm 6')
    num_data = [i.split('\t')[6:] for i in num_data.split('\n')[:-1]]
    num_data = [i for i in num_data if (i[-1]!='' and i[-1]!='.')]
    num_data = [[i.replace('A','4').replace('I','1').replace('°','').replace('T','7') for i in j] for j in num_data]
    try:
        num_data = [[int(float(j)) for j in i] for i in num_data[1:]]
    except:
        continue
    given_numbers = np.zeros((y_dim,x_dim))
    
    for i in num_data:
        given_numbers[bisect(y_axes,i[1])][bisect(x_axes,i[0])] = i[-1]
    given_numbers[0][0]=1
    vector_lengths = [-1 if not all_arrow_destinations[i][j] else len(all_arrow_destinations[i][j]) for i in range(y_dim) for j in range(x_dim)]
    given_numbers_set = set(given_numbers.flatten())
    
    for i in range(y_dim):
        for j in range(x_dim):
            if all_arrow_destinations[i][j]==0:
                if given_numbers[i][j]==0:
                    given_numbers[i][j]=-1
    present_status = np.copy(given_numbers)
    missing_numbers = [i for i in range(1,x_dim*y_dim) if i not in given_numbers.flatten()]
    guess_account = []
    
    def guess_account_init():
        num_now = min(missing_numbers)
        last_num_position = [(i,j) for i in range(y_dim) for j in range(x_dim) if (given_numbers[i][j]==(num_now-1))][0]
        next_num_positions = all_arrow_destinations[last_num_position[0]][last_num_position[1]]
        for i in next_num_positions:
            if given_numbers[i[0]][i[1]]==0:
                present_status[i[0]][i[1]]=num_now
                missing_numbers.pop(0)
                return guess_account.append([num_now,next_num_positions,next_num_positions.index(i)])  
    
    guess_account_init()
    
    def check_arrow_matching(num_now,pos):
        for i in all_arrow_destinations[pos[0]][pos[1]]:
            if given_numbers[i[0]][i[1]]== (num_now+1):
                return True
        return False

    def next_position():
        num_now = missing_numbers[0]
        last_num_position = [(i,j) for i in range(y_dim) for j in range(x_dim) if (present_status[i][j]==(num_now-1))][0]
        num_now_positions = all_arrow_destinations[last_num_position[0]][last_num_position[1]]
        check_req = (num_now+1) not in missing_numbers
        for i in num_now_positions:
            if present_status[i[0]][i[1]]==0:
                if check_req:
                    if check_arrow_matching(num_now,i):
                        pass
                    else:
                        continue
                present_status[i[0]][i[1]]=num_now
                missing_numbers.pop(0)
                return guess_account.append([num_now,num_now_positions,num_now_positions.index(i)])
        while True:
            #print("running while loop in next_position function")
            last_num_details = guess_account.pop()
            last_number = last_num_details[0]
            last_num_position = last_num_details[1][last_num_details[2]]
            present_status[last_num_position[0]][last_num_position[1]]=0
            last_num_positions = last_num_details[1][last_num_details[2]+1:]
            check_req = (last_number + 1) not in missing_numbers
            for i in last_num_positions:
                if present_status[i[0]][i[1]]==0:
                    if check_req:
                        if check_arrow_matching(last_number, i):
                            pass
                        else:
                            continue
                    present_status[i[0]][i[1]]=last_number
                    return guess_account.append([last_number,last_num_details[1],last_num_details[1].index(i)])
            missing_numbers.insert(0,last_number)
        return "it is bad"
    step = 1
    while (len(missing_numbers)!=0):
        step+=1
        next_position()     
    print(step,'\t',round(time()-start_time,2))
    blank_canvas = np.zeros(im.shape)
    blank_canvas.fill(255)
    cv.drawContours(blank_canvas, [contours[i] for i in range(len(contours)) if all_arrows[i]!=0], -1, (0,255,0), 2)
    for i in range(y_dim):
        for j in range(x_dim):
            cv.putText(blank_canvas,str(int(present_status[i][j])),(int(y_axes[j])-25,int(x_axes[i])-25),cv.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv.LINE_4)
    pyauto.moveTo(1614,132)
    cv.imshow('Contours', blank_canvas) 
    cv.waitKey(0) 
    cv.destroyAllWindows() 
    pyauto.click(63,955)
