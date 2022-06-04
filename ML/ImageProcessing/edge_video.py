"""
This code is used for detecting image edges using "Structured Forest".
For the purpose of benchmarking auther used Canny method.

The code is taken from:
https://debuggercafe.com/edge-detection-using-structured-forests-with-opencv/

The sample data cab be found on the rep.

Future update:
It is required to check if there is a method for the quantification of the 
reults of methods using grouand truth.

The original publication for Staructure Forest cab be downloaded from:
https://openaccess.thecvf.com/content_iccv_2013/papers/Dollar_Structured_Forests_for_2013_ICCV_paper.pdf

Date: 2/15/2021
version: 1.0.0 
"""

import argparse
import numpy as np
import cv2
import time
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to the input video', 
                    required=True)
args = vars(parser.parse_args())
cap = cv2.VideoCapture(args['input'])
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"

# define codec and create VideoWriter object 
out1 = cv2.VideoWriter(f"outputs/{save_name}_forests.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                      (frame_width, frame_height))

out2 = cv2.VideoWriter(f"outputs/{save_name}_canny.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                      (frame_width, frame_height))



# initialize the structured edge detector with the model
edge_detector = cv2.ximgproc.createStructuredEdgeDetection('model/model.yml')
frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second


# read video until end of frames
while cap.isOpened(): 
    ret, frame = cap.read()
    if ret == True:
        # keep a copy of the original frame
        orig_frame = frame.copy()
        # convert to RGB frame and convert to float32
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # normalize to bring all pixel values between 0.0 and 1.0
        frame = frame.astype(np.float32) / 255.0
        # grayscale and blurring for canny edge detection
        gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # carry out Canny edge detection
        canny = cv2.Canny(blurred, 50, 200)
        # start time counter before Structured Forests edge detection starts
        start_time = time.time()
        # detect the edges
        edges = edge_detector.detectEdges(frame)
        # end time after Structured Forests edge detection ends
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        # show and save the results
        cv2.imshow('Structured forests', edges)
        cv2.imshow('Canny', canny)
        # make the gray scale output into an output with 3 color channels
        forests_edges_3d = np.repeat(edges[:, :, np.newaxis], 3, axis=2)
        canny_edges_3d = np.repeat(canny[:, :, np.newaxis], 3, axis=2)
        out1.write((forests_edges_3d*255.0).astype(np.uint8))
        out2.write((canny_edges_3d).astype(np.uint8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


    # release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")