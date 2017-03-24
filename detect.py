from imutils.object_detection import non_max_suppression
import numpy as np
import numpy
import imutils
import cv2
import subprocess as sp
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", type=str, help="Path to the video file")
args = vars(parser.parse_args())

videoFile = None
if args.get("video", None) is None or not os.path.isfile(args.get("video", None)):
    print("Please check the path of the video file")
    exit()
else:
    videoFile = args.get("video", None)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))

cap = cv2.VideoCapture(videoFile)

index = 0
frameRate = cap.get(cv2.CAP_PROP_FPS)
print(frameRate)
import time
start = time.time()
import csv
with open('people.csv', 'w', ) as csvfile:
    peoplewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    while cap.isOpened():
        curr_frame = cap.get(1)
        print("frame: ", curr_frame)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # if curr_frame != 0:
        #     print("elapsed: ", fps/curr_frame)
        # import copy
        # raw_image = pipe.stdout.read(640*360*3) # read 1280*720*3 bytes (= 1 frame)
        # frame =  numpy.fromstring(raw_image, dtype='uint8').reshape((360,640,3))

        ret, frame = cap.read()

        # frame1 = frame.clone()
        # frame1 = np.array(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = imutils.resize(gray )

        (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4),
                padding=(8, 8), scale=1.05)
        
        # for (x, y, w, h) in rects:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # Write to csv, first column: frame number, 2nd column: no. of peoples
        peoplewriter.writerow([cap.get(1), len(pick)])
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
        # cv2.imshow('orig', frame)
        # cv2.imshow('non_max', frame)

        # index += 1
        # if index == 500:
        #     break
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()