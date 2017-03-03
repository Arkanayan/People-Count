from imutils.object_detection import non_max_suppression
import numpy as np
import numpy
import imutils
import cv2
import subprocess as sp



# VIDEO_URL = "https://manifest.googlevideo.com/api/manifest/hls_playlist/id/l5vUW5ZRHK0.0/itag/94/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/cmbypass/yes/goi/160/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D135/hls_chunk_host/r2---sn-5jucgv5qc5oq-cagz.googlevideo.com/gcr/in/playlist_type/DVR/mm/32/mn/sn-5jucgv5qc5oq-cagz/ms/lv/mv/m/pl/19/dover/6/upn/WqYvM_XDWOM/beids/%5B9452306%5D/mt/1488196396/ip/106.51.66.100/ipbits/0/expire/1488218103/sparams/ip,ipbits,expire,id,itag,source,requiressl,ratebypass,live,cmbypass,goi,sgoap,sgovp,hls_chunk_host,gcr,playlist_type,mm,mn,ms,mv,pl/signature/252042428928E4725F652130CC6C7E97B35F0B87.339052FF9F19DC67DFCD137B184856DF60CCC64D/key/dg_yt0/playlist/index.m3u8"
# VIDEO_URL = 'https://r2---sn-5jucgv5qc5oq-cagz.googlevideo.com/videoplayback/id/853532cc5e2ff4a0/itag/134/source/youtube/requiressl/yes/mm/31/initcwndbps/2950000/mn/sn-5jucgv5qc5oq-cagz/pcm2cms/yes/pl/19/ms/au/mv/m/ratebypass/yes/mime/video%2Fmp4/otfp/1/gir/yes/clen/11779214/lmt/1459875326930074/dur/156.990/signature/0E3EE9CCFB826CAF4FE4D403BAB00B4F4150E6FA.989C85682F9DB78BB15AFD5BDA17AD37CBDF31D1/key/dg_yt0/mt/1488534207/beids/%5B9465692%5D/upn/GTh4RMqGqlc/ip/106.51.66.100/ipbits/0/expire/1488555913/sparams/ip,ipbits,expire,id,itag,source,requiressl,mm,initcwndbps,mn,pcm2cms,pl,ms,mv,ratebypass,mime,otfp,gir,clen,lmt,dur/'

# FFMPEG_BIN = '/usr/bin/ffmpeg'

# pipe = sp.Popen([ FFMPEG_BIN, "-i", VIDEO_URL,

#            "-an",   # disable audio
#            "-f", "image2pipe",
#            "-pix_fmt", "bgr24",
#            "-vcodec", "rawvideo", "-"],
#            stdin = sp.PIPE, stdout = sp.PIPE, bufsize=10**9)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))

cap = cv2.VideoCapture('/home/arka/MantraLabs/Emotion/random_people_walk.webm')

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