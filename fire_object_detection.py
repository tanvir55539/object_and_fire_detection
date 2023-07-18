import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
import time
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth

def nothing(x): #----
    pass #-----

url = 'http://192.168.94.25/cam-hi.jpg'
# url = 'http://192.168.71.74/cam-hi.jpg'
#im = Nonef
# count = 0

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
folder = "1Z0jeH9tS40r3ykdo-yWBktN8RfUzInN3"

def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    l_h, l_s, l_v = 0, 0, 255
    u_h, u_s, u_v = 41, 100, 255
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)
        # _, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_b = np.array([l_h, l_s, l_v])
        u_b = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, l_b, u_b)

        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area > 2000:
                cv2.drawContours(frame, [c], -1, (255, 0, 0), 3)
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                cv2.putText(frame, "fire", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("live transmission", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("res", res)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def run2():
    count=0
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)##------------try this one leter by shifting upword
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        bbox, label, conf = cv.detect_common_objects(im)
        im = draw_bbox(im, bbox, label, conf)

        cv2.imshow('detection', im)
        key = cv2.waitKey(5)

        if "person" in label:
            count += 1
            t = str(count) + '.png'
            cv2.imwrite(t, im)
            print("image saved as: " + t)
            f = drive.CreateFile({'parents': [{'id': folder}], 'title': t})
            f.SetContentFile('1.png')
            f.Upload()
            print("image uploaded as: " + t)

        if key == ord('f'):
            run1()

        else:
            continue

        # if key == ord('f'):
        #     run1()

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executer:
        executer.submit(run2)
