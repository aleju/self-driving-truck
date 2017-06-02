from __future__ import print_function, division
import cv2
import numpy as np
import sys
from lib import speed as speedlib

if sys.version_info[0] == 3:
    raw_input = input

def main():
    cv2.namedWindow("segment", cv2.WINDOW_NORMAL)

    db = speedlib.SpeedSegmentsDatabase.get_instance()
    for key in db.segments:
        seg = db.segments[key]
        cv2.imshow("segment", seg.get_image())
        cv2.waitKey(100)
        label = raw_input("Enter label [current label: '%s']: " % (str(seg.label),))
        seg.label = label
        db.save()

    print("Finished.")

if __name__ == "__main__":
    main()
