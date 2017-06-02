from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib import windowhandling
from lib import ets2window
from lib import ets2game
from lib import screenshot
from lib import speed as speedlib

import time
import cv2
import numpy as np
import os

def main():
    cv2.namedWindow("speed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("speed_bin", cv2.WINDOW_NORMAL)

    def on_route_advisor_visible(game, scr):
        speed_image = game.win.get_speed_image(scr)
        som = speedlib.SpeedOMeter(speed_image)

        speed_image_bin = som.get_postprocessed_image()

        cv2.imshow("speed", speed_image)
        cv2.imshow("speed_bin", speed_image_bin.astype(np.uint8)*255)
        cv2.waitKey(1)

        segments = som.split_to_segments()
        print("Found %d segments" % (len(segments),), [s.arr.shape for s in segments])

        for seg in segments:
            if not seg.is_in_database():
                speedlib.SpeedSegmentsDatabase.get_instance().add(seg)
                print("Added new segment with key: %s" % (seg.get_key(),))
            else:
                print("Segment already in database.")

    game = ets2game.ETS2Game()
    game.on_route_advisor_visible = on_route_advisor_visible
    game.run()

if __name__ == "__main__":
    main()
