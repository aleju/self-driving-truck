from __future__ import division, print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib import replay_memory
from lib import steering_wheel as swlib

import numpy as np
import time
import sqlite3
from scipy import misc
import cv2
import imgaug as ia

if sys.version_info[0] == 3:
    raw_input = input

try:
    xrange
except NameError:
    xrange = range

MAX_TIMEDIFF_MS = 500
MAX_PIXELDIFF = 800

def main():
    #cv2.namedWindow("overview", cv2.WINDOW_NORMAL)

    key = raw_input("This will add the approximated steering wheel location to all states in the database. Press 'y' to continue.")
    assert key == "y"

    tracker_classical = swlib.SteeringWheelTracker()
    tracker_cnn = swlib.SteeringWheelTrackerCNN()
    memories = [
        (replay_memory.ReplayMemory.create_instance_reinforced(val=False), 10**6),
        (replay_memory.ReplayMemory.create_instance_reinforced(val=True), 10**6),
        (replay_memory.ReplayMemory.create_instance_supervised(val=False), 60*10),
        (replay_memory.ReplayMemory.create_instance_supervised(val=True), 60*10)
    ]
    for memory, memory_reset_every in memories:
        add_columns(memory)

        last_datetime = None
        last_scr = None
        last_was_new_scene = False
        last_autoreset = 0

        for idx in xrange(memory.id_min, memory.id_max+1):
            state = memory.get_state_by_id(idx)
            if state.steering_wheel_classical is None or state.steering_wheel_cnn is None:
                timediff_ms = 0 if last_datetime is None else (state.from_datetime - last_datetime).total_seconds() * 1000
                """
                if last_scr is not None:
                    from scipy import misc
                    print(state.screenshot_rs.shape, last_scr.shape)
                    print(state.screenshot_rs[20:-30, ...] - last_scr[20:-30, ...])
                    misc.imshow(state.screenshot_rs[20:-30, ...])
                """
                if last_scr is None:
                    #pixeldiff = 0
                    same_scene = True
                else:
                    img1 = np.average(np.average(state.screenshot_rs[20:-30, ...], axis=1), axis=0)
                    img2 = np.average(np.average(last_scr[20:-30, ...], axis=1), axis=0)
                    #pixeldiff = 0 if last_scr is None else np.average(np.abs(img1 - img2))
                    same_scene = screens_show_same_scene(last_scr, state.screenshot_rs)

                if last_was_new_scene:
                    print("resetting (last was new)")
                    tracker_classical.reset()
                    tracker_cnn.reset()
                    last_was_new_scene = False
                elif last_autoreset > memory_reset_every:
                    print("resetting (auto)...")
                    tracker_classical.reset()
                    tracker_cnn.reset()
                    last_autoreset = 0
                    last_was_new_scene = False
                elif timediff_ms > MAX_TIMEDIFF_MS or not same_scene:
                    print("resetting (new scene)...")
                    tracker_classical.reset()
                    tracker_cnn.reset()
                    last_was_new_scene = True
                else:
                    last_was_new_scene = False
                (wheel_deg_classical, (wheel_deg_raw1_classical, wheel_deg_raw2_classical)), time_req_classical = track(tracker_classical, state.screenshot_rs)
                (wheel_deg_cnn, wheel_deg_raw_cnn), time_req_cnn = track(tracker_cnn, state.screenshot_rs)

                img_viz = np.copy(state.screenshot_rs)
                img_viz = util.draw_text(img_viz, x=0, y=0, text="%.2f | %.2f / %.2f\n%.2f | %.2f" % (wheel_deg_classical, wheel_deg_raw1_classical, wheel_deg_raw2_classical, wheel_deg_cnn, wheel_deg_raw_cnn))
                #cv2.imshow("overview", img_viz[:,:,::-1])
                #cv2.waitKey(150)
                #if timediff_ms > 1000:
                #    from scipy import misc
                #    misc.imshow(np.hstack([last_scr, state.screenshot_rs]))

                state.steering_wheel_classical = wheel_deg_classical
                state.steering_wheel_raw_one_classical = wheel_deg_raw1_classical
                state.steering_wheel_raw_two_classical = wheel_deg_raw2_classical
                state.steering_wheel_cnn = wheel_deg_cnn
                state.steering_wheel_raw_cnn = wheel_deg_raw_cnn
                memory.update_state(state.idx, state, commit=False)

                last_datetime = state.from_datetime
                last_scr = state.screenshot_rs
                last_autoreset += 1
            else:
                timediff_ms = 0
                wheel_deg_classical = 0
                wheel_deg_cnn = 0
                time_req_classical = 0
                time_req_cnn = 0

            if idx % 1000 == 0:
                print("#%d/%d timediff=%d wheel_cl=%.2f wheel_cnn=%.2f in_cl=%.4fs in_cnn=%.4f" % (idx, memory.id_max, timediff_ms, wheel_deg_classical, wheel_deg_cnn, time_req_classical, time_req_cnn))
                memory.commit()

        memory.commit()

def track(tracker, scr):
    time_start = time.time()
    result = tracker.estimate_angle(scr)
    time_end = time.time()
    return result, time_end - time_start

def screens_show_same_scene(scr1, scr2):
    scr1 = ia.imresize_single_image(scr1, (50, 70))
    scr2 = ia.imresize_single_image(scr2, (50, 70))
    hist1 = cv2.calcHist([scr1[...,0], scr1[...,1], scr1[...,2]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([scr2[...,0], scr2[...,1], scr2[...,2]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    diff = np.sum(np.abs(hist1 - hist2))
    return diff <= MAX_PIXELDIFF

def add_columns(memory):
    column_names = [
        "steering_wheel",
        "steering_wheel_raw_one",
        "steering_wheel_raw_two",
        "steering_wheel_cnn",
        "steering_wheel_raw_cnn"
    ]
    for name in column_names:
        try:
            memory.conn.execute("ALTER TABLE states ADD COLUMN %s REAL" % (name,))
        except sqlite3.OperationalError as exc:
            if "duplicate column" in str(exc):
                print("Column '%s' already added to table." % (name,))
            else:
                raise exc

    memory.commit()

if __name__ == "__main__":
    main()
