from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import windowhandling
import screenshot
import numpy as np
from scipy import misc, ndimage
import cv2
import util
from config import Config
import time

class ETS2Window(object):
    def __init__(self, win_id, coordinates=None):
        self.win_id = win_id
        if coordinates is None:
            self.coordinates = self.get_coordinates()
        else:
            self.coordinates = coordinates

        self.offence_ff_image = ndimage.imread(Config.OFFENCE_FF_IMAGE, mode="RGB")
        self.damage_single_digit_image = ndimage.imread(Config.DAMAGE_SINGLE_DIGIT_IMAGE, mode="RGB")
        self.damage_double_digit_image = ndimage.imread(Config.DAMAGE_DOUBLE_DIGIT_IMAGE, mode="RGB")
        self.reverse_image = ndimage.imread(Config.REVERSE_IMAGE, mode="RGB")

    def is_activated(self):
        active_win_id = windowhandling.get_active_window_id()
        return active_win_id == self.win_id

    def get_coordinates(self):
        # x1, y1, x2, y2
        return windowhandling.get_window_coordinates(self.win_id)

    def get_image(self):
        x1, y1, x2, y2 = self.coordinates
        return screenshot.make_screenshot(x1=x1, y1=y1, x2=x2, y2=y2)

    def get_speed_image(self, scr=None):
        scr = scr if scr is not None else self.get_image()
        return get_speed_image(scr)

    def get_route_advisor_image(self, scr=None):
        scr = scr if scr is not None else self.get_image()
        return get_route_advisor_image(scr)

    def is_route_advisor_visible(self, scr, threshold=2):
        ra = self.get_route_advisor_image(scr)
        #misc.imshow(ra)
        #print("ra_shape", ra.shape)
        #assert ra.shape == (9, 3)
        #ra1d = np.average(ra, axis=2)
        ra_rows = np.average(ra, axis=1)
        #print("ra_rows.shape", ra_rows.shape)
        #print("ra_rows", ra_rows)
        expected = np.array([[ 25.33766234,  22.92207792,  21.94805195],
                    [ 31.79220779,  29.50649351,  28.58441558],
                    [ 70.32467532,  68.96103896,  68.32467532],
                    [ 63.51948052,  61.97402597,  61.2987013 ],
                    [ 66.20779221,  64.72727273,  64.14285714],
                    [ 64.12987013,  62.51948052,  62.01298701],
                    [ 60.61038961,  58.94805195,  58.20779221],
                    [ 65.31168831,  63.74025974,  63.12987013],
                    [ 18.18181818,  15.66233766,  14.51948052]], dtype=np.float32)

        #print("expected", ra_rows)
        #print("diff", ra_rows - expected)

        # evade brightness differences
        observed_normalized = ra_rows - np.average(ra_rows)
        expected_normalized = expected - np.average(expected)

        #print("observed_normalized", observed_normalized)
        #print("expected_normalized", expected_normalized)

        dist = np.abs(observed_normalized - expected_normalized)
        dist_avg = np.average(dist)
        #print("dist", dist)
        #print("dist_avg", dist_avg)
        return dist_avg < threshold

    # quite close scores even for some non-paused images
    def is_paused(self, scr, threshold=4):
        """
        Pause mode shows a message at the top center and at
        the bottom right (above the route advisor)
        Top center message:
           Yellowish string "F1"
             top left     | x=605 y=164
             top right    | x=617 y=164
             bottom left  | x=605 y=174
             bottom right | x=617 y=174
           Yellowish string "Pause"
             top left     | x=639 y=164
             top right    | x=675 y=164
             bottom left  | x=639 y=174
             bottom right | x=675 y=174
        Bottom right message
           Yellowish string "Advisor"
             top left     | x=1100, y=433
             top right    | x=1146, y=433
             bottom left  | x=1100, y=440
             bottom right | x=1146, y=440
        """
        y1 = 430
        y2 = 439 + 1
        x1 = 1100
        x2 = 1148 + 1
        bt_right_img = scr[y1:y2, x1:x2, :]
        #misc.imsave("images/pause_bottom_right_advisor.png", bt_right_img)
        #misc.imshow(bt_right_img)
        #print(np.average(bt_right_img, axis=1))
        """
        import cv2
        import matplotlib.pyplot as plt
        chans = cv2.split(bt_right_img)
        for col, chan in zip(["red", "green", "blue"], chans):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])

            print("chan=", col)
            print(hist)

            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.show()
        """
        expected = [[  42.40816327,   34.91836735,    3.26530612],
                    [  42.63265306,   34.57142857,    3.10204082],
                    [  48.06122449,   37.08163265,    3.28571429],
                    [ 130.0,          91.02040816,    2.18367347],
                    [ 130.36734694,   91.14285714,    2.44897959],
                    [ 112.44897959,   78.85714286,    3.24489796],
                    [ 131.65306122,   91.57142857,    3.20408163],
                    [ 120.75510204,   84.30612245,    4.        ],
                    [ 127.97959184,   89.28571429,    3.59183673],
                    [ 111.57142857,   78.24489796,    4.69387755]]
        observed = np.average(bt_right_img, axis=1)
        expected_normalized = expected - np.average(expected, axis=0)
        observed_normalized = observed - np.average(observed, axis=0)
        dist = np.abs(observed_normalized - expected_normalized)
        dist_avg = np.average(dist)
        #print("is_paused", dist_avg)
        return dist_avg < threshold

    # seems to also fire for "You have a new mail"
    def is_offence_shown(self, scr, threshold=0.97):
        time_start = time.time()
        y1 = 584
        y2 = 591 + 1
        x1 = 1119
        x2 = 1180 + 1
        offence_area = scr[y1:y2, x1:x2, :]
        x, y, score = util.template_match(needle=self.offence_ff_image, haystack=offence_area)
        time_req = time.time() - time_start
        #print("in %.4fs" % (time_req,))
        #print("is_offence_shown", x, y, score)
        #misc.imshow(offence_area)
        return score >= threshold

    def is_damage_shown(self, scr, threshold=2):
        time_start = time.time()
        coords_double = (1092, 567)
        coords_single = (1095, 567)
        h_double, w_double = self.damage_double_digit_image.shape[0:2]
        h_single, w_single = self.damage_single_digit_image.shape[0:2]
        for (w, h), (x1, y1), expected in zip([(w_double, h_double), (w_single, h_single)], [coords_double, coords_single], [self.damage_double_digit_image, self.damage_single_digit_image]):
            observed = scr[y1:y1+h, x1:x1+w, :]
            expected_normalized = expected - np.average(expected, axis=(0, 1))
            observed_normalized = observed - np.average(observed, axis=(0, 1))
            dist = np.abs(observed_normalized - expected_normalized)
            dist_avg = np.average(dist)
            #print("is_damage_shown", dist_avg)
            if dist_avg < threshold:
                return True
        time_req = time.time() - time_start
        return False

    def is_reverse(self, scr, threshold=2):
        expected = self.reverse_image
        x1 = 1070
        y1 = 481
        #x2 = 1076
        #y2 = 489
        h, w = expected.shape[0:2]
        observed = scr[y1:y1+h, x1:x1+w, :]
        expected_normalized = expected - np.average(expected, axis=(0, 1))
        observed_normalized = observed - np.average(observed, axis=(0, 1))
        dist = np.abs(observed_normalized - expected_normalized)
        dist_avg = np.average(dist)
        #print("is_reverse", dist_avg)
        return dist_avg < threshold

    def keys(self, up=None, press=None, down=None):
        #windowhandling.sendkeys(self.win_id, up=up, press=press, down=down)
        windowhandling.sendkeys_pykb(up=up, tap=press, down=down)

def find_ets2_window_id():
    win_ids = windowhandling.find_window_ids("Truck")
    # for some reason, there always seem to be listed multiple ETS2 windows
    # only one of them is correct and has height and width >1,
    # all other's height and width seems to always be 1
    # so we find here all windows with name ETS2 and pick the one with the highest
    # size (height*width)
    candidates = []
    for win_id in win_ids:
        winname = windowhandling.get_window_name(win_id)
        if winname == "Euro Truck Simulator 2":
            #print("window %d has title 'Euro Truck Simulator 2'" % (win_id,))
            x1, y1, x2, y2 = windowhandling.get_window_coordinates(win_id)
            h, w = y2 - y1, x2 - x1
            size = h * w
            candidates.append((win_id, size))
    if len(candidates) == 0:
        return None
    else:
        candidates = sorted(candidates, key=lambda t: t[1], reverse=True)
        best = candidates[0]
        if best[1] > 1:
            return best[0]
        else:
            return None

def get_speed_image_coords(scr, mode="extended", border=(2, 2, 2, 2)):
    h, w = scr.shape[0:2]
    assert w == 1280 and h == 720

    """
    if mode == "whole":
        x1_speed_rel = (994 - border[3]) / 1280
        x2_speed_rel = (1052 + border[1]) / 1280
        y1_speed_rel = (477 - border[0]) / 720
        y2_speed_rel = (494 + border[2]) / 720
    elif mode == "small":
        x1_speed_rel = (1005 - border[3]) / 1280
        x2_speed_rel = (1017 + border[1]) / 1280
        y1_speed_rel = (481 - border[0]) / 720
        y2_speed_rel = (489 + border[2]) / 720

    x1_speed = int(x1_speed_rel * w)
    x2_speed = int(x2_speed_rel * w)
    y1_speed = int(y1_speed_rel * h)
    y2_speed = int(y2_speed_rel * h)
    """

    if mode == "whole":
        x1 = 994 - border[3]
        x2 = 1052 + border[1]
        y1 = 477 - border[0]
        y2 = 494 + border[2]
    elif mode == "extended":
        x1 = 997 - border[3]
        x2 = 1048 + border[1]
        y1 = 481 - border[0]
        y2 = 489 + border[2]
    elif mode == "small":
        x1 = 1005 - border[3]
        x2 = 1017 + border[1]
        y1 = 481 - border[0]
        y2 = 489 + border[2]

    return x1, y1, x2, y2

def get_route_advisor_image_coords(scr):
    assert scr.shape[0:2] == (720, 1280)
    x1 = 1086
    x2 = 1162
    y1 = 689
    y2 = 697
    return x1, y1, x2, y2

def get_route_advisor_image(scr):
    x1, y1, x2, y2 = get_route_advisor_image_coords(scr)
    return scr[y1:y2+1, x1:x2+1, :]

def get_speed_image(scr):
    x1, y1, x2, y2 = get_speed_image_coords(scr)
    return scr[y1:y2+1, x1:x2+1, :]
