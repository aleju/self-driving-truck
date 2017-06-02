from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from train_steering_wheel.train import (
    MODEL_HEIGHT as CNN_MODEL_HEIGHT,
    MODEL_HEIGHT as CNN_MODEL_WIDTH,
    ANGLE_BIN_SIZE as CNN_ANGLE_BIN_SIZE,
    extract_steering_wheel_image as cnn_extract_steering_wheel_image,
    downscale_image as cnn_downscale_image
)
from train_steering_wheel import models as cnn_models
from lib import util
from config import Config

from scipy import misc, ndimage
import cv2
import imgaug as ia
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage import filters
from skimage import morphology
import math
import torch

#cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)

STEERING_WHEEL_TRACKER_CNN_FP = os.path.join(Config.MAIN_DIR, "train_steering_wheel/steering_wheel.tar")

class SteeringWheelTrackerCNN(object):
    def __init__(self, default_angle=0):
        assert os.path.isfile(STEERING_WHEEL_TRACKER_CNN_FP)
        checkpoint = torch.load(STEERING_WHEEL_TRACKER_CNN_FP)
        self.model = cnn_models.SteeringWheelTrackerCNNModel()
        self.model.load_state_dict(checkpoint["tracker_cnn_state_dict"])
        self.model.cuda(Config.GPU)
        self.model.eval()

        self.default_angle = default_angle

        self.reset()

    def reset(self):
        self.last_match = None
        self.last_angle = self.default_angle
        self.last_angle_raw = self.default_angle
        self.overflow_degrees = 45
        self.overflow_max_count = 3
        self.overflow_counter = 0

    def estimate_angle(self, image):
        #from scipy import misc
        subimg = cnn_extract_steering_wheel_image(image)
        #misc.imshow(subimg)
        subimg = cnn_downscale_image(subimg)
        #misc.imshow(subimg)
        angle_raw_bins = self.model.forward_image(subimg, volatile=True, requires_grad=False, gpu=Config.GPU, softmax=True)
        angle_raw_bins = angle_raw_bins.data[0].cpu().numpy()
        angle_raw_bin = np.argmax(angle_raw_bins)
        #print(angle_raw_bins.data.cpu().numpy())

        """
        angle_raw_center = angle_raw_bin * CNN_ANGLE_BIN_SIZE + CNN_ANGLE_BIN_SIZE * 0.5 - 180
        angle_raw_left = angle_raw_center - CNN_ANGLE_BIN_SIZE
        angle_raw_right = angle_raw_center + CNN_ANGLE_BIN_SIZE
        angle_raw_center_p = angle_raw_bins[angle_raw_bin]
        angle_raw_left_p = angle_raw_bins[angle_raw_bin-1] if angle_raw_bin-1 > 0 else 0
        angle_raw_right_p = angle_raw_bins[angle_raw_bin+1] if angle_raw_bin+1 < angle_raw_bins.shape[0] else 0

        angle_raw = angle_raw_left_p * angle_raw_left + angle_raw_center_p * angle_raw_center + angle_raw_right_p * angle_raw_right
        """
        angle_raw = angle_raw_bin * CNN_ANGLE_BIN_SIZE + CNN_ANGLE_BIN_SIZE * 0.5 - 180

        #print(angle_raw)
        possible_angles = [angle_raw]
        if angle_raw < 0:
            possible_angles.append(180+(180-abs(angle_raw)))
            possible_angles.append(-360-abs(angle_raw))
        if angle_raw > 0:
            possible_angles.append(-180-(180-abs(angle_raw)))
            possible_angles.append(360+abs(angle_raw))
        possible_angles_dist = [(a, abs(self.last_angle - a)) for a in possible_angles]
        possible_angles_dist_sort = sorted(possible_angles_dist, key=lambda t: t[1])
        angle = possible_angles_dist_sort[0][0]

        if angle > Config.STEERING_WHEEL_MAX:
            angle = angle - 360
        elif angle < Config.STEERING_WHEEL_MIN:
            angle = angle + 360

        if abs(angle - self.last_angle) >= self.overflow_degrees:
            if self.overflow_counter >= self.overflow_max_count:
                self.last_angle = angle
                self.last_angle_raw = angle_raw
                self.overflow_counter = 0
            else:
                angle = self.last_angle
                angle_raw = self.last_angle_raw
            self.overflow_counter += 1
        else:
            self.last_angle = angle
            self.last_angle_raw = angle_raw
            self.overflow_counter = 0

        return angle, angle_raw

class SteeringWheelTracker(object):
    def __init__(self, default_angle=0):
        self.default_angle = default_angle
        self.max_angle = Config.STEERING_WHEEL_MAX # wheel can be turned by roughly +/- 360+90 degrees, add 40deg tolerance
        self.reset()

    def reset(self):
        self.last_match = None
        self.last_angle = self.default_angle
        self.last_angle_raw1 = self.default_angle
        self.last_angle_raw2 = self.default_angle
        self.overflow_degrees = 45
        self.overflow_max_count = 3
        self.overflow_counter = 0

    def estimate_angle(self, image, visualize=False):
        if visualize:
            match, image_viz = estimate_by_last_match(image, last_match=self.last_match, visualize=visualize)
            #if match is None and self.last_match is not None:
            #    match, image_viz = estimate_by_last_match(image, last_match=None, visualize=visualize)
        else:
            match = estimate_by_last_match(image, last_match=self.last_match, visualize=visualize)
            #if match is None and self.last_match is not None:
            #    match = estimate_by_last_match(image, last_match=None, visualize=visualize)

        if match is None:
            #print("no match")
            self.reset()
        else:
            v1 = np.float32([1, 0])
            #print(match)
            v2a = np.float32([
                match["right_x"] - match["left_x"],
                match["right_y"] - match["left_y"]
            ])
            v2b = np.float32([
                match["left_x"] - match["right_x"],
                match["left_y"] - match["right_y"]
            ])
            angle_raw1 = get_angle(v1, v2a)
            angle_raw2 = get_angle(v1, v2b)
            #print("early angle_raw1", angle_raw1, "angle_raw2", angle_raw2)
            if angle_raw1 > 180:
                angle_raw1 = -(360 - angle_raw1)
            if angle_raw2 > 180:
                angle_raw2 = -(360 - angle_raw2)
            #distance_from_90 = (angle_raw1 % 90)
            #p_flipped = 1 - (min(distance_from_90, 90-distance_from_90) / 45)
            #flip_distance = min(abs(abs(angle_raw1) - 270), abs(abs(angle_raw1) - 90)) / 90
            #maxp = 0.95
            #p_flipped = np.clip(1 - flip_distance, 0, maxp)

            # maxp and p_flipped is legacy stuff, can be removed
            maxp = 0.95
            p_flipped = maxp
            possible_angles = [
                (-360+angle_raw1, maxp),
                (-360+angle_raw2, p_flipped),
                (angle_raw1, maxp),
                (angle_raw2, p_flipped),
                (360+angle_raw1, maxp),
                (360+angle_raw2, p_flipped),
            ]
            possible_angles = [(r, p) for (r, p) in possible_angles if r < self.max_angle]
            possible_angles_dist = [(poss, abs(poss - self.last_angle), p) for (poss, p) in possible_angles]
            possible_angles_dist_sort = sorted(possible_angles_dist, key=lambda t: t[1]*(1-t[2]))
            angle = possible_angles_dist_sort[0][0]

            #print("angle_raw1 %.2f | angle_raw2 %.2f | after add %.2f | poss %s | poss sort %s" % (angle_raw1, angle_raw2, angle, str(possible_angles_dist), str(possible_angles_dist_sort)))
            #print("after add", angle)

            if abs(angle - self.last_angle) >= self.overflow_degrees:
                if self.overflow_counter >= self.overflow_max_count:
                    self.last_match = match
                    self.last_angle = angle
                    self.last_angle_raw1 = angle_raw1
                    self.last_angle_raw2 = angle_raw2
                    self.overflow_counter = 0
                else:
                    self.last_match = None
                    angle = self.last_angle
                    angle_raw1 = self.last_angle_raw1
                    angle_raw2 = self.last_angle_raw2
                self.overflow_counter += 1
            else:
                self.last_match = match
                self.last_angle = angle
                self.last_angle_raw1 = angle_raw1
                self.last_angle_raw2 = angle_raw2
                self.overflow_counter = 0
        if visualize:
            return self.last_angle, (self.last_angle_raw1, self.last_angle_raw2), image_viz
        else:
            return self.last_angle, (self.last_angle_raw1, self.last_angle_raw2)


def get_angle(v1, v2):
    v1_theta = math.atan2(v1[1], v1[0])
    v2_theta = math.atan2(v2[1], v2[0])
    r = (v2_theta - v1_theta) * (180.0 / math.pi)
    if r < 0:
        r += 360.0
    return r

def estimate_by_last_match(image, last_match=None, visualize=False):
    if last_match is None:
        return estimate(image, visualize=visualize)
    else:
        #search_rect = (
        #    last_match["min_x"], last_match["min_y"],
        #    last_match["max_x"], last_match["max_y"]
        #)
        search_rect = None
        #expected_position = (last_match["center_x"], last_match["center_y"])
        expected_position = None
        return estimate(image, expected_position=expected_position, search_rect=search_rect, visualize=visualize)

#@profile
def estimate(image, expected_position=None, search_rect=None, search_rect_border=0.05, expected_pixels=(10, 200), optimal_size=90, visualize=False):
    downscale_factor = 1 # legacy stuff
    h, w = image.shape[0:2]

    if search_rect is not None:
        x1, y1, x2, y2 = search_rect[0], search_rect[1], search_rect[2], search_rect[3]
        if search_rect_border > 0:
            clip = np.clip
            bx = int(w * search_rect_border)
            by = int(h * search_rect_border)
            x1 = clip(x1 - bx, 0, w-2)
            y1 = clip(y1 - by, 0, h-2)
            x2 = clip(x2 + bx, x1, w-1)
            y2 = clip(y2 + by, y1, h-1)
    else:
        # full wheel: x1=440, x2=870, y1=440, y2=720
        # wheel w=440, h=270
        x1 = int(w * (480/1280))
        x2 = int(w * (830/1280))
        y1 = int(h * (520/720))
        y2 = int(h * (720/720))
    rect_h = y2 - y1
    rect_w = x2 - x1

    if expected_position is None:
        expected_position = (
            int(w * (646/1280)),
            int(h * (684/720))
        )

    img_wheel = image[y1:y2+1, x1:x2+1, :]
    img_wheel_rs = img_wheel
    img_wheel_rsy = cv2.cvtColor(img_wheel_rs, cv2.COLOR_RGB2GRAY)
    expected_position_rs = (
        int((expected_position[0]-x1) * downscale_factor),
        int((expected_position[1]-y1) * downscale_factor)
    )

    #thresh_mask = filters.threshold_li(img_wheel_rsy)
    thresh_mask = filters.threshold_isodata(img_wheel_rsy)
    thresh = img_wheel_rsy > thresh_mask #40
    #cv2.imshow("thresh", thresh.astype(np.uint8)*255)
    #cv2.waitKey(10)
    thresh = morphology.binary_dilation(thresh, morphology.square(3))

    img_labeled, num_labels = morphology.label(
        thresh, background=0, connectivity=1, return_num=True
    )

    segments = []
    for label in range(1, num_labels+1):
        img_seg = (img_labeled == label)
        (yy, xx) = np.nonzero(img_seg)

        # size of correct segment is around 60 pixels without dilation and 90 with dilation
        # position is at around x=21, y=13
        # (both numbers for screenshots after jpg-compression/decompression at 1/4 the original
        # size, i.e. 1280/4 x 720/4)
        if expected_pixels[0] <= len(yy) <= expected_pixels[1]:
            center_x = np.average(xx)
            center_y = np.average(yy)

            # euclidean distance to expected position
            # segments which's center is at the expected position get a 0
            # segments which a further away get higher values
            dist_pos = 0.1 * math.sqrt((center_x - expected_position_rs[0]) ** 2 + (center_y - expected_position_rs[1])**2)

            # distance to optimal size (number of pixels)
            # segments that have the same number of pixels as the expected size
            # get a 0, segments with 50pecent more/less pixels get a 0
            dist_size = np.clip(
                1/(optimal_size*0.5) * abs(len(yy) - optimal_size),
                0, 1
            )
            dist = dist_pos + dist_size

            segments.append({
                "xx": xx,
                "yy": yy,
                "center_x": center_x,
                "center_y": center_y,
                "dist_pos": dist_pos,
                "dist_size": dist_size,
                "dist": dist,
                "img_seg": img_seg
            })

    if len(segments) == 0:
        return (None, None) if visualize else None

    segments = sorted(segments, key=lambda d: d["dist"])
    best_match = segments[0]
    xx = x1 + (best_match["xx"].astype(np.float32) * (1/downscale_factor)).astype(np.int32)
    yy = y1 + (best_match["yy"].astype(np.float32) * (1/downscale_factor)).astype(np.int32)

    image_segment = best_match["img_seg"]
    image_segment = morphology.binary_erosion(image_segment, morphology.square(3))

    cx, cy = int(best_match["center_x"]), int(best_match["center_y"])
    sy, sx = 10, 10
    hx1 = np.clip(cx - sx, 0, image_segment.shape[1])
    hx2 = np.clip(cx + sx + 1, 0, image_segment.shape[1])
    hy1 = np.clip(cy - sy, 0, image_segment.shape[0])
    hy2 = np.clip(cy + sy + 1, 0, image_segment.shape[0])
    hough_segment = image_segment[hy1:hy2, hx1:hx2]
    h, theta, d = hough_line(hough_segment)
    if len(h) == 0:
        return (None, None) if visualize else None

    hspaces, angles, dists = hough_line_peaks(h, theta, d, num_peaks=1)
    if len(hspaces) == 0:
        return (None, None) if visualize else None

    hspace, angle, dist = hspaces[0], angles[0], dists[0]
    line_y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    line_y1 = (dist - hough_segment.shape[1] * np.cos(angle)) / np.sin(angle)
    slope = (line_y1 - line_y0) / (hx2 - hx1)

    left_x = cx - 3
    right_x = cx + 3
    left_y = cy + (-3) * slope
    right_y = cy + 3 * slope

    #print("x1 %d x2 %d y1 %d y2 %d | cx %d cy %d | hx1 %d hx2 %d hy1 %d hy2 %d | line_y0 %.2f line_y1 %.2f | left_x %d right_x %d left_y %d right_y %d | hs %s | is %s" % (
    #    x1, x2, y1, y2, cx, cy, hx1, hx2, hy1, hy2, line_y0, line_y1, left_x, right_x, left_y, right_y, hough_segment.shape, image_segment.shape
    #))

    #fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    #ax.imshow(image_segment, cmap=plt.cm.gray)
    #ax.plot([left_x, right_x], [left_y, right_y], "-r")
    #plt.show()

    best_match["min_x"] = int(np.min(xx))
    best_match["max_x"] = int(np.max(xx))
    best_match["min_y"] = int(np.min(yy))
    best_match["max_y"] = int(np.max(yy))
    best_match["center_x"] = x1 + (best_match["center_x"] * (1/downscale_factor))
    best_match["center_y"] = y1 + (best_match["center_y"] * (1/downscale_factor))
    best_match["left_x"] = x1 + (left_x * (1/downscale_factor))
    best_match["right_x"] = x1 + (right_x * (1/downscale_factor))
    best_match["left_y"] = y1 + (left_y * (1/downscale_factor))
    best_match["right_y"] = y1 + (right_y * (1/downscale_factor))

    if visualize:
        upf = 2
        image_viz = ia.imresize_single_image(np.copy(image), (image.shape[0]*upf, image.shape[1]*upf))
        image_viz = util.draw_point(image_viz, x=int(x1)*upf, y=int(y1)*upf, size=7, color=[255, 0, 0])
        image_viz = util.draw_point(image_viz, x=(int(x2)-1)*upf, y=int(y1)*upf, size=7, color=[255, 0, 0])
        image_viz = util.draw_point(image_viz, x=(int(x2)-1)*upf, y=(int(y2)-1)*upf, size=7, color=[255, 0, 0])
        image_viz = util.draw_point(image_viz, x=int(x1)*upf, y=(int(y2)-1)*upf, size=7, color=[255, 0, 0])
        image_viz = util.draw_point(image_viz, x=int(expected_position[0])*upf, y=int(expected_position[1])*upf, size=7, color=[0, 0, 255])
        image_viz = util.draw_point(image_viz, x=best_match["min_x"]*upf, y=best_match["min_y"]*upf, size=7, color=[128, 0, 0])
        image_viz = util.draw_point(image_viz, x=(best_match["max_x"]-1)*upf, y=best_match["min_y"]*upf, size=7, color=[128, 0, 0])
        image_viz = util.draw_point(image_viz, x=(best_match["max_x"]-1)*upf, y=(best_match["max_y"]-1)*upf, size=7, color=[128, 0, 0])
        image_viz = util.draw_point(image_viz, x=best_match["min_x"]*upf, y=(best_match["max_y"]-1)*upf, size=7, color=[128, 0, 0])
        image_viz = util.draw_point(image_viz, x=int(best_match["center_x"])*upf, y=int(best_match["center_y"])*upf, size=7, color=[0, 0, 128])
        image_viz = util.draw_point(image_viz, x=int(best_match["left_x"])*upf, y=int(best_match["left_y"])*upf, size=7, color=[0, 255, 0])
        image_viz = util.draw_point(image_viz, x=int(best_match["right_x"])*upf, y=int(best_match["right_y"])*upf, size=7, color=[0, 128, 0])
        return best_match, image_viz
    else:
        return best_match
