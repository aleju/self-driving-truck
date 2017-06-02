"""See ets2window.py for reading out the speed-o-meter from screen."""
from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from skimage import morphology
import numpy as np
import cv2
import os

class SpeedSegmentsDatabase(object):
    instance = None

    def __init__(self, filepath=Config.SPEED_IMAGE_SEGMENTS_DB_FILEPATH):
        self.filepath = filepath
        self.segments = SpeedSegmentsDatabase.load_dict_from_file(filepath)

    @staticmethod
    def get_instance():
        if SpeedSegmentsDatabase.instance is None:
            SpeedSegmentsDatabase.instance = SpeedSegmentsDatabase()
        return SpeedSegmentsDatabase.instance

    @staticmethod
    def load_dict_from_file(filepath):
        if not os.path.isfile(filepath):
            print("No SpeedSegments database found at file '%s'. Initializing database empty." % (filepath,))
            return dict()
        else:
            lines = open(filepath, "r").readlines()
            #lines = [line.strip().split("\t") for line in lines]
            lines = [line.strip() for line in lines]
            segments = dict()
            nb_found = 0
            nb_labeled = 0
            #segments_found_keys = set()
            #segments_found = []
            for line in lines:
                seg = SpeedSegment.from_string(line)
                segments[seg.get_key()] = seg
                nb_found += 1
                nb_labeled += (seg.label is not None)
            """
            for h, w, vals in lines:
                seg = np.zeros((int(h) * int(w)), dtype=np.bool)
                for i in range(len(vals)):
                    seg[i] = int(vals[i])
                seg = seg.reshape((int(h), int(w)))
                seg_key = SpeedSegment.segment_image_to_key(seg)
                segments[seg_key] = seg
                #segments_found.append(seg)
                #segments_found_keys.add(seg_key)
                nb_found += 1
                nb_labeled += (seg.label is not None)
            #return segments_found_keys, segments_found
            """

            print("Loaded SpeedSegments database from file '%s' with %d segments, %d of which are labeled." % (filepath, nb_found, nb_labeled))
            return segments

    def save(self):
        with open(self.filename, "w") as f:
            for key in self.segments:
                segment = self.segments[key]

                #sr = segment.arr.astype(np.int32).ravel()
                #vals = []
                #for v in sr:
                #    vals.append(str(v))
                #label = segment.label if segment.label is not None else "?"
                #f.write("%s\t%d\t%d\t%s" % (label, segment.shape[0], segment.shape[1], "".join(vals)))
                f.write("%s\n" % (segment.to_string(),))

    def add(self, segment, save=True, force=False):
        key = segment.get_key()
        if key not in self.segments or force:
            self.segments[key] = segment
            if save:
                self.save()

    def get_by_key(self, key):
        #print("get_by_key(%s) => %s" % (key, str(self.segments.get(key))))
        return self.segments.get(key)

    def contains_key(self, key):
        #print("contains_key(%s) => %s" % (key, str(self.get_by_key(key) is not None)))
        return self.get_by_key(key) is not None

class SpeedOMeter(object):
    def __init__(self, image):
        assert image.dtype == np.uint8
        assert image.ndim == 3
        self.image = image

    def get_postprocessed_image(self):
        return SpeedOMeter.postprocess_speed_image(self.image)

    def get_postprocessed_image_rgb(self):
        img_bin = self.get_postprocessed_image()
        img2d = img_bin.astype(np.uint8) * 255
        return np.tile(img2d[:, :, np.newaxis], (1, 1, 3))

    def split_to_segments(self):
        post = self.get_postprocessed_image()
        return SpeedOMeter.segment_speed_image(post)

    def predict_speed_raw(self):
        segs = self.split_to_segments()
        #assert len(segs) > 0
        if len(segs) == 0:
            print("[WARNING] [SpeedOMeter.predict_speed_raw()] length of segs is zero")
            return 0

        result = []
        for seg in segs:
            label = seg.predict_label()
            result.append(label)
        return result

    def predict_speed(self):
        labels = self.predict_speed_raw()
        digits = set("0123456789")
        result = []
        for label in labels:
            if label is None:
                return None
            for c in label:
                if c in digits:
                    result.append(c)
        if len(result) == 0:
            return None
        else:
            return int("".join(result))

    @staticmethod
    def postprocess_speed_image(speed_image):
        speed_image_gray = cv2.cvtColor(speed_image, cv2.COLOR_RGB2GRAY)
        #assert speed_image_gray.dtype == np.uint8
        #speed_image_edges = filters.prewitt_v(speed_image_gray) # float, roughly -1 to 1

        #speed_image = speed_image[:, :, 1:] # remove red
        #speed_image_avg = np.average(speed_image, axis=2)
        return (speed_image_gray > 140)
        #print("speed_image_edges.dtype", speed_image_edges.dtype, np.min(speed_image_edges), np.max(speed_image_edges))
        #return np.logical_or(speed_image_edges < (-64/255), speed_image_edges > (64/255))

    @staticmethod
    def segment_speed_image(speed_image_bin):
        speed_image_labeled, num_labels = morphology.label(
            speed_image_bin, background=0, connectivity=1, return_num=True
        )

        segments = []
        for label in range(1, num_labels+1):
            (yy, xx) = np.nonzero(speed_image_labeled == label)
            min_y, max_y = np.min(yy), np.max(yy)
            min_x, max_x = np.min(xx), np.max(xx)
            seg_img = speed_image_bin[min_y:max_y+1, min_x:max_x+1]
            segments.append(SpeedSegment(seg_img))

        return segments

class SpeedSegment(object):
    def __init__(self, arr, label=None):
        assert arr.dtype == np.bool
        assert arr.ndim == 2
        self.arr = arr
        self.label = label

    def get_image(self):
        return self.arr.astype(np.uint8) * 255

    def predict_label(self):
        key = self.get_key()
        annotated_seg = SpeedSegmentsDatabase.get_instance().get_by_key(key)
        if annotated_seg is not None:
            return annotated_seg.label
        else:
            return None

    def is_in_database(self):
        key = self.get_key()
        return SpeedSegmentsDatabase.get_instance().contains_key(key)

    def get_key(self):
        return SpeedSegment.segment_image_to_key(self.arr)

    def to_string(self):
        sr = self.arr.astype(np.int32).ravel()
        vals = []
        for v in sr:
            vals.append(str(v))
        label = self.label if self.label is not None else "?"
        h, w = self.arr.shape
        return "%s||%d||%d||%s" % (label, h, w, "".join(vals))

    @staticmethod
    def from_string(line):
        label, h, w, vals = line.split("||")
        seg_image = np.zeros((int(h) * int(w)), dtype=np.bool)
        for i in range(len(vals)):
            seg_image[i] = int(vals[i])
        seg_image = seg_image.reshape((int(h), int(w)))
        if label == "?":
            label = None
        seg = SpeedSegment(seg_image, label=label)
        return seg

    @staticmethod
    def segment_image_to_key(arr):
        assert arr.dtype == np.bool
        assert arr.ndim == 2
        # int conversion here so that the string does not become
        # "TrueFalseFalseTrue..." but instead "1001..."
        sr = arr.ravel().astype(np.int32)
        vals = []
        for v in sr:
            vals.append(str(v))
        return "%dx%d_%s" % (arr.shape[0], arr.shape[1], str("".join(vals)),)
