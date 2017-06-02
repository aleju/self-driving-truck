from __future__ import print_function, division
import os

class Config(object):
    MAIN_DIR = os.path.dirname(__file__)
    GPU = 0
    MODEL_HEIGHT = 720 // 4 # 180
    MODEL_WIDTH = 1280 // 4 # 320
    #MODEL_HEIGHT_SMALL = MODEL_HEIGHT // 4 # 45
    #MODEL_WIDTH_SMALL = MODEL_WIDTH // 4 # 80
    #REPLAY_MEMORIES_DIR = MAIN_DIR
    ANNOTATIONS_DIR = "/media/aj/grab/ml/ets2ai/annotations" if os.path.isdir("/media/aj/grab/ml/ets2ai/annotations") else os.path.join(MAIN_DIR, "annotations/")
    SPEED_IMAGE_SEGMENTS_DB_FILEPATH = os.path.join(MAIN_DIR, "annotations/speed_image_segments.txt")
    REPLAY_MEMORIES_DIR = os.path.join(MAIN_DIR, "../ets2ai/")
    REPLAY_MEMORY_CFGS = {
        "reinforced-train": {"filepath": os.path.join(REPLAY_MEMORIES_DIR, "replay_memory_reinforced.sqlite"), "max_size": 650*1000, "max_size_tolerance": 20000},
        "reinforced-val": {"filepath": os.path.join(REPLAY_MEMORIES_DIR, "replay_memory_reinforced_val.sqlite"), "max_size": 150*1000, "max_size_tolerance": 20000},
        "supervised-train": {"filepath": os.path.join(REPLAY_MEMORIES_DIR, "replay_memory_supervised.sqlite"), "max_size": 200*1000, "max_size_tolerance": 20000},
        "supervised-val": {"filepath": os.path.join(REPLAY_MEMORIES_DIR, "replay_memory_supervised_val.sqlite"), "max_size": 50*1000, "max_size_tolerance": 20000},
    }
    #MODEL_NB_PREVIOUS_STATES = 10
    #MODEL_NB_REWARD_BINS = 21
    MAX_SPEED = 150
    STEERING_WHEEL_MAX = 360+90+40 # wheel can be turned by roughly +/- 360+90 degrees, add 40deg tolerance
    STEERING_WHEEL_MIN = -STEERING_WHEEL_MAX # steering_wheel.py currently only looks at max value
    STEERING_WHEEL_RAW_ONE_MAX = 180
    STEERING_WHEEL_RAW_ONE_MIN = -STEERING_WHEEL_RAW_ONE_MAX
    STEERING_WHEEL_RAW_TWO_MAX = STEERING_WHEEL_RAW_ONE_MAX
    STEERING_WHEEL_RAW_TWO_MIN = STEERING_WHEEL_RAW_ONE_MIN
    STEERING_WHEEL_CNN_MAX = 360+90+40
    STEERING_WHEEL_CNN_MIN = -STEERING_WHEEL_CNN_MAX
    STEERING_WHEEL_RAW_CNN_MAX = 180
    STEERING_WHEEL_RAW_CNN_MIN = -STEERING_WHEEL_RAW_CNN_MAX
    MIN_REWARD = -100
    MAX_REWARD = 100
    KEY_PAUSE = "F1"
    KEY_QUICKLOAD = "F11"
    RELOAD_MAX_SAVEGAME_NUMBER = 6
    OFFENCE_FF_IMAGE = os.path.join(MAIN_DIR, "images/offence_ff.png")
    DAMAGE_SINGLE_DIGIT_IMAGE = os.path.join(MAIN_DIR, "images/damage_single_digit.png")
    DAMAGE_DOUBLE_DIGIT_IMAGE = os.path.join(MAIN_DIR, "images/damage_double_digit.png")
    REVERSE_IMAGE = os.path.join(MAIN_DIR, "images/reverse.png")
