"""Configuration file.
Note that this config file does not contain all constants, some are
in the train.py scripts."""
from __future__ import print_function, division
import os

class Config(object):
    """Class containing some global configuration constants."""

    # path to the directory of the project
    MAIN_DIR = os.path.dirname(__file__)

    # GPU id to use
    GPU = 0

    # height and width at which ingame screenshots are saved to memory
    MODEL_HEIGHT = 720 // 4 # 180
    MODEL_WIDTH = 1280 // 4 # 320

    # directory of the annotations
    ANNOTATIONS_DIR = "/media/aj/grab/ml/ets2ai/annotations" if os.path.isdir("/media/aj/grab/ml/ets2ai/annotations") else os.path.join(MAIN_DIR, "annotations/")
    # filepath to the speed image annotations
    SPEED_IMAGE_SEGMENTS_DB_FILEPATH = os.path.join(MAIN_DIR, "annotations/speed_image_segments.txt")

    # directory where the replay memory files are saved
    REPLAY_MEMORIES_DIR = MAIN_DIR

    # configurations of the replay memories
    REPLAY_MEMORY_CFGS = {
        "reinforced-train": {"filepath": os.path.join(REPLAY_MEMORIES_DIR, "replay_memory_reinforced.sqlite"), "max_size": 650*1000, "max_size_tolerance": 20000},
        "reinforced-val": {"filepath": os.path.join(REPLAY_MEMORIES_DIR, "replay_memory_reinforced_val.sqlite"), "max_size": 150*1000, "max_size_tolerance": 20000},
        "supervised-train": {"filepath": os.path.join(REPLAY_MEMORIES_DIR, "replay_memory_supervised.sqlite"), "max_size": 200*1000, "max_size_tolerance": 20000},
        "supervised-val": {"filepath": os.path.join(REPLAY_MEMORIES_DIR, "replay_memory_supervised_val.sqlite"), "max_size": 50*1000, "max_size_tolerance": 20000},
    }

    # max speed, model inputs are clipped to that maximum
    MAX_SPEED = 150

    # max/min steering wheel positions to use for the classical and CNN-based
    # steering wheel prediction. values beyond this are viewed as somehow wrong
    # and either ignored or lead to the assumption that the current approximated
    # position is wrong.
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

    # min and maximum direct reward range, used for computing bins
    MIN_REWARD = -100
    MAX_REWARD = 100

    # keys for pause and quickload to use
    KEY_PAUSE = "F1"
    KEY_QUICKLOAD = "F11"

    # Number of savegames to use during training. If set to N, the AI will
    # load a random one of the first N savegames during reinforcement learning.
    # (This happens many times during the training.)
    RELOAD_MAX_SAVEGAME_NUMBER = 6

    # Directory of an example image from an offence message.
    OFFENCE_FF_IMAGE = os.path.join(MAIN_DIR, "images/offence_ff.png")
    # Directory of an example image from a damage message (XX% damage).
    DAMAGE_SINGLE_DIGIT_IMAGE = os.path.join(MAIN_DIR, "images/damage_single_digit.png")
    # Directory of an example image from a damage message (X% damage).
    DAMAGE_DOUBLE_DIGIT_IMAGE = os.path.join(MAIN_DIR, "images/damage_double_digit.png")
    # Directory of an example image showing and activated reverse gear.
    REVERSE_IMAGE = os.path.join(MAIN_DIR, "images/reverse.png")
