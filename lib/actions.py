from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
import numpy as np

ACTION_UP = "W"
ACTION_DOWN = "S"
ACTION_LEFT = "A"
ACTION_RIGHT = "D"
ACTION_UP_DOWN_NONE = "~WS"
ACTION_LEFT_RIGHT_NONE = "~AD"

ACTIONS_UP_DOWN = [ACTION_UP, ACTION_DOWN, ACTION_UP_DOWN_NONE]
ACTIONS_LEFT_RIGHT = [ACTION_LEFT, ACTION_RIGHT, ACTION_LEFT_RIGHT_NONE]
ALL_ACTIONS = ACTIONS_UP_DOWN + ACTIONS_LEFT_RIGHT

ACTION_TO_KEY = {
    ACTION_UP: "w",
    ACTION_DOWN: "s",
    ACTION_LEFT: "a",
    ACTION_RIGHT: "d",
    ACTION_UP_DOWN_NONE: None,
    ACTION_LEFT_RIGHT_NONE: None
}

ACTION_UP_VECTOR_INDEX = 0
ACTION_DOWN_VECTOR_INDEX = 1
ACTION_LEFT_VECTOR_INDEX = 0
ACTION_RIGHT_VECTOR_INDEX = 1
ACTION_UP_DOWN_NONE_VECTOR_INDEX = 2
ACTION_LEFT_RIGHT_NONE_VECTOR_INDEX = 2

def make_one_hot_vector(size, idx_one):
    vec = np.zeros((size,), dtype=np.int32)
    vec[idx_one] = 1
    return vec

ACTION_UP_VECTOR = make_one_hot_vector(3, ACTION_UP_VECTOR_INDEX)
ACTION_DOWN_VECTOR = make_one_hot_vector(3, ACTION_DOWN_VECTOR_INDEX)
ACTION_LEFT_VECTOR = make_one_hot_vector(3, ACTION_LEFT_VECTOR_INDEX)
ACTION_RIGHT_VECTOR = make_one_hot_vector(3, ACTION_RIGHT_VECTOR_INDEX)
ACTION_UP_DOWN_NONE_VECTOR = make_one_hot_vector(3, ACTION_UP_DOWN_NONE_VECTOR_INDEX)
ACTION_LEFT_RIGHT_NONE_VECTOR = make_one_hot_vector(3, ACTION_LEFT_RIGHT_NONE_VECTOR_INDEX)

ACTION_UP_DOWN_TO_VECTOR_INDEX = {
    ACTION_UP: ACTION_UP_VECTOR_INDEX,
    ACTION_DOWN: ACTION_DOWN_VECTOR_INDEX,
    ACTION_UP_DOWN_NONE: ACTION_UP_DOWN_NONE_VECTOR_INDEX
}

ACTION_LEFT_RIGHT_TO_VECTOR_INDEX = {
    ACTION_LEFT: ACTION_LEFT_VECTOR_INDEX,
    ACTION_RIGHT: ACTION_RIGHT_VECTOR_INDEX,
    ACTION_LEFT_RIGHT_NONE: ACTION_LEFT_RIGHT_NONE_VECTOR_INDEX
}

ACTION_UP_DOWN_FROM_VECTOR_INDEX = [
    ACTION_UP,
    ACTION_DOWN,
    ACTION_UP_DOWN_NONE
]

ACTION_LEFT_RIGHT_FROM_VECTOR_INDEX = [
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_LEFT_RIGHT_NONE
]

ACTION_UP_DOWN_TO_VECTOR = {
    ACTION_UP: ACTION_UP_VECTOR,
    ACTION_DOWN: ACTION_DOWN_VECTOR,
    ACTION_UP_DOWN_NONE: ACTION_UP_DOWN_NONE_VECTOR
}

ACTION_LEFT_RIGHT_TO_VECTOR = {
    ACTION_LEFT: ACTION_LEFT_VECTOR,
    ACTION_RIGHT: ACTION_RIGHT_VECTOR,
    ACTION_LEFT_RIGHT_NONE: ACTION_LEFT_RIGHT_NONE_VECTOR
}

ALL_MULTIACTIONS = []
ACTIONS_TO_MULTIVEC = dict()
ACTIONS_FROM_MULTIVEC_INDEX = []
i = 0
for action_up_down in ACTIONS_UP_DOWN:
    for action_left_right in ACTIONS_LEFT_RIGHT:
        ALL_MULTIACTIONS.append((action_up_down, action_left_right))
        vec = np.zeros((len(ACTIONS_UP_DOWN) * len(ACTIONS_LEFT_RIGHT),), dtype=np.float32)
        vec[i] = 1
        ACTIONS_TO_MULTIVEC[(action_up_down, action_left_right)] = vec
        ACTIONS_FROM_MULTIVEC_INDEX.append((action_up_down, action_left_right))
        i += 1

def get_random_action_up_down():
    return random.choice(ACTIONS_UP_DOWN)

def get_random_action_left_right():
    return random.choice(ACTIONS_LEFT_RIGHT)

def keys_to_action_up_down(keys):
    if keys is None:
        return ACTION_UP_DOWN_NONE
    elif "w" in keys:
        return ACTION_UP
    elif "s" in keys:
        return ACTION_DOWN
    else:
        return ACTION_UP_DOWN_NONE

def keys_to_action_left_right(keys):
    if keys is None:
        return ACTION_LEFT_RIGHT_NONE
    elif "a" in keys:
        return ACTION_LEFT
    elif "d" in keys:
        return ACTION_RIGHT
    else:
        return ACTION_LEFT_RIGHT_NONE

def action_up_down_to_vector(action):
    return np.copy(ACTION_UP_DOWN_TO_VECTOR[action])

def action_left_right_to_vector(action):
    return np.copy(ACTION_LEFT_RIGHT_TO_VECTOR[action])

def action_up_down_to_vector_index(action):
    return np.copy(ACTION_UP_DOWN_TO_VECTOR_INDEX[action])

def action_left_right_to_vector_index(action):
    return np.copy(ACTION_LEFT_RIGHT_TO_VECTOR_INDEX[action])

def action_up_down_from_vector_index(index):
    return ACTION_UP_DOWN_FROM_VECTOR_INDEX[index]

def action_left_right_from_vector_index(index):
    return ACTION_LEFT_RIGHT_FROM_VECTOR_INDEX[index]

def action_to_key(action):
    return ACTION_TO_KEY[action]

def actions_to_multivec(action_up_down, action_left_right):
    return ACTIONS_TO_MULTIVEC[(action_up_down, action_left_right)]
