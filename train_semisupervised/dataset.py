from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import train
from lib import replay_memory
from lib import util

import numpy as np
import cPickle as pickle
import gzip as gz
import random

try:
    xrange
except NameError:
    xrange = range

def dataset_state_idx_to_example(state_idx, memory):
    prev = max(max(train.PREVIOUS_STATES_DISTANCES), 10)
    states = memory.get_states_by_ids(range(state_idx-prev, state_idx+1+10))
    states_past = states[0:prev]
    state = states[prev]
    states_future = states[prev+1:]

    if any([s is None for s in states_past]):
        print("[INFO] a state past is None for idx %d" % (state_idx,))
    if any([s is None for s in states_future]):
        print("[INFO] a state future is None for idx %d" % (state_idx,))
    states_past = [s if s is not None else state for s in states_past]
    states_future = [s if s is not None else state for s in states_future]

    previous_multiaction_vecs = [state_past.actions_multivec for state_past in states_past]
    previous_multiaction_vec = previous_multiaction_vecs[0]
    previous_multiaction_vecs_avg = np.average(np.array(previous_multiaction_vecs, dtype=np.float32), 0)
    multiaction_vec = state.actions_multivec
    next_multiaction_vecs = [state_future.actions_multivec for state_future in states_future]
    next_multiaction_vec = next_multiaction_vecs[0]
    next_multiaction_vecs_avg = np.average(np.array(next_multiaction_vecs, dtype=np.float32), 0)

    ex = Example(state_idx, state.screenshot_rs_jpg)
    ex.previous_screenshots_rs_jpg = [states_past[len(states_past)-d].screenshot_rs_jpg for d in train.PREVIOUS_STATES_DISTANCES]
    ex.previous_multiaction_vec = previous_multiaction_vec
    ex.previous_multiaction_vecs_avg = previous_multiaction_vecs_avg
    ex.multiaction_vec = multiaction_vec
    ex.next_multiaction_vec = next_multiaction_vec
    ex.next_multiaction_vecs_avg = next_multiaction_vecs_avg

    return ex

def load_dataset_annotated():
    memory = replay_memory.ReplayMemory.create_instance_supervised(val=False)

    examples = dict()
    if not train.DEBUG:
        print("[load_dataset] Loading grids examples...")
        for (grid_key, fp) in train.ANNOTATIONS_GRIDS_FPS:
            if not os.path.isfile(fp):
                print("[WARNING] Could not find annotations (grids) file '%s'" % (fp,))
            else:
                dataset = pickle.load(open(fp, "r"))
                for key in dataset:
                    state_idx = dataset[key]["idx"]
                    from_datetime = dataset[key]["from_datetime"]
                    screenshot_rs = dataset[key]["screenshot_rs"]
                    grid = dataset[key][grid_key]
                    if not key in examples:
                        examples[key] = dataset_state_idx_to_example(state_idx, memory)
                    examples[key].add_grid(grid_key, grid)

    print("[load_dataset] Loading atts examples...")
    for (att_group_key, fp) in train.ANNOTATIONS_ATTS_FPS:
        dataset = pickle.load(open(fp, "r"))
        for key in dataset:
            if not os.path.isfile(fp):
                print("[WARNING] Could not find annotations (atts) file '%s'" % (fp,))
            else:
                state_idx = dataset[key]["idx"]
                from_datetime = dataset[key]["from_datetime"]
                screenshot_rs = dataset[key]["screenshot_rs"]
                attributes = dataset[key][att_group_key]
                if not key in examples:
                    examples[key] = dataset_state_idx_to_example(state_idx, memory)
                examples[key].add_attributes(attributes)

    return examples.values()

def load_dataset_annotated_compressed():
    fp = train.ANNOTATIONS_COMPRESSED_FP
    assert os.path.isfile(fp)
    with gz.open(fp, "rb") as f:
        examples = pickle.load(f)
    return examples

def load_dataset_autogen(val, nb_load, not_in=None):
    not_in = not_in if not_in is not None else set()
    memory = replay_memory.ReplayMemory.create_instance_supervised(val=val)

    print("[load_dataset_autogen] Loading random examples...")
    examples = dict()
    for i in xrange(nb_load):
        rnd_idx = random.randint(memory.id_min+max(train.PREVIOUS_STATES_DISTANCES), memory.id_max-10)
        key = str(rnd_idx)
        if key not in not_in:
            examples[key] = dataset_state_idx_to_example(rnd_idx, memory)

    return examples.values()

class Example(object):
    def __init__(self, state_idx, screenshot_rs_jpg):
        self.state_idx = state_idx
        self.screenshot_rs_jpg = screenshot_rs_jpg
        self.previous_screenshots_rs_jpg = None
        self.previous_multiaction_vec = None
        #self.previous_multiaction_vecs = None
        self.previous_multiaction_vecs_avg = None
        self.multiaction_vec = None
        self.next_multiaction_vec = None
        #self.next_multiaction_vecs = None
        self.next_multiaction_vecs_avg = None
        self.grids = dict()
        self.attributes = dict()

    @property
    def screenshot_rs(self):
        return util.decompress_img(self.screenshot_rs_jpg)

    @property
    def previous_screenshots_rs(self):
        return [util.decompress_img(scr) for scr in self.previous_screenshots_rs_jpg]

    def add_grid(self, grid_key, grid):
        self.grids[grid_key] = grid

    def add_attributes(self, atts):
        for att_group_name in atts:
            self.attributes[att_group_name] = atts[att_group_name]

    def get_grids_array(self):
        h, w = self.screenshot_rs.shape[0:2]
        lst = []
        grids_annotated = []
        for key in train.GRIDS_ORDER:
            if key in self.grids:
                lst.append(self.grids[key])
                grids_annotated.append(True)
            else:
                lst.append(np.zeros((h, w), dtype=np.float32))
                grids_annotated.append(False)
        arr = np.array(lst, dtype=np.float32)
        return arr.transpose((1, 2, 0)), grids_annotated

    def get_attributes_array(self):
        if not self.has_attributes():
            return np.zeros((train.NB_ATTRIBUTE_VALUES,), dtype=np.float32), False
        else:
            lst = []
            for att_group_name in train.ATTRIBUTES_ORDER:
                vec = attribute_to_onehot(att_group_name, self.attributes[att_group_name])
                lst.extend(vec)
            return np.array(lst, dtype=np.float32), True

    def has_attributes(self):
        return len(self.attributes) > 0

def attribute_to_onehot(att_group_name, att_name):
    group = train.ATTRIBUTE_GROUPS_BY_NAME[att_group_name]
    l = len(group.attributes)
    idx = [i for i, att in enumerate(group.attributes) if att.name == att_name]
    #print(att_group_name, att_name, [(i, att.name) for i, att in enumerate(group.attributes)])
    assert len(idx) == 1, len(idx)
    vec = np.zeros((l,), dtype=np.float32)
    vec[idx[0]] = 1
    return vec
