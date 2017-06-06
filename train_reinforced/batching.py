"""Functions to generate batches for the reinforcement learning part.
Mainly intended for training, though during the playing phase, the same
functions are used."""
from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import models as models_reinforced

import cv2
from config import Config
import imgaug as ia
from lib.util import to_variable, to_cuda, to_numpy
from lib import util
from lib import actions as actionslib
from lib import replay_memory
import numpy as np
from scipy import misc
import multiprocessing
import threading
import random
import time

if sys.version_info[0] == 2:
    import cPickle as pickle
    from Queue import Full as QueueFull
elif sys.version_info[0] == 3:
    import pickle
    from queue import Full as QueueFull
    xrange = range

GPU = Config.GPU
NB_REWARD_BINS = 101

class BatchData(object):
    """Method encapsulating the data of a single batch.

    TODO some of the functions are named like properties, rename
    """

    def __init__(self, curr_idx, images_by_timestep, images_prev_by_timestep, multiactions, rewards, speeds, is_reverse, steering_wheel, steering_wheel_raw, previous_states_distances):
        self.curr_idx = curr_idx
        self.images_by_timestep = images_by_timestep
        self.images_prev_by_timestep = images_prev_by_timestep
        self.multiactions = multiactions
        self.rewards = rewards
        self.speeds = speeds
        self.is_reverse = is_reverse
        self.steering_wheel = steering_wheel
        self.steering_wheel_raw = steering_wheel_raw
        self.previous_states_distances = previous_states_distances

    @property
    def batch_size(self):
        return self.images_by_timestep.shape[1]

    @property
    def nb_future(self):
        return self.images_prev_by_timestep.shape[0] - 1

    @property
    def nb_prev_per_image(self):
        return self.images_prev_by_timestep.shape[2]

    def reward_bin_idx(self, timestep, inbatch_idx):
        timestep = self.curr_idx + timestep
        reward = self.rewards[timestep, inbatch_idx]
        reward_norm = (reward - Config.MIN_REWARD) / (Config.MAX_REWARD - Config.MIN_REWARD)
        reward_norm = 1 - reward_norm # top to bottom
        rewbin = np.clip(int(reward_norm * NB_REWARD_BINS), 0, NB_REWARD_BINS-1) # clip here, because MAX_REWARD ends up giving bin NB_REWARD_BINS, which is 1 too high
        return rewbin

    def rewards_bins(self, timestep):
        timestep = self.curr_idx + timestep
        T, B = self.rewards.shape
        result = np.zeros((B, NB_REWARD_BINS), dtype=np.float32)
        for b in xrange(B):
            rewbin = self.reward_bin_idx(timestep-self.curr_idx, b)
            result[b, rewbin] = 1
        return result

    def rewards_bins_all(self):
        T, B = self.rewards.shape
        bins_over_time = [self.rewards_bins(t) for t in xrange(-self.curr_idx, T-self.curr_idx)]
        return np.array(bins_over_time, dtype=np.float32)

    def inputs_supervised(self, volatile=False, requires_grad=True, gpu=GPU):
        images = to_cuda(to_variable(self.images_by_timestep[0], volatile=volatile, requires_grad=requires_grad), gpu)
        images_prev = to_cuda(to_variable(self.images_prev_by_timestep[0], volatile=volatile, requires_grad=requires_grad), gpu)
        return images, images_prev

    def inputs_reinforced_add_numpy(self, timestep=0):
        timestep = self.curr_idx + timestep
        B = self.batch_size

        prev_indices_exclusive = [timestep - d for d in self.previous_states_distances]
        prev_indices_inclusive = [timestep] + prev_indices_exclusive

        ma_vecs = np.zeros((self.batch_size, len(prev_indices_exclusive), 9), dtype=np.float32)
        for i, idx in enumerate(prev_indices_exclusive):
            mas = self.multiactions[idx]
            for b, ma in enumerate(mas):
                ma_vecs[b, i, :] = actionslib.ACTIONS_TO_MULTIVEC[ma]
        ma_vecs = ma_vecs.reshape(self.batch_size, -1) # (B, P*9) with P=number of previous images

        speeds = self.speeds[prev_indices_inclusive, :]
        steering_wheel = (self.steering_wheel[prev_indices_inclusive, :] - Config.STEERING_WHEEL_CNN_MIN) / (Config.STEERING_WHEEL_CNN_MAX - Config.STEERING_WHEEL_CNN_MIN)
        steering_wheel_raw = (self.steering_wheel_raw[prev_indices_inclusive, :] - Config.STEERING_WHEEL_RAW_CNN_MIN) / (Config.STEERING_WHEEL_RAW_CNN_MAX - Config.STEERING_WHEEL_RAW_CNN_MIN)
        vals = {
            "speeds": np.squeeze(np.clip(speeds / Config.MAX_SPEED, 0, 1)),
            "is_reverse": np.squeeze(self.is_reverse[prev_indices_inclusive, :]),
            "steering_wheel": np.squeeze(steering_wheel*2 - 1),
            "steering_wheel_raw": np.squeeze(steering_wheel_raw*2 - 1),
            "multiactions_vecs": ma_vecs
        }
        if B == 1:
            vals["speeds"] = vals["speeds"][:, np.newaxis]
            vals["is_reverse"] = vals["is_reverse"][:, np.newaxis]
            vals["steering_wheel"] = vals["steering_wheel"][:, np.newaxis]
            vals["steering_wheel_raw"] = vals["steering_wheel_raw"][:, np.newaxis]
        vals["speeds"] = vals["speeds"].transpose((1, 0)) # (P, B) => (B, P) with P=number of previous images
        vals["is_reverse"] = vals["is_reverse"].transpose((1, 0)) # (P, B) => (B, P) with P=number of previous images
        vals["steering_wheel"] = vals["steering_wheel"].transpose((1, 0)) # (P, B) => (B, P) with P=number of previous images
        vals["steering_wheel_raw"] = vals["steering_wheel_raw"].transpose((1, 0)) # (P, B) => (B, P) with P=number of previous images

        return vals

    def inputs_reinforced_add(self, volatile=False, requires_grad=True, gpu=GPU):
        return to_cuda(to_variable(self.inputs_reinforced_add_numpy(), volatile=volatile, requires_grad=requires_grad), gpu)

    def future_inputs_supervised(self, volatile=False, requires_grad=True, gpu=GPU):
        images = to_cuda(to_variable(self.images_by_timestep[1:], volatile=volatile, requires_grad=requires_grad), gpu)
        images_prev = to_cuda(to_variable(self.images_prev_by_timestep[1:], volatile=volatile, requires_grad=requires_grad), gpu)
        return images, images_prev

    def future_reinforced_add(self, volatile=False, requires_grad=True, gpu=GPU):
        vals = {
            "speeds": [],
            "is_reverse": [],
            "steering_wheel": [],
            "steering_wheel_raw": [],
            "multiactions_vecs": []
        }
        for timestep in xrange(1, self.nb_future+1):
            inputs_ts = self.inputs_reinforced_add_numpy(timestep=timestep)
            vals["speeds"].append(inputs_ts["speeds"])
            vals["is_reverse"].append(inputs_ts["is_reverse"])
            vals["steering_wheel"].append(inputs_ts["steering_wheel"])
            vals["steering_wheel_raw"].append(inputs_ts["steering_wheel_raw"])
            vals["multiactions_vecs"].append(inputs_ts["multiactions_vecs"])
        vals["speeds"] = np.array(vals["speeds"], dtype=np.float32)
        vals["is_reverse"] = np.array(vals["is_reverse"], dtype=np.float32)
        vals["steering_wheel"] = np.array(vals["steering_wheel"], dtype=np.float32)
        vals["steering_wheel_raw"] = np.array(vals["steering_wheel_raw"], dtype=np.float32)
        vals["multiactions_vecs"] = np.array(vals["multiactions_vecs"], dtype=np.float32)

        T, B, _ = vals["speeds"].shape
        vals_flat = {
            "speeds": vals["speeds"].reshape((T*B, -1)),
            "is_reverse": vals["is_reverse"].reshape((T*B, -1)),
            "steering_wheel": vals["steering_wheel"].reshape((T*B, -1)),
            "steering_wheel_raw": vals["steering_wheel_raw"].reshape((T*B, -1)),
            "multiactions_vecs": vals["multiactions_vecs"].reshape((T*B, -1))
        }

        return to_cuda(to_variable(vals_flat, volatile=volatile, requires_grad=requires_grad), gpu)

    def inputs_successor_multiactions_vecs(self, volatile=False, requires_grad=True, gpu=GPU):
        # the successor gets in actions a and has to predict the next
        # state, i.e. for tuples (s, a, r, s') it gets a and predicts s',
        # hence the future actions here start at curr_idx (current state index)
        # and end at -1
        arr = models_reinforced.SuccessorPredictor.multiactions_to_vecs(self.multiactions[self.curr_idx:-1])
        assert arr.shape == (self.nb_future, self.batch_size, 9)
        return to_cuda(to_variable(arr, volatile=volatile, requires_grad=requires_grad), gpu)

    def direct_rewards_values(self, volatile=False, requires_grad=True, gpu=GPU):
        rews = self.rewards[self.curr_idx, :][:,np.newaxis]
        rews = np.tile(rews, (1, 9))
        return to_cuda(to_variable(rews, volatile=volatile, requires_grad=requires_grad), gpu)

    def future_direct_rewards_values(self, volatile=False, requires_grad=True, gpu=GPU):
        rews = self.rewards[self.curr_idx+1:, :][:, :, np.newaxis]
        rews = np.tile(rews, (1, 1, 9))
        return to_cuda(to_variable(rews, volatile=volatile, requires_grad=requires_grad), gpu)

    def outputs_dr_gt(self, volatile=False, requires_grad=True, gpu=GPU):
        # for a tuple (s, a, r, s'), the reward r is ought to be predicted
        # that is, the reward for the previous action, which is dependent
        # on the new state s' that it created
        # it is saved at the previous timestep, i.e. at state s, hence
        # here -1
        bins = self.rewards_bins(-1)
        return to_cuda(to_variable(bins, volatile=volatile, requires_grad=requires_grad), gpu)

    def outputs_dr_future_gt(self, volatile=False, requires_grad=True, gpu=GPU):
        # starting at curr_idx and ending at -1 here for the same reason
        # as above
        bins = self.rewards_bins_all()
        bins = bins[self.curr_idx:-1]
        return to_cuda(to_variable(bins, volatile=volatile, requires_grad=requires_grad), gpu)

    def outputs_ae_gt(self, volatile=False, requires_grad=True, gpu=GPU):
        imgs = self.images_by_timestep[0, ...]
        imgs = np.clip(imgs*255, 0, 255).astype(np.uint8).transpose((0, 2, 3, 1))
        imgs_rs = ia.imresize_many_images(imgs, (45, 80))
        imgs_rs = (imgs_rs / 255.0).astype(np.float32).transpose((0, 3, 1, 2))
        return to_cuda(to_variable(imgs_rs, volatile=volatile, requires_grad=requires_grad), gpu)

    def chosen_action_indices(self):
        mas_timestep = self.multiactions[self.curr_idx]
        indices = [np.argmax(actionslib.ACTIONS_TO_MULTIVEC[ma]) for ma in mas_timestep]
        return indices

    def chosen_action_indices_future(self):
        indices_by_timestep = []
        for t_idx in xrange(self.nb_future):
            mas_timestep = self.multiactions[t_idx]
            indices = [np.argmax(actionslib.ACTIONS_TO_MULTIVEC[ma]) for ma in mas_timestep]
            indices_by_timestep.append(indices)
        return indices_by_timestep

    def draw(self, timestep=0, inbatch_idx=0):
        timestep = self.curr_idx + timestep
        img = self.images_by_timestep[timestep-self.curr_idx, inbatch_idx, :, :, :]
        img = (img.transpose((1, 2, 0))*255).astype(np.uint8)
        imgs_prev = self.images_prev_by_timestep[timestep-self.curr_idx, inbatch_idx, :, :, :]
        imgs_prev = (imgs_prev.transpose((1, 2, 0))*255).astype(np.uint8)

        h, w = img.shape[0:2]
        imgs_viz = [img] + [np.tile(imgs_prev[..., i][:, :, np.newaxis], (1, 1, 3)) for i in xrange(imgs_prev.shape[2])]
        imgs_viz = [ia.imresize_single_image(im, (h, w), interpolation="cubic") for im in imgs_viz]
        imgs_viz = np.hstack(imgs_viz)

        rewards_bins = self.rewards_bins_all()
        mas = [self.multiactions[i][inbatch_idx] for i in xrange(timestep-self.nb_prev_per_image, timestep)]
        pos = [timestep] + [timestep-d for d in self.previous_states_distances]
        reinforced_add = self.inputs_reinforced_add_numpy(timestep=timestep-self.curr_idx)
        outputs_dr_gt = self.outputs_dr_gt()[inbatch_idx]
        texts = [
            "pos: " + " ".join([str(i) for i in pos]),
            "Rewards:      " + " ".join(["%.2f" % (self.rewards[i, inbatch_idx],) for i in pos]),
            "Rewards bins: " + " ".join(["%d" % (np.argmax(rewards_bins[i, inbatch_idx]),) for i in pos]),
            "Speeds:       " + " ".join(["%.2f" % (self.speeds[i, inbatch_idx],) for i in pos]),
            "Multiactions: " + " ".join(["%s%s" % (ma[0], ma[1]) for ma in mas]),
            "Speeds RA:    " + " ".join(["%.3f" % (reinforced_add["speeds"][inbatch_idx, i],) for i in xrange(reinforced_add["speeds"].shape[1])]),
            "outputs_dr_gt[t=-1]: " + "%d" % (np.argmax(to_numpy(outputs_dr_gt)),)
        ]
        texts = "\n".join(texts)

        result = np.zeros((imgs_viz.shape[0]*3, imgs_viz.shape[1], 3), dtype=np.uint8)
        util.draw_image(result, x=0, y=0, other_img=imgs_viz, copy=False)
        result = util.draw_text(result, x=0, y=imgs_viz.shape[0]+4, text=texts, size=9)
        return result

def states_to_batch(previous_states_list, states_list, augseq, previous_states_distances, model_height, model_width, model_prev_height, model_prev_width):
    """Convert multiple chains of states into a batch.

    Parameters
    ----------
    previous_states_list : list of list of State
        Per chain of states a list of the previous states.
        First index of the list is the batch index,
        second index is the timestep. The oldest states come first.
    states_list : list of list of State
        Per chain of states a list of states that contain the "current"
        state at the start, followed by future states.
        First index is batch index, second timestep.
    augseq : Augmenter
        Sequence of augmenters to apply to each image. Use Noop() to make
        no changes.
    previous_states_distances : list of int
        List of distances relative to the current state. Each distance
        refers to one previous state to add to the model input.
        E.g. [2, 1] adds the state 200ms and 100ms before the current "state".
    model_height : int
        Height of the model input images (current state).
    model_width : int
        Width of the model input images (current state).
    model_prev_height : int
        Height of the model input images (previous states).
    model_prev_width : int
        Width of the model input images (previous states).

    Returns
    ----------
    List of BatchData
    """
    assert isinstance(previous_states_list, list)
    assert isinstance(states_list, list)
    assert isinstance(previous_states_list[0], list)
    assert isinstance(states_list[0], list)
    assert len(previous_states_list) == len(states_list)

    B = len(states_list)
    H, W = model_height, model_width
    Hp, Wp = model_prev_height, model_prev_width

    nb_prev_load = max(previous_states_distances)
    nb_future_states = len(states_list[0]) - 1
    nb_timesteps = nb_prev_load + 1 + nb_future_states
    #images = np.zeros((nb_timesteps, B, H, W, 3), dtype=np.uint8)
    #images_gray = np.zeros((nb_timesteps, B, Hp, Wp), dtype=np.float32)
    images_by_timestep = np.zeros((1+nb_future_states, B, H, W, 3), dtype=np.float32)
    images_gray = np.zeros((nb_timesteps, B, Hp, Wp), dtype=np.float32)
    multiactions = [[] for i in xrange(nb_timesteps)]
    rewards = np.zeros((nb_timesteps, B), dtype=np.float32)
    speeds = np.zeros((nb_timesteps, B), dtype=np.float32)
    is_reverse = np.zeros((nb_timesteps, B), dtype=np.float32)
    steering_wheel = np.zeros((nb_timesteps, B), dtype=np.float32)
    steering_wheel_raw = np.zeros((nb_timesteps, B), dtype=np.float32)

    augseqs_det = [augseq.to_deterministic() for _ in xrange(len(states_list))]

    for b, (previous_states, states) in enumerate(zip(previous_states_list, states_list)):
        augseq_det = augseqs_det[b]

        all_states = previous_states + states
        for t, state in enumerate(all_states):
            imgy = cv2.cvtColor(state.screenshot_rs, cv2.COLOR_RGB2GRAY)
            imgy_rs = downscale(imgy, Hp, Wp)
            imgy_rs_aug = augseq_det.augment_image(imgy_rs)
            images_gray[t, b, ...] = imgy_rs

            multiactions[t].append(state.multiaction)
            rewards[t, b] = state.reward
            if state.speed is not None:
                speeds[t, b] = state.speed
            if state.is_reverse is not None:
                is_reverse[t, b] = int(state.is_reverse)
            if state.steering_wheel_cnn is not None:
                steering_wheel[t, b] = state.steering_wheel_cnn
            if state.steering_wheel_raw_cnn is not None:
                steering_wheel_raw[t, b] = state.steering_wheel_raw_cnn
    images_gray = images_gray[..., np.newaxis]

    for b, states in enumerate(states_list):
        augseq_det = augseqs_det[b]

        for i, state in enumerate(states):
            state = states[i]
            images_by_timestep[i, b, ...] = augseq_det.augment_image(downscale(state.screenshot_rs, H, W))

    nb_prev_per_img = len(previous_states_distances)
    images_prev_by_timestep = np.zeros((1+nb_future_states, B, Hp, Wp, nb_prev_per_img), dtype=np.float32)
    for t in xrange(1 + nb_future_states):
        indices = [nb_prev_load+t-d for d in previous_states_distances]
        prev = images_gray[indices]
        prev = prev.transpose((1, 2, 3, 4, 0)).reshape((B, Hp, Wp, nb_prev_per_img))
        images_prev_by_timestep[t] = prev
    images_by_timestep = (images_by_timestep.astype(np.float32) / 255.0).transpose((0, 1, 4, 2, 3))
    images_prev_by_timestep = (images_prev_by_timestep.astype(np.float32) / 255.0).transpose((0, 1, 4, 2, 3))

    return BatchData(nb_prev_load, images_by_timestep, images_prev_by_timestep, multiactions, rewards, speeds, is_reverse, steering_wheel, steering_wheel_raw, previous_states_distances)

def downscale(im, h, w):
    if im.ndim == 2:
        im = im[:, :, np.newaxis]
        return np.squeeze(ia.imresize_single_image(im, (h, w), interpolation="cubic"))
    else:
        return ia.imresize_single_image(im, (h, w), interpolation="cubic")

class BatchLoader(object):
    """Class to load batches from the replay memory."""

    def __init__(self, val, batch_size, augseq, previous_states_distances, nb_future_states, model_height, model_width, model_prev_height, model_prev_width):
        self.val = val
        self.batch_size = batch_size
        self.augseq = augseq.deepcopy()
        self.augseq.reseed(random.randint(0, 10**6))
        self.previous_states_distances = previous_states_distances
        self.nb_future_states = nb_future_states
        self.model_height = model_height
        self.model_width = model_width
        self.model_prev_height = model_prev_height
        self.model_prev_width = model_prev_width
        self._memory = None

    def load_random_batch(self):
        if self._memory is None:
            self._memory = replay_memory.ReplayMemory.create_instance_reinforced(val=self.val)
            self._memory.update_caches()
            print("Connected memory to %s, idmin=%d, idmax=%d" % ("val" if self.val else "train", self._memory.id_min, self._memory.id_max))
        memory = self._memory

        nb_prev = max(self.previous_states_distances)
        nb_timesteps = nb_prev + 1 + self.nb_future_states

        previous_states_list = []
        states_list = []
        for b in xrange(self.batch_size):
            statechain = memory.get_random_state_chain(nb_timesteps)
            previous_states_list.append(statechain[:nb_prev])
            states_list.append(statechain[nb_prev:])

        return states_to_batch(previous_states_list, states_list, self.augseq, self.previous_states_distances, self.model_height, self.model_width, self.model_prev_height, self.model_prev_width)

class BackgroundBatchLoader(object):
    """Class that takes a BatchLoader and executes it many times in background
    processes."""

    def __init__(self, batch_loader, queue_size, nb_workers, threaded=False):
        self.queue = multiprocessing.Queue(queue_size)
        self.workers = []
        self.exit_signal = multiprocessing.Event()
        for i in range(nb_workers):
            seed = random.randint(1, 10**6)
            if threaded:
                worker = threading.Thread(target=self._load_batches, args=(batch_loader, self.queue, self.exit_signal, None))
            else:
                worker = multiprocessing.Process(target=self._load_batches, args=(batch_loader, self.queue, self.exit_signal, seed))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def get_batch(self):
        return pickle.loads(self.queue.get())

    def _load_batches(self, batch_loader, queue, exit_signal, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        while not exit_signal.is_set():
            batch = batch_loader.load_random_batch()

            start_time = time.time()
            batch_str = pickle.dumps(batch, protocol=-1)
            added_to_queue = False # without this, it will add the batch countless times to the queue
            while not added_to_queue and not exit_signal.is_set():
                try:
                    queue.put(batch_str, timeout=1)
                    added_to_queue = True
                except QueueFull as e:
                    pass
            end_time = time.time()
        batch_loader._memory.close()

    def join(self):
        self.exit_signal.set()
        time.sleep(5)

        while not self.queue.empty():
            _ = self.queue.get()
        #self.queue.join()

        for worker in self.workers:
            #worker.join()
            worker.terminate()

if __name__ == "__main__":
    from scipy import misc
    from imgaug import augmenters as iaa

    MODEL_HEIGHT = 90
    MODEL_WIDTH = 160
    MODEL_PREV_HEIGHT = 45
    MODEL_PREV_WIDTH = 80

    loader = BatchLoader(
        val=False, batch_size=8, augseq=iaa.Noop(),
        previous_states_distances=[2, 4, 6, 8, 10],
        nb_future_states=10,
        model_height=MODEL_HEIGHT, model_width=MODEL_WIDTH,
        model_prev_height=MODEL_PREV_HEIGHT, model_prev_width=MODEL_PREV_HEIGHT
    )
    for _ in xrange(1000):
        for t in xrange(3):
            imgs = []
            for b in xrange(3):
                print(t, b)
                batch = loader.load_random_batch()
                imgs.append(batch.draw(timestep=t, inbatch_idx=b))
            misc.imshow(np.vstack(imgs))
