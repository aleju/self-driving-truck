"""Trains a CNN to detect the current steering wheel angle from images."""
from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import models
from lib import replay_memory
from lib import util
from lib.util import to_variable, to_cuda, to_numpy
from lib import plotting
from config import Config

from scipy import misc
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing
import threading
import argparse
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

if sys.version_info[0] == 2:
    import cPickle as pickle
elif sys.version_info[0] == 3:
    import pickle
    xrange = range

# train for N batches
NB_BATCHES = 50000

# size of each batch
BATCH_SIZE = 128

# save/val/plot every N batches
SAVE_EVERY = 500
VAL_EVERY = 500
PLOT_EVERY = 500

# use N batches for validation (loss will be averaged)
NB_VAL_BATCHES = 128

# input image height/width
MODEL_HEIGHT = 32
MODEL_WIDTH = 64

# size of each bin in degrees
ANGLE_BIN_SIZE = 5

def main():
    """Function that initializes the training (e.g. models)
    and runs the batches."""

    parser = argparse.ArgumentParser(description="Train steering wheel tracker")
    parser.add_argument('--nocontinue', default=False, action="store_true", help="Whether to NOT continue the previous experiment", required=False)
    args = parser.parse_args()

    if os.path.isfile("steering_wheel.tar") and not args.nocontinue:
        checkpoint = torch.load("steering_wheel.tar")
    else:
        checkpoint = None

    if checkpoint is not None:
        history = plotting.History.from_string(checkpoint["history"])
    else:
        history = plotting.History()
        history.add_group("loss", ["train", "val"], increasing=False)
        history.add_group("acc", ["train", "val"], increasing=True)
    loss_plotter = plotting.LossPlotter(
        history.get_group_names(),
        history.get_groups_increasing(),
        save_to_fp="train_plot.jpg"
    )
    loss_plotter.start_batch_idx = 100

    tracker_cnn = models.SteeringWheelTrackerCNNModel()
    tracker_cnn.train()

    optimizer = optim.Adam(tracker_cnn.parameters())

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
    if checkpoint is not None:
        tracker_cnn.load_state_dict(checkpoint["tracker_cnn_state_dict"])

    if Config.GPU >= 0:
        tracker_cnn.cuda(Config.GPU)
        criterion.cuda(Config.GPU)

    # initialize image augmentation cascade
    rarely = lambda aug: iaa.Sometimes(0.1, aug)
    sometimes = lambda aug: iaa.Sometimes(0.2, aug)
    often = lambda aug: iaa.Sometimes(0.4, aug)
    augseq = iaa.Sequential([
            sometimes(iaa.Crop(percent=(0, 0.025))),
            rarely(iaa.GaussianBlur((0, 1.0))), # blur images with a sigma between 0 and 3.0
            rarely(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255), per_channel=0.5)), # add gaussian noise to images
            often(iaa.Dropout(
                iap.FromLowerResolution(
                    other_param=iap.Binomial(1 - 0.2),
                    size_px=(2, 16)
                ),
                per_channel=0.2
            )),
            often(iaa.Add((-20, 20), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
            often(iaa.Multiply((0.8, 1.2), per_channel=0.25)), # change brightness of images (50-150% of original value)
            often(iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5)), # improve or worsen the contrast
            often(iaa.Affine(
                scale={"x": (0.8, 1.3), "y": (0.8, 1.3)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-0, 0),
                shear=(-0, 0),
                order=[0, 1],
                cval=(0, 255),
                mode=["constant", "edge"]
            )),
            rarely(iaa.Grayscale(alpha=(0.0, 1.0)))
        ],
        random_order=True # do all of the above in random order
    )

    #memory = replay_memory.ReplayMemory.get_instance_supervised()
    batch_loader_train = BatchLoader(val=False, augseq=augseq, queue_size=15, nb_workers=4)
    batch_loader_val = BatchLoader(val=True, augseq=iaa.Noop(), queue_size=NB_VAL_BATCHES, nb_workers=2)

    start_batch_idx = 0 if checkpoint is None else checkpoint["batch_idx"] + 1
    for batch_idx in xrange(start_batch_idx, NB_BATCHES):
        run_batch(batch_idx, False, batch_loader_train, tracker_cnn, criterion, optimizer, history, (batch_idx % 20) == 0)

        if (batch_idx+1) % VAL_EVERY == 0:
            for i in xrange(NB_VAL_BATCHES):
                run_batch(batch_idx, True, batch_loader_val, tracker_cnn, criterion, optimizer, history, i == 0)

        if (batch_idx+1) % PLOT_EVERY == 0:
            loss_plotter.plot(history)

        # every N batches, save a checkpoint
        if (batch_idx+1) % SAVE_EVERY == 0:
            torch.save({
                "batch_idx": batch_idx,
                "history": history.to_string(),
                "tracker_cnn_state_dict": tracker_cnn.state_dict()
            }, "steering_wheel.tar")

def run_batch(batch_idx, val, batch_loader, tracker_cnn, criterion, optimizer, history, save_debug_image):
    """Train or validate on a single batch."""
    train = not val
    time_cbatch_start = time.time()
    inputs, outputs_gt = batch_loader.get_batch()
    if Config.GPU >= 0:
        inputs = to_cuda(to_variable(inputs, volatile=val), Config.GPU)
        outputs_gt_bins = to_cuda(to_variable(np.argmax(outputs_gt, axis=1), volatile=val, requires_grad=False), Config.GPU)
        outputs_gt = to_cuda(to_variable(outputs_gt, volatile=val, requires_grad=False), Config.GPU)
    time_cbatch_end = time.time()

    time_fwbw_start = time.time()
    if train:
        optimizer.zero_grad()
    outputs_pred = tracker_cnn(inputs)
    outputs_pred_sm = F.softmax(outputs_pred)
    loss = criterion(outputs_pred, outputs_gt_bins)
    if train:
        loss.backward()
        optimizer.step()
    time_fwbw_end = time.time()

    loss = loss.data.cpu().numpy()[0]
    outputs_pred_np = to_numpy(outputs_pred_sm)
    outputs_gt_np = to_numpy(outputs_gt)
    acc = np.sum(np.equal(np.argmax(outputs_pred_np, axis=1), np.argmax(outputs_gt_np, axis=1))) / BATCH_SIZE
    history.add_value("loss", "train" if train else "val", batch_idx, loss, average=val)
    history.add_value("acc", "train" if train else "val", batch_idx, acc, average=val)
    print("[%s] Batch %05d | loss %.8f | acc %.2f | cbatch %.04fs | fwbw %.04fs" % ("T" if train else "V", batch_idx, loss, acc, time_cbatch_end - time_cbatch_start, time_fwbw_end - time_fwbw_start))

    if save_debug_image:
        debug_img = generate_debug_image(inputs, outputs_gt, outputs_pred_sm)
        misc.imsave("debug_img_%s.jpg" % ("train" if train else "val"), debug_img)

def generate_debug_image(inputs, outputs_gt, outputs_pred):
    """Draw an image with current ground truth and predictions for debug purposes."""
    current_image = inputs.data[0].cpu().numpy()
    current_image = np.clip(current_image * 255, 0, 255).astype(np.uint8).transpose((1, 2, 0))
    current_image = ia.imresize_single_image(current_image, (32*4, 64*4))
    h, w = current_image.shape[0:2]
    outputs_gt = to_numpy(outputs_gt)[0]
    outputs_pred = to_numpy(outputs_pred)[0]

    binwidth = 6
    outputs_grid = np.zeros((20+2, outputs_gt.shape[0]*binwidth, 3), dtype=np.uint8)
    for angle_bin_idx in xrange(outputs_gt.shape[0]):
        val = outputs_pred[angle_bin_idx]
        x_start = angle_bin_idx*binwidth
        x_end = (angle_bin_idx+1)*binwidth
        fill_start = 1
        fill_end = 1 + int(20*val)
        #print(angle_bin_idx, x_start, x_end, fill_start, fill_end, outputs_grid.shape, outputs_grid[fill_start:fill_end, x_start+1:x_end].shape)
        if fill_start < fill_end:
            outputs_grid[fill_start:fill_end, x_start+1:x_end] = [255, 255, 255]

        bordercol = [128, 128, 128] if outputs_gt[angle_bin_idx] < 1 else [0, 0, 255]
        outputs_grid[0:22, x_start:x_start+1] = bordercol
        outputs_grid[0:22, x_end:x_end+1] = bordercol
        outputs_grid[0, x_start:x_end+1] = bordercol
        outputs_grid[21, x_start:x_end+1] = bordercol

    outputs_grid = outputs_grid[::-1, :, :]

    bin_gt = np.argmax(outputs_gt)
    bin_pred = np.argmax(outputs_pred)
    angles = [(binidx*ANGLE_BIN_SIZE) - 180 for binidx in [bin_gt, bin_pred]]

    #print(outputs_grid.shape)
    current_image = np.pad(current_image, ((0, 128), (0, 400), (0, 0)), mode="constant")
    current_image[h+4:h+4+22, 4:4+outputs_grid.shape[1], :] = outputs_grid
    current_image = util.draw_text(current_image, x=4, y=h+4+22+4, text="GT: %03.2fdeg\nPR: %03.2fdeg" % (angles[0], angles[1]), size=10)

    return current_image

def extract_steering_wheel_image(screenshot_rs):
    """Extract the part of a screenshot (resized to 180x320 HxW) that usually
    contains the steering wheel."""
    h, w = screenshot_rs.shape[0:2]
    x1 = int(w * (470/1280))
    x2 = int(w * (840/1280))
    y1 = int(h * (500/720))
    y2 = int(h * (720/720))
    return screenshot_rs[y1:y2+1, x1:x2+1, :]

def downscale_image(steering_wheel_image):
    """Downscale an image to the model's input sizes (height, width)."""
    return ia.imresize_single_image(
        steering_wheel_image,
        (MODEL_HEIGHT, MODEL_WIDTH),
        interpolation="linear"
    )

def load_random_state(memory, depth=0):
    """Load a single random state from the replay memory which has a steering
    wheel position attached (estimated via classical means)."""
    rndidx = random.randint(memory.id_min, memory.id_max)
    state = memory.get_state_by_id(rndidx)
    if state.steering_wheel_classical is None:
        if depth+1 >= 200:
            raise Exception("Maximum depth reached in load_random_state(), \
                too many states with None in column steering_wheel_classical. \
                Use scripts/add_steering_wheel.py to recalculate missing values.")
        return load_random_state(memory, depth=depth+1)
    else:
        return state

def load_random_batch(memory, augseq, batch_size):
    """Load a random batch from the replay memory for training.
    augseq contains the image augmentation sequence to use."""
    inputs = np.zeros((batch_size, MODEL_HEIGHT, MODEL_WIDTH, 3), dtype=np.uint8)
    outputs = np.zeros((batch_size, 360//ANGLE_BIN_SIZE), dtype=np.float32)

    for b_idx in xrange(batch_size):
        state = load_random_state(memory)
        subimg = extract_steering_wheel_image(state.screenshot_rs)
        subimg = augseq.augment_image(subimg)
        subimg = downscale_image(subimg)
        inputs[b_idx] = subimg
        deg = state.steering_wheel_classical % 360
        if -360 <= deg < -180:
            deg = 360 - deg
        elif -180 <= deg < 0:
            pass
        elif 0 <= deg < 180:
            pass
        elif 180 <= deg < 360:
            deg = -360 + deg
        deg = 180 + deg
        bin_idx = int(deg / ANGLE_BIN_SIZE)
        outputs[b_idx, bin_idx] = 1

    inputs = (inputs / 255).astype(np.float32).transpose((0, 3, 1, 2))

    return inputs, outputs

class BatchLoader(object):
    """Class to load batches in multiple background processes."""
    def __init__(self, val, queue_size, augseq, nb_workers, threaded=False):
        self.queue = multiprocessing.Queue(queue_size)
        self.workers = []
        for i in range(nb_workers):
            seed = random.randint(0, 10**6)
            augseq_worker = augseq.deepcopy()
            if threaded:
                worker = threading.Thread(target=self._load_batches, args=(val, self.queue, augseq_worker, None))
            else:
                worker = multiprocessing.Process(target=self._load_batches, args=(val, self.queue, augseq_worker, seed))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def get_batch(self):
        return pickle.loads(self.queue.get())

    def _load_batches(self, val, queue, augseq_worker, seed):
        if seed is None:
            random.seed(seed)
            np.random.seed(seed)
            augseq_worker.reseed(seed)
            ia.seed(seed)
        memory = replay_memory.ReplayMemory.create_instance_reinforced(val=val)

        while True:
            batch = load_random_batch(memory, augseq_worker, BATCH_SIZE)
            queue.put(pickle.dumps(batch, protocol=-1))

if __name__ == "__main__":
    main()
