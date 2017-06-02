"""Train a semi-supervised model."""
from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import models
from batching import BatchLoader
from visualization import generate_debug_image
from dataset import (
    load_dataset_annotated,
    load_dataset_annotated_compressed,
    load_dataset_autogen
)

from annotate.annotate_attributes import ATTRIBUTE_GROUPS
from lib.util import to_variable, to_cuda, to_numpy
from lib import plotting
from config import Config

from scipy import misc
import imgaug as ia
from imgaug import augmenters as iaa
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

try:
    xrange
except NameError:
    xrange = range

DEBUG = False
MODEL_HEIGHT = 90
MODEL_WIDTH = 160
MODEL_PREV_HEIGHT = 45
MODEL_PREV_WIDTH = 80
NB_BATCHES = 50000
BATCH_SIZE = 32
GPU = 0
SAVE_EVERY = 100
VAL_EVERY = 250
NB_VAL_BATCHES = 16
PLOT_EVERY = 100
NB_VAL_SPLIT = 128
NB_AUTOGEN_VAL = 512
NB_AUTOGEN_TRAIN = 20000 if not DEBUG else 256
LOSS_AE_WEIGHTING = 0.2
LOSS_GRIDS_WEIGHTING = 0.3
LOSS_ATTRIBUTES_WEIGHTING = 0.1
LOSS_MULTIACTIONS_WEIGHTING = 0.1
LOSS_FLOW_WEIGHTING = 0.1
LOSS_CANNY_WEIGHTING = 0.1
LOSS_FLIPPED_WEIGHTING = 0.1
P_FLIP = 0.2
CANNY_SIGMAS = [1.0]
ANNOTATIONS_GRIDS_FPS = [
    ("street_boundary_grid", os.path.join(Config.ANNOTATIONS_DIR, "annotations.pickle")),
    ("cars_grid", os.path.join(Config.ANNOTATIONS_DIR, "annotations_cars.pickle")),
    ("cars_mirrors_grid", os.path.join(Config.ANNOTATIONS_DIR, "annotations_cars_mirrors.pickle")),
    ("crashables_grid", os.path.join(Config.ANNOTATIONS_DIR, "annotations_crashables.pickle")),
    ("current_lane_grid", os.path.join(Config.ANNOTATIONS_DIR, "annotations_current_lane.pickle")),
    ("lanes_same_direction_grid", os.path.join(Config.ANNOTATIONS_DIR, "annotations_lanes_same_direction.pickle")),
    ("steering_wheel_grid", os.path.join(Config.ANNOTATIONS_DIR, "annotations_steering_wheel.pickle")),
    ("street_markings_grid", os.path.join(Config.ANNOTATIONS_DIR, "annotations_street_markings.pickle"))
]

ANNOTATIONS_ATTS_FPS = [
    ("attributes", os.path.join(Config.ANNOTATIONS_DIR, "annotations_attributes.pickle"))
]

ANNOTATIONS_COMPRESSED_FP = os.path.join(Config.ANNOTATIONS_DIR, "semisupervised_annotations.pickle.gz")
USE_COMPRESSED_ANNOTATIONS = True

GRIDS_ORDER = [key for (key, fp) in ANNOTATIONS_GRIDS_FPS]
ATTRIBUTES_ORDER = []
for att_group in ATTRIBUTE_GROUPS:
    ATTRIBUTES_ORDER.append(att_group.name)

ATTRIBUTE_GROUPS_BY_NAME = dict([(att_group.name, att_group) for att_group in ATTRIBUTE_GROUPS])
NB_ATTRIBUTE_VALUES = sum([len(att_group.attributes) for att_group in ATTRIBUTE_GROUPS])

GRIDS_DOWNSCALE_FACTOR = 1
AE_DOWNSCALE_FACTOR = 1
FLOW_DOWNSCALE_FACTOR = 1
CANNY_DOWNSCALE_FACTOR = 1

PREVIOUS_STATES_DISTANCES = [1, 2]

def main():
    """Initialize/load model, dataset, optimizers, history and loss
    plotter, augmentation sequence. Then start training loop."""

    parser = argparse.ArgumentParser(description="Train semisupervised model")
    parser.add_argument('--nocontinue', default=False, action="store_true", help="Whether to NOT continue the previous experiment", required=False)
    parser.add_argument('--withshortcuts', default=False, action="store_true", help="Whether to train a model with shortcuts from downscaling to upscaling layers.", required=False)
    args = parser.parse_args()

    checkpoint_fp = "train_semisupervised_model%s.tar" % ("_withshortcuts" if args.withshortcuts else "",)
    if os.path.isfile(checkpoint_fp) and not args.nocontinue:
        checkpoint = torch.load(checkpoint_fp)
    else:
        checkpoint = None

    # load or initialize loss history
    if checkpoint is not None:
        history = plotting.History.from_string(checkpoint["history"])
    else:
        history = plotting.History()
        history.add_group("loss-ae", ["train", "val"], increasing=False)
        history.add_group("loss-grids", ["train", "val"], increasing=False)
        history.add_group("loss-atts", ["train", "val"], increasing=False)
        history.add_group("loss-multiactions", ["train", "val"], increasing=False)
        history.add_group("loss-flow", ["train", "val"], increasing=False)
        history.add_group("loss-canny", ["train", "val"], increasing=False)
        history.add_group("loss-flipped", ["train", "val"], increasing=False)

    # initialize loss plotter
    loss_plotter = plotting.LossPlotter(
        history.get_group_names(),
        history.get_groups_increasing(),
        save_to_fp="train_semisupervised_plot%s.jpg" % ("_withshortcuts" if args.withshortcuts else "",)
    )
    loss_plotter.start_batch_idx = 100

    # initialize and load model
    predictor = models.Predictor() if not args.withshortcuts else models.PredictorWithShortcuts()
    if checkpoint is not None:
        predictor.load_state_dict(checkpoint["predictor_state_dict"])
    predictor.train()

    # initialize optimizer
    optimizer_predictor = optim.Adam(predictor.parameters())

    # initialize losses
    criterion_ae = nn.MSELoss()
    criterion_grids = nn.BCELoss()
    criterion_atts = nn.BCELoss()
    criterion_multiactions = nn.BCELoss()
    criterion_flow = nn.BCELoss()
    criterion_canny = nn.BCELoss()
    criterion_flipped = nn.BCELoss()

    # send everything to gpu
    if GPU >= 0:
        predictor.cuda(GPU)
        criterion_ae.cuda(GPU)
        criterion_grids.cuda(GPU)
        criterion_atts.cuda(GPU)
        criterion_multiactions.cuda(GPU)
        criterion_flow.cuda(GPU)
        criterion_canny.cuda(GPU)
        criterion_flipped.cuda(GPU)

    # initialize image augmentation cascade
    rarely = lambda aug: iaa.Sometimes(0.1, aug)
    sometimes = lambda aug: iaa.Sometimes(0.2, aug)
    often = lambda aug: iaa.Sometimes(0.3, aug)
    augseq = iaa.Sequential([
            #iaa.Fliplr(0.5),
            often(iaa.Crop(percent=(0, 0.05))),
            sometimes(iaa.GaussianBlur((0, 0.2))), # blur images with a sigma between 0 and 3.0
            often(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5)), # add gaussian noise to images
            often(iaa.Dropout((0.0, 0.05), per_channel=0.5)),
            #often(iaa.Sometimes(0.5,
            #    iaa.Dropout((0.0, 0.05), per_channel=0.5),
            #    iaa.Dropout(
            #        iap.FromLowerResolution(
            #            other_param=iap.Binomial(1 - 0.2),
            #            size_px=(2, 16)
            #        ),
            #        per_channel=0.2
            #    )
            #)),
            rarely(iaa.Sharpen(alpha=(0, 0.7), lightness=(0.75, 1.5))), # sharpen images
            rarely(iaa.Emboss(alpha=(0, 0.7), strength=(0, 2.0))), # emboss images
            rarely(iaa.Sometimes(0.5,
                iaa.EdgeDetect(alpha=(0, 0.4)),
                iaa.DirectedEdgeDetect(alpha=(0, 0.4), direction=(0.0, 1.0)),
            )),
            often(iaa.Add((-20, 20), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
            often(iaa.Multiply((0.8, 1.2), per_channel=0.25)), # change brightness of images (50-150% of original value)
            often(iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5)), # improve or worsen the contrast
            #sometimes(iaa.Grayscale(alpha=(0.0, 1.0))),
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.07, 0.07), "y": (-0.07, 0.07)},
                rotate=(0, 0),
                shear=(0, 0),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            ))
        ],
        random_order=True # do all of the above in random order
    )

    # load datasets
    print("Loading dataset...")
    if USE_COMPRESSED_ANNOTATIONS:
        examples = load_dataset_annotated_compressed()
    else:
        examples = load_dataset_annotated()
    #examples_annotated_ids = set([ex.state_idx for ex in examples])
    examples_annotated_ids = set()
    examples_autogen_val = load_dataset_autogen(val=True, nb_load=NB_AUTOGEN_VAL, not_in=examples_annotated_ids)
    examples_autogen_train = load_dataset_autogen(val=False, nb_load=NB_AUTOGEN_TRAIN, not_in=examples_annotated_ids)
    random.shuffle(examples)
    random.shuffle(examples_autogen_val)
    random.shuffle(examples_autogen_train)
    examples_val = examples[0:NB_VAL_SPLIT]
    examples_train = examples[NB_VAL_SPLIT:]

    # initialize background batch loaders
    #memory = replay_memory.ReplayMemory.get_instance_supervised()
    batch_loader_train = BatchLoader(examples_train, examples_autogen_train, augseq=augseq, queue_size=15, nb_workers=4, threaded=False)
    batch_loader_val = BatchLoader(examples_val, examples_autogen_val, augseq=iaa.Noop(), queue_size=NB_VAL_BATCHES, nb_workers=1, threaded=False)

    # training loop
    print("Training...")
    start_batch_idx = 0 if checkpoint is None else checkpoint["batch_idx"] + 1
    for batch_idx in xrange(start_batch_idx, NB_BATCHES):
        # train on batch

        # load batch data
        time_cbatch_start = time.time()
        (inputs, inputs_prev), (outputs_ae_gt, outputs_grids_gt_orig, outputs_atts_gt_orig, outputs_multiactions_gt, outputs_flow_gt, outputs_canny_gt, outputs_flipped_gt), (grids_annotated, atts_annotated) = batch_loader_train.get_batch()
        inputs = to_cuda(to_variable(inputs), GPU)
        inputs_prev = to_cuda(to_variable(inputs_prev), GPU)
        outputs_ae_gt = to_cuda(to_variable(outputs_ae_gt, requires_grad=False), GPU)
        outputs_multiactions_gt = to_cuda(to_variable(outputs_multiactions_gt, requires_grad=False), GPU)
        outputs_flow_gt = to_cuda(to_variable(outputs_flow_gt, requires_grad=False), GPU)
        outputs_canny_gt = to_cuda(to_variable(outputs_canny_gt, requires_grad=False), GPU)
        outputs_flipped_gt = to_cuda(to_variable(outputs_flipped_gt, requires_grad=False), GPU)
        time_cbatch_end = time.time()

        # predict and compute losses
        time_fwbw_start = time.time()
        optimizer_predictor.zero_grad()
        (outputs_ae_pred, outputs_grids_pred, outputs_atts_pred, outputs_multiactions_pred, outputs_flow_pred, outputs_canny_pred, outputs_flipped_pred, emb) = predictor(inputs, inputs_prev)
        # zero-grad some outputs where annotations are not available for specific examples
        outputs_grids_gt = remove_unannotated_grids_gt(outputs_grids_pred, outputs_grids_gt_orig, grids_annotated)
        outputs_grids_gt = to_cuda(to_variable(outputs_grids_gt, requires_grad=False), GPU)
        outputs_atts_gt = remove_unannotated_atts_gt(outputs_atts_pred, outputs_atts_gt_orig, atts_annotated)
        outputs_atts_gt = to_cuda(to_variable(outputs_atts_gt, requires_grad=False), GPU)
        loss_ae = criterion_ae(outputs_ae_pred, outputs_ae_gt)
        loss_grids = criterion_grids(outputs_grids_pred, outputs_grids_gt)
        loss_atts = criterion_atts(outputs_atts_pred, outputs_atts_gt)
        loss_multiactions = criterion_multiactions(outputs_multiactions_pred, outputs_multiactions_gt)
        loss_flow = criterion_flow(outputs_flow_pred, outputs_flow_gt)
        loss_canny = criterion_canny(outputs_canny_pred, outputs_canny_gt)
        loss_flipped = criterion_flipped(outputs_flipped_pred, outputs_flipped_gt)
        losses_grad_lst = [
            loss.data.new().resize_as_(loss.data).fill_(w) for loss, w in zip(
                [loss_ae, loss_grids, loss_atts, loss_multiactions, loss_flow, loss_canny, loss_flipped],
                [LOSS_AE_WEIGHTING, LOSS_GRIDS_WEIGHTING, LOSS_ATTRIBUTES_WEIGHTING, LOSS_MULTIACTIONS_WEIGHTING, LOSS_FLOW_WEIGHTING, LOSS_CANNY_WEIGHTING, LOSS_FLIPPED_WEIGHTING]
            )
        ]
        torch.autograd.backward([loss_ae, loss_grids, loss_atts, loss_multiactions, loss_flow, loss_canny, loss_flipped], losses_grad_lst)
        optimizer_predictor.step()
        time_fwbw_end = time.time()

        # add losses to history and output a message
        loss_ae_value = to_numpy(loss_ae)[0]
        loss_grids_value = to_numpy(loss_grids)[0]
        loss_atts_value = to_numpy(loss_atts)[0]
        loss_multiactions_value = to_numpy(loss_multiactions)[0]
        loss_flow_value = to_numpy(loss_flow)[0]
        loss_canny_value = to_numpy(loss_canny)[0]
        loss_flipped_value = to_numpy(loss_flipped)[0]
        history.add_value("loss-ae", "train", batch_idx, loss_ae_value)
        history.add_value("loss-grids", "train", batch_idx, loss_grids_value)
        history.add_value("loss-atts", "train", batch_idx, loss_atts_value)
        history.add_value("loss-multiactions", "train", batch_idx, loss_multiactions_value)
        history.add_value("loss-flow", "train", batch_idx, loss_flow_value)
        history.add_value("loss-canny", "train", batch_idx, loss_canny_value)
        history.add_value("loss-flipped", "train", batch_idx, loss_flipped_value)
        print("[T] Batch %05d L[ae=%.4f, grids=%.4f, atts=%.4f, multiactions=%.4f, flow=%.4f, canny=%.4f, flipped=%.4f] T[cbatch=%.04fs, fwbw=%.04fs]" % (batch_idx, loss_ae_value, loss_grids_value, loss_atts_value, loss_multiactions_value, loss_flow_value, loss_canny_value, loss_flipped_value, time_cbatch_end - time_cbatch_start, time_fwbw_end - time_fwbw_start))

        # genrate a debug image showing batch predictions and ground truths
        if (batch_idx+1) % 20 == 0:
            debug_img = generate_debug_image(
                inputs, inputs_prev,
                outputs_ae_gt, outputs_grids_gt_orig, outputs_atts_gt_orig, outputs_multiactions_gt, outputs_flow_gt, outputs_canny_gt, outputs_flipped_gt,
                outputs_ae_pred, outputs_grids_pred, outputs_atts_pred, outputs_multiactions_pred, outputs_flow_pred, outputs_canny_pred, outputs_flipped_pred,
                grids_annotated, atts_annotated
            )
            misc.imsave(
                "train_semisupervised_debug_img%s.jpg" % ("_withshortcuts" if args.withshortcuts else "",),
                debug_img
            )

        # run N validation batches
        # TODO merge this with training stuff above (one function for both)
        if (batch_idx+1) % VAL_EVERY == 0:
            predictor.eval()
            loss_ae_total = 0
            loss_grids_total = 0
            loss_atts_total = 0
            loss_multiactions_total = 0
            loss_flow_total = 0
            loss_canny_total = 0
            loss_flipped_total = 0
            for i in xrange(NB_VAL_BATCHES):
                time_cbatch_start = time.time()
                (inputs, inputs_prev), (outputs_ae_gt, outputs_grids_gt_orig, outputs_atts_gt_orig, outputs_multiactions_gt, outputs_flow_gt, outputs_canny_gt, outputs_flipped_gt), (grids_annotated, atts_annotated) = batch_loader_val.get_batch()
                inputs = to_cuda(to_variable(inputs, volatile=True), GPU)
                inputs_prev = to_cuda(to_variable(inputs_prev, volatile=True), GPU)
                outputs_ae_gt = to_cuda(to_variable(outputs_ae_gt, volatile=True), GPU)
                outputs_multiactions_gt = to_cuda(to_variable(outputs_multiactions_gt, volatile=True), GPU)
                outputs_flow_gt = to_cuda(to_variable(outputs_flow_gt, volatile=True), GPU)
                outputs_canny_gt = to_cuda(to_variable(outputs_canny_gt, volatile=True), GPU)
                outputs_flipped_gt = to_cuda(to_variable(outputs_flipped_gt, volatile=True), GPU)
                time_cbatch_end = time.time()

                time_fwbw_start = time.time()
                (outputs_ae_pred, outputs_grids_pred, outputs_atts_pred, outputs_multiactions_pred, outputs_flow_pred, outputs_canny_pred, outputs_flipped_pred, emb) = predictor(inputs, inputs_prev)
                outputs_grids_gt = remove_unannotated_grids_gt(outputs_grids_pred, outputs_grids_gt_orig, grids_annotated)
                outputs_grids_gt = to_cuda(to_variable(outputs_grids_gt, volatile=True), GPU)
                outputs_atts_gt = remove_unannotated_atts_gt(outputs_atts_pred, outputs_atts_gt_orig, atts_annotated)
                outputs_atts_gt = to_cuda(to_variable(outputs_atts_gt, volatile=True), GPU)
                loss_ae = criterion_ae(outputs_ae_pred, outputs_ae_gt)
                loss_grids = criterion_grids(outputs_grids_pred, outputs_grids_gt)
                loss_atts = criterion_atts(outputs_atts_pred, outputs_atts_gt)
                loss_multiactions = criterion_multiactions(outputs_multiactions_pred, outputs_multiactions_gt)
                loss_flow = criterion_flow(outputs_flow_pred, outputs_flow_gt)
                loss_canny = criterion_canny(outputs_canny_pred, outputs_canny_gt)
                loss_flipped = criterion_flipped(outputs_flipped_pred, outputs_flipped_gt)
                time_fwbw_end = time.time()

                loss_ae_value = to_numpy(loss_ae)[0]
                loss_grids_value = to_numpy(loss_grids)[0]
                loss_atts_value = to_numpy(loss_atts)[0]
                loss_multiactions_value = to_numpy(loss_multiactions)[0]
                loss_flow_value = to_numpy(loss_flow)[0]
                loss_canny_value = to_numpy(loss_canny)[0]
                loss_flipped_value = to_numpy(loss_flipped)[0]
                loss_ae_total += loss_ae_value
                loss_grids_total += loss_grids_value
                loss_atts_total += loss_atts_value
                loss_multiactions_total += loss_multiactions_value
                loss_flow_total += loss_flow_value
                loss_canny_total += loss_canny_value
                loss_flipped_total += loss_flipped_value
                print("[V] Batch %05d L[ae=%.4f, grids=%.4f, atts=%.4f, multiactions=%.4f, flow=%.4f, canny=%.4f, flipped=%.4f] T[cbatch=%.04fs, fwbw=%.04fs]" % (batch_idx, loss_ae_value, loss_grids_value, loss_atts_value, loss_multiactions_value, loss_flow_value, loss_canny_value, loss_flipped_value, time_cbatch_end - time_cbatch_start, time_fwbw_end - time_fwbw_start))

                if i == 0:
                    debug_img = generate_debug_image(
                        inputs, inputs_prev,
                        outputs_ae_gt, outputs_grids_gt_orig, outputs_atts_gt_orig, outputs_multiactions_gt, outputs_flow_gt, outputs_canny_gt, outputs_flipped_gt,
                        outputs_ae_pred, outputs_grids_pred, outputs_atts_pred, outputs_multiactions_pred, outputs_flow_pred, outputs_canny_pred, outputs_flipped_pred,
                        grids_annotated, atts_annotated
                    )
                    misc.imsave(
                        "train_semisupervised_debug_img_val%s.jpg" % ("_withshortcuts" if args.withshortcuts else "",),
                        debug_img
                    )
            history.add_value("loss-ae", "val", batch_idx, loss_ae_total / NB_VAL_BATCHES)
            history.add_value("loss-grids", "val", batch_idx, loss_grids_total / NB_VAL_BATCHES)
            history.add_value("loss-atts", "val", batch_idx, loss_atts_total / NB_VAL_BATCHES)
            history.add_value("loss-multiactions", "val", batch_idx, loss_multiactions_total / NB_VAL_BATCHES)
            history.add_value("loss-flow", "val", batch_idx, loss_flow_total / NB_VAL_BATCHES)
            history.add_value("loss-canny", "val", batch_idx, loss_canny_total / NB_VAL_BATCHES)
            history.add_value("loss-flipped", "val", batch_idx, loss_flipped_total / NB_VAL_BATCHES)
            predictor.train()

        # generate loss plot
        if (batch_idx+1) % PLOT_EVERY == 0:
            loss_plotter.plot(history)

        # every N batches, save a checkpoint
        if (batch_idx+1) % SAVE_EVERY == 0:
            checkpoint_fp = "train_semisupervised_model%s.tar" % ("_withshortcuts" if args.withshortcuts else "",)
            torch.save({
                "batch_idx": batch_idx,
                "history": history.to_string(),
                "predictor_state_dict": predictor.state_dict(),
            }, checkpoint_fp)

        # refresh automatically generated examples (autoencoder, canny edge stuff etc.)
        if (batch_idx+1) % 1000 == 0:
            print("Refreshing autogen dataset...")
            batch_loader_train.join()
            examples_autogen_train = load_dataset_autogen(val=False, nb_load=NB_AUTOGEN_TRAIN, not_in=examples_annotated_ids)
            batch_loader_train = BatchLoader(examples_train, examples_autogen_train, augseq=augseq, queue_size=15, nb_workers=4, threaded=False)

def remove_unannotated_grids_gt(outputs_grids_pred, outputs_grids_gt, grids_annotated):
    """Zero-grad grid outputs for which there is no annotation data for an
    example."""
    gt2 = np.copy(outputs_grids_gt)
    pred = to_numpy(outputs_grids_pred)
    for b_idx in xrange(grids_annotated.shape[0]):
        for grid_idx in xrange(grids_annotated.shape[1]):
            if grids_annotated[b_idx, grid_idx] == 0:
                gt2[b_idx, grid_idx, ...] = pred[b_idx, grid_idx, ...]
    return gt2

def remove_unannotated_atts_gt(outputs_atts_pred, outputs_atts_gt, atts_annotated):
    """Zero-grad attribute outputs for which there is no annotation data for an
    example."""
    gt2 = np.copy(outputs_atts_gt)
    pred = to_numpy(outputs_atts_pred)
    for b_idx in xrange(atts_annotated.shape[0]):
        if atts_annotated[b_idx, 0] == 0:
            gt2[b_idx, ...] = pred[b_idx, ...]
    return gt2

if __name__ == "__main__":
    main()
