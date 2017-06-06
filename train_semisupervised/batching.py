"""Functions/classes related to loading batches for training."""
from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import train

import numpy as np
import cv2
import imgaug as ia
import multiprocessing
import threading
import cPickle as pickle
import random
from skimage import feature

try:
    xrange
except NameError:
    xrange = range

def create_batch(examples, examples_autogen, augseq):
    """Convert datasets of examples to batches.

    This method will pick random examples from the datasets "examples"
    (25 percent per batch) and "examples_autogen" (75 percent per batch).

    Parameters
    ----------
    examples : list of Example
        One or more hand-annotated examples to use. The Example class
        is defined in dataset.py.
    examples_autogen : list of Example
        One or more automatically annoated examples to use.
    augseq : Augmenter
        Augmentation sequence to use.

    Returns
    ----------
    (images, images_prev), (outputs_ae, outputs_grids, outputs_atts, outputs_multiactions, outputs_flow, outputs_canny, outputs_flips), (grids_annotated, atts_annotated)
    where
        images are the current timestep input images
        images_prev are for each batch element the previous timestep input images
        outputs_ae are the ground truth outputs for the autoencoder (decoder)
        outputs_grids are the ground truth hand-annotated grids (e.g. car positions in images)
        outputs_atts are the ground truth hand-annotated attributes (e.g. number of lanes)
        outputs_multiactions are the ground truth action outputs (e.g. next action one-hot-vector)
        outputs_flow are the ground truth optical flow outputs
        outputs_canny are the ground truth canny edge outputs
        outputs_flips are the ground truth of flipped input images (temporal flips)
        grids_annotated is a (B, G) array that contains a 1 if for batch element b grid g was annotated
        atts_annotated is a (B, 1) array that contains a 1 if for batch element b attributes were annotated
    """
    B = train.BATCH_SIZE

    img_h, img_w = train.MODEL_HEIGHT, train.MODEL_WIDTH
    img_prev_h, img_prev_w = train.MODEL_PREV_HEIGHT, train.MODEL_PREV_WIDTH
    ae_h, ae_w = int(img_h*(1/train.AE_DOWNSCALE_FACTOR)), int(img_w*(1/train.AE_DOWNSCALE_FACTOR))
    grids_h, grids_w = int(img_h*(1/train.GRIDS_DOWNSCALE_FACTOR)), int(img_w*(1/train.GRIDS_DOWNSCALE_FACTOR))
    flow_h, flow_w = int(img_h*(1/train.FLOW_DOWNSCALE_FACTOR)), int(img_w*(1/train.FLOW_DOWNSCALE_FACTOR))
    canny_h, canny_w = int(img_h*(1/train.CANNY_DOWNSCALE_FACTOR)), int(img_w*(1/train.CANNY_DOWNSCALE_FACTOR))

    images = np.zeros((B, img_h, img_w, 3), dtype=np.float32)
    images_prev = np.zeros((B, img_prev_h, img_prev_w, len(train.PREVIOUS_STATES_DISTANCES)), dtype=np.float32)
    outputs_ae = np.zeros((B, ae_h, ae_w, 3 + len(train.PREVIOUS_STATES_DISTANCES)), dtype=np.float32)
    outputs_grids = np.zeros((B, grids_h, grids_w, len(train.ANNOTATIONS_GRIDS_FPS)), dtype=np.float32)
    outputs_atts = np.zeros((B, train.NB_ATTRIBUTE_VALUES), dtype=np.float32)
    outputs_multiactions = np.zeros((B, 9 + 9 + 9 + 9), dtype=np.float32)
    outputs_flow = np.zeros((B, flow_h, flow_w, 1), dtype=np.float32)
    outputs_canny = np.zeros((B, canny_h, canny_w, len(train.CANNY_SIGMAS)), dtype=np.float32)
    outputs_flips = np.zeros((B, len(train.PREVIOUS_STATES_DISTANCES)), dtype=np.float32)
    grids_annotated = np.zeros((B, len(train.GRIDS_ORDER)), dtype=np.int32)
    atts_annotated = np.zeros((B, 1), dtype=np.int32)
    for b_idx in xrange(B):
        if b_idx == 0 or b_idx % 4 == 0:
            rnd_idx = random.randint(0, len(examples)-1)
            example = examples[rnd_idx]
        else:
            rnd_idx = random.randint(0, len(examples_autogen)-1)
            example = examples_autogen[rnd_idx]

        img_curr = downscale(example.screenshot_rs, img_h, img_w)
        img_curr_y = gray(img_curr)
        imgs_prev = downscale(example.previous_screenshots_rs, img_prev_h, img_prev_w)
        imgs_prev_y = gray(imgs_prev)

        if random.random() < train.P_FLIP:
            flip_ids = random.sample(range(len(imgs_prev)), 2)
            imgs_prev[flip_ids[1]], imgs_prev[flip_ids[0]] = imgs_prev[flip_ids[0]], imgs_prev[flip_ids[1]]
            outputs_flips[b_idx, flip_ids[0]] = 1
            outputs_flips[b_idx, flip_ids[1]] = 1
            #misc.imshow(
            #    np.vstack([
            #        np.hstack(imgs_prev_orig),
            #        np.hstack(imgs_prev)
            #    ])
            #)

        grids, grids_annotated_one = example.get_grids_array()
        grids = (grids*255).astype(np.uint8)
        atts, has_attributes = example.get_attributes_array()

        flow_settings = [
            {"pyr_scale": 0.5, "levels": 3, "winsize": 3, "iterations": 2, "poly_n": 5, "poly_sigma": 1.2, "flags": 0}
        ]
        flow_imgs = []
        flow_curr_y_rs = downscale(img_curr_y, flow_h, flow_w)
        flow_prev_y_rs = gray(downscale(example.previous_screenshots_rs[0], flow_h, flow_w))
        for settings in flow_settings:
            # API of calcOpticalFlowFarneback is apparently different in python 3
            if sys.version_info[0] == 2:
                flow = cv2.calcOpticalFlowFarneback(
                    flow_prev_y_rs,
                    flow_curr_y_rs,
                    **settings
                )
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    flow_prev_y_rs,
                    flow_curr_y_rs,
                    None,
                    **settings
                )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            """
            def to_rgb(im):
                return np.clip(np.tile(im[...,np.newaxis], (1, 1, 3)), 0, 255).astype(np.uint8)
            misc.imshow(np.vstack([
                np.hstack([imgs_prev[0], img_curr]),
                np.hstack([to_rgb(mag*2), to_rgb(ang*2)]),
                np.hstack([to_rgb(np.log(mag+0.00001)*255), to_rgb(np.log(ang*0.5)*255)]),
                np.hstack([util.draw_heatmap_overlay(img_curr, np.log(mag+0.00001)), util.draw_heatmap_overlay(img_curr, np.log(ang*0.5))])
            ]))
            """

            mag = np.clip(np.log(mag + 0.00001), 0, 1)
            ang = np.clip(np.log(ang*0.5 + 0.00001), 0, 1)
            flow_img = mag[:, :, np.newaxis]
            flow_img = (flow_img * 255).astype(np.uint8)
            #misc.imshow(np.hstack([to_rgb(flow_img[...,0]), to_rgb(flow_img[...,1])]))
            flow_imgs.append(flow_img)

        augseq_det = augseq.to_deterministic()
        augseq_det_alt = augseq_det.deepcopy()
        augs_to_keep = augseq_det_alt.find_augmenters_by_name(".*(Affine|Crop|Fliplr).*", regex=True)
        augseq_det_alt = augseq_det_alt.remove_augmenters(lambda a, parents: a not in augs_to_keep)
        img_curr_aug = augseq_det.augment_image(img_curr)

        imgs_prev_y_aug = [augseq_det.augment_image(img) for img in imgs_prev_y]
        imgs_prev_y_aug = np.array(imgs_prev_y_aug, dtype=np.uint8).transpose((1, 2, 0))

        img_aug = img_curr_aug
        img_ae_aug = np.dstack([img_curr_aug, downscale(imgs_prev_y_aug, ae_h, ae_w)])

        img_ae_aug_rs = downscale(img_ae_aug, ae_h, ae_w)
        grids_rs = downscale(grids, grids_h, grids_w)
        flow_imgs_rs = downscale(flow_imgs, flow_h, flow_w)

        flow_imgs_rs_aug = [augseq_det_alt.augment_image(flow_img) for flow_img in flow_imgs_rs]
        grids_rs_aug = augseq_det_alt.augment_image(grids_rs)

        #misc.imshow(np.hstack([to_rgb(flow_img_aug_rs[...,0]*255), to_rgb(flow_img_aug_rs[...,1]*255)]))

        img_curr_y_canny = downscale(img_curr_y, canny_h, canny_w)
        imgs_canny = []
        for sidx, sigma in enumerate(train.CANNY_SIGMAS):
            imgs_canny.append(feature.canny(img_curr_y_canny, sigma=sigma))
        imgs_canny = (np.array(imgs_canny) * 255).astype(np.uint8).transpose((1, 2, 0))
        imgs_canny_aug = augseq_det_alt.augment_image(imgs_canny)
        """
        misc.imshow(
            np.vstack([
                np.hstack([imgs_canny[...,i] for i in xrange(imgs_canny.shape[2])]),
                np.hstack([imgs_canny_aug[...,i] for i in xrange(imgs_canny_aug.shape[2])])
            ])
        )
        """

        images[b_idx, ...] = img_aug / 255
        images_prev[b_idx, ...] = imgs_prev_y_aug / 255
        outputs_ae[b_idx, ...] = img_ae_aug_rs / 255
        outputs_grids[b_idx, ...] = grids_rs_aug / 255
        outputs_atts[b_idx, ...] = atts
        outputs_multiactions[b_idx, 0:9] = example.previous_multiaction_vecs_avg
        outputs_multiactions[b_idx, 9:18] = example.next_multiaction_vecs_avg
        outputs_multiactions[b_idx, 18:27] = example.multiaction_vec
        outputs_multiactions[b_idx, 27:36] = example.next_multiaction_vec
        for i, flow_img_rs_aug in enumerate(flow_imgs_rs_aug):
            outputs_flow[b_idx, :, :, i] = np.squeeze(flow_img_rs_aug / 255)
        outputs_canny[b_idx, ...] = imgs_canny_aug / 255
        grids_annotated[b_idx, ...] = grids_annotated_one
        atts_annotated[b_idx, ...] = has_attributes

        """
        def to_rgb(im):
            return np.tile(im[:,:,np.newaxis], (1, 1, 3))
        misc.imshow(
            np.hstack([
                images[b_idx, ..., 0:3],
                to_rgb(images[b_idx, ..., 3]),
                to_rgb(images[b_idx, ..., 4]),
                to_rgb(images[b_idx, ..., 5])
            ])
        )
        """

    images = images.transpose(0, 3, 1, 2)
    images_prev = images_prev.transpose(0, 3, 1, 2)
    outputs_ae = outputs_ae.transpose(0, 3, 1, 2)
    outputs_grids = outputs_grids.transpose(0, 3, 1, 2)
    outputs_flow = outputs_flow.transpose(0, 3, 1, 2)
    outputs_canny = outputs_canny.transpose(0, 3, 1, 2)

    #for arr in [images, outputs_ae, outputs_grids, outputs_atts, outputs_multiactions, outputs_flow, outputs_flips]:
    #    print(np.min(arr), np.average(arr), np.max(arr))

    return (images, images_prev), (outputs_ae, outputs_grids, outputs_atts, outputs_multiactions, outputs_flow, outputs_canny, outputs_flips), (grids_annotated, atts_annotated)

def downscale(im, h, w):
    """Downscale one or more images to size (h, w)."""
    if isinstance(im, list):
        return [downscale(i, h, w) for i in im]
    else:
        if im.ndim == 2:
            im = im[:, :, np.newaxis]
            im_rs = ia.imresize_single_image(im, (h, w), interpolation="cubic")
            return np.squeeze(im)
        else:
            return ia.imresize_single_image(im, (h, w), interpolation="cubic")

def gray(im):
    """Convert one or more images to grayscale."""
    if isinstance(im, list):
        return [gray(i) for i in im]
    else:
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

def to_rgb(im):
    """Convert an image from (h, w) to (h, w, 3)."""
    if im.ndim == 3:
        if im.shape[2] == 3:
            return im
        else:
            return np.tile(im, (1, 1, 3))
    else:
        return np.tile(im[:, :, np.newaxis], (1, 1, 3))

class BatchLoader(object):
    """Class to generate batches in multiple background processes."""

    def __init__(self, dataset, dataset_autogen, queue_size, augseq, nb_workers, threaded=False):
        self.queue = multiprocessing.Queue(queue_size)
        self.workers = []
        for i in range(nb_workers):
            seed = random.randint(0, 10**6)
            augseq_worker = augseq.deepcopy()
            augseq_worker.reseed()
            if threaded:
                worker = threading.Thread(target=self._load_batches, args=(dataset, dataset_autogen, self.queue, augseq_worker, seed))
            else:
                worker = multiprocessing.Process(target=self._load_batches, args=(dataset, dataset_autogen, self.queue, augseq_worker, seed))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def get_batch(self):
        return pickle.loads(self.queue.get())

    def _load_batches(self, dataset, dataset_autogen, queue, augseq_worker, seed):
        random.seed(seed)
        np.random.seed(seed)
        while True:
            batch = create_batch(dataset, dataset_autogen, augseq_worker)
            queue.put(pickle.dumps(batch, protocol=-1))

    def join(self):
        for worker in self.workers:
            worker.terminate()
