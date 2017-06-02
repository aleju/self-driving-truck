from __future__ import division, print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import train
from annotate.annotate_attributes import ATTRIBUTE_GROUPS
from lib.util import to_numpy
from lib import util

import imgaug as ia
import numpy as np

try:
    xrange
except NameError:
    xrange = range

def generate_debug_image(images, images_prev, \
    outputs_ae_gt, outputs_grids_gt, outputs_atts_gt, \
    outputs_multiactions_gt, outputs_flow_gt, outputs_canny_gt, \
    outputs_flipped_gt, \
    outputs_ae_pred, outputs_grids_pred, outputs_atts_pred, \
    outputs_multiactions_pred, outputs_flow_pred, outputs_canny_pred, \
    outputs_flipped_pred, \
    grids_annotated, atts_annotated):
    image = to_numpy(images)[0]
    grids_annotated = grids_annotated[0]
    atts_annotated = atts_annotated[0]

    ae_gt = to_numpy(outputs_ae_gt)[0]
    grids_gt = to_numpy(outputs_grids_gt)[0]
    atts_gt = to_numpy(outputs_atts_gt)[0]
    multiactions_gt = to_numpy(outputs_multiactions_gt)[0]
    flow_gt = to_numpy(outputs_flow_gt)[0]
    canny_gt = to_numpy(outputs_canny_gt)[0]
    flipped_gt = to_numpy(outputs_flipped_gt)[0]

    ae_pred = to_numpy(outputs_ae_pred)[0]
    grids_pred = to_numpy(outputs_grids_pred)[0]
    atts_pred = to_numpy(outputs_atts_pred)[0]
    multiactions_pred = to_numpy(outputs_multiactions_pred)[0]
    flow_pred = to_numpy(outputs_flow_pred)[0]
    canny_pred = to_numpy(outputs_canny_pred)[0]
    flipped_pred = to_numpy(outputs_flipped_pred)[0]

    image = (np.squeeze(image).transpose(1, 2, 0) * 255).astype(np.uint8)
    ae_pred = (np.squeeze(ae_pred).transpose(1, 2, 0) * 255).astype(np.uint8)
    grids_gt = (np.squeeze(grids_gt).transpose(1, 2, 0) * 255).astype(np.uint8)
    grids_pred = (np.squeeze(grids_pred).transpose(1, 2, 0) * 255).astype(np.uint8)
    atts_gt = np.squeeze(atts_gt)
    atts_pred = np.squeeze(atts_pred)
    multiactions_gt = np.squeeze(multiactions_gt)
    multiactions_pred = np.squeeze(multiactions_pred)
    flow_gt = (flow_gt.transpose(1, 2, 0) * 255).astype(np.uint8)
    flow_pred = (flow_pred.transpose(1, 2, 0) * 255).astype(np.uint8)
    canny_gt = (canny_gt.transpose(1, 2, 0) * 255).astype(np.uint8)
    canny_pred = (canny_pred.transpose(1, 2, 0) * 255).astype(np.uint8)

    #print(image.shape, grid_gt.shape, grid_pred.shape)

    h, w = int(image.shape[0]*0.5), int(image.shape[1]*0.5)
    #h, w = image.shape[0:2]
    image_rs = ia.imresize_single_image(image, (h, w), interpolation="cubic")
    grids_vis = []
    for i in xrange(grids_gt.shape[2]):
        grid_gt_rs = ia.imresize_single_image(grids_gt[..., i][:, :, np.newaxis], (h, w), interpolation="cubic")
        grid_pred_rs = ia.imresize_single_image(grids_pred[..., i][:, :, np.newaxis], (h, w), interpolation="cubic")
        grid_gt_hm = util.draw_heatmap_overlay(image_rs, np.squeeze(grid_gt_rs) / 255)
        grid_pred_hm = util.draw_heatmap_overlay(image_rs, np.squeeze(grid_pred_rs) / 255)
        if grids_annotated[i] == 0:
            grid_gt_hm[::4, ::4, :] = [255, 0, 0]
        grids_vis.append(np.hstack((grid_gt_hm, grid_pred_hm)))

    """
    lst = [image[0:3]] \
        + [image[3:6]] \
        + [ia.imresize_single_image(ae_pred[:, :, 0:3], (image.shape[0], image.shape[1]), interpolation="cubic")] \
        + [ia.imresize_single_image(ae_pred[:, :, 3:6], (image.shape[0], image.shape[1]), interpolation="cubic")] \
        + [ia.imresize_single_image(np.tile(flow_pred[:, :, 0][:, :, np.newaxis], (1, 1, 3)), (image.shape[0], image.shape[1]), interpolation="cubic")] \
        + [ia.imresize_single_image(np.tile(flow_pred[:, :, 1][:, :, np.newaxis], (1, 1, 3)), (image.shape[0], image.shape[1]), interpolation="cubic")] \
        + grids_vis
    print([s.shape for s in lst])
    """

    def downscale(im):
        return ia.imresize_single_image(im, (image.shape[0]//2, image.shape[1]//2), interpolation="cubic")

    def to_rgb(im):
        if im.ndim == 2:
            im = im[:, :, np.newaxis]
        return np.tile(im, (1, 1, 3))

    #print(canny_gt.shape, canny_gt[...,0].shape, to_rgb(canny_gt[...,0]).shape, downscale(to_rgb(canny_gt[...,0])).shape)
    #print(canny_pred.shape, canny_gt[...,0].shape, to_rgb(canny_pred[...,0]).shape, downscale(to_rgb(canny_pred[...,0])).shape)
    current_image = np.vstack(
        #[image[:, :, 0:3]]
        [image[:, :, 0:3]]
        + grids_vis
        + [np.hstack([
            downscale(ae_pred[:, :, 0:3]),
            downscale(to_rgb(ae_pred[:, :, 3]))
        ])]
        + [np.hstack([
            downscale(to_rgb(ae_pred[:, :, 4])),
        #    downscale(to_rgb(ae_pred[:, :, 5]))
            np.zeros_like(downscale(to_rgb(ae_pred[:, :, 4])))
        ])]
        + [np.hstack([
            downscale(to_rgb(flow_gt[..., 0])),
            downscale(to_rgb(flow_pred[..., 0]))
        ])]
        + [np.hstack([
            downscale(to_rgb(canny_gt[..., 0])),
            downscale(to_rgb(canny_pred[..., 0]))
        ])]
    )
    y_grids_start = image.shape[0]
    grid_height = grids_vis[0].shape[0]
    for i, name in enumerate(train.GRIDS_ORDER):
        current_image = util.draw_text(current_image, x=2, y=y_grids_start+(i+1)*grid_height-12, text=name, size=8, color=[0, 255, 0])

    current_image = np.pad(current_image, ((0, 280), (0, 280), (0, 0)), mode="constant", constant_values=0)
    texts = []
    att_idx = 0
    for i, att_group in enumerate(ATTRIBUTE_GROUPS):
        texts.append(att_group.name_shown)
        for j, att in enumerate(att_group.attributes):
            if atts_annotated[0] == 0:
                texts.append(" %s | ? | %.2f" % (att.name, atts_pred[att_idx]))
            else:
                texts.append(" %s | %.2f | %.2f" % (att.name, atts_gt[att_idx], atts_pred[att_idx]))
            att_idx += 1
    current_image = util.draw_text(current_image, x=current_image.shape[1]-256+1, y=1, text="\n".join(texts), size=8, color=[0, 255, 0])

    ma_texts = ["multiactions (prev avg, next avg, curr, next)"]
    counter = 0
    while counter < multiactions_gt.shape[0]:
        ma_texts_sub = ([], [])
        for j in xrange(9):
            ma_texts_sub[0].append("%.2f" % (multiactions_gt[counter],))
            ma_texts_sub[1].append("%.2f" % (multiactions_pred[counter],))
            counter += 1
        ma_texts.append(" ".join(ma_texts_sub[0]))
        ma_texts.append(" ".join(ma_texts_sub[1]))
        ma_texts.append("")
    current_image = util.draw_text(current_image, x=current_image.shape[1]-256+1, y=650, text="\n".join(ma_texts), size=8, color=[0, 255, 0])

    flipped_texts = [
        "flipped",
        " ".join(["%.2f" % (flipped_gt[i],) for i in xrange(flipped_gt.shape[0])]),
        " ".join(["%.2f" % (flipped_pred[i],) for i in xrange(flipped_pred.shape[0])])
    ]
    current_image = util.draw_text(current_image, x=current_image.shape[1]-256+1, y=810, text="\n".join(flipped_texts), size=8, color=[0, 255, 0])

    return current_image
