from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib import actions as actionslib
from lib import util
from lib.util import to_numpy

import imgaug as ia
import numpy as np
import torch.nn.functional as F

try:
    xrange
except NameError:
    xrange = range

def generate_overview_image(current_state, last_state, \
    action_up_down_bpe, action_left_right_bpe, \
    memory, memory_val, \
    ticks, last_train_tick, \
    plans, plan_to_rewards_direct, plan_to_reward_indirect, \
    plan_to_reward, plans_ranking, current_plan, best_plan_ae_decodings,
    idr_v, idr_adv,
    grids, args):
    h, w = current_state.screenshot_rs.shape[0:2]
    scr = np.copy(current_state.screenshot_rs)
    scr = ia.imresize_single_image(scr, (h//2, w//2))

    if best_plan_ae_decodings is not None:
        ae_decodings = (to_numpy(best_plan_ae_decodings) * 255).astype(np.uint8).transpose((0, 2, 3, 1))
        ae_decodings = [ia.imresize_single_image(ae_decodings[i, ...], (h//4, w//4)) for i in xrange(ae_decodings.shape[0])]
        ae_decodings = ia.draw_grid(ae_decodings, cols=5)
        #ae_decodings = np.vstack([
        #    np.hstack(ae_decodings[0:5]),
        #    np.hstack(ae_decodings[5:10])
        #])
    else:
        ae_decodings = np.zeros((1, 1, 3), dtype=np.uint8)

    if grids is not None:
        scr_rs = ia.imresize_single_image(scr, (h//4, w//4))
        grids = (to_numpy(grids)[0] * 255).astype(np.uint8)
        grids = [ia.imresize_single_image(grids[i, ...][:,:,np.newaxis], (h//4, w//4)) for i in xrange(grids.shape[0])]
        grids = [util.draw_heatmap_overlay(scr_rs, np.squeeze(grid/255).astype(np.float32)) for grid in grids]
        grids = ia.draw_grid(grids, cols=4)
    else:
        grids = np.zeros((1, 1, 3), dtype=np.uint8)

    plans_text = []

    if idr_v is not None and idr_adv is not None:
        idr_v = to_numpy(idr_v[0])
        idr_adv = to_numpy(idr_adv[0])
        plans_text.append("V(s): %+07.2f" % (idr_v[0],))
        adv_texts = []
        curr = []
        for i, ma in enumerate(actionslib.ALL_MULTIACTIONS):
            curr.append("A(%s%s): %+07.2f" % (ma[0] if ma[0] != "~WS" else "_", ma[1] if ma[1] != "~AD" else "_", idr_adv[i]))
            if (i+1) % 3 == 0 or (i+1) == len(actionslib.ALL_MULTIACTIONS):
                adv_texts.append(" ".join(curr))
                curr = []
        plans_text.extend(adv_texts)

    if current_plan is not None:
        plans_text.append("")
        plans_text.append("Current Plan:")
        actions_ud_text = []
        actions_lr_text = []
        for multiaction in current_plan:
            actions_ud_text.append("%s" % (multiaction[0] if multiaction[0] != "~WS" else "_",))
            actions_lr_text.append("%s" % (multiaction[1] if multiaction[1] != "~AD" else "_",))
        plans_text.extend([" ".join(actions_ud_text), " ".join(actions_lr_text)])

    plans_text.append("")
    plans_text.append("Best Plans:")
    if plan_to_rewards_direct is not None:
        for plan_idx in plans_ranking[::-1][0:5]:
            plan = plans[plan_idx]
            rewards_direct = plan_to_rewards_direct[plan_idx]
            reward_indirect = plan_to_reward_indirect[plan_idx]
            reward = plan_to_reward[plan_idx]
            actions_ud_text = []
            actions_lr_text = []
            rewards_text = []
            for multiaction in plan:
                actions_ud_text.append("%s" % (multiaction[0] if multiaction[0] != "~WS" else "_",))
                actions_lr_text.append("%s" % (multiaction[1] if multiaction[1] != "~AD" else "_",))
            for rewards_t in rewards_direct:
                rewards_text.append("%+04.1f" % (rewards_t,))
            rewards_text.append("| %+07.2f (V(s')=%+07.2f)" % (reward, reward_indirect))
            plans_text.extend(["", " ".join(actions_ud_text), " ".join(actions_lr_text), " ".join(rewards_text)])
    plans_text = "\n".join(plans_text)

    stats_texts = [
        "u/d bpe: %s" % (action_up_down_bpe.rjust(5)),
        "  l/r bpe: %s" % (action_left_right_bpe.rjust(5)),
        "u/d ape: %s %s" % (current_state.action_up_down.rjust(5), "[C]" if action_up_down_bpe != current_state.action_up_down else ""),
        "  l/r ape: %s %s" % (current_state.action_left_right.rjust(5), "[C]" if action_left_right_bpe != current_state.action_left_right else ""),
        "speed: %03d" % (current_state.speed,) if current_state.speed is not None else "speed: None",
        "is_reverse: yes" if current_state.is_reverse else "is_reverse: no",
        "is_damage_shown: yes" if current_state.is_damage_shown else "is_damage_shown: no",
        "is_offence_shown: yes" if current_state.is_offence_shown else "is_offence_shown: no",
        "steering wheel: %05.2f (%05.2f)" % (current_state.steering_wheel_cnn, current_state.steering_wheel_raw_cnn),
        "reward for last state: %05.2f" % (last_state.reward,) if last_state is not None else "reward for last state: None",
        "p_explore: %.2f%s" % (current_state.p_explore if args.p_explore is None else args.p_explore, "" if args.p_explore is None else " (constant)"),
        "memory size (train/val): %06d / %06d" % (memory.size, memory_val.size),
        "ticks: %06d" % (ticks,),
        "last train: %06d" % (last_train_tick,)
    ]
    stats_text = "\n".join(stats_texts)

    all_texts = plans_text + "\n\n\n" + stats_text

    result = np.zeros((720, 590, 3), dtype=np.uint8)
    util.draw_image(result, x=0, y=0, other_img=scr, copy=False)
    util.draw_image(result, x=0, y=scr.shape[0]+10, other_img=ae_decodings, copy=False)
    util.draw_image(result, x=0, y=scr.shape[0]+10+ae_decodings.shape[0]+10, other_img=grids, copy=False)
    result = util.draw_text(result, x=0, y=scr.shape[0]+10+ae_decodings.shape[0]+10+grids.shape[0]+10, size=8, text=all_texts, color=[255, 255, 255])
    return result

def generate_training_debug_image(inputs_supervised, inputs_supervised_prev, \
    outputs_dr_preds, outputs_dr_gt, \
    outputs_idr_preds, outputs_idr_gt, \
    outputs_successor_preds, outputs_successor_gt, \
    outputs_ae_preds, outputs_ae_gt, \
    outputs_dr_successors_preds, outputs_dr_successors_gt, \
    outputs_idr_successors_preds, outputs_idr_successors_gt,
    multiactions):
    imgs_in = to_numpy(inputs_supervised)[0]
    imgs_in = np.clip(imgs_in * 255, 0, 255).astype(np.uint8).transpose((1, 2, 0))
    imgs_in_prev = to_numpy(inputs_supervised_prev)[0]
    imgs_in_prev = np.clip(imgs_in_prev * 255, 0, 255).astype(np.uint8).transpose((1, 2, 0))
    h, w = imgs_in.shape[0:2]
    imgs_in = np.vstack([
        np.hstack([downscale(imgs_in[..., 0:3]), downscale(to_rgb(imgs_in_prev[..., 0]))]),
        #np.hstack([downscale(to_rgb(imgs_in_prev[..., 1])), downscale(to_rgb(imgs_in_prev[..., 2]))])
        np.hstack([downscale(to_rgb(imgs_in_prev[..., 1])), np.zeros_like(imgs_in[..., 0:3])])
    ])
    h_imgs = imgs_in.shape[0]

    ae_gt = np.clip(to_numpy(outputs_ae_gt)[0] * 255, 0, 255).astype(np.uint8).transpose((1, 2, 0))
    ae_preds = np.clip(to_numpy(outputs_ae_preds)[0] * 255, 0, 255).astype(np.uint8).transpose((1, 2, 0))
    """
    imgs_ae = np.vstack([
        downscale(ae_preds[..., 0:3]),
        downscale(to_rgb(ae_preds[..., 3])),
        downscale(to_rgb(ae_preds[..., 4])),
        downscale(to_rgb(ae_preds[..., 5]))
    ])
    """
    imgs_ae = np.hstack([downscale(ae_gt), downscale(ae_preds)])
    h_ae = imgs_ae.shape[0]

    outputs_successor_dr_grid = draw_successor_dr_grid(
        to_numpy(F.softmax(outputs_dr_successors_preds[:, 0, :])),
        to_numpy(outputs_dr_successors_gt[:, 0]),
        upscale_factor=(2, 4)
    )

    outputs_dr_preds = to_numpy(F.softmax(outputs_dr_preds))[0]
    outputs_dr_gt = to_numpy(outputs_dr_gt)[0]
    grid_preds = output_grid_to_image(outputs_dr_preds[np.newaxis, :], upscale_factor=(2, 4))
    grid_gt = output_grid_to_image(outputs_dr_gt[np.newaxis, :], upscale_factor=(2, 4))
    imgs_dr = np.hstack([
        grid_gt,
        np.zeros((grid_gt.shape[0], 4, 3), dtype=np.uint8),
        grid_preds,
        np.zeros((grid_gt.shape[0], 8, 3), dtype=np.uint8),
        outputs_successor_dr_grid
    ])
    successor_multiactions_str = " ".join(["%s%s" % (ma[0] if ma[0] != "~WS" else "_", ma[1] if ma[1] != "~AD" else "_") for ma in multiactions[0]])
    imgs_dr = np.pad(imgs_dr, ((30, 0), (0, 300), (0, 0)), mode="constant", constant_values=0)
    imgs_dr = util.draw_text(imgs_dr, x=0, y=0, text="DR curr bins gt:%s, pred:%s | successor preds\nsucc. mas: %s" % (str(np.argmax(outputs_dr_gt)), str(np.argmax(outputs_dr_preds)), successor_multiactions_str), size=9)
    h_dr = imgs_dr.shape[0]

    outputs_idr_preds = np.squeeze(to_numpy(outputs_idr_preds)[0])
    outputs_idr_gt = np.squeeze(to_numpy(outputs_idr_gt)[0])
    idr_text = [
        "[IndirectReward A0]",
        "  gt: %.2f" % (outputs_idr_gt[..., 0],),
        "  pr: %.2f" % (outputs_idr_preds[..., 0],),
        "[IndirectReward A1]",
        "  gt: %.2f" % (outputs_idr_gt[..., 1],),
        "  pr: %.2f" % (outputs_idr_preds[..., 1],),
        "[IndirectReward A2]",
        "  gt: %.2f" % (outputs_idr_gt[..., 2],),
        "  pr: %.2f" % (outputs_idr_preds[..., 2],)
    ]
    idr_text = "\n".join(idr_text)

    outputs_successor_preds = np.squeeze(to_numpy(outputs_successor_preds)[:, 0, :])
    outputs_successor_gt = np.squeeze(to_numpy(outputs_successor_gt)[:, 0, :])
    distances = np.average((outputs_successor_preds - outputs_successor_gt) ** 2, axis=1)
    successors_text = [
        "[Successors]",
        "  Distances:",
        "    " + " ".join(["%02.2f" % (d,) for d in distances]),
        "  T=0 gt/pred:",
        "    " + " ".join(["%+02.2f" % (val,) for val in outputs_successor_gt[0, 0:25]]),
        "    " + " ".join(["%+02.2f" % (val,) for val in outputs_successor_preds[0, 0:25]]),
        "  T=1 gt/pred:",
        "    " + " ".join(["%+02.2f" % (val,) for val in outputs_successor_gt[1, 0:25]]),
        "    " + " ".join(["%+02.2f" % (val,) for val in outputs_successor_preds[1, 0:25]]),
        "  T=2 gt/pred:",
        "    " + " ".join(["%+02.2f" % (val,) for val in outputs_successor_gt[2, 0:25]]),
        "    " + " ".join(["%+02.2f" % (val,) for val in outputs_successor_preds[2, 0:25]]),
    ]
    successors_text = "\n".join(successors_text)

    outputs_dr_successors_preds = np.squeeze(to_numpy(outputs_dr_successors_preds)[:, 0, :])
    outputs_dr_successors_gt = np.squeeze(to_numpy(outputs_dr_successors_gt)[:, 0, :])
    bins_dr_successors_preds = np.argmax(outputs_dr_successors_preds, axis=1)
    bins_dr_successors_gt = np.argmax(outputs_dr_successors_gt, axis=1)
    successors_dr_text = [
        "[Direct rewards bins of successors]",
        "  gt:   " + " ".join(["%d" % (b,) for b in bins_dr_successors_gt]),
        "  pred: " + " ".join(["%d" % (b,) for b in bins_dr_successors_preds])
    ]
    successors_dr_text = "\n".join(successors_dr_text)

    outputs_idr_successors_preds = np.squeeze(to_numpy(outputs_idr_successors_preds)[:, 0, :])
    outputs_idr_successors_gt = np.squeeze(to_numpy(outputs_idr_successors_gt)[:, 0, :])
    successors_idr_text = [
        "[Indirect rewards of successors A0]",
        "  gt:   " + " ".join(["%+03.2f" % (v,) for v in outputs_idr_successors_gt[..., 0]]),
        "  pred: " + " ".join(["%+03.2f" % (v,) for v in outputs_idr_successors_preds[..., 0]]),
        "[Indirect rewards of successors A1]",
        "  gt:   " + " ".join(["%+03.2f" % (v,) for v in outputs_idr_successors_gt[..., 1]]),
        "  pred: " + " ".join(["%+03.2f" % (v,) for v in outputs_idr_successors_preds[..., 1]]),
        "[Indirect rewards of successors A2]",
        "  gt:   " + " ".join(["%+03.2f" % (v,) for v in outputs_idr_successors_gt[..., 2]]),
        "  pred: " + " ".join(["%+03.2f" % (v,) for v in outputs_idr_successors_preds[..., 2]])
    ]
    successors_idr_text = "\n".join(successors_idr_text)

    result = np.zeros((950, 320, 3), dtype=np.uint8)
    spacing = 4
    util.draw_image(result, x=0, y=0, other_img=imgs_in, copy=False)
    util.draw_image(result, x=0, y=h_imgs+spacing, other_img=imgs_ae, copy=False)
    util.draw_image(result, x=0, y=h_imgs+spacing+h_ae+spacing, other_img=imgs_dr, copy=False)
    result = util.draw_text(result, x=0, y=h_imgs+spacing+h_ae+spacing+h_dr+spacing, text=idr_text + "\n" + successors_text + "\n" + successors_dr_text + "\n" + successors_idr_text, size=9)

    return result

def to_rgb(im):
    return np.tile(im[:,:,np.newaxis], (1, 1, 3))

def downscale(im):
    return ia.imresize_single_image(im, (90, 160), interpolation="cubic")

def output_grid_to_image(output_grid, upscale_factor=(4, 4)):
    if output_grid is None:
        grid_vis = np.zeros((Config.MODEL_NB_REWARD_BINS, Config.MODEL_NB_FUTURE_BLOCKS), dtype=np.uint8)
    else:
        if output_grid.ndim == 3:
            output_grid = output_grid[0]
        grid_vis = (output_grid.transpose((1, 0)) * 255).astype(np.uint8)
    grid_vis = np.tile(grid_vis[:, :, np.newaxis], (1, 1, 3))
    if output_grid is None:
        grid_vis[::2, ::2, :] = [255, 0, 0]
    grid_vis = ia.imresize_single_image(grid_vis, (grid_vis.shape[0]*upscale_factor[0], grid_vis.shape[1]*upscale_factor[1]), interpolation="nearest")
    grid_vis = np.pad(grid_vis, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=128)
    return grid_vis

def draw_successor_dr_grid(outputs_dr_successors_preds, outputs_dr_successors_gt, upscale_factor=(4, 4)):
    T, S = outputs_dr_successors_preds.shape
    cols = []
    for t in range(T):
        col = (outputs_dr_successors_preds[t][np.newaxis, :].transpose((1, 0)) * 255).astype(np.uint8)
        col = np.tile(col[:, :, np.newaxis], (1, 1, 3))
        correct_bin_idx = np.argmax(outputs_dr_successors_gt[t])
        col[correct_bin_idx, 0, 2] = 255
        col = ia.imresize_single_image(col, (col.shape[0]*upscale_factor[0], col.shape[1]*upscale_factor[1]), interpolation="nearest")
        col = np.pad(col, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=128)
        cols.append(col)
    return np.hstack(cols)
