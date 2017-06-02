from __future__ import division, print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from train_semisupervised.train import (
    PREVIOUS_STATES_DISTANCES,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    MODEL_PREV_HEIGHT,
    MODEL_PREV_WIDTH
)
from batching import states_to_batch
from train_semisupervised import models as models_semisupervised
from lib.util import to_variable, to_cuda, to_numpy
from lib import util
from lib import actions as actionslib
from config import Config

import imgaug as ia
from imgaug import augmenters as iaa
import cPickle as pickle
import argparse
import numpy as np
import cv2
import gzip as gz
import torch
import torch.nn.functional as F
import collections
from scipy import misc, ndimage
from skimage import draw
import glob

try:
    xrange
except NameError:
    xrange = range

def main():
    parser = argparse.ArgumentParser(description="Generate frames for a video. This requires a previous recording done via train.py --record='filename'.")
    parser.add_argument("--record", default=None, help="Filepath to recording.", required=True)
    parser.add_argument("--outdir", default="video_output", help="Output directory name.", required=False)
    parser.add_argument("--y", default=False, action="store_true", help="Always let ffmpeg/avconv overwrite existing video files without asking.", required=False)
    parser.add_argument("--noredraw", default=False, action="store_true", help="Skip drawing frames for videos for which the target directory already exists.", required=False)
    args = parser.parse_args()
    assert args.record is not None

    if "*" in args.record:
        record_fps = glob.glob(args.record)
        assert len(record_fps) > 0
        for record_fp in record_fps:
            fn = os.path.basename(record_fp)
            assert fn != ""
            fn_root = fn[0:fn.index(".")] #os.path.splitext(fn)
            assert fn_root != ""
            outdir = os.path.join(args.outdir, fn_root + "-" + str(abs(hash(record_fp)))[0:8])

            process(record_fp, outdir, args.y, args.noredraw)
    else:
        process(args.record, args.outdir, args.y, args.noredraw)

def process(record_fp, outdir_frames, y, noredraw):
    if outdir_frames[-1] == "/":
        outdir_videos = os.path.dirname(outdir_frames[:-1])
        dirname_frames_last = os.path.basename(outdir_frames[:-1])
    else:
        outdir_videos = os.path.dirname(outdir_frames)
        dirname_frames_last = os.path.basename(outdir_frames)
    assert os.path.isfile(record_fp)

    print("Processing recording '%s'..." % (record_fp,))
    print("Writing frames to '%s', videos to '%s' with filename start '%s'..." % (outdir_frames, outdir_videos, dirname_frames_last))

    with gz.open(record_fp, "rb") as f:
        recording = pickle.load(f)

    if os.path.exists(outdir_frames) and noredraw:
        print("Video frames were already drawn, not redrawing")
    else:
        if not os.path.exists(outdir_frames):
            print("Target directory for frames does not exist, creating it...")
            os.makedirs(outdir_frames)

        fd = FrameDrawer(outdir_videos)
        for fidx, frames_drawn in enumerate(fd.draw_frames(recording)):
            print("Frame %06d of around %06d..." % (fidx, len(recording["frames"])))
            frame_plans = frames_drawn[0]
            frame_atts = frames_drawn[1]
            frame_grids = frames_drawn[2]
            if frame_plans is not None:
                misc.imsave(os.path.join(outdir_frames, "plans_%06d.jpg" % (fidx,)), frame_plans)
            if frame_atts is not None:
                misc.imsave(os.path.join(outdir_frames, "atts_%06d.jpg" % (fidx,)), frame_atts)
            if frame_grids is not None:
                misc.imsave(os.path.join(outdir_frames, "grids_%06d.jpg" % (fidx,)), frame_grids)
            #if fidx > 200:
            #    break

    if not os.path.exists(outdir_videos):
        print("Target directory for videos does not exist, creating it...")
        os.makedirs(outdir_videos)

    frame_fps = ["plans_%06d.jpg", "atts_%06d.jpg", "grids_%06d.jpg"]
    frame_fps = [os.path.join(outdir_frames, fp) for fp in frame_fps]
    video_fps = ["plans.mp4", "atts.mp4", "grids.mp4"]
    video_fps = [os.path.join(outdir_videos, "%s-%s" % (dirname_frames_last, fp)) for fp in video_fps]

    for frame_fp, video_fp in zip(frame_fps, video_fps):
        #os.system('avconv %s -framerate 10 -i "%s" -crf 25 -b:v 2000k -vcodec mpeg4 %s' % ("-y" if y else "", frame_fp, video_fp))
        os.system('avconv %s -framerate 10 -i "%s" -crf 25 -b:v 2000k -vcodec h264 %s' % ("-y" if y else "", frame_fp, video_fp))

class FrameDrawer(object):
    def __init__(self, outdir):
        self.outdir = outdir

        checkpoint_supervised = torch.load("../train_semisupervised/train_semisupervised_model_withshortcuts.tar")
        embedder_supervised = models_semisupervised.PredictorWithShortcuts()
        embedder_supervised.eval()
        embedder_supervised.load_state_dict(checkpoint_supervised["predictor_state_dict"])
        if Config.GPU >= 0:
            embedder_supervised.cuda(Config.GPU)
        self.embedder_supervised = embedder_supervised

    def draw_frames(self, recording):
        previous_states = collections.deque(maxlen=max(PREVIOUS_STATES_DISTANCES))
        for frame in recording["frames"]:
            scr = util.decompress_img(frame["scr"])
            scr = np.clip(scr.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
            current_state = frame["state"]

            current_plan_idx = frame["current_plan_idx"]
            current_plan_step_idx = frame["current_plan_step_idx"]
            idr_v = frame["idr_v"]
            idr_adv = frame["idr_adv"]
            plan_to_rewards_direct = frame["plan_to_rewards_direct"]
            plan_to_reward_indirect = frame["plan_to_reward_indirect"]
            plan_to_reward = frame["plan_to_reward"]
            plans_ranking = frame["plans_ranking"]

            if current_plan_idx is not None:
                frame_plans = self.draw_frame_plans(
                    scr, current_state,
                    recording["plans"],
                    current_plan_idx, current_plan_step_idx,
                    idr_v, idr_adv,
                    plan_to_rewards_direct, plan_to_reward_indirect, plan_to_reward,
                    plans_ranking
                )
            else:
                frame_plans = None

            if len(previous_states) == previous_states.maxlen:
                batch = states_to_batch([list(previous_states)], [[current_state]], iaa.Noop(), PREVIOUS_STATES_DISTANCES, MODEL_HEIGHT, MODEL_WIDTH, MODEL_PREV_HEIGHT, MODEL_PREV_WIDTH)
                inputs_supervised = batch.inputs_supervised(volatile=True, gpu=Config.GPU)

                x_ae, x_grids, x_atts, x_ma, x_flow, x_canny, x_flipped, x_emb = self.embedder_supervised.forward(inputs_supervised[0], inputs_supervised[1])

                frame_attributes = self.draw_frame_attributes(scr, x_atts)
                frame_grids = self.draw_frame_grids(scr, x_grids)
            else:
                frame_attributes = None
                frame_grids = None

            yield (frame_plans, frame_attributes, frame_grids)

            previous_states.append(current_state)

    def draw_frame_plans(self, scr, state, plans, current_plan_idx, current_plan_step_idx, idr_v, idr_adv, plan_to_rewards_direct, plan_to_reward_indirect, plan_to_reward, plans_ranking):
        mincolf = 0.2
        bgcolor = [0, 0, 0]
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        scr_main = ia.imresize_single_image(scr, (int(720*0.58), int(1280*0.58)))
        util.draw_image(
            image,
            y=int((image.shape[0]-scr_main.shape[0])/2),
            x=1280-scr_main.shape[1]-2,
            other_img=scr_main,
            copy=False
        )
        image = util.draw_text(
            image,
            x=1280-(scr_main.shape[1]//2)-125,
            y=image.shape[0] - int((image.shape[0]-scr_main.shape[0])/2) + 10,
            text="Framerate matches the one that the model sees (10fps).",
            size=10,
            color=[128, 128, 128]
        )

        def draw_key(key):
            btn = [
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
            ]
            btn = np.array(btn, dtype=np.uint8) * 255
            btn = np.tile(btn[:, :, np.newaxis], (1, 1, 3))
            if key is None:
                return np.zeros_like(btn)
            elif key == "":
                return btn
            else:
                return util.draw_text(btn, x=3, y=3, text=key, size=9, color=[255, 255, 255])

        def multiaction_idx_to_image(multiaction_idx):
            #btn = np.pad(btn, ((0, 0), (0, 4), (0, 0)), mode="constant", constant_values=0)
            key_to_img = dict()
            for key in ["W", "A", "S", "D", None]:
                key_to_img[key] = draw_key(key)

            multiaction = actionslib.ALL_MULTIACTIONS[multiaction_idx]
            sw = 1.0 if multiaction[0] == "W" else mincolf
            sa = 1.0 if multiaction[1] == "A" else mincolf
            ss = 1.0 if multiaction[0] == "S" else mincolf
            sd = 1.0 if multiaction[1] == "D" else mincolf
            buttons = [
                [key_to_img[None], key_to_img["W"]*sw, key_to_img[None]],
                [key_to_img["A"]*sa, key_to_img["S"]*ss, key_to_img["D"]*sd]
            ]
            buttons_img = np.vstack([
                np.hstack([btn.astype(np.uint8) for btn in buttons[0]]),
                np.hstack([btn.astype(np.uint8) for btn in buttons[1]])
            ])
            buttons_img = np.pad(buttons_img, ((0, 0), (0, 4), (0, 0)), mode="constant", constant_values=0)
            return buttons_img

        multiaction_idx_to_image_dict = dict([(i, multiaction_idx_to_image(i)) for i in range(len(actionslib.ALL_MULTIACTIONS))])
        multiaction_to_image_dict = dict([(ma, multiaction_idx_to_image(i)) for i, ma in enumerate(actionslib.ALL_MULTIACTIONS)])

        def plan_to_image(p_multiactions, p_direct_rewards, p_v, padding_bottom=8, minwidth=200):
            plan_viz = [multiaction_to_image_dict[ma] for ma in p_multiactions]
            #plan_viz = [np.pad(a, ((0, 20), (2, 2), (0, 0)), mode="constant", constant_values=0) for a in plan_viz]
            plan_viz = [np.pad(a, ((0, 20), (0, 1), (0, 0)), mode="constant", constant_values=0) for a in plan_viz]
            if p_direct_rewards is not None:
                for j in xrange(len(plan_viz)):
                    #plan_viz[j] = util.draw_text(plan_viz[j], x=9, y=plan_viz[j].shape[0]-16, text="r", size=9, color=[128, 128, 128])
                    plan_viz[j] = util.draw_text(plan_viz[j], x=11, y=plan_viz[j].shape[0]-13, text="r %.1f" % (p_direct_rewards[j],), size=9, color=[128, 128, 128])

            if p_v is not None:
                plan_viz.append(np.zeros_like(plan_viz[-1]))
                #plan_viz[-1] = util.draw_text(plan_viz[-1], x=3, y=5, text="V", size=9, color=[128, 128, 128])
                #plan_viz[-1] = util.draw_text(plan_viz[-1], x=9, y=11, text="V %.1f" % (p_v,), size=9, color=[255, 255, 255])
                plan_viz[-1] = util.draw_text(plan_viz[-1], x=5, y=16, text="V %.1f" % (p_v,), size=9, color=[255, 255, 255])

            plan_viz = np.hstack(plan_viz)
            width_extend = minwidth - plan_viz.shape[1] if plan_viz.shape[1] < minwidth else 0
            #print("width_extend", width_extend, minwidth, plan_viz.shape[0])
            plan_viz = np.pad(plan_viz, ((0, padding_bottom), (0, width_extend), (0, 0)), mode="constant", constant_values=0)

            return plan_viz

        # -------------
        # current plan
        # -------------
        current_plan_viz = plan_to_image(
            plans[current_plan_idx][current_plan_step_idx:],
            None, None
        )
        #current_plan_viz = np.pad(current_plan_viz, ((50, 0), (20, 0), (0, 0)), mode="constant", constant_values=0)
        current_plan_viz = np.pad(current_plan_viz, ((50, 0), (2, 0), (0, 0)), mode="constant", constant_values=0)
        current_plan_viz = util.draw_text(current_plan_viz, x=4, y=4, text="Current Plan", color=[255, 255, 255])
        util.draw_image(image, y=10, x=10, other_img=current_plan_viz, copy=False)

        # -------------
        # best plans
        # -------------
        best_plans_viz = []
        for i in range(4):
            plan_idx = plans_ranking[::-1][i]
            plan = plans[plan_idx]
            r = plan_to_rewards_direct[plan_idx]
            v = plan_to_reward_indirect[plan_idx]
            plan_viz = plan_to_image(plan, r, v)
            best_plans_viz.append(plan_viz)

        best_plans_viz = np.vstack(best_plans_viz)

        #best_plans_viz = np.pad(best_plans_viz, ((50, 30), (20, 0), (0, 0)), mode="constant", constant_values=0)
        best_plans_viz = np.pad(best_plans_viz, ((50, 30), (2, 0), (0, 0)), mode="constant", constant_values=0)
        best_plans_viz = util.draw_text(best_plans_viz, x=4, y=4, text="Best Plans", color=[255, 255, 255])
        best_plans_viz = util.draw_text(best_plans_viz, x=30, y=best_plans_viz.shape[0]-20, text="r = expected direct reward at timestep (discounted)\nV = expected indirect reward at last timestep (discounted)", color=[128, 128, 128], size=9)

        util.draw_image(image, y=110, x=10, other_img=best_plans_viz, copy=False)

        # --------------
        # top15
        # --------------
        n = 15
        top_viz = []
        counts_ud = dict([(action, 0) for action in actionslib.ACTIONS_UP_DOWN])
        counts_lr = dict([(action, 0) for action in actionslib.ACTIONS_LEFT_RIGHT])
        for i in range(n):
            plan_idx = plans_ranking[::-1][i]
            plan = plans[plan_idx]
            for ma in plan:
                counts_ud[ma[0]] += 1
                counts_lr[ma[1]] += 1
        sum_ud = np.sum(list(counts_ud.values()))
        sum_lr = np.sum(list(counts_lr.values()))
        fracs_ud = [counts_ud["W"]/sum_ud, counts_ud["S"]/sum_ud, counts_ud["~WS"]/sum_ud]
        fracs_lr = [counts_lr["A"]/sum_lr, counts_lr["D"]/sum_lr, counts_lr["~AD"]/sum_lr]
        def draw_bar(frac, key, h=30, w=20, margin_right=15):
            bar = np.zeros((h, 1), dtype=np.uint8) + 32
            bar[0:int(h*frac)+1] = 255
            bar = np.flipud(bar)
            bar = np.tile(bar[:, :, np.newaxis], (1, w, 3))
            bar = np.pad(bar, ((20, 30), (0, margin_right), (0, 0)), mode="constant", constant_values=0)
            textx = 5
            if frac*100 >= 10:
                textx = textx - 3
            elif frac*100 >= 100:
                textx = textx - 6
            bar = ia.draw_text(bar, x=textx, y=2, text="%.0f%%" % (frac*100,), size=8, color=[255, 255, 255])
            keyimg = draw_key(key)
            util.draw_image(bar, x=(w//2)-keyimg.shape[1]//2, y=bar.shape[0]-keyimg.shape[0]-8, other_img=keyimg, copy=False)
            return bar

        bars_ud = [draw_bar(fracs_ud[0], "W"), draw_bar(fracs_ud[1], "S"), draw_bar(fracs_ud[2], "", margin_right=55)]
        bars_lr = [draw_bar(fracs_lr[0], "A"), draw_bar(fracs_lr[1], "D"), draw_bar(fracs_lr[2], "")]
        top_viz = np.hstack(bars_ud + bars_lr)
        top_viz = np.pad(top_viz, ((50, 30), (20, 180), (0, 0)), mode="constant", constant_values=0)
        top_viz = util.draw_text(top_viz, x=4, y=4, text="Share Of Keys (Top %d Plans)" % (n,), color=[255, 255, 255])
        top_viz = util.draw_text(top_viz, x=4, y=top_viz.shape[0]-20, text="Percent of actions among top %d plans that contain a top/down or left/right key" % (n,), color=[128, 128, 128], size=9)

        util.draw_image(image, y=430, x=10, other_img=top_viz, copy=False)

        # --------------
        # other
        # --------------
        other_viz = np.zeros((300, 500, 3), dtype=np.uint8)
        other_viz = util.draw_text(other_viz, x=4, y=4, text="Speed", color=[255, 255, 255])
        other_viz = util.draw_text(other_viz, x=150, y=4, text="Steering Wheel", color=[255, 255, 255])

        other_viz = util.draw_text(other_viz, x=12, y=65, text="%d km/h" % (state.speed if state.speed is not None else -1), color=[255, 255, 255])

        sw_angle = state.steering_wheel_cnn if state.steering_wheel_cnn is not None else 0
        sw_circle = np.zeros((80, 80, 3), dtype=np.int32)
        if sw_angle <= -360 or sw_angle >= 360:
            rr, cc = draw.circle(r=40, c=40, radius=30)
            sw_circle[rr, cc, :] = 128
        col = [128, 128, 128] if -360 < sw_angle < 360 else [255, 255, 255]
        if abs(sw_angle % 360) > 1:
            if sw_angle < 0:
                sw_circle = util.draw_direction_circle(
                    sw_circle,
                    y=40, x=40,
                    r_inner=0, r_outer=30,
                    angle_start=360-(abs(int(sw_angle)) % 360), angle_end=360,
                    color_border=col,
                    color_fill=col
                    #color_fill=[255,0,0]
                )
                #sw_circle = util.draw_text(sw_circle, x=5, y=5, text="%.2f\n%.2f" % (abs(int(sw_angle)) % 360, 360-(abs(int(sw_angle)) % 360)), size=12, color=[255, 255, 255])
            else:
                sw_circle = util.draw_direction_circle(
                    sw_circle,
                    y=40, x=40,
                    r_inner=0, r_outer=30,
                    angle_start=0, angle_end=int(sw_angle) % 360,
                    color_border=col,
                    color_fill=col
                    #color_fill=[0,255,0]
                )
        rr, cc, val = draw.circle_perimeter_aa(40, 40, radius=30)
        #sw_circle[rr, cc, :] = sw_circle[rr, cc, :] + np.tile((val * 255)[:,:,np.newaxis], (1, 1, 3))
        sw_circle[rr, cc, :] += np.tile((val * 255).astype(np.int32)[:,np.newaxis], (1, 3))
        sw_circle = np.clip(sw_circle, 0, 255).astype(np.uint8)
        sw_circle = np.pad(sw_circle, ((0, 0), (0, 140), (0, 0)), mode="constant", constant_values=0)
        sw_circle = util.draw_text(sw_circle, x=92, y=27, text="%d deg" % (sw_angle,), color=[255, 255, 255])
        util.draw_image(other_viz, x=150, y=40, other_img=sw_circle, copy=False)

        util.draw_image(image, y=590, x=10, other_img=other_viz, copy=False)

        return image

    def draw_frame_attributes(self, scr, atts):
        atts = atts[0]
        mincolf = 0.2

        #print("space_front raw", atts[33:37], F.softmax(atts[33:37]))
        #print("space_left raw", atts[37:41], F.softmax(atts[37:41]))
        #print("space_right raw", atts[41:45], F.softmax(atts[41:45].unsqueeze(0)).squeeze(0))
        road_type = simplesoftmax(to_numpy(atts[0:10]))
        intersection = simplesoftmax(to_numpy(atts[10:17]))
        direction = simplesoftmax(to_numpy(atts[17:20]))
        lane_count = simplesoftmax(to_numpy(atts[20:25]))
        curve = simplesoftmax(to_numpy(atts[25:33]))
        space_front = simplesoftmax(to_numpy(atts[33:37]))
        space_left = simplesoftmax(to_numpy(atts[37:41]))
        space_right = simplesoftmax(to_numpy(atts[41:45]))
        offroad = simplesoftmax(to_numpy(atts[45:48]))

        bgcolor = [0, 0, 0]
        image = np.zeros((720, 1280, 3), dtype=np.uint8) + bgcolor
        scr_main = ia.imresize_single_image(scr, (int(720*0.58), int(1280*0.58)))
        util.draw_image(
            image,
            y=int((image.shape[0]-scr_main.shape[0])/2),
            x=1280-scr_main.shape[1]-2,
            other_img=scr_main,
            copy=False
        )
        image = util.draw_text(
            image,
            x=1280-(scr_main.shape[1]//2)-125,
            y=image.shape[0] - int((image.shape[0]-scr_main.shape[0])/2) + 10,
            text="Framerate matches the one that the model sees (10fps).",
            size=10,
            color=[128, 128, 128]
        )

        # ---------------
        # Curve
        # ---------------
        """
        street = np.zeros((65, 65, 3), dtype=np.float32)
        street[:, 0:2, :] = 255
        street[:, -2:, :] = 255
        street[:, 32:35, :] = 255

        street_left_strong = curve(street
        """
        curve_left_strong = 255 - ndimage.imread("../images/video/curve-left-strong.png", mode="RGB")
        curve_left_medium = 255 - ndimage.imread("../images/video/curve-left-medium.png", mode="RGB")
        curve_left_slight = 255 - ndimage.imread("../images/video/curve-left-slight.png", mode="RGB")
        curve_straight = 255 - ndimage.imread("../images/video/curve-straight.png", mode="RGB")
        curve_right_strong = np.fliplr(curve_left_strong)
        curve_right_medium = np.fliplr(curve_left_medium)
        curve_right_slight = np.fliplr(curve_left_slight)

        curve_straight = (curve_straight * np.clip(curve[0], mincolf, 1.0)).astype(np.uint8)
        curve_left_slight = (curve_left_slight * np.clip(curve[1], mincolf, 1.0)).astype(np.uint8)
        curve_left_medium = (curve_left_medium * np.clip(curve[2], mincolf, 1.0)).astype(np.uint8)
        curve_left_strong = (curve_left_strong * np.clip(curve[3], mincolf, 1.0)).astype(np.uint8)
        curve_right_slight = (curve_right_slight * np.clip(curve[4], mincolf, 1.0)).astype(np.uint8)
        curve_right_medium = (curve_right_medium * np.clip(curve[5], mincolf, 1.0)).astype(np.uint8)
        curve_right_strong = (curve_right_strong * np.clip(curve[6], mincolf, 1.0)).astype(np.uint8)

        def add_perc(curve_img, perc, x_correct):
            col = np.clip(255 * perc, mincolf*255, 255)
            col = np.array([col, col, col], dtype=np.uint8)

            curve_img_pad = np.pad(curve_img, ((0, 20), (0, 0), (0, 0)), mode="constant", constant_values=0)

            x = int(curve_img_pad.shape[1]/2) - 6
            if (perc*100) >= 100:
                x = x - 9
            elif (perc*100) >= 10:
                x = x - 6
            x = x + x_correct

            curve_img_pad = util.draw_text(
                curve_img_pad,
                x=x,
                y=curve_img_pad.shape[0]-15,
                text="%.0f%%" % (perc*100,),
                color=col,
                size=9
            )
            return curve_img_pad

        curve_straight = add_perc(curve_straight, curve[0], x_correct=0)
        curve_left_slight = add_perc(curve_left_slight, curve[1], x_correct=3)
        curve_left_medium = add_perc(curve_left_medium, curve[2], x_correct=1)
        curve_left_strong = add_perc(curve_left_strong, curve[3], x_correct=-1)
        curve_right_slight = add_perc(curve_right_slight, curve[4], x_correct=-3)
        curve_right_medium = add_perc(curve_right_medium, curve[5], x_correct=-2)
        curve_right_strong = add_perc(curve_right_strong, curve[6], x_correct=0)

        curves = np.hstack([
            curve_left_strong, curve_left_medium, curve_left_slight,
            curve_straight,
            curve_right_slight, curve_right_medium, curve_right_strong
        ])

        curves = np.pad(curves, ((50, 0), (20, 0), (0, 0)), mode="constant", constant_values=0)
        curves = util.draw_text(curves, x=4, y=4, text="Curve", color=[255, 255, 255])

        util.draw_image(image, y=50, x=2, other_img=curves, copy=False)

        # ---------------
        # Lane count
        # ---------------
        pics = []
        for lc_idx in range(4):
            col = int(np.clip(255*lane_count[lc_idx], 255*mincolf, 255))
            col = np.array([col, col, col], dtype=np.uint8)
            lc = lc_idx + 1
            marking_width = 2
            street = np.zeros((64, 64, 3), dtype=np.float32)
            street[:, 0:marking_width, :] = col
            street[:, -marking_width:, :] = col
            inner_width = street.shape[1] - 2*marking_width
            lane_width = int((inner_width - (lc-1)*marking_width) // lc)
            start = marking_width
            for i in range(lc-1):
                mstart = start + lane_width
                mend = mstart + marking_width
                street[1::6, mstart:mend, :] = col
                street[2::6, mstart:mend, :] = col
                street[3::6, mstart:mend, :] = col
                start = mend

            x = 14 + 24
            if lane_count[lc_idx]*100 >= 10:
                x = x - 8
            elif lane_count[lc_idx]*100 >= 100:
                x = x - 12

            street = np.pad(street, ((0, 20), (14, 14), (0, 0)), mode="constant", constant_values=0)
            street = util.draw_text(street, x=x, y=street.shape[0]-14, text="%.0f%%" % (lane_count[lc_idx]*100,), size=9, color=col)
            pics.append(street)

        pics = np.hstack(pics)
        pics = np.pad(pics, ((55, 0), (20, 0), (0, 0)), mode="constant", constant_values=0)
        pics = util.draw_text(pics, x=4, y=4, text="Lane Count", color=[255, 255, 255])
        util.draw_image(image, y=250, x=2, other_img=pics, copy=False)

        # ---------------
        # Space
        # ---------------
        truck = np.zeros((100, 55, 3), dtype=np.uint8)
        truck[0:2, :, :] = 255
        truck[0:20, 0:2, :] = 255
        truck[0:20, -2:, :] = 255
        truck[20:22, :, :] = 255

        truck[22:25, 25:27, :] = 255
        truck[22:25, 29:31, :] = 255

        truck[24:26, :, :] = 255
        truck[24:, 0:2, :] = 255
        truck[24:, -2:, :] = 255
        truck[24:, -2:, :] = 255
        truck[-2:, :, :] = 255

        truck_full = np.pad(truck, ((50, 50), (100, 50), (0, 0)), mode="constant", constant_values=np.average(bgcolor))

        #print("space_front", space_front)
        #print("space_right", space_right)
        #print("space_left", space_left)
        fill_top = 1 * space_front[0] + 0.6 * space_front[1] + 0.25 * space_front[2] + 0 * space_front[3]
        fill_right = 1 * space_right[0] + 0.6 * space_right[1] + 0.25 * space_right[2] + 0 * space_right[3]
        fill_left = 1 * space_left[0] + 0.6 * space_left[1] + 0.25 * space_left[2] + 0 * space_left[3]

        r_outer_top = 8 + int((30-8) * fill_top)
        r_outer_right = 8 + int((30-8) * fill_right)
        r_outer_left = 8 + int((30-8) * fill_left)

        def fill_to_text(fill):
            col = np.array([255, 255, 255], dtype=np.uint8)
            if fill > 0.75:
                text = "plenty"
            elif fill > 0.5:
                text = "some"
            elif fill > 0.25:
                text = "low"
            else:
                text = "minimal"
            return text, col

        #top
        truck_full = util.draw_direction_circle(
            truck_full,
            y=33, x=100+27,
            r_inner=8, r_outer=30,
            angle_start=-60, angle_end=60,
            color_border=[255, 255, 255],
            color_fill=[0, 0, 0]
        )
        truck_full = util.draw_direction_circle(
            truck_full,
            y=33, x=100+27,
            r_inner=8, r_outer=r_outer_top,
            angle_start=-60, angle_end=60,
            color_border=[255, 255, 255],
            color_fill=[255, 255, 255]
        )
        #text, col = fill_to_text(fill_top)
        #truck_full = util.draw_text(truck_full, x=100+27, y=15, text=text, size=9, color=col)

        # right
        truck_full = util.draw_direction_circle(
            truck_full,
            y=100, x=170,
            r_inner=8, r_outer=30,
            angle_start=30, angle_end=180-30,
            color_border=[255, 255, 255],
            color_fill=[0, 0, 0]
        )
        truck_full = util.draw_direction_circle(
            truck_full,
            y=100, x=170,
            r_inner=8, r_outer=r_outer_right,
            angle_start=30, angle_end=180-30,
            color_border=[255, 255, 255],
            color_fill=[255, 255, 255]
        )
        #text, col = fill_to_text(fill_right)
        #truck_full = util.draw_text(truck_full, x=170, y=100, text=text, size=9, color=col)

        # left
        truck_full = util.draw_direction_circle(
            truck_full,
            y=100, x=83,
            r_inner=8, r_outer=30,
            angle_start=180+30, angle_end=360-30,
            color_border=[255, 255, 255],
            color_fill=[0, 0, 0]
        )
        truck_full = util.draw_direction_circle(
            truck_full,
            y=100, x=83,
            r_inner=8, r_outer=r_outer_left,
            angle_start=180+30, angle_end=360-30,
            color_border=[255, 255, 255],
            color_fill=[255, 255, 255]
        )
        #text, col = fill_to_text(fill_left)
        #truck_full = util.draw_text(truck_full, x=75, y=100, text=text, size=9, color=col)

        truck_full = np.pad(truck_full, ((50, 0), (110, 0), (0, 0)), mode="constant", constant_values=0)
        truck_full = util.draw_text(truck_full, x=4, y=4, text="Space", color=[255, 255, 255])

        util.draw_image(image, y=450, x=10, other_img=truck_full, copy=False)

        return image

    def draw_frame_grids(self, scr, grids):
        grids_meta = [
            (0, "street boundaries"),
            (3, "crashables (except cars)"),
            (7, "street markings"),
            (4, "current lane"),
            (1, "cars"),
            (2, "cars in mirrors")
        ]
        titles = [title for idx, title in grids_meta]
        grids = to_numpy(grids[0])
        grids = [grids[idx] for idx, title in grids_meta]
        #self.grid_to_graph(scr, grids[0])

        bgcolor = [0, 0, 0]
        image = np.zeros((720, 1280, 3), dtype=np.uint8) + bgcolor
        scr_main = ia.imresize_single_image(scr, (int(720*0.58), int(1280*0.58)))
        #util.draw_image(image, y=720-scr_main.shape[0], x=1080-scr_main.shape[1], other_img=scr_main, copy=False)
        util.draw_image(
            image,
            y=int((image.shape[0]-scr_main.shape[0])/2),
            x=1280-scr_main.shape[1]-2,
            other_img=scr_main,
            copy=False
        )
        image = util.draw_text(
            image,
            x=1280-(scr_main.shape[1]//2)-125,
            y=image.shape[0] - int((image.shape[0]-scr_main.shape[0])/2) + 10,
            text="Framerate matches the one that the model sees (10fps).",
            size=10,
            color=[128, 128, 128]
        )

        grid_rel_size = 0.19
        scr_small = ia.imresize_single_image(scr, (int(720*grid_rel_size), int(1280*grid_rel_size)))
        grid_hms = []
        for grid, title in zip(grids, titles):
            grid = (grid*255).astype(np.uint8)[:,:,np.newaxis]
            grid = ia.imresize_single_image(grid, (int(720*grid_rel_size), int(1280*grid_rel_size)), interpolation="nearest")
            grid_hm = util.draw_heatmap_overlay(scr_small, grid/255)
            grid_hm = np.pad(grid_hm, ((2, 0), (2, 2), (0, 0)), mode="constant", constant_values=np.average(bgcolor))
            #grid_hm = np.pad(grid_hm, ((0, 20), (0, 0), (0, 0)), mode="constant", constant_values=0)
            #grid_hm[-20:, 2:-2, :] = [128, 128, 255]
            #grid_hm = util.draw_text(grid_hm, x=4, y=grid_hm.shape[0]-16, text=title, size=10, color=[255, 255, 255])
            grid_hm = np.pad(grid_hm, ((40, 0), (0, 0), (0, 0)), mode="constant", constant_values=0)
            grid_hm = util.draw_text(grid_hm, x=4, y=20, text=title, size=12, color=[255, 255, 255])
            grid_hms.append(grid_hm)
        grid_hms = ia.draw_grid(grid_hms, cols=2)

        util.draw_image(image, y=70, x=0, other_img=grid_hms, copy=False)

        return image

    def tsalesman(self, graph):
        if len(graph) <= 2:
            return graph

        paths = []
        for _ in xrange(1000):
            ids = list(range(len(graph)))
            np.random.shuffle(ids)
            path = [graph[idx] for idx in ids]
            paths.append(path)

        def length_edge(node1, node2):
            d = math.sqrt((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2)
            return d

        def length_path(path):
            length_sum = length_edge(path[0], path[-1])
            for i in xrange(1, len(path)):
                length_sum += length_edge(path[i-1], path[i])
            return length_sum

        paths_l = [(path, length_path(path)) for path in paths]
        paths = sorted(paths_l, key=lambda t: t[1])
        return paths[0][0]

    def grid_to_graph(self, scr, grid):
        import scipy.ndimage as ndimage
        import scipy.ndimage.filters as filters
        data = grid
        neighborhood_size = 7
        #threshold_max = 0.5
        threshold_diff = 0.1
        threshold_score = 0.2

        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size)
        #diff = ((data_max - data_min) > threshold_diff)
        #maxima[diff == 0] = 0
        maxima[data_max < 0.2] = 0

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        xx, yy, score = [], [], []
        for dy, dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            y_center = (dy.start + dy.stop - 1)/2
            s = np.average(data[dy.start:dy.stop+1, dx.start:dx.stop+1])
            if s > threshold_score:
                xx.append(x_center / grid.shape[1])
                yy.append(y_center / grid.shape[0])
                score.append(s)

        graph = list(zip(xx, yy, score))
        path = tsalesman(graph)
        paths_final = [path]

        scr_viz = np.copy(scr)
        h, w = scr.shape[0:2]
        #hup, wup = h/grid.shape[0], w/grid.shape[1]
        hup, wup = h, w
        for i, (x, y, s) in enumerate(zip(xx, yy, score)):
            size = 3*int(s*10)
            size = size if size % 2 != 0 else size - 1
            scr_viz = util.draw_point(scr_viz, y=int(y*hup), x=int(x*wup), size=size, color=[0, 255, 0])
            scr_viz = util.draw_text(scr_viz, y=int(y*hup), x=int(x*wup), text=str(i), color=[0, 255, 0])

        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]
        for path, col in zip(paths_final, colors):
            last_x = None
            last_y = None
            for (x, y, s) in path:
                if last_x is not None:
                    scr_viz = util.draw_line(scr_viz, y1=int(last_y*hup), x1=int(last_x*wup), y2=int(y*hup), x2=int(x*wup), color=col, thickness=2)
                last_x = x
                last_y = y
        misc.imshow(scr_viz)

        """
        paths_final = []
        graph = list(zip(xx, yy, score))
        for _ in range(1):
            paths_final_flat = flatten_list(paths_final)
            print("paths_final_flat", paths_final_flat)
            source_candidates = [(i, (x, y, s)) for i, (x, y, s) in enumerate(graph) if (x, y, s) not in paths_final_flat]
            if len(source_candidates) == 0:
                break
            else:
                #print("source_candidates", source_candidates)
                #source_score = max([s for (i, (x, y, s)) in source_candidates])
                #print("source_score", source_score)
                #source_id = [i for (i, (x, y, s)) in source_candidates if s == source_score][0]
                source_val = min([x for (i, (x, y, s)) in source_candidates])
                source_id = [i for (i, (x, y, s)) in source_candidates if x == source_val][0]
                print("source_id", source_id)
                _, _, paths = self.dijkstra(graph, source_id, already_done=[i for i, (x, y, s) in enumerate(paths_final_flat)])
                if len(paths) == 0:
                    break
                else:
                    print("paths", paths)
                    #best_path = sorted(paths, key=lambda t: -t[1]+t[2], reverse=True)[0]
                    best_path = sorted(paths, key=lambda t: t[2], reverse=True)[0]
                    best_path = best_path[0]
                    print("best_path ids", best_path)
                    best_path = [graph[idx] for idx in best_path]
                    print("best_path", best_path)
                    paths_final.append(best_path)
        paths_final = [path for path in paths_final if len(path) > 1]

        scr_viz = np.copy(scr)
        h, w = scr.shape[0:2]
        #hup, wup = h/grid.shape[0], w/grid.shape[1]
        hup, wup = h, w
        for i, (x, y, s) in enumerate(zip(xx, yy, score)):
            size = 3*int(s*10)
            size = size if size % 2 != 0 else size - 1
            scr_viz = util.draw_point(scr_viz, y=int(y*hup), x=int(x*wup), size=size, color=[0, 255, 0])
            scr_viz = util.draw_text(scr_viz, y=int(y*hup), x=int(x*wup), text=str(i), color=[0, 255, 0])

        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]
        for path, col in zip(paths_final, colors):
            last_x = None
            last_y = None
            for (x, y, s) in path:
                if last_x is not None:
                    scr_viz = util.draw_line(scr_viz, y1=int(last_y*hup), x1=int(last_x*wup), y2=int(y*hup), x2=int(x*wup), color=col, thickness=2)
                last_x = x
                last_y = y
        misc.imshow(scr_viz)
        """

        #misc.imshow(np.hstack([scr, scr_viz]))

    """
    def shortest(self, graph):
        edges = []


    def dijkstra(self, graph, source_id, distance_threshold=0.5, already_done=None):
        already_done = set() if already_done is None else set(already_done)
        id_to_vertex = dict([(i, v) for i, v in enumerate(graph)])
        vertex_to_id = dict([(v, i) for i, v in enumerate(graph)])
        graph_ids = [i for i, _ in enumerate(graph)]
        def length(id1, id2):
            d = (graph[id1][0]-graph[id2][0])**2 + (graph[id1][1]-graph[id2][1])**2
            if id1 in already_done or id2 in already_done:
                d = d + 0.2
            d = d / (0.5*(id_to_vertex[id1][2] + id_to_vertex[id2][2]))
            print("length", id1, id2, d)
            return d
        def neighbours(id1):
            n = []
            for id2, v in id_to_vertex.items():
                #print(id1, id2, length(id1, id2))
                if id1 != id2 and length(id1, id2) < distance_threshold:
                    n.append(id2)
            return n
        def mindist(dist, Q):
            mindist_val = 999999
            mindist_id = -1
            for vid in Q:
                if dist[vid] < mindist_val:
                    mindist_val = mindist_val
                    mindist_id = vid
            return mindist_id

        dist = dict()
        prev = dict()
        Q = set()
        for vid in graph_ids:
            dist[vid] = 999999
            prev[vid] = None
            Q.add(vid)

        dist[source_id] = 0
        prev[source_id] = None

        while len(Q) > 0:
            uid = mindist(dist, Q)
            print("do", uid)
            if uid == -1:
                print(Q)
                print(dist)
            if uid == -1:
                break
            else:
                Q.remove(uid)

                for vid in neighbours(uid):
                    alt = dist[uid] + length(uid, vid)
                    print("neighbour %d -> %d | d=%.2f" % (uid, vid, alt))
                    if alt < dist[vid]:
                        print("closer!")
                        dist[vid] = alt
                        prev[vid] = uid

        print("dist", dist)
        print("prev", prev)
        # paths
        ps = []
        for i, v in enumerate(graph):
            last_node_id = i
            p = [last_node_id]
            sum_dist = 0
            count_nodes = 1
            while True:
                curr_node_id = prev[last_node_id]
                if curr_node_id is None:
                    break
                else:
                    p.append(curr_node_id)
                    count_nodes += 1
                    sum_dist += length(last_node_id, curr_node_id)
                last_node_id = curr_node_id
            ps.append((p, sum_dist, count_nodes))
        print("ps", ps)
        ps = [p for p in ps if p[0][-1] == source_id]
        print("ps red", ps)

        return dist, prev, ps
    """

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def simplesoftmax(l):
    s = np.sum(l)
    if s > 0:
        return l/s
    else:
        return np.zeros_like(s)

if __name__ == "__main__":
    main()
