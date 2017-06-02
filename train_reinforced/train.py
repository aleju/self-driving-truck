"""Train a self-driving truck via reinforcement learning."""
from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import models as models_reinforced
from plans import create_plans
from batching import BatchLoader, BackgroundBatchLoader, states_to_batch
from visualization import generate_overview_image, generate_training_debug_image

from train_semisupervised.train import (
    PREVIOUS_STATES_DISTANCES,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    MODEL_PREV_HEIGHT,
    MODEL_PREV_WIDTH
)
from train_semisupervised import models as models_semisupervised
from lib import ets2game
from lib import speed as speedlib
from lib import replay_memory
from lib import states as stateslib
from lib import actions as actionslib
from lib import steering_wheel as swlib
from lib import util
from lib.util import to_variable, to_cuda, to_numpy
from lib import plotting
from lib import rewards as rewardslib
from config import Config

from scipy import misc
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import time
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import cPickle as pickle
import argparse
import collections
from datetime import datetime
import copy
import gzip as gz

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

try:
    xrange
except NameError:
    xrange = range

# number of batches per training sessions and batch size
NB_BATCHES_PER_TRAIN = 30000
BATCH_SIZE = 8

# when to start training sessions
# wait until minimum number of experiences, then every 30k experiences
def DO_TRAIN(ora, game):
    if game.tick_ingame_route_advisor_idx == 0 and ora.args.instanttrain:
        return True
    elif ora.args.onlyapply:
        return False
    elif ora.memory.size < 30000 or ora.memory_val.size < 2000:
        return False
    else:
        return (game.tick_ingame_route_advisor_idx+1) % 30000 == 0
# when to save during training
DO_SAVE = lambda epoch, batch_idx, batch_idx_total: (batch_idx+1) % min(2000, NB_BATCHES_PER_TRAIN-1) == 0
# when to validate during training
DO_VALIDATE = lambda epoch, batch_idx, batch_idx_total: (batch_idx+1) % min(500, NB_BATCHES_PER_TRAIN-1) == 0
# when to plot loss curves during training
DO_PLOT = lambda epoch, batch_idx, batch_idx_total: (batch_idx+1) % min(500, NB_BATCHES_PER_TRAIN-1) == 0
# whether to train the model which was initialized by semi-supervised training
DO_TRAIN_SUPERVISED = lambda epoch, batch_idx, batch_idx_total: batch_idx_total > 50000
# whether to backpropagate losses of direct and indirect reward prediction
# applied to the successors. if false, the reward prediction models are only
# trained to handle the true embeddings, not the noisier successor embeddings.
DO_BACKPROP_SUCCESSOR_REWARDS = lambda epoch, batch_idx, batch_idx_total: batch_idx_total > 2000
# whether to NOT backpropagate the reward gradients from the successors beyond
# the direct/indirect reward models. If False, the successors should become more
# suited towards predicting correctly factors that influence rewards.
DO_CUTOFF_BACKPROP_SUCCESSOR_REWARDS = lambda epoch, batch_idx, batch_idx_total: batch_idx_total < 5000
# whether to NOT backpropagate all successor gradietns beyond the successors (into
# the embeddings). If False, the embeddings should become more suited for
# successor prediction.
DO_CUTOFF_BACKPROP_SUCCESSOR = lambda epoch, batch_idx, batch_idx_total: batch_idx_total < 10000
# whether to train an autoencoder on the embeddings
DO_TRAIN_AE = lambda epoch, batch_idx, batch_idx_total: batch_idx_total < 25000

# weightings of gradients
# idr is here lowered to avoid problems from high gradients related to MSE
LOSS_DR_WEIGHTING = 1
LOSS_DR_FUTURE_WEIGHTING = 1
LOSS_IDR_WEIGHTING = 0.01
LOSS_IDR_FUTURE_WEIGHTING = 0.01
LOSS_SUCCESSOR_WEIGHTING = 0.5
LOSS_DR_SUCCESSORS_WEIGHTING = 1
LOSS_IDR_SUCCESSORS_WEIGHTING = 0.01
LOSS_AE_WEIGHTING = 1

# multiplier for all Adam learning rates
# might be a good idea to increase this value to 1 at the start
LEARNING_RATE_MULTIPLIER = 0.1

# number of validation batches to run per validation
NB_VAL_BATCHES = 128
# how often to fix weights for IDR/Q-value models during training
TRAIN_FIX_EVERY_N_BATCHES = 100
# whether to add a state to the training or validation replay memory
MEMORY_ADD_STATE_TO_TRAIN = lambda tick_idx: True if tick_idx % 10000 > 1000 else False
# whether to commit changes to the databases (replay memories)
ENABLE_COMMITS = True
# commit every Nth state
COMMIT_EVERY = 110

# number of future timesteps to predict when applying the mdoel, training it
# or validating it
NB_FUTURE_STATES_APPLY = 10
NB_FUTURE_STATES_TRAIN = 10
NB_FUTURE_STATES_VAL = 10
PICK_NEW_PLAN_EVERY_N_STEPS = 10

# number of reward bins for the direct reward prediction,
# the sizes of these bins and their average values
NB_REWARD_BINS = 101
BIN_SIZE = (Config.MAX_REWARD - Config.MIN_REWARD) / NB_REWARD_BINS
BIN_AVERAGES = np.array([Config.MIN_REWARD + (i*BIN_SIZE + (i+1)*BIN_SIZE)/2 for i in xrange(NB_REWARD_BINS)], dtype=np.float32)[::-1]

# discount factor and discount over time
GAMMA = 0.95
REWARDS_FUTURE_GAMMAS = np.power(
    np.array([GAMMA] * NB_FUTURE_STATES_APPLY, dtype=np.float32),
    1 + np.arange(NB_FUTURE_STATES_APPLY)
)

# whether to use argmax on the predicted direct reward bins to compute the
# direct reward (otherwise: average of activated bins, multiplied by probability
# of each bin)
DIRECT_REWARD_ARGMAX = False

# whether to add the computed direct reward to the final reward, which is used
# to determine the ranking of plans (and therefore the plan to pick)
# if false, the ranking is only determined by the final timesteps V-value
# (indirect reward)
REWARD_ADD_DIRECT_REWARD = False

# plans of actions to check during validation (for each plan, expected rewards
# are calculated, then the best plan is chosen)
PLANS, PLANS_VECS = create_plans(NB_FUTURE_STATES_APPLY)
PLANS_VECS_VAR = to_cuda(to_variable(PLANS_VECS, volatile=True), Config.GPU)

# epsilon greedy constants (=p_explore)
# chance in case p_explore hits that the last state's action is repeated
P_EXPLORE_CHANCE_REDO = 0.85
# minimum and maximum chances to explore
P_EXPLORE_MIN = 0.2
P_EXPLORE_MAX = 0.8
# reach minimum p_explore after around 12 hours / 430k experiences
P_EXPLORE_STEPSIZE = 1/(12*60*60*10)

def main():
    """Initialize/load model, replay memories, optimizers, history and loss
    plotter, augmentation sequence. Then start play and train loop."""

    # -----------
    # test loading a batch, for faster debugging
    loader = BatchLoader(
        val=False, batch_size=BATCH_SIZE, augseq=iaa.Noop(),
        previous_states_distances=PREVIOUS_STATES_DISTANCES,
        nb_future_states=NB_FUTURE_STATES_TRAIN,
        model_height=MODEL_HEIGHT, model_width=MODEL_WIDTH,
        model_prev_height=MODEL_PREV_HEIGHT, model_prev_width=MODEL_PREV_HEIGHT
    )
    batch = loader.load_random_batch()
    # -----------

    parser = argparse.ArgumentParser(description="Train trucker via reinforcement learning.")
    parser.add_argument("--nocontinue", default=False, action="store_true", help="Whether to NOT continue the previous experiment", required=False)
    parser.add_argument("--instanttrain", default=False, action="store_true", help="Start first training instantly", required=False)
    parser.add_argument("--onlytrain", default=False, action="store_true", help="Do nothing but training, ignore the game", required=False)
    parser.add_argument("--onlyapply", default=False, action="store_true", help="Never train the models", required=False)
    parser.add_argument("--noinsert", default=False, action="store_true", help="Whether to not add experiences to database.", required=False)
    parser.add_argument("--nospeedkill", default=False, action="store_true", help="Whether to not autokill tries that end in low speed for a long time.", required=False)
    # TODO this uses the standard semi-supervised model, not the one with shortcuts
    parser.add_argument("--showgrids", default=False, action="store_true", help="Whether to show grids in the overview window (takes extra time per state).", required=False)
    parser.add_argument("--p_explore", default=None, help="Constant p_explore value to use.", required=False)
    parser.add_argument("--record", default=None, help="Record screenshots and states, then save them to the given filepath, e.g. record='records/testrun17.pickle.gz'.", required=False)
    args = parser.parse_args()
    if args.p_explore is not None:
        args.p_explore = float(args.p_explore)
    if args.record and os.path.isfile(args.record):
        print("[WARNING] --record file already exists. This will overwrite the file!")

    if not args.onlytrain:
        cv2.namedWindow("overview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("overview", 590, 720)
        cv2.moveWindow("overview", 30, 1080-720)
        cv2.waitKey(100)

    # ---------------------
    # initialize variables
    # ---------------------
    checkpoint_reinforced = torch.load("train_reinforced_model.tar") if os.path.isfile("train_reinforced_model.tar") and not args.nocontinue else None
    checkpoint_supervised = torch.load("../train_semisupervised/train_semisupervised_model.tar") if os.path.isfile("../train_semisupervised/train_semisupervised_model.tar") else None
    if checkpoint_reinforced is None:
        print("[WARNING] No checkpoint from reinforcement training found.")

    # connection to write states and check size
    # training creates several new connections
    memory = replay_memory.ReplayMemory.create_instance_reinforced()
    memory_val = replay_memory.ReplayMemory.create_instance_reinforced(val=True)

    history = plotting.History()
    embedder_supervised_orig = models_semisupervised.Predictor()
    models = {
        "embedder_supervised": models_semisupervised.Predictor(),
        "embedder_reinforced": models_reinforced.Embedder(),
        "direct_reward_predictor": models_reinforced.DirectRewardPredictor(NB_REWARD_BINS),
        "indirect_reward_predictor": models_reinforced.IndirectRewardPredictor(),
        "successor_predictor": models_reinforced.SuccessorPredictor(),
        "ae_decoder": models_reinforced.AEDecoder()
    }

    embedder_supervised_orig.eval()

    print("Embedder supervised:")
    print(models["embedder_supervised"])
    print("Embedder reinforced:")
    print(models["embedder_reinforced"])
    optimizers = {
        "embedder_supervised": optim.Adam(models["embedder_supervised"].parameters(), lr=0.001*LEARNING_RATE_MULTIPLIER),
        "embedder_reinforced": optim.Adam(models["embedder_reinforced"].parameters(), lr=0.001*LEARNING_RATE_MULTIPLIER),
        "direct_reward_predictor": optim.Adam(models["direct_reward_predictor"].parameters(), lr=0.001*LEARNING_RATE_MULTIPLIER),
        "indirect_reward_predictor": optim.Adam(models["indirect_reward_predictor"].parameters(), lr=0.001*LEARNING_RATE_MULTIPLIER),
        "successor_predictor": optim.Adam(models["successor_predictor"].parameters(), lr=0.001*LEARNING_RATE_MULTIPLIER),
        "ae_decoder": optim.Adam(models["ae_decoder"].parameters(), lr=0.001*LEARNING_RATE_MULTIPLIER)
    }

    # initialize loss value hsitory
    history.add_group("loss-dr", ["train", "val"], increasing=False)
    history.add_group("loss-dr-future", ["train", "val"], increasing=False)
    history.add_group("loss-dr-successors", ["train", "val"], increasing=False)
    history.add_group("loss-idr", ["train", "val"], increasing=False)
    history.add_group("loss-idr-future", ["train", "val"], increasing=False)
    history.add_group("loss-idr-successors", ["train", "val"], increasing=False)
    history.add_group("loss-successor", ["train", "val"], increasing=False)
    history.add_group("loss-ae", ["train", "val"], increasing=False)

    # initialize loss plotter
    loss_plotter = plotting.LossPlotter(
        history.get_group_names(),
        history.get_groups_increasing(),
        save_to_fp="train_reinforced_plot.jpg"
    )
    # this limits the plots to the last 100k batches
    for group_name in history.get_group_names():
        loss_plotter.xlim[group_name] = (-100000, None)
    loss_plotter.start_batch_idx = 100

    # load model parameters and loss history
    if checkpoint_reinforced is not None:
        print("Loading old checkpoint from reinforced training....")
        models["embedder_supervised"].load_state_dict(checkpoint_reinforced["embedder_supervised_state_dict"])
        models["embedder_reinforced"].load_state_dict(checkpoint_reinforced["embedder_reinforced_state_dict"])
        models["direct_reward_predictor"].load_state_dict(checkpoint_reinforced["direct_reward_predictor_state_dict"])
        models["indirect_reward_predictor"].load_state_dict(checkpoint_reinforced["indirect_reward_predictor_state_dict"])
        models["successor_predictor"].load_state_dict(checkpoint_reinforced["successor_predictor_state_dict"])
        models["ae_decoder"].load_state_dict(checkpoint_reinforced["ae_decoder_state_dict"])
        history = plotting.History.from_string(checkpoint_reinforced["history"])
    elif checkpoint_supervised is not None:
        models["embedder_supervised"].load_state_dict(checkpoint_supervised["predictor_state_dict"])
    else:
        print("[WARNING] No checkpoint from reinforcement training or semisupervised training found. Starting from zero.")

    if checkpoint_supervised is not None:
        embedder_supervised_orig.load_state_dict(checkpoint_supervised["predictor_state_dict"])

    # initialize loss criterions
    criterions = {
        "direct_reward_predictor": nn.CrossEntropyLoss(),
        "indirect_reward_predictor": nn.MSELoss(),
        "successor_predictor": nn.MSELoss(),
        "ae_decoder": nn.MSELoss(),
    }

    # send everything to the GPU
    if Config.GPU >= 0:
        embedder_supervised_orig.cuda(Config.GPU)
        for key in models:
            models[key].cuda(Config.GPU)
        for key in criterions:
            criterions[key].cuda(Config.GPU)

    # initialize image augmentation cascade
    rarely = lambda aug: iaa.Sometimes(0.1, aug)
    sometimes = lambda aug: iaa.Sometimes(0.2, aug)
    often = lambda aug: iaa.Sometimes(0.3, aug)
    augseq = iaa.Sequential([
            often(iaa.Crop(percent=(0, 0.05))),
            sometimes(iaa.GaussianBlur((0, 0.2))), # blur images with a sigma between 0 and 3.0
            often(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5)), # add gaussian noise to images
            often(
                iaa.Dropout(
                    iap.FromLowerResolution(
                        other_param=iap.Binomial(1 - 0.2),
                        size_px=(2, 16)
                    ),
                    per_channel=0.2
                )
            ),
            rarely(iaa.Sharpen(alpha=(0, 0.7), lightness=(0.75, 1.5))), # sharpen images
            rarely(iaa.Emboss(alpha=(0, 0.7), strength=(0, 2.0))), # emboss images
            rarely(iaa.Sometimes(0.5,
                iaa.EdgeDetect(alpha=(0, 0.4)),
                iaa.DirectedEdgeDetect(alpha=(0, 0.4), direction=(0.0, 1.0)),
            )),
            often(iaa.Add((-20, 20), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
            often(iaa.Multiply((0.8, 1.2), per_channel=0.25)), # change brightness of images (50-150% of original value)
            often(iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5)), # improve or worsen the contrast
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(0, 0),
                shear=(0, 0),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            ))
        ],
        random_order=True # do all of the above in random order
    )

    # ---------------------
    # game loop
    # ---------------------
    ora = OnRouteAdvisorVisible(
        memory,
        memory_val,
        models,
        embedder_supervised_orig,
        optimizers,
        criterions,
        history,
        loss_plotter,
        augseq,
        nb_ticks_done=0 if checkpoint_reinforced is None else checkpoint_reinforced["nb_ticks_done"],
        last_train_tick=-1 if checkpoint_reinforced is None else checkpoint_reinforced["last_train_tick"],
        batch_idx_total=0 if checkpoint_reinforced is None else checkpoint_reinforced["batch_idx_total"],
        epoch=0 if checkpoint_reinforced is None else checkpoint_reinforced["epoch"],
        p_explore=P_EXPLORE_MAX if checkpoint_reinforced is None else checkpoint_reinforced["p_explore"],
        args=args
    )
    if args.onlytrain:
        while True:
            ora.train()
    game = ets2game.ETS2Game()
    game.on_route_advisor_visible = ora
    game.min_interval = 100 / 1000
    try:
        game.run()
    except KeyboardInterrupt:
        if args.record is not None and len(ora.recording["frames"]) > 0:
            wait_sec = 10
            print("Saving recording of %d frames to filepath '%s' in %d seconds. Press CTRL+C to cancel." % (len(ora.recording["frames"]), args.record, wait_sec))
            if os.path.isfile(args.record):
                print("[WARNING] File already exists! This will overwrite the file!")
            else:
                print("The file '%s' does to not exist yet." % (args.record,))
            time.sleep(wait_sec)
            print("Saving...")
            #path_pieces = os.path.split(args.record)
            directory = os.path.dirname(args.record)
            #if len(path_pieces) > 1:
            if directory != "":
                if not os.path.exists(directory):
                    os.makedirs(directory)
            for frame in ora.recording["frames"]:
                frame["scr"] = util.compress_to_jpg(frame["scr"])
            with gz.open(args.record, "wb") as f:
                pickle.dump(ora.recording, f, protocol=-1)

class OnRouteAdvisorVisible(object):
    def __init__(self, memory, memory_val, models, embedder_supervised_orig, optimizers, criterions, history, loss_plotter, augseq, nb_ticks_done, last_train_tick, batch_idx_total, epoch, args, p_explore=P_EXPLORE_MAX):
        self.memory = memory
        self.memory_val = memory_val
        self.models = models
        self.embedder_supervised_orig = embedder_supervised_orig
        self.optimizers = optimizers
        self.criterions = criterions
        self.history = history
        self.loss_plotter = loss_plotter
        self.augseq = augseq
        self.previous_states = collections.deque(maxlen=max(PREVIOUS_STATES_DISTANCES))
        self.last_scr = None # used for recording
        self.p_explore = collections.deque([p_explore], maxlen=NB_FUTURE_STATES_APPLY)
        self._fill_p_explore()
        self.last_speeds = collections.deque(maxlen=500)
        self.nb_ticks_done = nb_ticks_done
        self.last_train_tick = last_train_tick
        self.batch_idx_total = batch_idx_total
        self.epoch = epoch
        self.sw_tracker = swlib.SteeringWheelTrackerCNN()

        self.current_plan_idx = None
        self.current_plan_step_idx = None
        self.current_plan_reward = None

        self.args = args
        self.recording = {
            "time": time.time(),
            "args": args,
            "plans": PLANS,
            "frames": []
        }

        self.switch_models_to_eval()
        #self.switch_models_to_train()

    def switch_models_to_train(self):
        for key in self.models:
            self.models[key].train()

    def switch_models_to_eval(self):
        for key in self.models:
            self.models[key].eval()

    def _fill_p_explore(self):
        while len(self.p_explore) < self.p_explore.maxlen:
            self._add_p_explore()

    def _add_p_explore(self):
        self.p_explore.append(np.clip(self.p_explore[-1] - P_EXPLORE_STEPSIZE, P_EXPLORE_MIN, P_EXPLORE_MAX))

    def __call__(self, game, scr):
        """Function called every 100ms while playing.
        Applies the model in order to play, records experiences/states to
        the replay memory and calls the training function every now and then."""

        if game.win.is_paused(scr):
            print("[OnRouteAdvisorVisible] Game paused, skipping execution/training. Unpause the game.")
            return

        # decrease replay memory size to max size
        if self.memory.is_above_tolerance() or self.memory_val.is_above_tolerance():
            game.reset_actions()
            time.sleep(1)
            game.pause()
            time.sleep(1)
            self.memory.shrink_to_max_size(force=True)
            self.memory_val.shrink_to_max_size(force=True)
            time.sleep(1)
            game.unpause()
            time.sleep(1)
            return

        # train the model
        load_save = False
        if DO_TRAIN(self, game):
            game.reset_actions()
            time.sleep(1)
            self.memory.commit()
            self.memory_val.commit()
            time.sleep(0.5)
            self.memory.close()
            self.memory_val.close()
            game.pause()
            self.train()
            game.unpause()
            time.sleep(0.5)
            self.memory.connect()
            self.memory_val.connect()
            time.sleep(1)
            load_save = True
            self.last_train_tick = game.tick_ingame_route_advisor_idx

        # game load must come after save, because loading might take too long,
        # then training pause is called during load screen, which then causes problems
        if not self.args.nospeedkill and len(self.last_speeds) == self.last_speeds.maxlen and np.average(self.last_speeds) < 3:
            load_save = True
            print("[OnRouteAdvisorVisible] Low speeds for a long time, auto-reloading save game.")
            print("[OnRouteAdvisorVisible] average speed: %.4f, max speed: %d" % (np.average(self.last_speeds), np.max(self.last_speeds)))
            print("[OnRouteAdvisorVisible] last speeds: ", self.last_speeds)

        # load savegame
        if load_save:
            game.reset_actions()
            time.sleep(1)
            self.last_speeds.clear()
            self.previous_states.clear()
            self.sw_tracker.reset()
            game.load_random_save_game()
            # if return is not used here, then the screenshot (scr) should be
            # update as some time might have passed
            return

        times = dict()

        # estimate current speed
        times["images"] = time.time()
        speed_image = game.win.get_speed_image(scr)
        som = speedlib.SpeedOMeter(speed_image)
        times["images"] = time.time() - times["images"]

        # initialize current state
        times["state"] = time.time()
        from_datetime = datetime.utcnow()
        screenshot_rs = ia.imresize_single_image(scr, (Config.MODEL_HEIGHT, Config.MODEL_WIDTH), interpolation="cubic")
        screenshot_rs_jpg = util.compress_to_jpg(screenshot_rs)
        speed = som.predict_speed()
        is_reverse = game.win.is_reverse(scr)
        is_offence_shown = game.win.is_offence_shown(scr) # 4-6ms
        is_damage_shown = game.win.is_damage_shown(scr)
        sw, sw_raw = self.sw_tracker.estimate_angle(util.decompress_img(screenshot_rs_jpg)) # ~2.5ms

        current_state = stateslib.State(
            from_datetime=from_datetime,
            screenshot_rs_jpg=screenshot_rs_jpg,
            speed=speed,
            is_reverse=is_reverse,
            is_offence_shown=is_offence_shown,
            is_damage_shown=is_damage_shown,
            reward=None,
            action_left_right=None,
            action_up_down=None,
            p_explore=self.p_explore[0],
            steering_wheel_classical=None,
            steering_wheel_raw_one_classical=None,
            steering_wheel_raw_two_classical=None,
            steering_wheel_cnn=sw,
            steering_wheel_raw_cnn=sw_raw,
            allow_cache=True
        )
        times["state"] = time.time() - times["state"]

        # apply model once enough previous states were observed
        if len(self.previous_states) >= self.previous_states.maxlen:
            times["model"] = time.time()

            # transform current and previous states to batch
            # same function here as in training to reduce probability of bugs
            #batch = states_to_batch([list(self.previous_states)], [[current_state]], self.augseq, PREVIOUS_STATES_DISTANCES, MODEL_HEIGHT, MODEL_WIDTH, MODEL_PREV_HEIGHT, MODEL_PREV_WIDTH)
            batch = states_to_batch([list(self.previous_states)], [[current_state]], iaa.Noop(), PREVIOUS_STATES_DISTANCES, MODEL_HEIGHT, MODEL_WIDTH, MODEL_PREV_HEIGHT, MODEL_PREV_WIDTH)

            # generate embeddings of semi-supervised model part
            inputs_supervised = batch.inputs_supervised(volatile=True, gpu=Config.GPU)
            embeddings_supervised = self.models["embedder_supervised"].embed(inputs_supervised[0], inputs_supervised[1])

            # generate embeddings of reinforced model part
            inputs_reinforced_add = batch.inputs_reinforced_add(volatile=True, gpu=Config.GPU)
            embeddings_reinforced = self.models["embedder_reinforced"].forward_dict(embeddings_supervised, inputs_reinforced_add)

            # predict successors for each plan (chain of future actions)
            # embeddings_reinforced: (1, 512), PLANS_VECS_VAR: (T, N_PLANS, 9), future_embeddings_rf: (T, N_PLANS, 512)
            future_embeddings_rf, hidden = self.models["successor_predictor"].forward_apply(embeddings_reinforced, PLANS_VECS_VAR, gpu=Config.GPU)
            # must transpose here first, otherwise the flattened array is
            #   <B=0,T=0>,<B=1,T=0>,...
            # instead of
            #   <B=0,T=0>,<B=0,T=1>,...
            future_embeddings_rf_flat = future_embeddings_rf.transpose(0, 1).contiguous().view(PLANS_VECS_VAR.size(1) * NB_FUTURE_STATES_APPLY, 512) # (N_PLANS*T, 512)

            # predict future direct rewards
            direct_rewards = self.models["direct_reward_predictor"].forward(future_embeddings_rf_flat, softmax=True)

            # predict current state indirect rewards (V(s) and A(s, a))
            _, (idr_v, idr_adv) = self.models["indirect_reward_predictor"].forward(embeddings_reinforced, return_v_adv=True)

            # predict future/successor indirect rewards
            _, (plan_to_v, _) = self.models["indirect_reward_predictor"].forward(future_embeddings_rf[-1, ...].view(PLANS_VECS_VAR.size(1), -1), return_v_adv=True)

            # estimate plan ranking based on predicted rewards
            plan_to_reward_bins = to_numpy(direct_rewards.view(len(PLANS), NB_FUTURE_STATES_APPLY, -1))
            plan_to_rewards_direct, plan_to_reward_indirect, plan_to_reward = calculate_future_rewards(plan_to_reward_bins, to_numpy(plan_to_v))
            plans_ranking = np.argsort(plan_to_reward, axis=0, kind="mergesort")
            best_plan = PLANS[plans_ranking[-1]]

            # predict AE decodings for future states of best plan
            #best_plan_ae_decodings = self.models["ae_decoder"](future_embeddings_rf[:, plans_ranking[-1], :].contiguous().view(NB_FUTURE_STATES_APPLY, 512))
            best_plan_ae_decodings = None
            if self.args.showgrids:
                grids = self.embedder_supervised_orig.predict_grids(inputs_supervised[0], inputs_supervised[1])
            else:
                grids = None

            # use the calculated plan or stick with the old one?
            use_this_plan = (self.current_plan_idx is None) or (self.current_plan_step_idx >= PICK_NEW_PLAN_EVERY_N_STEPS-1)
            if use_this_plan:
                self.current_plan_idx = plans_ranking[-1]
                self.current_plan_step_idx = 0
            else:
                best_plan = PLANS[self.current_plan_idx]
                self.current_plan_step_idx += 1
            times["model"] = time.time() - times["model"]

            # choose actions accordings to current plan,
            # add epsilon-greedy to that
            times["ai-control"] = time.time()
            action_up_down_bpe, action_left_right_bpe = best_plan[self.current_plan_step_idx]
            action_up_down, action_left_right = p_explore_to_actions(
                self.p_explore[0], action_up_down_bpe, action_left_right_bpe,
                self.previous_states[-1] if len(self.previous_states) > 0 else None,
                constant_p_explore=self.args.p_explore)
            times["ai-control"] = time.time() - times["ai-control"]
        else:
            plan_to_rewards_direct = None
            plan_to_reward_indirect = None
            plan_to_reward = None
            plans_ranking = None
            best_plan_ae_decodings = None
            grids = None
            action_up_down = action_up_down_bpe = actionslib.ACTION_UP_DOWN_NONE
            action_left_right = action_left_right_bpe = actionslib.ACTION_LEFT_RIGHT_NONE
            outputs_to_rewards = None
            idr_v, idr_adv = None, None

        game.set_actions_of_interval([action_up_down, action_left_right])
        current_state.action_up_down = action_up_down
        current_state.action_left_right = action_left_right

        times["reward"] = time.time()
        # last state reward
        if len(self.previous_states) > 0:
            # calculate last state's reward, i.e. for
            #   (s, a, r) (s', a', r')
            # we compute r (derived from s') and attach it to s
            last_state = self.previous_states[-1]
            last_state_reward = rewardslib.calculate_reward(last_state, current_state)
            last_state.reward = last_state_reward

            # save last state to train/val DB
            if not self.args.noinsert:
                if MEMORY_ADD_STATE_TO_TRAIN(game.tick_ingame_route_advisor_idx):
                    self.memory.add_state(last_state, commit=False, shrink=False)
                else:
                    self.memory_val.add_state(last_state, commit=False, shrink=False)
        else:
            last_state = None
            last_state_reward = None
        times["reward"] = time.time() - times["reward"]

        # commit last DB inserts of states
        times["commit"] = time.time()
        do_commit = (game.tick_ingame_route_advisor_idx % COMMIT_EVERY == 0)
        if not self.args.noinsert and do_commit:
            self.memory.commit()
            self.memory_val.commit()
        times["commit"] = time.time() - times["commit"]

        if self.args.record is not None and len(self.previous_states) >= 1:
            self.recording["frames"].append({
                #"scr": util.compress_to_jpg(self.last_scr),
                "scr": self.last_scr,
                "scr_time": game.last_scr_time,
                "state": self.previous_states[-1],
                "current_plan_idx": self.current_plan_idx,
                "current_plan_step_idx": self.current_plan_step_idx,
                "idr_v": to_numpy(idr_v[0]) if idr_v is not None else None,
                "idr_adv": to_numpy(idr_adv[0]) if idr_adv is not None else None,
                "plan_to_rewards_direct": plan_to_rewards_direct,
                "plan_to_reward_indirect": plan_to_reward_indirect,
                "plan_to_reward": plan_to_reward,
                "plans_ranking": plans_ranking
            })

        self.previous_states.append(current_state)
        self.last_scr = scr
        self._add_p_explore()
        if speed is not None:
            self.last_speeds.append(speed)
        else:
            # do not add None to last speeds, otherwise functions like mean()
            # dont work any more
            print("[WARNING] Speed is None! Last speeds: ", self.last_speeds)
            if len(self.last_speeds) > 0:
                self.last_speeds.append(self.last_speeds[-1])
            else:
                self.last_speeds.append(0)

        # generate overview image
        if game.tick_ingame_route_advisor_idx % 1 == 0:
            times["overview-img"] = time.time()
            img = generate_overview_image(
                current_state, last_state,
                action_up_down_bpe, action_left_right_bpe,
                self.memory,
                self.memory_val,
                game.tick_ingame_route_advisor_idx,
                self.last_train_tick,
                PLANS,
                plan_to_rewards_direct,
                plan_to_reward_indirect,
                plan_to_reward,
                plans_ranking,
                PLANS[self.current_plan_idx][self.current_plan_step_idx:] if self.current_plan_idx is not None else None,
                best_plan_ae_decodings,
                idr_v, idr_adv,
                grids,
                self.args
            )
            times["overview-img"] = time.time() - times["overview-img"]

            times["overview-img-show"] = time.time()
            cv2.imshow("overview", img[:,:,::-1])
            cv2.waitKey(1)
            times["overview-img-show"] = time.time() - times["overview-img-show"]

        #for time_name in times:
        #    print("%s: %.4fs" % (time_name, times[time_name]))

        self.nb_ticks_done += 1

    def train(self):
        """Training function."""

        print("[OnRouteAdvisorVisible] Training.")
        print("[OnRouteAdvisorVisible] Memory size: %d train, %d val" % (self.memory.size, self.memory_val.size))

        # initialize background batch loaders that generate batches on other
        # CPU cores
        batch_loader_train = BatchLoader(
            val=False, batch_size=BATCH_SIZE, augseq=self.augseq,
            previous_states_distances=PREVIOUS_STATES_DISTANCES, nb_future_states=NB_FUTURE_STATES_TRAIN, model_height=MODEL_HEIGHT, model_width=MODEL_WIDTH, model_prev_height=MODEL_PREV_HEIGHT, model_prev_width=MODEL_PREV_WIDTH
        )
        batch_loader_val = BatchLoader(
            val=True, batch_size=BATCH_SIZE, augseq=iaa.Noop(),
            previous_states_distances=PREVIOUS_STATES_DISTANCES, nb_future_states=NB_FUTURE_STATES_VAL, model_height=MODEL_HEIGHT, model_width=MODEL_WIDTH, model_prev_height=MODEL_PREV_HEIGHT, model_prev_width=MODEL_PREV_WIDTH
        )
        batch_loader_train = BackgroundBatchLoader(batch_loader_train, queue_size=25, nb_workers=6)
        batch_loader_val = BackgroundBatchLoader(batch_loader_val, queue_size=15, nb_workers=2)

        self.switch_models_to_train()

        for batch_idx in xrange(NB_BATCHES_PER_TRAIN):
            # fix model parameters every N batches
            if batch_idx == 0 or batch_idx % TRAIN_FIX_EVERY_N_BATCHES == 0:
                models_fixed = dict([(key, copy.deepcopy(model)) for (key, model) in self.models.items() if key in set(["indirect_reward_predictor"])])

            self._run_batch(batch_loader_train, batch_idx, models_fixed, val=False)

            if DO_VALIDATE(self.epoch, batch_idx, self.batch_idx_total):
                self.switch_models_to_eval()
                for i in xrange(NB_VAL_BATCHES):
                    self._run_batch(batch_loader_val, batch_idx, models_fixed, val=True)
                self.switch_models_to_train()

            # every N batches, plot loss curves
            if DO_PLOT(self.epoch, batch_idx, self.batch_idx_total):
                self.loss_plotter.plot(self.history)

            # every N batches, save a checkpoint
            if DO_SAVE(self.epoch, batch_idx, self.batch_idx_total):
                self._save()

            self.batch_idx_total += 1

        print("Joining batch loaders...")
        batch_loader_train.join()
        batch_loader_val.join()

        print("Saving...")
        self._save()

        self.switch_models_to_eval()
        self.epoch += 1

        print("Finished training.")

    def _run_batch(self, batch_loader, batch_idx, models_fixed, val):
        """Train/Validate on a single batch."""
        time_prep_start = time.time()
        train = not val

        embedder_supervised = self.models["embedder_supervised"]
        embedder_reinforced = self.models["embedder_reinforced"]
        direct_reward_predictor = self.models["direct_reward_predictor"]
        indirect_reward_predictor = self.models["indirect_reward_predictor"]
        successor_predictor = self.models["successor_predictor"]
        ae_decoder = self.models["ae_decoder"]

        #embedder_supervised_fixed = models_fixed["embedder_supervised"]
        #embedder_reinforced_fixed = models_fixed["embedder_reinforced"]
        #direct_reward_predictor_fixed = models_fixed["direct_reward_predictor"]
        indirect_reward_predictor_fixed = models_fixed["indirect_reward_predictor"]
        #successor_predictor_fixed = models_fixed["successor_predictor"]
        #ae_decoder_fixed = models_fixed["ae_decoder"]

        backward_supervised = not val and DO_TRAIN_SUPERVISED(self.epoch, batch_idx, self.batch_idx_total)
        backward_successor_rewards = DO_BACKPROP_SUCCESSOR_REWARDS(self.epoch, batch_idx, self.batch_idx_total)
        cutoff_backward_successor = DO_CUTOFF_BACKPROP_SUCCESSOR(self.epoch, batch_idx, self.batch_idx_total)
        cutoff_backward_successor_rewards = DO_CUTOFF_BACKPROP_SUCCESSOR_REWARDS(self.epoch, batch_idx, self.batch_idx_total)
        train_ae = DO_TRAIN_AE(self.epoch, batch_idx, self.batch_idx_total)
        #print(
        #    "backward_supervised", backward_supervised,
        #    "backward_successor_rewards", backward_successor_rewards,
        #    "cutoff_backward_successor", cutoff_backward_successor,
        #    "cutoff_backward_successor_rewards", cutoff_backward_successor_rewards,
        #    "train_ae", train_ae
        #)
        time_prep_end = time.time()

        # ----------
        # collect batch
        # ----------
        time_cbatch_start = time.time()
        batch = batch_loader.get_batch()
        time_cbatch_end = time.time()

        time_cbatchf_start = time.time()
        inputs_supervised = batch.inputs_supervised(volatile=not backward_supervised, requires_grad=False, gpu=Config.GPU)
        inputs_reinforced_add = batch.inputs_reinforced_add(volatile=val, requires_grad=False, gpu=Config.GPU)
        future_inputs_supervised = batch.future_inputs_supervised(volatile=not backward_supervised, requires_grad=False, gpu=Config.GPU)
        future_reinforced_add = batch.future_reinforced_add(volatile=val, requires_grad=False, gpu=Config.GPU)
        inputs_successor_multiactions_vecs = batch.inputs_successor_multiactions_vecs(volatile=val, requires_grad=False, gpu=Config.GPU)
        outputs_dr_gt = batch.outputs_dr_gt(volatile=val, requires_grad=False, gpu=Config.GPU)
        outputs_dr_future_gt = batch.outputs_dr_future_gt(volatile=val, requires_grad=False, gpu=Config.GPU)
        direct_reward_values = batch.direct_rewards_values(volatile=val, requires_grad=False, gpu=Config.GPU)
        future_direct_reward_values = batch.future_direct_rewards_values(volatile=val, requires_grad=False, gpu=Config.GPU)
        #future_direct_reward_values = batch.future_direct_rewards_values(volatile=val, requires_grad=False, gpu=GPU)
        #next_direct_rewards_values = future_direct_reward_values[0, ...]
        chosen_action_indices = batch.chosen_action_indices()
        chosen_action_indices_future = batch.chosen_action_indices_future()
        outputs_ae_gt = batch.outputs_ae_gt(volatile=val, requires_grad=False, gpu=Config.GPU)
        time_cbatchf_end = time.time()

        # ----------
        # forward/backward
        # ----------
        time_fwbw_start = time.time()

        if train:
            # zero grad of all optimizers
            for key in self.optimizers:
                self.optimizers[key].zero_grad()

        # embed future states
        T, B, C, H, W = future_inputs_supervised[0].size()
        T, B, Cprev, Hprev, Wprev = future_inputs_supervised[1].size()
        future_emb_sup = embedder_supervised.embed(
            future_inputs_supervised[0].view(T*B, C, H, W),
            future_inputs_supervised[1].view(T*B, Cprev, Hprev, Wprev)
        )
        if not backward_supervised:
            future_emb_sup = Variable(future_emb_sup.data, volatile=val, requires_grad=True if not val else False)
        future_emb_rf = embedder_reinforced.forward_dict(future_emb_sup, future_reinforced_add)
        future_emb_rf_by_time = future_emb_rf.view(T, B, -1)
        outputs_successor_gt = to_cuda(Variable(future_emb_rf_by_time.data, volatile=val, requires_grad=False), Config.GPU)

        # embed current states
        emb_sup = embedder_supervised.embed(inputs_supervised[0], inputs_supervised[1])
        if not backward_supervised:
            emb_sup = to_cuda(Variable(emb_sup.data, volatile=val, requires_grad=True if not val else False), Config.GPU)
        emb_rf = embedder_reinforced.forward_dict(emb_sup, inputs_reinforced_add)

        # predict indirect rewards
        outputs_idr_preds = indirect_reward_predictor(emb_rf)
        outputs_idr_preds_next_fixed = indirect_reward_predictor_fixed(future_emb_rf_by_time[0])

        outputs_idr_gt = direct_reward_values + GAMMA * outputs_idr_preds_next_fixed
        outputs_idr_gt = idr_zerograd_unchosen_actions(outputs_idr_gt, outputs_idr_preds, chosen_action_indices)
        outputs_idr_gt = to_cuda(Variable(outputs_idr_gt.data, volatile=val, requires_grad=False), Config.GPU)

        # predict direct reward, successor states, AE decodings
        outputs_dr_preds = direct_reward_predictor(emb_rf, softmax=False)
        successor_input = emb_rf
        if cutoff_backward_successor:
            successor_input = to_cuda(Variable(successor_input.data, volatile=val, requires_grad=True if not val else False), Config.GPU)
        outputs_successor_preds = successor_predictor(
            successor_input,
            inputs_successor_multiactions_vecs,
            volatile=False, gpu=Config.GPU
        )[0]
        if train_ae:
            outputs_ae_preds = ae_decoder(emb_rf)
        else:
            outputs_ae_preds = outputs_ae_gt

        # predict direct rewards on future states
        # (embeddings for these are already available for successors anyways)
        outputs_dr_future_preds = direct_reward_predictor(future_emb_rf_by_time.view(T*B, -1), softmax=False)

        outputs_idr_future_preds_all = indirect_reward_predictor(future_emb_rf_by_time.view(T*B, -1)).view(T, B, -1)
        outputs_idr_future_preds_all_fixed = indirect_reward_predictor_fixed(future_emb_rf_by_time.view(T*B, -1)).view(T, B, -1)
        outputs_idr_future_preds = outputs_idr_future_preds_all[0:-1]
        outputs_idr_future_preds_next = outputs_idr_future_preds_all_fixed[1:]
        outputs_idr_future_gt = future_direct_reward_values[:-1] + GAMMA * outputs_idr_future_preds_next
        outputs_idr_future_gt = idr_zerograd_unchosen_actions_future(outputs_idr_future_gt, outputs_idr_future_preds, chosen_action_indices_future)
        outputs_idr_future_preds = outputs_idr_future_preds.view((T-1)*B, -1)
        outputs_idr_future_gt = outputs_idr_future_gt.view((T-1)*B, -1)
        outputs_idr_future_gt = to_cuda(Variable(outputs_idr_future_gt.data, volatile=val, requires_grad=False), gpu=Config.GPU)

        # predict the direct and indirect reward for successors
        # (i.e. predictions of future states)
        # these losses are currently not backpropagated through the successor
        # and embedder
        dr_successors_input = outputs_successor_preds
        idr_successors_input = outputs_successor_preds[0:-1]
        if cutoff_backward_successor_rewards:
            dr_successors_input = to_cuda(Variable(dr_successors_input.data, volatile=val, requires_grad=True if not val else False), Config.GPU)
            idr_successors_preds = to_cuda(Variable(idr_successors_input.data, volatile=val, requires_grad=True if not val else False), Config.GPU)
        outputs_dr_successors_preds = direct_reward_predictor(dr_successors_input.view(T*B, -1), softmax=False)
        outputs_idr_successors_preds = indirect_reward_predictor(idr_successors_input.view((T-1)*B, -1)).view((T-1), B, -1)
        outputs_dr_successors_gt = outputs_dr_future_gt
        outputs_idr_successors_gt = outputs_idr_future_gt.clone().view((T-1), B, -1)
        outputs_idr_successors_gt = idr_zerograd_unchosen_actions_future(outputs_idr_successors_gt, outputs_idr_successors_preds, chosen_action_indices_future)
        outputs_idr_successors_gt = outputs_idr_successors_gt.view((T-1)*B, -1)
        outputs_idr_successors_gt = to_cuda(Variable(outputs_idr_successors_gt.data, volatile=val, requires_grad=False), gpu=Config.GPU)

        #print("outputs_dr_preds", outputs_dr_preds.size(), outputs_dr_gt.size())
        #print("outputs_dr_gt.max(dim=1)[1]", outputs_dr_gt.max(dim=1)[1])
        #print("outputs_dr_future_preds", outputs_dr_future_preds.size(), outputs_dr_future_gt.size())
        #print("outputs_dr_successors_preds", outputs_dr_successors_preds.size(), outputs_dr_successors_gt.size())
        #print("outputs_idr_successors_preds", outputs_idr_successors_preds.size(), outputs_idr_successors_gt.size())

        loss_dr = self.criterions["direct_reward_predictor"](outputs_dr_preds, outputs_dr_gt.max(dim=1)[1].squeeze(dim=1))
        loss_dr_future = self.criterions["direct_reward_predictor"](outputs_dr_future_preds, outputs_dr_future_gt.view(T*B, -1).max(dim=1)[1].squeeze(dim=1))
        loss_dr_successors = self.criterions["direct_reward_predictor"](outputs_dr_successors_preds, outputs_dr_successors_gt.view(T*B, -1).max(dim=1)[1].squeeze(dim=1))
        loss_idr = self.criterions["indirect_reward_predictor"](outputs_idr_preds, outputs_idr_gt)
        loss_idr_future = self.criterions["indirect_reward_predictor"](outputs_idr_future_preds, outputs_idr_future_gt)
        loss_idr_successors = self.criterions["indirect_reward_predictor"](outputs_idr_successors_preds, outputs_idr_successors_gt)
        loss_successor = self.criterions["successor_predictor"](outputs_successor_preds, outputs_successor_gt)
        loss_ae = self.criterions["ae_decoder"](outputs_ae_preds, outputs_ae_gt) if train_ae else None

        if train:
            lw = [
                (loss_dr, LOSS_DR_WEIGHTING),
                (loss_dr_future, LOSS_DR_FUTURE_WEIGHTING),
                (loss_idr, LOSS_IDR_WEIGHTING),
                (loss_idr_future, LOSS_IDR_FUTURE_WEIGHTING),
                (loss_successor, LOSS_SUCCESSOR_WEIGHTING),
                (loss_dr_successors, LOSS_DR_SUCCESSORS_WEIGHTING if backward_successor_rewards else 0),
                (loss_idr_successors, LOSS_IDR_SUCCESSORS_WEIGHTING if backward_successor_rewards else 0),
                (loss_ae, LOSS_AE_WEIGHTING if train_ae else 0)
            ]
            lw_reduced = [(loss, weighting) for (loss, weighting) in lw if loss is not None and weighting > 0]

            losses = [loss for (loss, weighting) in lw_reduced]
            losses_grad = [loss.data.new().resize_as_(loss.data).fill_(weighting) for (loss, weighting) in lw_reduced]

            if len(lw_reduced) == 0:
                print("[WARNING] After removing losses without weights there was nothing left to train on.")
            else:
                torch.autograd.backward(losses, losses_grad)
                for key in self.optimizers:
                    if key == "embedder_supervised":
                        if backward_supervised:
                            self.optimizers[key].step()
                    elif key == "ae_decoder":
                        if train_ae:
                            self.optimizers[key].step()
                    else:
                        self.optimizers[key].step()

        torch.cuda.synchronize()
        time_fwbw_end = time.time()

        time_finish_start = time.time()
        # add average loss values to history and output message
        self.history.add_value("loss-dr", "train" if train else "val", self.batch_idx_total, loss_dr.data[0], average=val)
        self.history.add_value("loss-dr-future", "train" if train else "val", self.batch_idx_total, loss_dr_future.data[0], average=val)
        self.history.add_value("loss-dr-successors", "train" if train else "val", self.batch_idx_total, loss_dr_successors.data[0], average=val)
        self.history.add_value("loss-idr", "train" if train else "val", self.batch_idx_total, loss_idr.data[0], average=val)
        self.history.add_value("loss-idr-future", "train" if train else "val", self.batch_idx_total, loss_idr_future.data[0], average=val)
        self.history.add_value("loss-idr-successors", "train" if train else "val", self.batch_idx_total, loss_idr_successors.data[0], average=val)
        self.history.add_value("loss-successor", "train" if train else "val", self.batch_idx_total, loss_successor.data[0], average=val)
        self.history.add_value("loss-ae", "train" if train else "val", self.batch_idx_total, loss_ae.data[0] if train_ae else 0, average=val)

        msg_losses = "dr=%.4f dr-future=%.4f dr-successors=%.4f idr=%.4f idr-future=%.4f idr-successors=%.4f successor=%.4f ae=%.4f" % (loss_dr.data[0], loss_dr_future.data[0], loss_dr_successors.data[0], loss_idr.data[0], loss_idr_future.data[0], loss_idr_successors.data[0], loss_successor.data[0], loss_ae.data[0] if train_ae else 0,)

        # generate an overview image of the last batch
        if (batch_idx+1) % 20 == 0 or val:
            debug_img = generate_training_debug_image(
                inputs_supervised[0], inputs_supervised[1],
                outputs_dr_preds, outputs_dr_gt,
                outputs_idr_preds, outputs_idr_gt,
                outputs_successor_preds, outputs_successor_gt,
                outputs_ae_preds, outputs_ae_gt,
                outputs_dr_successors_preds.view(T, B, -1), outputs_dr_successors_gt,
                outputs_idr_successors_preds.view((T-1), B, 9), outputs_idr_successors_gt.view((T-1), B, 9),
                batch.multiactions
            )
            misc.imsave("train_reinforced_debug_img_%s.png" % ("train" if train else "val",), debug_img)
        time_finish_end = time.time()

        print("%s%d/%04d L[%s] T[prep=%.02fs, cb=%.02fs cbf=%.2fs fwbw=%.02fs, fin=%.2fs]" % (
            "T" if train else "V",
            self.epoch,
            batch_idx,
            msg_losses,
            time_prep_end - time_prep_start,
            time_cbatch_end - time_cbatch_start,
            time_cbatchf_end - time_cbatchf_start,
            time_fwbw_end - time_fwbw_start,
            time_finish_end - time_finish_start
        ))

    def _save(self):
        """Function to save a checkpoint."""
        torch.save({
            "history": self.history.to_string(),
            "embedder_supervised_state_dict": self.models["embedder_supervised"].state_dict(),
            "embedder_reinforced_state_dict": self.models["embedder_reinforced"].state_dict(),
            "direct_reward_predictor_state_dict": self.models["direct_reward_predictor"].state_dict(),
            "indirect_reward_predictor_state_dict": self.models["indirect_reward_predictor"].state_dict(),
            "successor_predictor_state_dict": self.models["successor_predictor"].state_dict(),
            "ae_decoder_state_dict": self.models["ae_decoder"].state_dict(),
            "p_explore": self.p_explore[0],
            "nb_ticks_done": self.nb_ticks_done,
            "last_train_tick": self.last_train_tick,
            "batch_idx_total": self.batch_idx_total,
            "epoch": self.epoch
        }, "train_reinforced_model.tar")

def idr_zerograd_unchosen_actions(outputs_idr_gt, outputs_idr_preds, chosen_action_indices):
    """Set gradients to zero for indirect rewards which's actions were not
    chosen. This is done by setting the ground truth to the prediction."""
    B, _ = outputs_idr_gt.size()
    for b_idx in xrange(B):
        pred = outputs_idr_preds[b_idx]
        for a_idx in xrange(9):
            chosen_action_idx = chosen_action_indices[b_idx]
            if a_idx != chosen_action_idx:
                outputs_idr_gt[b_idx, a_idx] = pred[a_idx]
    return outputs_idr_gt

def idr_zerograd_unchosen_actions_future(outputs_idr_future_gt, outputs_idr_future_preds, chosen_action_indices_future):
    """Set gradients to zero for future indirect rewards which's actions were not
    chosen. This is done by setting the ground truth to the prediction."""
    T, B, _ = outputs_idr_future_gt.size()
    for t_idx in xrange(T):
        for b_idx in xrange(B):
            pred = outputs_idr_future_preds[t_idx, b_idx]
            for a_idx in xrange(9):
                chosen_action_idx = chosen_action_indices_future[t_idx][b_idx]
                if a_idx != chosen_action_idx:
                    outputs_idr_future_gt[t_idx, b_idx, a_idx] = pred[a_idx]
    return outputs_idr_future_gt

"""
def calculate_future_rewards_argmax(plan_to_reward_bins):
    B, T, S = plan_to_reward_bins.shape
    plan_to_reward_bins_ids = np.argmax(plan_to_reward_bins, axis=2)
    plan_to_rewards = BIN_AVERAGES[plan_to_reward_bins_ids.flatten()].reshape(B, T)
    plan_to_rewards_direct = plan_to_rewards * REWARDS_FUTURE_GAMMAS
    plan_to_reward = np.sum(plan_to_rewards_direct, axis=1)
    return plan_to_rewards_direct, plan_to_reward
"""

def calculate_future_rewards(plan_to_reward_bins, indirect_rewards):
    """Compute expected rewards of plans by predicted direct and indirect
    rewards (latter: only of last successor)."""
    B, T, S = plan_to_reward_bins.shape
    if DIRECT_REWARD_ARGMAX:
        plan_to_reward_bins_ids = np.argmax(plan_to_reward_bins, axis=2)
        plan_to_rewards = BIN_AVERAGES[plan_to_reward_bins_ids.flatten()].reshape(B, T)
    else:
        plan_to_rewards = np.sum(plan_to_reward_bins * BIN_AVERAGES, axis=2)
    plan_to_rewards_direct = plan_to_rewards * REWARDS_FUTURE_GAMMAS
    #plan_to_reward_indirect = np.average(indirect_rewards, axis=1) * np.power(GAMMA, REWARDS_FUTURE_GAMMAS.shape[0]+1)
    plan_to_reward_indirect = np.squeeze(indirect_rewards[:, 0]) * np.power(GAMMA, REWARDS_FUTURE_GAMMAS.shape[0]+1)
    if REWARD_ADD_DIRECT_REWARD:
        plan_to_reward = np.sum(plan_to_rewards_direct, axis=1) + plan_to_reward_indirect
    else:
        plan_to_reward = plan_to_reward_indirect
    #plan_to_reward = np.sum(plan_to_rewards_direct, axis=1)
    #plan_to_reward = np.squeeze(plan_to_reward_indirect)
    return plan_to_rewards_direct, plan_to_reward_indirect, plan_to_reward

"""
def calculate_future_rewards(plan_to_reward_bins, indirect_rewards):
    plan_to_rewards = np.sum(plan_to_reward_bins * BIN_AVERAGES, axis=2)
    plan_to_rewards_direct = plan_to_rewards * REWARDS_FUTURE_GAMMAS
    plan_to_reward_indirect = indirect_rewards * np.power(GAMMA, REWARDS_FUTURE_GAMMAS.shape[0]+1)
    plan_to_reward = np.sum(plan_to_rewards_direct, axis=1) + np.squeeze(plan_to_reward_indirect)
    #plan_to_reward = np.squeeze(plan_to_reward_indirect)
    return plan_to_rewards_direct, plan_to_reward_indirect, plan_to_reward
"""

def p_explore_to_actions(p_explore, action_up_down, action_left_right, last_state, constant_p_explore):
    """Change actions according to epsilon-greedy policy."""
    # if a constant p_explore value was chosen via --p_explore flag,
    # then use that value
    if constant_p_explore is not None:
        p_explore = constant_p_explore

    # in p_explore of all cases randomize up/down action
    if random.random() < p_explore:
        if last_state is not None and random.random() < P_EXPLORE_CHANCE_REDO:
            # in p percent of all cases, simply repeat the last action
            action_up_down = last_state.action_up_down
        else:
            # in (1-p) percent of all cases, chose a random action
            action_up_down = random.choice(actionslib.ACTIONS_UP_DOWN)

    # same for left/right action
    if random.random() < p_explore:
        if last_state is not None and random.random() < P_EXPLORE_CHANCE_REDO:
            action_left_right = last_state.action_left_right
        else:
            action_left_right = random.choice(actionslib.ACTIONS_LEFT_RIGHT)

    return action_up_down, action_left_right

if __name__ == "__main__":
    main()
