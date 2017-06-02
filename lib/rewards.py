from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
import numpy as np

def calculate_reward(state, next_state):
    if next_state.speed is not None:
        next_speed = next_state.speed
    elif state.speed is not None:
        next_speed = state.speed
    else:
        next_speed = 0
    return calculate_reward_raw(next_speed, next_state.is_reverse, next_state.is_offence_shown, next_state.is_damage_shown)

def calculate_reward_raw(next_speed, next_is_reverse, next_is_offence_shown, next_is_damage_shown):
    assert next_speed is not None
    #print(next_speed, next_is_reverse, next_is_offence_shown, next_is_damage_shown)
    #if next_is_damage_shown:
    #    return Config.MIN_REWARD
    #else:
    #speed_reward = Config.MAX_SPEED - (Config.MAX_SPEED / (1 + 0.1*np.clip(next_speed, 0, Config.MAX_SPEED))) - 20
    #speed_reward = 100 * (speed_reward / Config.MAX_SPEED)
    #lowest = (-20/Config.MAX_SPEED)*100

    speed_reward = next_speed
    lowest = 0

    speed_reward = np.clip(speed_reward, lowest, 100)
    if next_is_reverse:
        if speed_reward > 0:
            speed_reward *= 0.25

    #offence_reward = -50 if next_is_offence_shown else 0
    offence_reward = -10 if next_is_offence_shown else 0
    damage_reward = -50 if next_is_damage_shown else 0

    reward = speed_reward + offence_reward + damage_reward
    return np.clip(reward, Config.MIN_REWARD, Config.MAX_REWARD)

"""
class GridToRewardConverter(object):
    def __init__(self, gamma=Config.GAMMA):
        bins = Config.MODEL_NB_REWARD_BINS
        blocks = Config.MODEL_NB_FUTURE_BLOCKS
        block_sizes = Config.MODEL_FUTURE_BLOCK_SIZE

        # weight all bins the same way (might also give more weight to negatives)
        self.base_weighting = np.ones((blocks, bins), dtype=np.float32)
        #print("self.base_weighting", self.base_weighting)

        # average value of each bin
        bin_averages = []
        bin_start = Config.MAX_REWARD
        bin_size = (Config.MAX_REWARD - Config.MIN_REWARD) / Config.MODEL_NB_REWARD_BINS
        for i in range(Config.MODEL_NB_REWARD_BINS):
            bin_end = bin_start - bin_size
            bin_avg = (bin_end + bin_start) / 2
            bin_averages.append(bin_avg)
            bin_start = bin_end
        self.bin_averages = np.tile(np.array(bin_averages)[np.newaxis, :], (blocks, 1))
        #print("self.bin_averages", self.bin_averages)

        # correct for block sizes
        self.block_size_correct = np.tile(np.array(block_sizes, dtype=np.float32)[:, np.newaxis], (1, bins))
        #print("self.block_size_correct", self.block_size_correct)

        # gamma influence (weight future rewards less)
        # for each block of N states, calculate the gamma value of each
        # state and take their average
        gammas = []
        gamma_current = 1
        for bsz in block_sizes:
            gammas_block = []
            for i in range(bsz):
                gammas_block.append(gamma_current)
                gamma_current = gamma_current * gamma
            gammas.append(np.average(gammas_block))

            #gamma_end = gamma_start * (gamma ** bsz)
            #gamma_avg = (gamma_start + gamma_end) / 2
            #gammas.append(gamma_avg)
            #gamma_start = gamma_end * gamma
        self.gammas = np.tile(np.array(gammas, dtype=np.float32)[:, np.newaxis], (1, bins))
        #print("self.gammas", self.gammas)

        self.weighting = self.base_weighting * self.bin_averages * self.block_size_correct * self.gammas
        #print("self.weighting", self.weighting)

    def __call__(self, output_grid):
        return np.sum(output_grid * self.weighting)
"""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    speeds = np.arange(150)
    ys = []
    for speed in speeds:
        ys.append(calculate_reward_raw(speed, False, False, False))
    plt.plot(speeds, ys)
    plt.show()
