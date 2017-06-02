from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib import windowhandling
from lib import ets2window
from lib import ets2game
from lib import screenshot
from lib import speed as speedlib
from lib import replay_memory
from lib import states as stateslib
from lib import actions as actionslib
from lib import util
from lib import pykeylogger
from lib import steering_wheel as swlib
from lib import rewards as rewardslib
from config import Config
import time
from datetime import datetime
import cv2
import imgaug as ia
import numpy as np
import random
import argparse
np.random.seed(42)
random.seed(42)

def main():
    parser = argparse.ArgumentParser(description="Train trucker via reinforcement learning.")
    parser.add_argument("--val", default=False, action="store_true", help="Whether to add experiences to the validation memory.", required=False)
    args = parser.parse_args()

    cv2.namedWindow("overview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("overview", Config.MODEL_WIDTH + 256, Config.MODEL_HEIGHT + 128)
    cv2.moveWindow("overview", 55, 1080-720)
    cv2.waitKey(100)

    class OnRouteAdvisorVisible(object):
        def __init__(self):
            self.memory = replay_memory.ReplayMemory.create_instance_supervised(val=args.val)
            self.is_collecting = False
            self.is_collecting_keylisten = True
            self.last_state = None
            self.sw_tracker = swlib.SteeringWheelTrackerCNN()
            self.sw_tracker_classical = swlib.SteeringWheelTracker()

        def __call__(self, game, scr):
            speed_image = game.win.get_speed_image(scr)
            som = speedlib.SpeedOMeter(speed_image)

            keys_modifiers, keys = pykeylogger.fetch_keydowns()
            ctrl_pressed = (keys_modifiers["left ctrl"] == True or keys_modifiers["right ctrl"] == True)
            if ctrl_pressed:
                if self.is_collecting_keylisten:
                    self.is_collecting = not self.is_collecting
                self.is_collecting_keylisten = False
            else:
                self.is_collecting_keylisten = True
            #print("keys_any_changes", keys_any_changes)
            #print("keys:", keys)
            #print("keys_modifiers:", keys_modifiers)

            reward = None
            action_up_down = actionslib.keys_to_action_up_down(keys)
            action_left_right = actionslib.keys_to_action_left_right(keys)
            p_explore = 0

            from_datetime = datetime.utcnow()
            screenshot_rs = ia.imresize_single_image(scr, (Config.MODEL_HEIGHT, Config.MODEL_WIDTH), interpolation="cubic")
            screenshot_rs_jpg = util.compress_to_jpg(screenshot_rs)
            speed = som.predict_speed()
            is_reverse = game.win.is_reverse(scr)
            is_offence_shown = game.win.is_offence_shown(scr)
            is_damage_shown = game.win.is_damage_shown(scr)

            scr_de = util.decompress_img(screenshot_rs_jpg)
            sw_classical, (sw_raw_one_classical, sw_raw_two_classical) = self.sw_tracker_classical.estimate_angle(scr_de)
            sw, sw_raw = self.sw_tracker.estimate_angle(scr_de)

            current_state = stateslib.State(
                from_datetime=from_datetime,
                screenshot_rs_jpg=screenshot_rs_jpg,
                speed=speed,
                is_reverse=is_reverse,
                is_offence_shown=is_offence_shown,
                is_damage_shown=is_damage_shown,
                reward=None,
                action_left_right=action_left_right,
                action_up_down=action_up_down,
                p_explore=p_explore,
                steering_wheel_classical=sw_classical,
                steering_wheel_raw_one_classical=sw_raw_one_classical,
                steering_wheel_raw_two_classical=sw_raw_two_classical,
                steering_wheel_cnn=sw,
                steering_wheel_raw_cnn=sw_raw,
                allow_cache=False
            )

            if self.last_state is not None:
                self.last_state.reward = rewardslib.calculate_reward(self.last_state, current_state)
                if self.is_collecting:
                    self.memory.add_state(self.last_state, commit=False)

            do_commit = (game.tick_ingame_route_advisor_idx % 20 == 0)
            if do_commit:
                self.memory.commit()

            img = generate_overview_image(
                screenshot_rs,
                som, self.memory, do_commit,
                action_up_down, action_left_right,
                self.is_collecting
            )
            cv2.imshow("overview", img[:,:,::-1])
            cv2.waitKey(1)

            #print("[Main] Speed:", som.predict_speed(), som.predict_speed_raw())
            self.last_state = current_state

    game = ets2game.ETS2Game()
    game.on_route_advisor_visible = OnRouteAdvisorVisible()
    game.min_interval = 100 / 1000
    game.run()

def generate_overview_image(screenshot_rs, som, memory, do_commit, action_up_down, action_left_right, is_collecting):
    current_image = screenshot_rs
    speed_image = som.get_postprocessed_image_rgb()

    """
    time_start = time.time()
    screenshot_rs_small = ia.imresize_single_image(screenshot_rs, (Config.MODEL_HEIGHT//2, Config.MODEL_WIDTH//2), interpolation="cubic")
    screenshot_rs_small = cv2.cvtColor(screenshot_rs_small, cv2.COLOR_RGB2GRAY)
    screenshot_rs_small = np.tile(screenshot_rs_small[:, :, np.newaxis], (1, 1, 3))
    print("RS %.4f" % (time.time() - time_start,))

    from lib import util
    screenshot_rs_small_jpg = util.compress_to_jpg(screenshot_rs_small)
    time_start = time.time()
    screenshot_rs_small = util.decompress_img(screenshot_rs_small_jpg)
    print("JPG %.4f" % (time.time() - time_start,))
    """

    h, w = current_image.shape[0:2]
    #screenshot_rs = ia.imresize_single_image(screenshot_rs, ())
    h_small, w_small = screenshot_rs.shape[0:2]
    h_speed, w_speed = speed_image.shape[0:2]

    current_image = np.pad(current_image, ((0, h_small+h_speed), (0, 256), (0, 0)), mode="constant")
    current_image[h:h+h_small, 0:w_small, :] = screenshot_rs
    current_image[h+h_small:h+h_small+h_speed, 0:w_speed, :] = speed_image

    speed = som.predict_speed()

    texts = [
        "memory size: %06d" % (memory.size,),
        "commit: yes" if do_commit else "commit: no",
        "action u/d: %s" % (action_up_down,),
        "action l/r: %s" % (action_left_right,),
        "speed: %03d" % (speed,) if speed is not None else "speed: NONE",
        "is tracking (press CTRL to toggle)" if is_collecting else "NOT tracking (press CTRL to toggle)"
    ]
    texts = "\n".join(texts)

    for i, text in enumerate([texts]):
        current_image = util.draw_text(current_image, x=w+10, y=5+(i*15), size=10, text=text)

    return current_image

if __name__ == "__main__":
    main()
