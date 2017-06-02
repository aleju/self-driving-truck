from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import actions
import util
from config import Config
import imgaug as ia

class State(object):
    def __init__(self, from_datetime, screenshot_rs_jpg, speed, is_reverse, \
        is_offence_shown, is_damage_shown, reward, \
        action_left_right, action_up_down, p_explore, \
        steering_wheel_classical, steering_wheel_raw_one_classical, steering_wheel_raw_two_classical, \
        steering_wheel_cnn, steering_wheel_raw_cnn,
        allow_cache=False, idx=None):
        assert action_up_down is None or action_up_down in actions.ACTIONS_UP_DOWN
        assert action_left_right is None or action_left_right in actions.ACTIONS_LEFT_RIGHT
        self.from_datetime = from_datetime
        self.screenshot_rs_jpg = screenshot_rs_jpg
        self.speed = speed
        self.is_reverse = bool(is_reverse) # convert sqlite int to bool if necessary
        self.is_offence_shown = bool(is_offence_shown)
        self.is_damage_shown = bool(is_damage_shown)
        self.reward = reward
        self.action_left_right = action_left_right
        self.action_up_down = action_up_down
        self.p_explore = p_explore
        self.steering_wheel_classical = steering_wheel_classical
        self.steering_wheel_raw_one_classical = steering_wheel_raw_one_classical
        self.steering_wheel_raw_two_classical = steering_wheel_raw_two_classical
        self.steering_wheel_cnn = steering_wheel_cnn
        self.steering_wheel_raw_cnn = steering_wheel_raw_cnn

        self.allow_cache = allow_cache
        self._screenshot_rs = None
        self._screenshot_rs_small = None

        self.idx = idx

    @property
    def screenshot_rs(self):
        if self.allow_cache:
            if self._screenshot_rs is None:
                self._screenshot_rs = util.decompress_img(self.screenshot_rs_jpg)
            return self._screenshot_rs
        else:
            return util.decompress_img(self.screenshot_rs_jpg)

    """
    @property
    def screenshot_rs_small(self):
        if self.allow_cache:
            if self._screenshot_rs_small is None:
                scr_large = self.screenshot_rs
                h, w = Config.MODEL_HEIGHT_SMALL, Config.MODEL_WIDTH_SMALL
                self._screenshot_rs_small = ia.imresize_single_image(scr_large, (h, w), interpolation="cubic")
            return self._screenshot_rs_small
        else:
            scr_large = self.screenshot_rs
            h, w = Config.MODEL_HEIGHT_SMALL, Config.MODEL_WIDTH_SMALL
            return ia.imresize_single_image(scr_large, (h, w), interpolation="cubic")
    """

    @property
    def multiaction(self):
        return (self.action_up_down, self.action_left_right)

    @property
    def actions_multivec(self):
        return actions.actions_to_multivec(action_up_down=self.action_up_down, action_left_right=self.action_left_right)

    @staticmethod
    def from_row(row):
        # sqlite3 returns unicode strings, which causes problems in python2
        # when they are compared to non-unicode strings
        # in python3, all strings should be by default unicode so it shouldnt
        # cause any problems there
        if sys.version_info[0] >= 3:
            return State(
                idx=row[0],
                from_datetime=row[1],
                screenshot_rs_jpg=str(row[2]), # sqlite3 returns buffer object instead of string
                speed=row[3],
                is_reverse=row[4],
                is_offence_shown=row[5],
                is_damage_shown=row[6],
                reward=row[7],
                action_up_down=row[8],
                action_left_right=row[9],
                p_explore=row[10],
                steering_wheel_classical=row[11],
                steering_wheel_raw_one_classical=row[12],
                steering_wheel_raw_two_classical=row[13],
                steering_wheel_cnn=row[14],
                steering_wheel_raw_cnn=row[15]
            )
        else:
            return State(
                idx=row[0],
                from_datetime=row[1],
                screenshot_rs_jpg=str(row[2]), # sqlite3 returns buffer object instead of string
                speed=row[3],
                is_reverse=row[4],
                is_offence_shown=row[5],
                is_damage_shown=row[6],
                reward=row[7],
                action_up_down=str(row[8]), # make sure that these are not unicode
                action_left_right=str(row[9]), # make sure that these are not unicode
                p_explore=row[10],
                steering_wheel_classical=row[11],
                steering_wheel_raw_one_classical=row[12],
                steering_wheel_raw_two_classical=row[13],
                steering_wheel_cnn=row[14],
                steering_wheel_raw_cnn=row[15]
            )

    def to_string(self):
        return "State(dt=%s, img_shape=%s, speed=%s, off=%d, dmg=%d, reward=%.2f, alr=%s, aud=%s, pe=%.2f, w=%.2f, wr1=%.2f, wr2=%.2f)" % (self.from_datetime, self.screenshot_rs.shape, str(self.speed), int(self.is_offence_shown), int(self.is_damage_shown), self.reward, self.action_left_right, self.action_up_down, self.p_explore, self.steering_wheel, self.steering_wheel_raw_one, self.steering_wheel_raw_two)
