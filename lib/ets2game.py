from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#import thread
import time
import ets2window
import windowhandling
import actions as actionslib
from config import Config
import random

class ETS2Game(object):
    def __init__(self, win=None):
        if win is None:
            while True:
                win_id = ets2window.find_ets2_window_id()
                if win_id is None:
                    print("[ETS2Game] ETS2 window not found. Expected window with name 'Euro Truck Simulator 2'. (Not started yet?)")
                    time.sleep(1)
                else:
                    break

            win = ets2window.ETS2Window(win_id)

            print("---------------------")
            print("ETS2 window found.")
            print("Window ID: %d" % (win.win_id,))
            print("Coordinates:", win.get_coordinates())
            print("Activated:", win.is_activated())
            print("--------------------")

        self.win = win
        x1, y1, x2, y2 = win.coordinates
        h = y2 - y1
        w = x2 - x1
        assert w == 1280 and h == 720, "Detected %dx%d resolution instead of expected 1280x720" % (h, w)

        self.on_screenshot = None
        self.on_route_advisor_visible = None
        self.min_interval = 100 / 1000 # 100ms steps
        self.timeout_after_screenshot = 0
        self.tick_idx = 0
        self.tick_ingame_idx = 0
        self.tick_ingame_route_advisor_idx = 0

        self.scheduled_actions_of_interval = []
        self.past_actions_of_interval = []

    def run(self):
        win_is_activated = False

        while True:
            time_start = time.time()

            time_start_wa = time.time()
            if self.tick_idx % 20 == 0 or win_is_activated == False:
                win_is_activated = self.win.is_activated()
            time_req_wa = time.time() - time_start_wa

            if not win_is_activated:
                print("[ETS2Game.run] ETS2 window not activated (ETS2 win id: %d, active win id: %d)" % (self.win.win_id, windowhandling.get_active_window_id()))
                time.sleep(2)

                time_req_scr = 0
                time_req_onscr = 0
                time_req_onra = 0
                time_req_at = 0
            else:
                time_start_scr = time.time()
                scr = self.win.get_image()
                self.last_scr_time = time_start_scr
                time_req_scr = time.time() - time_start_scr

                time_start_onscr = time.time()
                if self.on_screenshot is not None:
                    self.on_screenshot(self, scr)
                time_req_scr = time.time() - time_start_onscr

                time_start_onra = time.time()
                if not self.win.is_route_advisor_visible(scr):
                    print("[ETS2Game.run] Route Advisor not visible. When ingame, press F3 to show it.")
                else:
                    if self.on_route_advisor_visible is not None:
                        self.on_route_advisor_visible(self, scr)
                    self.tick_ingame_route_advisor_idx += 1
                time_req_onra = time.time() - time_start_onra

                self.tick_ingame_idx += 1

                time_start_at = time.time()
                self._actions_tick()
                time_req_at = time.time() - time_start_at

            time_done = time.time() - time_start

            while time.time() - time_start < self.min_interval:
                time.sleep(1/1000)

            print("[ETS2Game.run] Step done in %.4fs, finished in %.4fs. (wa %dms, scr %dms, onscr %dms, onra %dms, at %dms)" % (
                    time_done,
                    time.time() - time_start,
                    int(time_req_wa*1000),
                    int(time_req_scr*1000),
                    int(time_req_scr*1000),
                    int(time_req_onra*1000),
                    int(time_req_at*1000)
                )
            )
            self.tick_idx += 1

    def reset_actions(self):
        self.set_actions_of_interval([actionslib.ACTION_UP_DOWN_NONE, actionslib.ACTION_LEFT_RIGHT_NONE])
        self._actions_tick()

    def set_actions_of_interval(self, actions):
        self.scheduled_actions_of_interval = actions

    """
    def _end_actions(self):
        keys = []
        for action in self.past_actions_of_interval:
            key = actionslib.action_to_key(action)
            if key is not None:
                keys.append(key)
        self.win.keyup(keyups)

    def _apply_actions(self):
        keys = []
        for action in self.scheduled_actions_of_interval:
            key = actionslib.action_to_key(action)
            if key is not None:
                keys.append(key)
        self.win.keydown(keys)
        self.past_actions_of_interval = self.scheduled_actions_of_interval
        self.scheduled_actions_of_interval = []
    """

    def _actions_tick(self):
        #print("[_actions_tick] ", self.past_actions_of_interval, "||", self.scheduled_actions_of_interval)
        keysup = set()
        for action in self.past_actions_of_interval:
            key = actionslib.action_to_key(action)
            if key is not None:
                keysup.add(key)

        keysdown = set()
        for action in self.scheduled_actions_of_interval:
            key = actionslib.action_to_key(action)
            if key is not None:
                keysdown.add(key)

        self.win.keys(up=keysup, down=keysdown)
        self.past_actions_of_interval = self.scheduled_actions_of_interval
        self.scheduled_actions_of_interval = []

    def pause(self):
        self.scheduled_actions_of_interval = []
        self.win.keys(press=set([Config.KEY_PAUSE]))

    def unpause(self):
        self.scheduled_actions_of_interval = []
        self.win.keys(press=set([Config.KEY_PAUSE]))

    def load_random_save_game(self):
        self.past_actions_of_interval = []
        self.scheduled_actions_of_interval = []
        self.win.keys(press=set([Config.KEY_QUICKLOAD]))
        time.sleep(5)
        nth_save = random.randint(0, Config.RELOAD_MAX_SAVEGAME_NUMBER-1) # randint is a<=x<=b
        i = 0
        while i < nth_save:
            print("Next savegame (switching to %d)" % (nth_save,))
            self.win.keys(press=set(["Down"]))
            time.sleep(1)
            i += 1
        time.sleep(3.0)
        for i in range(60):
            self.win.keys(press=set(["Return"]))
            time.sleep(2.0)
            scr = self.win.get_image()
            if self.win.is_route_advisor_visible(scr):
                print("Route advisor visible, save game apparently loaded.")
                break
