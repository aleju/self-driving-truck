from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import sqlite3
import states
from config import Config
import random
import numpy as np

class ReplayMemory(object):
    #instances = dict()

    def __init__(self, filepath, max_size, max_size_tolerance):
        self.filepath = filepath
        self.conn = sqlite3.connect(filepath, detect_types=sqlite3.PARSE_DECLTYPES)
        self.max_size = max_size
        self.max_size_tolerance = max_size_tolerance

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_datetime TIMESTAMP NOT NULL,
                screenshot_rs_jpg BLOB NOT NULL,
                speed INTEGER,
                is_reverse INTEGER(1),
                is_offence_shown INTEGER(1),
                is_damage_shown INTEGER(1),
                reward REAL NOT NULL,
                action_left_right VARCHAR(3),
                action_up_down VARCHAR(3),
                p_explore REAL NOT NULL,
                steering_wheel REAL,
                steering_wheel_raw_one REAL,
                steering_wheel_raw_two REAL,
                steering_wheel_cnn REAL,
                steering_wheel_raw_cnn REAL
            )
        """)

        self.size = 0
        self.nb_states_added = 0
        self.id_min = None
        self.id_max = None
        self.update_caches()

    """
    @staticmethod
    def get_instance(name, filepath, max_size, max_size_tolerance):
        if not name in ReplayMemory.instances:
            ReplayMemory.instances[name] = ReplayMemory(filepath=filepath, max_size=max_size, max_size_tolerance=max_size_tolerance)
        return ReplayMemory.instances[name]

    @staticmethod
    def get_instance_supervised():
        return ReplayMemory.get_instance(
            name="supervised",
            filepath=Config.REPLAY_MEMORY_SUPERVISED_FILEPATH,
            max_size=Config.REPLAY_MEMORY_SUPERVISED_MAX_SIZE,
            max_size_tolerance=Config.REPLAY_MEMORY_SUPERVISED_MAX_SIZE_TOLERANCE
        )

    @staticmethod
    def get_instance_reinforced():
        return ReplayMemory.get_instance(
            name="reinforced",
            filepath=Config.REPLAY_MEMORY_REINFORCED_FILEPATH,
            max_size=Config.REPLAY_MEMORY_REINFORCED_MAX_SIZE,
            max_size_tolerance=Config.REPLAY_MEMORY_REINFORCED_MAX_SIZE_TOLERANCE
        )
    """

    @staticmethod
    def create_instance_supervised(val=False):
        return ReplayMemory.create_instance_by_config("supervised%s" % ("-val" if val else "-train",))

    @staticmethod
    def create_instance_reinforced(val=False):
        return ReplayMemory.create_instance_by_config("reinforced%s" % ("-val" if val else "-train",))

    @staticmethod
    def create_instance_by_config(cfg_name):
        return ReplayMemory(
            filepath=Config.REPLAY_MEMORY_CFGS[cfg_name]["filepath"],
            max_size=Config.REPLAY_MEMORY_CFGS[cfg_name]["max_size"],
            max_size_tolerance=Config.REPLAY_MEMORY_CFGS[cfg_name]["max_size_tolerance"]
        )

    def add_states(self, states, shrink=True):
        for state in states:
            self.add_state(state, commit=False, shrink=False)
        if shrink:
            self.shrink_to_max_size()
        self.commit()

    def add_state(self, state, commit=True, shrink=True):
        idx = self.id_max + 1 if self.id_max is not None else 1
        from_datetime = state.from_datetime
        #scr_rs = state.screenshot_rs
        scr_rs_jpg = sqlite3.Binary(state.screenshot_rs_jpg)
        speed = state.speed
        is_reverse = state.is_reverse
        is_offence_shown = state.is_offence_shown
        is_damage_shown = state.is_damage_shown
        reward = state.reward
        action_left_right = state.action_left_right
        action_up_down = state.action_up_down
        p_explore = state.p_explore
        steering_wheel_classical = state.steering_wheel_classical
        steering_wheel_raw_one_classical = state.steering_wheel_raw_one_classical
        steering_wheel_raw_two_classical = state.steering_wheel_raw_two_classical
        steering_wheel_cnn = state.steering_wheel_cnn
        steering_wheel_raw_cnn = state.steering_wheel_raw_cnn

        stmt = """
        INSERT INTO states
               (id, from_datetime, screenshot_rs_jpg, speed, is_reverse, is_offence_shown, is_damage_shown, reward, action_left_right, action_up_down, p_explore, steering_wheel, steering_wheel_raw_one, steering_wheel_raw_two, steering_wheel_cnn, steering_wheel_raw_cnn)
        VALUES ( ?,             ?,                 ?,     ?,          ?,                ?,               ?,      ?,                 ?,              ?,         ?,              ?,                      ?,                      ?,                  ?,                      ?)
        """

        assert isinstance(action_left_right, str)
        assert isinstance(action_up_down, str)
        self.conn.execute(stmt, (
            idx, from_datetime, scr_rs_jpg,
            speed, is_reverse, is_offence_shown, is_damage_shown, reward,
            action_left_right, action_up_down, p_explore,
            steering_wheel_classical, steering_wheel_raw_one_classical, steering_wheel_raw_two_classical,
            steering_wheel_cnn, steering_wheel_raw_cnn
        ))
        self.id_max = idx
        self.size += 1
        self.nb_states_added += 1
        #print("[ReplayMemory] Added state (new size: %d)" % (self.size,))

        if commit:
            self.commit()
        if shrink:
            self.shrink_to_max_size()

    def update_state(self, idx, state, commit=True):
        new_idx = idx
        old_idx = state.idx
        from_datetime = state.from_datetime
        scr_rs_jpg = sqlite3.Binary(state.screenshot_rs_jpg)
        speed = state.speed
        is_reverse = state.is_reverse
        is_offence_shown = state.is_offence_shown
        is_damage_shown = state.is_damage_shown
        reward = state.reward
        action_left_right = state.action_left_right
        action_up_down = state.action_up_down
        p_explore = state.p_explore
        steering_wheel_classical = state.steering_wheel_classical
        steering_wheel_raw_one_classical = state.steering_wheel_raw_one_classical
        steering_wheel_raw_two_classical = state.steering_wheel_raw_two_classical
        steering_wheel_cnn = state.steering_wheel_cnn
        steering_wheel_raw_cnn = state.steering_wheel_raw_cnn

        stmt = """
        UPDATE states
        SET id=?,
            from_datetime=?,
            screenshot_rs_jpg=?,
            speed=?,
            is_reverse=?,
            is_offence_shown=?,
            is_damage_shown=?,
            reward=?,
            action_left_right=?,
            action_up_down=?,
            p_explore=?,
            steering_wheel=?,
            steering_wheel_raw_one=?,
            steering_wheel_raw_two=?,
            steering_wheel_cnn=?,
            steering_wheel_raw_cnn=?
        WHERE id=?
        """
        assert isinstance(action_left_right, str)
        assert isinstance(action_up_down, str)
        self.conn.execute(stmt, (
            new_idx, from_datetime, scr_rs_jpg,
            speed, is_reverse, is_offence_shown, is_damage_shown, reward,
            action_left_right, action_up_down, p_explore,
            steering_wheel_classical, steering_wheel_raw_one_classical, steering_wheel_raw_two_classical,
            steering_wheel_cnn, steering_wheel_raw_cnn,
            old_idx))

        if commit:
            self.commit()

    def get_state_by_id(self, idx):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT
                id, from_datetime, screenshot_rs_jpg, speed, is_reverse,
                is_offence_shown, is_damage_shown, reward,
                action_up_down, action_left_right, p_explore,
                steering_wheel, steering_wheel_raw_one, steering_wheel_raw_two,
                steering_wheel_cnn, steering_wheel_raw_cnn
            FROM states
            WHERE id=?
        """, (idx,))
        rows = cur.fetchall()
        result = []
        for row in rows:
            result.append(states.State.from_row(row))
        if len(result) == 0:
            return None
        else:
            assert len(result) == 1
            return result[0]

    def get_states_by_ids(self, ids):
        id_to_pos = dict([(idx, i) for i, idx in enumerate(ids)])

        cur = self.conn.cursor()
        cur.execute("""
            SELECT
                id, from_datetime, screenshot_rs_jpg, speed, is_reverse,
                is_offence_shown, is_damage_shown, reward,
                action_up_down, action_left_right, p_explore,
                steering_wheel, steering_wheel_raw_one, steering_wheel_raw_two,
                steering_wheel_cnn, steering_wheel_raw_cnn
            FROM states
            WHERE id IN (%s)
        """ % (", ".join([str(idx) for idx in ids]),))
        rows = cur.fetchall()


        result = [None] * len(ids)
        for row in rows:
            state = states.State.from_row(row)
            pos = id_to_pos[state.idx]
            result[pos] = state
        return result

    def get_random_states(self, count):
        assert self.size > 0
        id_min = self.id_min
        id_max = self.id_max
        ids = [str(v) for v in np.random.randint(id_min, id_max, size=(count,))]
        cur = self.conn.cursor()
        cur.execute("""
            SELECT
                id, from_datetime, screenshot_rs_jpg, speed, is_reverse,
                is_offence_shown, is_damage_shown, reward,
                action_up_down, action_left_right, p_explore,
                steering_wheel, steering_wheel_raw_one, steering_wheel_raw_two,
                steering_wheel_cnn, steering_wheel_raw_cnn
            FROM states
            WHERE id IN (%s)
        """ % (", ".join(ids)))
        rows = cur.fetchall()
        result = []
        for row in rows:
            result.append(states.State.from_row(row))
        while len(result) < count:
            result.append(random.choice(result))
        return result

    def get_states_range(self, pos_start, length):
        assert self.id_min is not None
        assert pos_start is not None
        id_start = self.id_min + pos_start
        id_end = id_start + length
        #print("[get_states_range] pos_start", pos_start, "length", length, "id_start", id_start, "id_end", id_end)
        cur = self.conn.cursor()
        cur.execute("""
            SELECT
                id, from_datetime, screenshot_rs_jpg, speed, is_reverse,
                is_offence_shown, is_damage_shown, reward,
                action_up_down, action_left_right, p_explore,
                steering_wheel, steering_wheel_raw_one, steering_wheel_raw_two,
                steering_wheel_cnn, steering_wheel_raw_cnn
            FROM states
            WHERE id >= ? and id < ?
            ORDER BY id ASC
        """, (id_start, id_end))
        rows = cur.fetchall()
        result = []
        for row in rows:
            result.append(states.State.from_row(row))
        #print("[get_states_range]", self.filepath, [state.idx for state in result])
        assert len(result) == length, "Wrong number of states found: pos_start=%d length=%d id_start=%d id_end=%d id_min=%d id_max=%d size=%d" % (pos_start, length, id_start, id_end, self.id_min, self.id_max, self.size)
        return result

    def get_random_state_chain(self, length, from_pos_range=None):
        #print("[get_random_state_chain] from_pos_range", from_pos_range)
        if length > self.size:
            raise Exception("Requested state chain length is larger than size of replay memory. Gather more experiences/states.")

        from_pos_range = list(from_pos_range) if from_pos_range is not None else [0, self.size]
        if from_pos_range[0] is None:
            from_pos_range[0] = 0
        #else:
        #    from_pos_range[0] = from_pos_range[0] if from_pos_range[0] >= self.id_min else self.id_min
        if from_pos_range[1] is None:
            from_pos_range[1] = self.size
        else:
            from_pos_range[1] = from_pos_range[1] if from_pos_range[1] <= self.size else self.size
        if length > (from_pos_range[1] - from_pos_range[0]):
            raise Exception("Requested state chain length is larger than allowed id range.")

        start_pos_min = from_pos_range[0]
        start_pos_max = from_pos_range[1] - length
        start_pos = random.randint(start_pos_min, start_pos_max)
        #print("get_random_state_chain", self.filepath, start_pos, length, start_pos_min, start_pos_max)
        return self.get_states_range(pos_start=start_pos, length=length)

    def get_random_state_chain_timediff(self, length, max_timediff_ms=500, depth=50):
        states = self.get_random_state_chain(length)
        maxdiff = 0
        last_time = states[0].from_datetime
        for state in states[1:]:
            if last_time < state.from_datetime:
                timediff = state.from_datetime - last_time
                timediff = timediff.total_seconds() * 1000
            else:
                print("[WARNING] load_random_state_chain: state from datetime %s is after state %s, expected reversed order" % (last_time, state.from_datetime))
                timediff = max_timediff_ms + 1
            maxdiff = max(timediff, maxdiff)
            last_time = state.from_datetime
        #print("maxdiff:", maxdiff)
        if maxdiff > max_timediff_ms:
            if depth == 0:
                print("[WARNING] reached max depth in load_random_state_chain", from_pos_range)
                return states
            else:
                return self.get_random_state_chain_timediff(length, max_timediff_ms=max_timediff_ms, depth=depth-1)
        else:
            return states

    def update_caches(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) as c FROM states")
        row = cur.fetchone()
        count = row[0]

        if count > 0:
            cur = self.conn.cursor()
            cur.execute("SELECT MIN(id) as id_min, MAX(id) as id_max FROM states")
            row = cur.fetchone()
            id_min = row[0]
            id_max = row[1]
        else:
            id_min = None
            id_max = None

        self.size = count
        self.id_min = id_min
        self.id_max = id_max

    def is_above_tolerance(self):
        return self.size > (self.max_size + self.max_size_tolerance)

    def shrink_to_max_size(self, force=False):
        is_over_size = (self.size > self.max_size)
        #is_over_tolerance = (self.size > self.max_size + self.max_size_tolerance)
        if self.is_above_tolerance() or (is_over_size and force):
            print("[ReplayMemory] Shrink to size (from %d to %d)" % (self.size, self.max_size))
            diff = self.size - self.max_size
            del_start = self.id_min
            del_end = self.id_min + diff
            cur = self.conn.cursor()
            cur.execute("DELETE FROM states WHERE id >= ? AND id < ?", (del_start, del_end))
            self.commit()
            #self.size -= diff
            #self.id_min += diff
            self.update_caches()
            print("[ReplayMemory] New size is %d" % (self.size,))

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    def connect(self):
        self.conn = sqlite3.connect(self.filepath, detect_types=sqlite3.PARSE_DECLTYPES)
