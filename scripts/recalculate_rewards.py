from __future__ import division, print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib import replay_memory
from lib import rewards as rewardslib

import collections

if sys.version_info[0] == 3:
    raw_input = input

try:
    xrange
except NameError:
    xrange = range

def main():
    key = raw_input("This will recalculate the rewards of all states in the database. Press 'y' to continue.")
    assert key == "y"

    memories = [
        replay_memory.ReplayMemory.create_instance_reinforced(val=False),
        #replay_memory.ReplayMemory.create_instance_reinforced(val=True)
    ]
    for memory in memories:
        state_curr = memory.get_state_by_id(memory.id_min)
        last_speeds = collections.deque(maxlen=20)
        for idx in xrange(memory.id_min+1, memory.id_max+1):
            state_next = memory.get_state_by_id(idx)
            assert state_curr is not None
            assert state_next is not None
            reward = rewardslib.calculate_reward(state_curr, state_next)
            print("Changing reward of state %d from %03.2f to %03.2f" % (idx-1, state_curr.reward, reward))
            state_curr.reward = reward
            memory.update_state(state_curr.idx, state_curr, commit=False)

            if state_curr.speed is None:
                print("[NOTE] speed is none, last speeds:", last_speeds)
            last_speeds.append(state_curr.speed)

            state_prev = state_curr

            if idx % 100 == 0:
                memory.commit()

if __name__ == "__main__":
    main()
