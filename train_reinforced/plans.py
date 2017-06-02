from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import models as models_reinforced
from lib import actions as actionslib

try:
    xrange
except NameError:
    xrange = range

def make_plans_unique(plans):
    added = set()
    result = []
    for plan in plans:
        plan_str = " ".join(["%s%s" % (a, b) for (a, b) in plan])
        if plan_str not in added:
            result.append(plan)
            added.add(plan_str)
    return result

def create_plans(nb_future_states):
    PLANS = []
    # plans that contain only a specific multiaction
    for multiaction in actionslib.ALL_MULTIACTIONS:
        plan = []
        for i in xrange(nb_future_states):
            plan.append(multiaction)
        PLANS.append(plan)
    # plans that contain multiaction A followed by B at every Nth step
    # e.g [U U UR U U UR U U UR] (U=up, UR=up+right)
    for n in [1, 2, 3]:
        for ma1 in actionslib.ALL_MULTIACTIONS:
            for ma2 in actionslib.ALL_MULTIACTIONS:
                if ma1 != ma2:
                    plan = []
                    for i in xrange(nb_future_states):
                        if i % n == 0:
                            plan.append(ma2)
                        else:
                            plan.append(ma1)
                    PLANS.append(plan)
    # plans that start with multiaction A followed by B after P percent of timesteps
    # e.g. [U U U U U UR UR UR UR UR] for P=50percent
    for p in [0.25, 0.5, 0.75]:
        for ma1 in actionslib.ALL_MULTIACTIONS:
            for ma2 in actionslib.ALL_MULTIACTIONS:
                if ma1 != ma2:
                    plan = []
                    for i in xrange(nb_future_states):
                        if i >= int(nb_future_states*p):
                            plan.append(ma1)
                        else:
                            plan.append(ma2)
                    PLANS.append(plan)

    print("Generated %d possible plans of multiactions." % (len(PLANS),))
    PLANS = make_plans_unique(PLANS)
    print("Reduced plans to %d after unique." % (len(PLANS),))

    PLANS_VECS = models_reinforced.SuccessorPredictor.multiactions_to_vecs(PLANS).transpose((1, 0, 2)) # BT9 -> TB9

    return PLANS, PLANS_VECS
