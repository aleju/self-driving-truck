"""Classes to handle plotting during the training."""
from __future__ import print_function, division
import math
import cPickle as pickle
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import time

GROWTH_BY = 500

class History(object):
    def __init__(self):
        self.line_groups = OrderedDict()

    @staticmethod
    def from_string(s):
        return pickle.loads(s)

    def to_string(self):
        return pickle.dumps(self, protocol=-1)

    @staticmethod
    def load_from_filepath(fp):
        #return json.loads(open(, "r").read())
        with open(fp, "r") as f:
            history = pickle.load(f)
        return history

    def save_to_filepath(self, fp):
        with open(fp, "w") as f:
            pickle.dump(self, f, protocol=-1)

    def add_group(self, group_name, line_names, increasing=True):
        self.line_groups[group_name] = LineGroup(group_name, line_names, increasing=increasing)

    def add_value(self, group_name, line_name, x, y, average=False):
        self.line_groups[group_name].lines[line_name].append(x, y, average=average)

    def get_group_names(self):
        return list(self.line_groups.iterkeys())

    def get_groups_increasing(self):
        return [group.increasing for group in self.line_groups.itervalues()]

    def get_max_x(self):
        return max([group.get_max_x() for group in self.line_groups.itervalues()])

    def get_recent_average(self, group_name, line_name, nb_points):
        ys = self.line_groups[group_name].lines[line_name].ys[-nb_points:]
        return np.average(ys)

class LineGroup(object):
    def __init__(self, group_name, line_names, increasing=True):
        self.group_name = group_name
        self.lines = OrderedDict([(name, Line()) for name in line_names])
        self.increasing = increasing
        self.xlim = (None, None)

    def get_line_names(self):
        return list(self.lines.iterkeys())

    def get_line_xs(self):
        #return [line.xs for line in self.lines.itervalues()]
        """
        for key, line in self.lines.items():
            if not hasattr(line, "last_index"):
                print(self.group_name, key, "no last index")
            else:
                print(self.group_name, key, "OK")
            print(type(line.xs), type(line.ys), type(line.counts), type(line.datetimes))
        """
        return [line.get_xs() for line in self.lines.itervalues()]

    def get_line_ys(self):
        #return [line.ys for line in self.lines.itervalues()]
        return [line.get_ys() for line in self.lines.itervalues()]

    def get_max_x(self):
        #return max([max(line.xs) if len(line.xs) > 0 else 0 for line in self.lines.itervalues()])
        return max([np.maximum(line.get_xs()) if line.last_index > -1 else 0 for line in self.lines.itervalues()])

"""
class Line(object):
    def __init__(self, xs=None, ys=None, counts=None, datetimes=None):
        self.xs = xs if xs is not None else []
        self.ys = ys if ys is not None else []
        self.counts = counts if counts is not None else []
        self.datetimes = datetimes if datetimes is not None else []
        self.last_index = -1

    def append(self, x, y, average=False):
        # legacy (for loading from pickle)
        #if not hasattr(self, "counts"):
        #    self.counts = [1] * len(self.xs)
        # ---

        if not average or len(self.xs) == 0 or self.xs[-1] != x:
            self.xs.append(x)
            self.ys.append(float(y)) # float to get rid of numpy
            self.counts.append(1)
            self.datetimes.append(time.time())
        else:
            count = self.counts[-1]
            self.ys[-1] = ((self.ys[-1] * count) + y) / (count+1)
            self.counts[-1] += 1
            self.datetimes[-1] = time.time()
"""

class Line(object):
    def __init__(self, xs=None, ys=None, counts=None, datetimes=None):
        zeros = np.tile(np.array([0], dtype=np.int32), GROWTH_BY)
        self.xs = xs if xs is not None else np.copy(zeros)
        self.ys = ys if ys is not None else zeros.astype(np.float32)
        self.counts = counts if counts is not None else zeros.astype(np.uint16)
        self.datetimes = datetimes if datetimes is not None else zeros.astype(np.uint64)
        self.last_index = -1

    # for legacy as functions, replace with properties
    def get_xs(self):
        # legacy
        if isinstance(self.xs, list):
            self._legacy_convert_from_list_to_np()

        return self.xs[0:self.last_index+1]

    def get_ys(self):
        return self.ys[0:self.last_index+1]

    def get_counts(self):
        return self.counts[0:self.last_index+1]

    def get_datetimes(self):
        return self.datetimes[0:self.last_index+1]

    def _legacy_convert_from_list_to_np(self):
        #print("is list!")
        print("[plotting] Converting from list to numpy...")
        self.last_index = len(self.xs) - 1
        self.xs = np.array(self.xs, dtype=np.int32)
        self.ys = np.array(self.ys, dtype=np.float32)
        self.counts = np.array(self.counts, dtype=np.uint16)
        self.datetimes = np.array([int(dt*1000) for dt in self.datetimes], dtype=np.uint64)

    def append(self, x, y, average=False):
        # legacy (for loading from pickle)
        #if not hasattr(self, "counts"):
        #    self.counts = [1] * len(self.xs)
        # ---

        #legacy
        if isinstance(self.xs, list):
            self._legacy_convert_from_list_to_np()

        if (self.last_index+1) == self.xs.shape[0]:
            #print("growing from %d by %d..." % (self.xs.shape[0], GROWTH_BY), self.xs.shape, self.ys.shape, self.counts.shape, self.datetimes.shape)
            zeros = np.tile(np.array([0], dtype=np.int32), GROWTH_BY)
            self.xs = np.append(self.xs, np.copy(zeros))
            self.ys = np.append(self.ys, zeros.astype(np.float32))
            self.counts = np.append(self.counts, zeros.astype(np.uint16))
            self.datetimes = np.append(self.datetimes, zeros.astype(np.uint64))
            #print("growing done", self.xs.shape, self.ys.shape, self.counts.shape, self.datetimes.shape)

        first_entry = (self.last_index == -1)
        if not average or first_entry or self.xs[self.last_index] != x:
            idx = self.last_index + 1
            self.xs[idx] = x
            self.ys[idx] = y
            self.counts[idx] = 1
            self.datetimes[idx] = int(time.time()*1000)
            self.last_index = idx
        else:
            idx = self.last_index
            count = self.counts[idx]
            self.ys[idx] = ((self.ys[idx] * count) + y) / (count+1)
            self.counts[idx] = count + 1
            self.datetimes[idx] = int(time.time()*1000)

        #print("added", x, y, average)
        #print(self.xs[self.last_index-10:self.last_index+10+1])
        #print(self.ys[self.last_index-10:self.last_index+10+1])
        #print(self.counts[self.last_index-10:self.last_index+10+1])
        #print(self.datetimes[self.last_index-10:self.last_index+10+1])

class LossPlotter(object):
    def __init__(self, titles, increasing, save_to_fp):
        assert len(titles) == len(increasing)
        n_plots = len(titles)
        self.titles = titles
        self.increasing = dict([(title, incr) for title, incr in zip(titles, increasing)])
        self.xlim = dict([(title, (None, None)) for title in titles])
        self.colors = ["red", "blue", "cyan", "magenta", "orange", "black"]

        self.nb_points_max = 500
        self.save_to_fp = save_to_fp
        self.start_batch_idx = 0
        self.autolimit_y = False
        self.autolimit_y_multiplier = 5

        #self.fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
        nrows = max(1, int(math.sqrt(n_plots)))
        ncols = int(math.ceil(n_plots / nrows))
        width = ncols * 10
        height = nrows * 10

        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))

        if nrows == 1 and ncols == 1:
            self.axes = [self.axes]
        else:
            self.axes = self.axes.flat

        title_to_ax = dict()
        for idx, (title, ax) in enumerate(zip(self.titles, self.axes)):
            title_to_ax[title] = ax
        self.title_to_ax = title_to_ax

        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.05)

    def plot(self, history):
        for plot_idx, title in enumerate(self.titles):
            ax = self.title_to_ax[title]
            group_name = title
            group_increasing = self.increasing[title]
            group = history.line_groups[title]
            line_names = group.get_line_names()
            #print("getting line x/y...", time.time())
            line_xs = group.get_line_xs()
            line_ys = group.get_line_ys()
            #print("getting line x/y FIN", time.time())

            """
            print("title", title)
            print("line_names", line_names)
            for i, xx in enumerate(line_xs):
                print("line_xs i: ", xx)
            for i, yy in enumerate(line_ys):
                print("line_ys i: ", yy)
            """
            if any([len(xx) > 0 for xx in line_xs]):
                xs_min = min([min(xx) for xx in line_xs if len(xx) > 0])
                xs_max = max([max(xx) for xx in line_xs if len(xx) > 0])
                xlim = self.xlim[title]
                xlim = [
                    max(xs_min, self.start_batch_idx) if xlim[0] is None else min(xlim[0], xs_max-1),
                    xs_max+1 if xlim[1] is None else xlim[1]
                ]
                if xlim[0] < 0:
                    xlim[0] = max(xs_max - abs(xlim[0]), 0)
                if xlim[1] < 0:
                    xlim[1] = max(xs_max - abs(xlim[1]), 1)
            else:
                # none of the lines has any value, so just use dummy values
                # to avoid min/max of empty sequence errors
                xlim = [
                    0 if self.xlim[title][0] is None else self.xlim[title][0],
                    1 if self.xlim[title][1] is None else self.xlim[title][1]
                ]

            self._plot_group(ax, group_name, group_increasing, line_names, line_xs, line_ys, xlim)
        self.fig.savefig(self.save_to_fp)

    # this seems to be slow sometimes
    def _line_to_xy(self, line_x, line_y, xlim, limit_y_min=None, limit_y_max=None):
        def _add_point(points_x, points_y, curr_sum, counter):
            points_x.append(batch_idx)
            y = curr_sum / counter
            if limit_y_min is not None and limit_y_max is not None:
                y = np.clip(y, limit_y_min, limit_y_max)
            elif limit_y_min is not None:
                y = max(y, limit_y_min)
            elif limit_y_max is not None:
                y = min(y, limit_y_max)
            points_y.append(y)

        nb_points = 0
        for i in range(len(line_x)):
            batch_idx = line_x[i]
            if xlim[0] <= batch_idx < xlim[1]:
                nb_points += 1

        point_every = max(1, int(nb_points / self.nb_points_max))
        points_x = []
        points_y = []
        curr_sum = 0
        counter = 0
        for i in range(len(line_x)):
            batch_idx = line_x[i]
            if xlim[0] <= batch_idx < xlim[1]:
                curr_sum += line_y[i]
                counter += 1
                if counter >= point_every:
                    _add_point(points_x, points_y, curr_sum, counter)
                    counter = 0
                    curr_sum = 0
        if counter > 0:
            _add_point(points_x, points_y, curr_sum, counter)

        return points_x, points_y

    def _plot_group(self, ax, group_name, group_increasing, line_names, line_xs, line_ys, xlim):
        ax.cla()
        ax.grid()

        if self.autolimit_y and any([len(line_xs) > 0 for line_xs in line_xs]):
            min_x = min([np.min(line_x) for line_x in line_xs])
            max_x = max([np.max(line_x) for line_x in line_xs])
            min_y = min([np.min(line_y) for line_y in line_ys])
            max_y = max([np.max(line_y) for line_y in line_ys])

            if group_increasing:
                if max_y > 0:
                    limit_y_max = None
                    limit_y_min = max_y / self.autolimit_y_multiplier
                    if min_y > limit_y_min:
                        limit_y_min = None
            else:
                if min_y > 0:
                    limit_y_max = min_y * self.autolimit_y_multiplier
                    limit_y_min = None
                    if max_y < limit_y_max:
                        limit_y_max = None

            if limit_y_min is not None:
                ax.plot((min_x, max_x), (limit_y_min, limit_y_min), c="purple")

            if limit_y_max is not None:
                ax.plot((min_x, max_x), (limit_y_max, limit_y_max), c="purple")

            # y achse range begrenzen
            yaxmin = min_y if limit_y_min is None else limit_y_min
            yaxmax = max_y if limit_y_max is None else limit_y_max
            yrange = yaxmax - yaxmin
            yaxmin = yaxmin - (0.05 * yrange)
            yaxmax = yaxmax + (0.05 * yrange)
            ax.set_ylim([yaxmin, yaxmax])
        else:
            limit_y_min = None
            limit_y_max = None

        for line_name, line_x, line_y, line_col in zip(line_names, line_xs, line_ys, self.colors):
            #print("line to xy...", time.time())
            x, y = self._line_to_xy(line_x, line_y, xlim, limit_y_min=limit_y_min, limit_y_max=limit_y_max)
            #print("line to xy FIN", time.time())
            #print("plotting ax...", time.time())
            ax.plot(x, y, color=line_col, linewidth=1.0)
            #print("plotting ax FIN", time.time())

        ax.set_title(group_name)
