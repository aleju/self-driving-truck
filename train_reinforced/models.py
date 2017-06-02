from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
import torch.optim as optim
from torch.autograd import Variable
from config import Config
import numpy as np
from lib import actions as actionslib
from lib.util import to_cuda, to_variable
import imgaug as ia
import random
import math

def add_white_noise(x, std, training):
    if training:
        noise = Variable(
            x.data.new().resize_as_(x.data).normal_(mean=0, std=std),
            volatile=x.volatile, requires_grad=False
        ).type_as(x)
        x = x + noise
    return x

def init_weights(module):
    for m in module.modules():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.02)
        elif classname.find('BatchNorm') != -1:
            #m.weight.data.fill_(1)
            #m.bias.data.zero_()
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()

        def identity(v):
            return lambda x: x
        #bn2d = nn.BatchNorm2d
        #bn1d = nn.BatchNorm1d
        bn2d = nn.InstanceNorm2d
        #bn1d = nn.InstanceNorm1d
        #bn2d = identity
        bn1d = identity

        self.nb_previous_images = 2

        self.emb_sup_c1    = nn.Conv2d(512, 1024, kernel_size=3, padding=0, stride=1)
        self.emb_sup_c1_bn = bn2d(1024)
        self.emb_sup_c1_sd = nn.Dropout2d(0.0)

        self.emb_add_fc1 = nn.Linear(
            (self.nb_previous_images+1) # speeds
            + (self.nb_previous_images+1) # is_reverse
            + (self.nb_previous_images+1) # steering wheel
            + (self.nb_previous_images+1) # steering wheel raw
            + self.nb_previous_images*9,
            128
        )
        self.emb_add_fc1_bn = bn1d(128)

        self.emb_fc1 = nn.Linear(1024*3 + 128, 512)
        self.emb_fc1_bn = bn1d(512)

        init_weights(self)

    """
    @staticmethod
    def states_to_inputs_reinforced_add(current_state, previous_states):
        all_states = [current_state] + previous_states

        last_date = None
        for i, s in enumerate(all_states):
            if last_date is not None:
                assert last_date > s.from_datetime, "#%d %s vs %s" % (i, last_date, s.from_datetime)
            last_date = s.from_datetime

        speeds = [s.speed for s in all_states]
        speeds = [np.clip(speed/Config.MAX_SPEED, 0, 1) if speed is not None else 0 for speed in speeds]
        speeds = np.array(speeds, dtype=np.float32)
        is_reverse = np.array([1 if s.is_reverse else 0 for s in all_states], dtype=np.float32)
        multiactions_vecs = np.array([s.actions_multivec for s in previous_states], dtype=np.float32).flatten()

        speeds = speeds[np.newaxis, ...]
        is_reverse = is_reverse[np.newaxis, ...]
        multiactions_vecs = multiactions_vecs[np.newaxis, ...]

        return {"speeds": speeds, "is_reverse": is_reverse, "multiactions_vecs": multiactions_vecs}

    def embed_state(self, emb_supervised, current_state, previous_states, volatile=False, gpu=GPU):
        input_reinforced_add = Embedder.states_to_inputs_reinforced_add(current_state, previous_states)

        return self.forward(
            emb_supervised,
            to_cuda(to_variable(input_reinforced_add["speeds"], volatile=volatile, requires_grad=False), gpu),
            to_cuda(to_variable(input_reinforced_add["is_reverse"], volatile=volatile, requires_grad=False), gpu)
            to_cuda(to_variable(input_reinforced_add["multiactions_vecs"], volatile=volatile, requires_grad=False), gpu)
        )
    """

    def forward(self, embeddings_supervised, speeds, is_reverse, steering_wheel, steering_wheel_raw, multiactions_vecs):
        #x_ae, x_grids, x_atts, x_ma, x_flow, x_emb = embeddings_supervised
        x_emb = embeddings_supervised

        def act(x):
            #return F.relu(x, inplace=True)
            return F.leaky_relu(x, negative_slope=0.2, inplace=True)

        x_emb_sup = embeddings_supervised # 512x3x5
        x_emb_sup = act(self.emb_sup_c1_sd(self.emb_sup_c1_bn(self.emb_sup_c1(x_emb_sup)))) # 1024x1x3
        x_emb_sup = x_emb_sup.view(-1, 1024*1*3)
        x_emb_sup = add_white_noise(x_emb_sup, 0.005, self.training)

        x_emb_add = torch.cat([speeds, is_reverse, steering_wheel, steering_wheel_raw, multiactions_vecs], 1)
        x_emb_add = act(self.emb_add_fc1_bn(self.emb_add_fc1(x_emb_add)))
        x_emb_add = add_white_noise(x_emb_add, 0.005, self.training)

        x_emb = torch.cat([x_emb_sup, x_emb_add], 1)
        x_emb = F.dropout(x_emb, p=0.05, training=self.training)

        embs = F.relu(self.emb_fc1_bn(self.emb_fc1(x_emb)))
        #embs = add_white_noise(embs, 0.005, self.training)
        embs = add_white_noise(embs, 0.005, True)

        return embs

    def forward_dict(self, embeddings_supervised, inputs_reinforced_add):
        return self.forward(
            embeddings_supervised,
            inputs_reinforced_add["speeds"],
            inputs_reinforced_add["is_reverse"],
            inputs_reinforced_add["steering_wheel"],
            inputs_reinforced_add["steering_wheel_raw"],
            inputs_reinforced_add["multiactions_vecs"]
        )

class DirectRewardPredictor(nn.Module):
    def __init__(self, nb_bins):
        super(DirectRewardPredictor, self).__init__()

        def identity(v):
            return lambda x: x
        #bn2d = nn.BatchNorm2d
        #bn1d = nn.BatchNorm1d
        bn2d = nn.InstanceNorm2d
        #bn1d = nn.InstanceNorm1d
        #bn2d = identity
        bn1d = identity

        self.fc1 = nn.Linear(512, 128)
        self.fc1_bn = bn1d(128)
        self.fc2 = nn.Linear(128, nb_bins)

        init_weights(self)

    """
    @staticmethod
    def preprocess_output(future_state, gpu=0, numpy=True):
        assert len(future_states) == Config.MODEL_NB_FUTURE_STATES

        output = np.zeros((Config.MODEL_NB_REWARD_BINS,), dtype=np.float32)
        hist, bin_edges = np.histogram(rewards, bins=Config.MODEL_NB_REWARD_BINS, range=(Config.MIN_REWARD, Config.MAX_REWARD), density=False)
        output[:] = hist[::-1] # first entries in hist are the lowest values so we reverse here

        if not numpy:
            output = Variable(torch.from_numpy(output))

        if not numpy and gpu >= 0:
            output = output.cuda(gpu)

        return output

    @staticmethod
    def preprocess_output_many(future_state_list, gpu=0, numpy=True):
        outputs = []
        for future_state in future_states_list:
            outputs.append(ReinforcedOutput.preprocess_outputs(future_state, gpu=-1, numpy=True))
        outputs = np.array(outputs, dtype=np.float32)

        if not numpy:
            outputs = Variable(torch.from_numpy(outputs))

        if not numpy and gpu >= 0:
            outputs = outputs.cuda(gpu)

        return outputs
    """

    def forward(self, embeddings, softmax):
        def act(x):
            return F.leaky_relu(x, negative_slope=0.2, inplace=True)

        x = act(self.fc1_bn(self.fc1(embeddings)))
        #x = act(self.fc1(embeddings))
        x = add_white_noise(x, 0.005, self.training)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)
        if softmax:
            return F.softmax(x)
        else:
            return x

class IndirectRewardPredictor(nn.Module):
    def __init__(self):
        super(IndirectRewardPredictor, self).__init__()

        def identity(v):
            return lambda x: x
        #bn2d = nn.BatchNorm2d
        #bn1d = nn.BatchNorm1d
        bn2d = nn.InstanceNorm2d
        #bn1d = nn.InstanceNorm1d
        #bn2d = identity
        bn1d = identity

        self.fc1 = nn.Linear(512, 128)
        self.fc1_bn = bn1d(128)

        self.fc_v = nn.Linear(128, 1)
        self.fc_advantage = nn.Linear(128, 9)

        init_weights(self)

    def forward(self, embeddings, return_v_adv=False):
        def act(x):
            #return F.tanh(x)
            return F.leaky_relu(x, negative_slope=0.2, inplace=True)

        B, _ = embeddings.size()

        x = act(self.fc1_bn(self.fc1(embeddings)))
        x = add_white_noise(x, 0.005, self.training)
        x = F.dropout(x, p=0.1, training=self.training)
        #x = self.fc2(x)

        x_v = self.fc_v(x)
        x_v_expanded = x_v.expand(B, 9)

        x_adv = self.fc_advantage(x)
        x_adv_mean = x_adv.mean(dim=1)
        x_adv_mean = x_adv_mean.expand(B, 9)
        x_adv = x_adv - x_adv_mean

        x = x_v_expanded + x_adv

        if return_v_adv:
            return x, (x_v, x_adv)
        else:
            return x

class SuccessorPredictor(nn.Module):
    def __init__(self):
        super(SuccessorPredictor, self).__init__()

        def identity(v):
            return lambda x: x
        #bn2d = nn.BatchNorm2d
        #bn1d = nn.BatchNorm1d
        bn2d = nn.InstanceNorm2d
        #bn1d = nn.InstanceNorm1d
        #bn2d = identity
        bn1d = identity

        #self.input_size = 9 + 512
        self.input_size = 9
        self.hidden_size = 512
        self.nb_layers = 1

        self.hidden_fc1 = nn.Linear(512, self.nb_layers*2*self.hidden_size)
        self.hidden_fc1_bn = bn1d(self.nb_layers*2*self.hidden_size)

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.nb_layers, dropout=0.1, batch_first=False)

        self.fc1 = nn.Linear(self.hidden_size, 512)
        #self.fc1_bn = bn1d(512)

        init_weights(self)

    @staticmethod
    def multiactions_to_vecs(multiactions_list):
        vecs = []
        for multiactions in multiactions_list:
            vecs_this = [actionslib.ACTIONS_TO_MULTIVEC[multiaction] for multiaction in multiactions]
            vecs.append(vecs_this)
        vecs = np.array(vecs, dtype=np.float32) # (T, B, 9) with B=number of plans, T=timesteps
        return vecs

    def forward_apply(self, embeddings_reinforced, multiactions_vecs, gpu=Config.GPU):
        assert len(embeddings_reinforced.size()) == 2
        #assert embeddings_reinforced.size(0) == 1
        T, B, _ = multiactions_vecs.size()
        _, S = embeddings_reinforced.size()
        # extend embeddings from (1, 256) to (N_PLANS, 256)
        embeddings_reinforced = embeddings_reinforced.expand(B, S)
        return self.forward(embeddings_reinforced, multiactions_vecs, volatile=True, gpu=gpu)

    def forward(self, embeddings_reinforced, multiactions_vecs, hidden=None, volatile=False, gpu=Config.GPU):
        inputs, embeddings_over_time = self._to_inputs(embeddings_reinforced, multiactions_vecs)
        if hidden is None:
            B = embeddings_reinforced.size(0)
            #hidden = self.create_hidden(inputs.size(1), volatile=volatile, gpu=gpu)
            hidden_vecs = self.hidden_fc1_bn(self.hidden_fc1(embeddings_reinforced))
            hidden_vecs = F.tanh(hidden_vecs)
            #h = F.tanh(hidden_vecs[:, 0:self.nb_layers*self.hidden_size])
            #c = F.sigmoid(hidden_vecs[:, self.nb_layers*self.hidden_size:])
            h = hidden_vecs[:, 0:self.nb_layers*self.hidden_size]
            c = hidden_vecs[:, self.nb_layers*self.hidden_size:]

            #h1 = h[:, 0*self.hidden_size:1*self.hidden_size]
            #h2 = h[:, 1*self.hidden_size:2*self.hidden_size]
            #c1 = c[:, 2*self.hidden_size:3*self.hidden_size]
            #c2 = c[:, 3*self.hidden_size:]
            hidden = (
                h.contiguous().view(self.nb_layers, B, self.hidden_size),
                c.contiguous().view(self.nb_layers, B, self.hidden_size)
            )

        output_lstm, hidden = self.rnn(inputs, hidden)

        T, B, S = output_lstm.size()
        x = output_lstm
        x = x.view(T*B, S)
        x = add_white_noise(x, 0.005, self.training)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc1(x)

        x = embeddings_over_time + x
        #x = self.fc1_bn(x)
        x = x.view(T, B, 512)

        x = F.relu(x, inplace=True)
        #x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

        return x, hidden

    def create_hidden(self, batch_size, volatile=False, gpu=Config.GPU):
        weight = next(self.parameters()).data
        return (
            to_cuda(Variable(weight.new(self.nb_layers, batch_size, self.hidden_size).zero_(), volatile=volatile), gpu),
            to_cuda(Variable(weight.new(self.nb_layers, batch_size, self.hidden_size).zero_(), volatile=volatile), gpu)
        )

    """
    def _to_inputs(self, embeddings_reinforced, multiactions_vecs):
        B, S = embeddings_reinforced.size()
        T = multiactions_vecs.size(0)
        # identical batch size
        assert embeddings_reinforced.size(0) == multiactions_vecs.size(1), "%s / %s" % (str(embeddings_reinforced.size()), str(multiactions_vecs.size()))
        embeddings_over_time = embeddings_reinforced.contiguous().view(1, B, S).expand(T, B, S)
        #print(embeddings_reinforced.size(), embeddings_over_time.size(), multiactions_vecs.size())
        inputs = torch.cat([embeddings_over_time, multiactions_vecs], 2)
        return inputs
    """

    def _to_inputs(self, embeddings_reinforced, multiactions_vecs):
        B, S = embeddings_reinforced.size()
        T = multiactions_vecs.size(0)
        # identical batch size
        assert embeddings_reinforced.size(0) == multiactions_vecs.size(1), "%s / %s" % (str(embeddings_reinforced.size()), str(multiactions_vecs.size()))
        embeddings_over_time = embeddings_reinforced.contiguous().view(1, B, S).expand(T, B, S)
        return multiactions_vecs, embeddings_over_time

    """
    def _to_inputs(self, embeddings_reinforced, multiactions_vecs):
        B, S = embeddings_reinforced.size()
        T = multiactions_vecs.size(0)
        # identical batch size
        assert embeddings_reinforced.size(0) == multiactions_vecs.size(1), "%s / %s" % (str(embeddings_reinforced.size()), str(multiactions_vecs.size()))
        embeddings_over_time = embeddings_reinforced.contiguous().view(1, B, S).expand(T, B, S)
        return torch.cat([embeddings_over_time, multiactions_vecs], 2), embeddings_over_time
    """

class AEDecoder(nn.Module):
    def __init__(self):
        super(AEDecoder, self).__init__()

        def identity(v):
            return lambda x: x
        #bn2d = nn.BatchNorm2d
        #bn1d = nn.BatchNorm1d
        bn2d = nn.InstanceNorm2d
        #bn1d = nn.InstanceNorm1d
        #bn2d = identity
        bn1d = identity

        self.ae_fc1 = nn.Linear(512, 128*3*5)
        self.ae_fc1_bn = bn1d(128*3*5)
        self.ae_c1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.ae_c1_bn = bn2d(128)
        self.ae_c2 = nn.Conv2d(128, 128, kernel_size=3, padding=(0, 1))
        self.ae_c2_bn = bn2d(128)
        self.ae_c3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.ae_c3_bn = bn2d(128)
        self.ae_c4 = nn.Conv2d(128, 3, kernel_size=3, padding=1)

        init_weights(self)

    def forward(self, embedding):
        def act(x):
            return F.relu(x, inplace=True)
        def up(x):
            m = nn.UpsamplingNearest2d(scale_factor=2)
            return m(x)
        #print(embedding.size())
        x_ae = embedding # Bx256
        x_ae = act(self.ae_fc1_bn(self.ae_fc1(x_ae))) # 128x3x5
        x_ae = x_ae.view(-1, 128, 3, 5)
        x_ae = up(x_ae) # 6x10
        #print(x_ae.size())
        x_ae = act(self.ae_c1_bn(self.ae_c1(x_ae))) # 6x10
        x_ae = up(x_ae) # 12x20
        #print(x_ae.size())
        x_ae = act(self.ae_c2_bn(self.ae_c2(x_ae))) # 12x20 -> 10x20
        x_ae = F.pad(x_ae, (0, 0, 1, 0)) # 11x20
        x_ae = up(x_ae) # 22x40
        #print(x_ae.size())
        x_ae = act(self.ae_c3_bn(self.ae_c3(x_ae))) # 22x40
        x_ae = up(x_ae) # 44x80
        #print(x_ae.size())
        x_ae = F.pad(x_ae, (0, 0, 1, 0)) # add 1px at top (from 44 to 45)
        x_ae = F.sigmoid(self.ae_c4(x_ae))
        return x_ae
