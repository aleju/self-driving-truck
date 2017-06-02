from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import train
from lib import actions
from lib.util import to_cuda, to_variable
from config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import imgaug as ia
import random
import math
import cv2

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

        def identity(_):
            return lambda x: x
        #bn2d = nn.BatchNorm2d
        #bn1d = nn.BatchNorm1d
        bn2d = nn.InstanceNorm2d
        bn1d = nn.InstanceNorm1d
        #bn2d = identity
        #bn1d = identity
        #bn2d = InstanceNormalization

        self.nb_previous_images = 2

        self.emb_c1_curr = nn.Conv2d(3, 128, kernel_size=7, padding=3, stride=2)
        self.emb_c1_bn_curr = bn2d(128)
        self.emb_c1_sd_curr = nn.Dropout2d(0.0)

        self.emb_c2_curr = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.emb_c2_bn_curr = bn2d(128)
        self.emb_c2_sd_curr = nn.Dropout2d(0.0)

        self.emb_c3_curr = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.emb_c3_bn_curr = bn2d(256)
        self.emb_c3_sd_curr = nn.Dropout2d(0.0)


        self.emb_c1_prev = nn.Conv2d(self.nb_previous_images, 64, kernel_size=3, padding=1, stride=1)
        self.emb_c1_bn_prev = bn2d(64)
        self.emb_c1_sd_prev = nn.Dropout2d(0.0)

        self.emb_c2_prev = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.emb_c2_bn_prev = bn2d(128)
        self.emb_c2_sd_prev = nn.Dropout2d(0.0)


        self.emb_c4 = nn.Conv2d(256+128+4, 256, kernel_size=5, padding=2, stride=2)
        self.emb_c4_bn = bn2d(256)
        self.emb_c4_sd = nn.Dropout2d(0.0)

        self.emb_c5 = nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=2)
        self.emb_c5_bn = bn2d(256)
        self.emb_c5_sd = nn.Dropout2d(0.0)

        self.emb_c6 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.emb_c6_bn = bn2d(512)
        self.emb_c6_sd = nn.Dropout2d(0.0)

        self.emb_c7 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.emb_c7_bn = bn2d(512)
        self.emb_c7_sd = nn.Dropout2d(0.0)

        self.maps_c1 = nn.Conv2d(512, 256, kernel_size=5, padding=2)
        self.maps_c1_bn = bn2d(256)
        self.maps_c2 = nn.Conv2d(256, 256, kernel_size=5, padding=(0, 2))
        self.maps_c2_bn = bn2d(256)
        self.maps_c3 = nn.Conv2d(256, 8+3+self.nb_previous_images+1+1, kernel_size=5, padding=2) # 8 grids, 3 for RGB AE, N prev for N grayscale AE, 1 flow, 1 canny

        # road_type: 10
        # intersection: 7
        # direction: 3
        # lane count: 5
        # curve: 8
        # space-front: 4
        # space-left: 4
        # space-right: 4
        # offroad: 3
        atts_size = 10 + 7 + 3 + 5 + 8 + 4 + 4 + 4 + 3
        ma_size = 9 + 9 + 9 + 9
        flipped_size = self.nb_previous_images
        self.vec_fc1 = nn.Linear(512*3*5, atts_size+ma_size+flipped_size, bias=False)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif classname.find('Linear') != -1:
                m.weight.data.normal_(0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #m.weight.data.normal_(1.0, 0.02)
                #m.bias.data.fill_(0)

    def downscale(self, img):
        return ia.imresize_single_image(img, (train.MODEL_HEIGHT, train.MODEL_WIDTH), interpolation="cubic")

    def downscale_prev(self, img):
        return ia.imresize_single_image(img, (train.MODEL_PREV_HEIGHT, train.MODEL_PREV_WIDTH), interpolation="cubic")

    def embed_state(self, previous_states, state, volatile=False, requires_grad=True, gpu=-1):
        prev_scrs = [self.downscale_prev(s.screenshot_rs) for s in previous_states]
        prev_scrs_y = [cv2.cvtColor(scr, cv2.COLOR_RGB2GRAY) for scr in prev_scrs]

        #inputs = np.dstack([self.downscale(state.screenshot_rs)] + list(reversed(prev_scrs_y)))
        inputs = np.array(self.downscale(state.screenshot_rs), dtype=np.float32)
        inputs = inputs / 255.0
        inputs = inputs.transpose((2, 0, 1))
        inputs = inputs[np.newaxis, ...]
        inputs = to_cuda(to_variable(inputs, volatile=volatile, requires_grad=requires_grad), gpu)

        inputs_prev = np.dstack(prev_scrs_y)
        inputs_prev = inputs_prev.astype(np.float32) / 255.0
        inputs_prev = inputs_prev.transpose((2, 0, 1))
        inputs_prev = inputs_prev[np.newaxis, ...]
        inputs_prev = to_cuda(to_variable(inputs_prev, volatile=volatile, requires_grad=requires_grad), gpu)

        return self.embed(inputs, inputs_prev)

    def embed(self, inputs, inputs_prev):
        return self.forward(inputs, inputs_prev, only_embed=True)

    def forward(self, inputs, inputs_prev, only_embed=False):
        def act(x):
            return F.relu(x, inplace=True)
        def lrelu(x, negative_slope=0.2):
            return F.leaky_relu(x, negative_slope=negative_slope, inplace=True)
        def up(x, f=2):
            m = nn.UpsamplingNearest2d(scale_factor=f)
            return m(x)
        def maxp(x):
            return F.max_pool2d(x, 2)

        B = inputs.size(0)
        pos_x = np.tile(np.linspace(0, 1, 40).astype(np.float32).reshape(1, 1, 40), (B, 1, 23, 1))
        pos_x = np.concatenate([pos_x, np.fliplr(pos_x)], axis=1)
        pos_y = np.tile(np.linspace(0, 1, 23).astype(np.float32).reshape(1, 23, 1), (B, 1, 1, 40))
        pos_y = np.concatenate([pos_y, np.flipud(pos_y)], axis=1)

        """
        print(pos_x_curr[0, 0, 0, 0])
        print(pos_x_curr[0, 0, 0, int(MODEL_WIDTH*(1/4))-1])
        print(pos_x_curr[0, 0, 0, int(MODEL_WIDTH*(2/4))-1])
        print(pos_x_curr[0, 0, 0, int(MODEL_WIDTH*(3/4))-1])
        print(pos_x_curr[0, 0, 0, int(MODEL_WIDTH*(4/4))-1])
        from scipy import misc
        misc.imshow(
            np.vstack([
                np.squeeze(pos_x_curr[0].transpose((1, 2, 0))) * 255,
                np.squeeze(pos_y_curr[0].transpose((1, 2, 0))) * 255
            ])
        )
        """

        pos_x = to_cuda(to_variable(pos_x, volatile=inputs.volatile, requires_grad=inputs.requires_grad), Config.GPU)
        pos_y = to_cuda(to_variable(pos_y, volatile=inputs.volatile, requires_grad=inputs.requires_grad), Config.GPU)

        x_emb0_curr = inputs # 3x90x160
        x_emb1_curr = lrelu(self.emb_c1_sd_curr(self.emb_c1_bn_curr(self.emb_c1_curr(x_emb0_curr)))) # 45x80
        x_emb2_curr = lrelu(self.emb_c2_sd_curr(self.emb_c2_bn_curr(self.emb_c2_curr(x_emb1_curr)))) # 45x80
        x_emb2_curr = F.pad(x_emb2_curr, (0, 0, 1, 0)) # 45x80 -> 46x80
        x_emb2_curr = maxp(x_emb2_curr) # 23x40
        x_emb3_curr = lrelu(self.emb_c3_sd_curr(self.emb_c3_bn_curr(self.emb_c3_curr(x_emb2_curr)))) # 23x40

        x_emb0_prev = inputs_prev # 2x45x80
        x_emb1_prev = lrelu(self.emb_c1_sd_prev(self.emb_c1_bn_prev(self.emb_c1_prev(x_emb0_prev)))) # 45x80
        x_emb1_prev = F.pad(x_emb1_prev, (0, 0, 1, 0)) # 45x80 -> 46x80
        x_emb1_prev = maxp(x_emb1_prev) # 23x40
        x_emb2_prev = lrelu(self.emb_c2_sd_prev(self.emb_c2_bn_prev(self.emb_c2_prev(x_emb1_prev)))) # 23x40

        x_emb3 = torch.cat([x_emb3_curr, x_emb2_prev, pos_x, pos_y], 1)
        x_emb3 = F.pad(x_emb3, (0, 0, 1, 0)) # 23x40 -> 24x40

        x_emb4 = lrelu(self.emb_c4_sd(self.emb_c4_bn(self.emb_c4(x_emb3)))) # 12x20
        x_emb5 = lrelu(self.emb_c5_sd(self.emb_c5_bn(self.emb_c5(x_emb4)))) # 6x10
        x_emb6 = lrelu(self.emb_c6_sd(self.emb_c6_bn(self.emb_c6(x_emb5)))) # 3x5
        x_emb7 = lrelu(self.emb_c7_sd(self.emb_c7_bn(self.emb_c7(x_emb6)))) # 3x5
        x_emb = x_emb7

        if only_embed:
            return x_emb
        else:
            x_maps = x_emb # 3x5
            x_maps = up(x_maps, 4) # 12x20
            x_maps = lrelu(self.maps_c1_bn(self.maps_c1(x_maps))) # 12x20
            x_maps = up(x_maps, 4) # 48x80
            x_maps = lrelu(self.maps_c2_bn(self.maps_c2(x_maps))) # 48x80 -> 44x80
            x_maps = F.pad(x_maps, (0, 0, 1, 0)) # 45x80
            x_maps = up(x_maps) # 90x160
            x_maps = F.sigmoid(self.maps_c3(x_maps)) # 90x160

            ae_size = 3 + self.nb_previous_images
            x_grids = x_maps[:, 0:8, ...]
            x_ae = x_maps[:, 8:8+ae_size, ...]
            x_flow = x_maps[:, 8+ae_size:8+ae_size+1, ...]
            x_canny = x_maps[:, 8+ae_size+1:8+ae_size+2, ...]

            x_vec = x_emb
            x_vec = x_vec.view(-1, 512*3*5)
            x_vec = F.dropout(x_vec, p=0.5, training=self.training)
            x_vec = F.sigmoid(self.vec_fc1(x_vec))

            atts_size = 10 + 7 + 3 + 5 + 8 + 4 + 4 + 4 + 3
            ma_size = 9 + 9 + 9 + 9
            x_atts = x_vec[:, 0:atts_size]
            x_ma = x_vec[:, atts_size:atts_size+ma_size]
            x_flipped = x_vec[:, atts_size+ma_size:]

            return x_ae, x_grids, x_atts, x_ma, x_flow, x_canny, x_flipped, x_emb

    def predict_grids(self, inputs, inputs_prev):
        x_ae, x_grids, x_atts, x_ma, x_flow, x_canny, x_flipped, x_emb = self.forward(inputs, inputs_prev)
        return x_grids

class PredictorWithShortcuts(nn.Module):
    def __init__(self):
        super(PredictorWithShortcuts, self).__init__()

        def identity(_):
            return lambda x: x
        #bn2d = nn.BatchNorm2d
        #bn1d = nn.BatchNorm1d
        bn2d = nn.InstanceNorm2d
        bn1d = nn.InstanceNorm1d
        #bn2d = identity
        #bn1d = identity
        #bn2d = InstanceNormalization

        self.nb_previous_images = 2

        self.emb_c1_curr    = nn.Conv2d(3, 128, kernel_size=7, padding=3, stride=2)
        self.emb_c1_bn_curr = bn2d(128)
        self.emb_c1_sd_curr = nn.Dropout2d(0.0)

        self.emb_c2_curr    = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.emb_c2_bn_curr = bn2d(128)
        self.emb_c2_sd_curr = nn.Dropout2d(0.0)

        self.emb_c3_curr    = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.emb_c3_bn_curr = bn2d(256)
        self.emb_c3_sd_curr = nn.Dropout2d(0.0)


        self.emb_c1_prev    = nn.Conv2d(self.nb_previous_images, 64, kernel_size=3, padding=1, stride=1)
        self.emb_c1_bn_prev = bn2d(64)
        self.emb_c1_sd_prev = nn.Dropout2d(0.0)

        self.emb_c2_prev    = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.emb_c2_bn_prev = bn2d(128)
        self.emb_c2_sd_prev = nn.Dropout2d(0.0)


        self.emb_c4    = nn.Conv2d(256+128+4, 256, kernel_size=5, padding=2, stride=2)
        self.emb_c4_bn = bn2d(256)
        self.emb_c4_sd = nn.Dropout2d(0.0)

        self.emb_c5    = nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=2)
        self.emb_c5_bn = bn2d(256)
        self.emb_c5_sd = nn.Dropout2d(0.0)

        self.emb_c6    = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.emb_c6_bn = bn2d(512)
        self.emb_c6_sd = nn.Dropout2d(0.0)

        self.emb_c7    = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.emb_c7_bn = bn2d(512)
        self.emb_c7_sd = nn.Dropout2d(0.0)

        self.maps_c1 = nn.Conv2d(512+256, 256, kernel_size=5, padding=2)
        self.maps_c1_bn = bn2d(256)
        self.maps_c2 = nn.Conv2d(256+128, 256, kernel_size=5, padding=(0, 2))
        self.maps_c2_bn = bn2d(256)
        self.maps_c3 = nn.Conv2d(256+3, 8+3+self.nb_previous_images+1+1, kernel_size=5, padding=2) # 8 grids, 3 for RGB AE, N prev for N grayscale AE, 1 flow, 1 canny

        # road_type: 10
        # intersection: 7
        # direction: 3
        # lane count: 5
        # curve: 8
        # space-front: 4
        # space-left: 4
        # space-right: 4
        # offroad: 3
        atts_size = 10 + 7 + 3 + 5 + 8 + 4 + 4 + 4 + 3
        ma_size = 9 + 9 + 9 + 9
        flipped_size = self.nb_previous_images
        self.vec_fc1 = nn.Linear(512*3*5, atts_size+ma_size+flipped_size, bias=False)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif classname.find('Linear') != -1:
                m.weight.data.normal_(0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #m.weight.data.normal_(1.0, 0.02)
                #m.bias.data.fill_(0)

    def downscale(self, img):
        return ia.imresize_single_image(img, (train.MODEL_HEIGHT, train.MODEL_WIDTH), interpolation="cubic")

    def downscale_prev(self, img):
        return ia.imresize_single_image(img, (train.MODEL_PREV_HEIGHT, train.MODEL_PREV_WIDTH), interpolation="cubic")

    def embed_state(self, previous_states, state, volatile=False, requires_grad=True, gpu=-1):
        prev_scrs = [self.downscale_prev(s.screenshot_rs) for s in previous_states]
        prev_scrs_y = [cv2.cvtColor(scr, cv2.COLOR_RGB2GRAY) for scr in prev_scrs]

        #inputs = np.dstack([self.downscale(state.screenshot_rs)] + list(reversed(prev_scrs_y)))
        inputs = np.array(self.downscale(state.screenshot_rs), dtype=np.float32)
        inputs = inputs / 255.0
        inputs = inputs.transpose((2, 0, 1))
        inputs = inputs[np.newaxis, ...]
        inputs = to_cuda(to_variable(inputs, volatile=volatile, requires_grad=requires_grad), gpu)

        inputs_prev = np.dstack(prev_scrs_y)
        inputs_prev = inputs_prev.astype(np.float32) / 255.0
        inputs_prev = inputs_prev.transpose((2, 0, 1))
        inputs_prev = inputs_prev[np.newaxis, ...]
        inputs_prev = to_cuda(to_variable(inputs_prev, volatile=volatile, requires_grad=requires_grad), gpu)

        return self.embed(inputs, inputs_prev)

    def embed(self, inputs, inputs_prev):
        return self.forward(inputs, inputs_prev, only_embed=True)

    def forward(self, inputs, inputs_prev, only_embed=False):
        def act(x):
            return F.relu(x, inplace=True)
        def lrelu(x, negative_slope=0.2):
            return F.leaky_relu(x, negative_slope=negative_slope, inplace=True)
        def up(x, f=2):
            m = nn.UpsamplingNearest2d(scale_factor=f)
            return m(x)
        def maxp(x):
            return F.max_pool2d(x, 2)

        B = inputs.size(0)
        pos_x = np.tile(np.linspace(0, 1, 40).astype(np.float32).reshape(1, 1, 40), (B, 1, 23, 1))
        pos_x = np.concatenate([pos_x, np.fliplr(pos_x)], axis=1)
        pos_y = np.tile(np.linspace(0, 1, 23).astype(np.float32).reshape(1, 23, 1), (B, 1, 1, 40))
        pos_y = np.concatenate([pos_y, np.flipud(pos_y)], axis=1)

        pos_x = to_cuda(to_variable(pos_x, volatile=inputs.volatile, requires_grad=inputs.requires_grad), Config.GPU)
        pos_y = to_cuda(to_variable(pos_y, volatile=inputs.volatile, requires_grad=inputs.requires_grad), Config.GPU)

        x_emb0_curr = inputs # 3x90x160
        x_emb1_curr = lrelu(self.emb_c1_sd_curr(self.emb_c1_bn_curr(self.emb_c1_curr(x_emb0_curr)))) # 45x80
        x_emb2_curr = lrelu(self.emb_c2_sd_curr(self.emb_c2_bn_curr(self.emb_c2_curr(x_emb1_curr)))) # 45x80
        x_emb2_curr = F.pad(x_emb2_curr, (0, 0, 1, 0)) # 45x80 -> 46x80
        x_emb2_curr_pool = maxp(x_emb2_curr) # 23x40
        x_emb3_curr = lrelu(self.emb_c3_sd_curr(self.emb_c3_bn_curr(self.emb_c3_curr(x_emb2_curr_pool)))) # 23x40

        x_emb0_prev = inputs_prev # 2x45x80
        x_emb1_prev = lrelu(self.emb_c1_sd_prev(self.emb_c1_bn_prev(self.emb_c1_prev(x_emb0_prev)))) # 45x80
        x_emb1_prev = F.pad(x_emb1_prev, (0, 0, 1, 0)) # 45x80 -> 46x80
        x_emb1_prev = maxp(x_emb1_prev) # 23x40
        x_emb2_prev = lrelu(self.emb_c2_sd_prev(self.emb_c2_bn_prev(self.emb_c2_prev(x_emb1_prev)))) # 23x40

        x_emb3 = torch.cat([x_emb3_curr, x_emb2_prev, pos_x, pos_y], 1)
        x_emb3 = F.pad(x_emb3, (0, 0, 1, 0)) # 23x40 -> 24x40

        x_emb4 = lrelu(self.emb_c4_sd(self.emb_c4_bn(self.emb_c4(x_emb3)))) # 12x20
        x_emb5 = lrelu(self.emb_c5_sd(self.emb_c5_bn(self.emb_c5(x_emb4)))) # 6x10
        x_emb6 = lrelu(self.emb_c6_sd(self.emb_c6_bn(self.emb_c6(x_emb5)))) # 3x5
        x_emb7 = lrelu(self.emb_c7_sd(self.emb_c7_bn(self.emb_c7(x_emb6)))) # 3x5
        x_emb = x_emb7

        if only_embed:
            return x_emb
        else:
            x_maps = x_emb # 3x5
            x_maps = up(x_maps, 4) # 12x20
            x_maps = lrelu(self.maps_c1_bn(self.maps_c1(
                torch.cat([x_maps, x_emb4], 1)
            ))) # 12x20
            x_maps = up(x_maps, 4) # 48x80
            x_maps = lrelu(self.maps_c2_bn(self.maps_c2(
                torch.cat([x_maps, F.pad(x_emb2_curr, (0, 0, 1, 1))], 1)
            ))) # 48x80 -> 44x80
            x_maps = F.pad(x_maps, (0, 0, 1, 0)) # 45x80
            x_maps = up(x_maps) # 90x160
            x_maps = F.sigmoid(self.maps_c3(
                torch.cat([x_maps, inputs], 1)
            )) # 90x160

            ae_size = 3 + self.nb_previous_images
            x_grids = x_maps[:, 0:8, ...]
            x_ae = x_maps[:, 8:8+ae_size, ...]
            x_flow = x_maps[:, 8+ae_size:8+ae_size+1, ...]
            x_canny = x_maps[:, 8+ae_size+1:8+ae_size+2, ...]

            x_vec = x_emb
            x_vec = x_vec.view(-1, 512*3*5)
            x_vec = F.dropout(x_vec, p=0.5, training=self.training)
            x_vec = F.sigmoid(self.vec_fc1(x_vec))

            atts_size = 10 + 7 + 3 + 5 + 8 + 4 + 4 + 4 + 3
            ma_size = 9 + 9 + 9 + 9
            x_atts = x_vec[:, 0:atts_size]
            x_ma = x_vec[:, atts_size:atts_size+ma_size]
            x_flipped = x_vec[:, atts_size+ma_size:]

            return x_ae, x_grids, x_atts, x_ma, x_flow, x_canny, x_flipped, x_emb

    def predict_grids(self, inputs, inputs_prev):
        x_ae, x_grids, x_atts, x_ma, x_flow, x_canny, x_flipped, x_emb = self.forward(inputs, inputs_prev)
        return x_grids
