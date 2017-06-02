from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cStringIO as StringIO
from scipy import misc, ndimage
from skimage import feature
import numpy as np
import torch
from torch.autograd import Variable
from config import Config
from matplotlib import pyplot as plt
import imgaug as ia
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import cv2

def template_match(needle, haystack):
    result = feature.match_template(haystack, needle)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[1], ij[0]
    score = result[y, x]
    return x, y, score

def compress_to_jpg(img):
    return compress_img(img, method="JPEG")

def compress_img(img, method):
    img_compressed_buffer = StringIO.StringIO()
    im = misc.toimage(img)
    im.save(img_compressed_buffer, format=method)
    img_compressed = img_compressed_buffer.getvalue()
    img_compressed_buffer.close()
    return img_compressed

def decompress_img(img_compressed):
    img_compressed_buffer = StringIO.StringIO()
    img_compressed_buffer.write(img_compressed)
    img = ndimage.imread(img_compressed_buffer, mode="RGB")
    img_compressed_buffer.close()
    return img

def draw_image(img, other_img, x, y, copy=True, raise_if_oob=False):
    if img.ndim == 3 and other_img.ndim == 2:
        other_img = np.tile(np.copy(other_img)[:, :, np.newaxis], (1, 1, 3))
    elif img.ndim == 3 and other_img.ndim == 3 and other_img.shape[2] == 1:
        other_img = np.tile(other_img, (1, 1, 3))
    else:
        assert img.ndim == other_img.ndim
        assert img.shape[2] == other_img.shape[2]

    result = np.copy(img) if copy else img
    imgh, imgw = img.shape[0], img.shape[1]
    if y < 0 or y >= imgh or x < 0 or x >= imgw:
        if raise_if_oob:
            raise Exception("Invalid coordinates y=%d, x=%d for img shape h=%d, w=%d" % (y, x, imgh, imgw))
        else:
            return result

    y2 = np.clip(y + other_img.shape[0], 0, imgh-1)
    x2 = np.clip(x + other_img.shape[1], 0, imgw-1)

    result[y:y2, x:x2, :] = other_img[0:y2-y, 0:x2-x, :]

    return result

def draw_point(img, y, x, size=5, color=[255, 0, 0], alpha=1.0, copy=True):
    assert img.dtype == np.uint8
    assert isinstance(y, int), "Invalid y-coordinate (type=%s val=%s)" % (type(y), str(y))
    assert isinstance(x, int), "Invalid x-coordinate (type=%s val=%s)" % (type(x), str(x))
    assert size % 2 != 0

    if copy:
        img = np.copy(img)

    if alpha < 0.99:
        img = img.astype(np.float32, copy=False)

    height, width = img.shape[0], img.shape[1]

    if y < 0 or y >= height:
        print("[WARNING] draw_point: y is out of bounds (y=%d vs bounds 0 to %d)" % (y, height,))
    if x < 0 or x >= width:
        print("[WARNING] draw_point: x is out of bounds (x=%d vs bounds 0 to %d)" % (x, width,))

    coords = []
    if size == 1:
        coords.append((y, x))
    else:
        sizeh = (size-1) // 2
        for yy in range(-sizeh, sizeh+1):
            for xx in range(-sizeh, sizeh+1):
                coords.append((y + yy, x + xx))

    coords = [(yy, xx) for (yy, xx) in coords if 0 <= yy < height and 0 <= xx < width]

    if len(coords) > 0:
        coords_y = [yy for yy, xx in coords]
        coords_x = [xx for yy, xx in coords]

        if alpha >= 0.99:
            img[coords_y, coords_x, :] = np.uint8(color)
        else:
            img[coords_y, coords_x, :] = (1-alpha) * img[coords_y, coords_x, :] + alpha * np.float32(color)

    if alpha < 0.99:
        img = np.clip(img, 0, 255, copy=False)
        img = img.astype(np.uint8)

    return img

def draw_line(img, y1, x1, y2, x2, color=[0, 255, 0], alpha=1.0, thickness=1, clip_result=True, copy=True):
    assert img.dtype == np.uint8
    assert 0 <= alpha <= 1.0
    assert thickness >= 1
    assert (thickness == 1 and alpha <= 1.0) or (thickness > 1 and alpha == 1.0) # wenn thickness > 1 dann muss auf PIL ausgewichen werden, welches derzeit kein alpha unterstuetzt

    if copy:
        result = np.copy(img)
    else:
        result = img

    img_height, img_width = result.shape[0], result.shape[1]
    y1 = np.clip(y1, 0, img_height-1)
    y2 = np.clip(y2, 0, img_height-1)
    x1 = np.clip(x1, 0, img_width-1)
    x2 = np.clip(x2, 0, img_width-1)

    if thickness == 1:
        rr, cc = draw.line(y1, x1, y2, x2)
        rr = np.clip(rr, 0, img_height - 1)
        cc = np.clip(cc, 0, img_width - 1)

        if alpha >= 1.0-0.01:
            result[rr, cc, :] = np.array(color)
        else:
            result = result.astype(np.float32)
            result[rr, cc, :] = (1 - alpha) * result[rr, cc, :] + alpha * np.array(color)
            if clip_result:
                result = np.clip(result, 0, 255)
            result = result.astype(np.uint8)
    else:
        # skimage does not support thickness, so use PIL here
        # here without alpha for now, would need RGBA image
        assert alpha >= 0.99
        result = Image.fromarray(result)
        context = ImageDraw.Draw(result)
        context.line([(x1, y1), (x2, y2)], fill=tuple(color), width=thickness)
        result = np.asarray(result)
        result.setflags(write=True) # PIL/asarray returns read-only array
        result = result.astype(np.uint8)

    return result

def draw_direction_circle(img, y, x, r_inner, r_outer, angle_start, angle_end, color_border=[0, 255, 0], color_fill=[0, 200, 0], alpha=1.0):
    assert r_inner <= r_outer
    input_dtype = img.dtype
    img = np.copy(img).astype(np.float32)
    color_border = np.array(color_border)
    color_fill = np.array(color_fill)

    #print(y, x, r_inner, r_outer, angle_start, angle_end)

    center = (x, y)
    axes_outer = (r_outer, r_outer)
    axes_inner = (r_inner, r_inner)
    angle = -90
    thickness = 1

    #cv2.ellipse(img, center=center, axes=axes_outer, angle=angle, startAngle=angle_start, endAngle=angle_end, color=color, thickness=thickness)
    outer_points = cv2.ellipse2Poly(center=center, axes=axes_outer, angle=angle, arcStart=angle_start, arcEnd=angle_end, delta=1)
    inner_points = cv2.ellipse2Poly(center=center, axes=axes_inner, angle=angle, arcStart=angle_start, arcEnd=angle_end, delta=1)
    # inner_points in umgekehrter reihenfolge, da sonst zwei linien schraeg durch den bogen gehen,
    # um inneren und aeusseren bogen zu verbinden
    full_arc = np.concatenate((outer_points, inner_points[::-1]), axis=0).astype(np.int32)

    if alpha >= 0.99:
        #points = [outer_points, inner_points]
        # polylines expects points as [(N, 2)] of dtype int32 (not int64)
        #cv2.polylines(img, points, isClosed=False, color=color, thickness=thickness)

        #points_connectors = [np.int32([outer_points[0], inner_points[0]]), np.int32([outer_points[-1], inner_points[-1]])]
        #cv2.polylines(img, points_connectors, isClosed=False, color=color, thickness=thickness)
        #cv2.fillConvexPoly(img, full_arc, color=color_fill.astype(np.int64))
        cv2.fillPoly(img, [full_arc], color=color_fill.astype(np.int64))
        cv2.polylines(img, [full_arc], isClosed=True, color=color_border.astype(np.int64), thickness=thickness, lineType=cv2.CV_AA)
        np.clip(img, 0, 255, out=img)
    else:
        """
        img_draw = np.zeros_like(img)

        #cv2.fillConvexPoly(img_draw, full_arc, color=color_fill.astype(np.int64))
        cv2.fillPoly(img_draw, [full_arc], color=color_fill.astype(np.int64))

        cv2.polylines(img_draw, [full_arc], isClosed=True, color=color_border.astype(np.int64), thickness=thickness, lineType=cv2.CV_AA)

        mask = img_draw > 0

        #mask_1d = np.tile(np.max(mask, axis=2), (1, 1, 3))
        mask_1d = np.tile(np.max(mask, axis=2)[:, :, np.newaxis], (1, 1, 3))
        img_outside_mask = img * (~mask_1d)
        img_inside_mask = img * mask_1d
        img = img_outside_mask + (1 - alpha) * img_inside_mask + alpha * img_draw
        """

        arc_x1, arc_y1 = np.min(full_arc, axis=0)
        arc_x2, arc_y2 = np.max(full_arc, axis=0)
        full_arc_shifted = full_arc - np.array([arc_x1, arc_y1])

        subimg = img[arc_y1:arc_y2+1, arc_x1:arc_x2+1, :]
        subimg_draw = np.zeros_like(subimg)

        #cv2.fillConvexPoly(img_draw, full_arc, color=color_fill.astype(np.int64))
        cv2.fillPoly(subimg_draw, [full_arc_shifted], color=color_fill.astype(np.int64))

        cv2.polylines(subimg_draw, [full_arc_shifted], isClosed=True, color=color_border.astype(np.int64), thickness=thickness, lineType=cv2.CV_AA)

        mask = subimg_draw > 0

        mask_1d = np.any(mask, axis=2)
        mask_3d = mask_1d[:, :, np.newaxis]
        subimg_outside_mask = subimg * (~mask_3d)
        subimg_inside_mask = subimg * mask_3d
        subimg = subimg_outside_mask + (1 - alpha) * subimg_inside_mask + alpha * subimg_draw

        np.clip(subimg, 0, 255, out=subimg)

        img[arc_y1:arc_y2+1, arc_x1:arc_x2+1, :] = subimg

    img = img.astype(input_dtype)
    return img

FONT_CACHE = dict()
def draw_text(img, y, x, text, color=[0, 255, 0], size=25):
    assert img.dtype in [np.uint8, np.int32, np.int64, np.float32, np.float64]

    input_dtype = img.dtype
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    for i in range(len(color)):
        val = color[i]
        if isinstance(val, float):
            val = int(val * 255)
            val = np.clip(val, 0, 255)
            color[i] = val

    shape = img.shape
    img = Image.fromarray(img)
    if size not in FONT_CACHE:
        FONT_CACHE[size] = ImageFont.truetype("DejaVuSans.ttf", size)
    context = ImageDraw.Draw(img)
    context.text((x, y), text, fill=tuple(color), font=FONT_CACHE[size])
    img_np = np.asarray(img)
    img_np.setflags(write=True)  # PIL/asarray returns read only array

    if img_np.dtype != input_dtype:
        img_np = img_np.astype(input_dtype)

    return img_np

def to_variable(inputs, volatile=False, requires_grad=True):
    if volatile:
        make_var = lambda x: Variable(x, volatile=True)
    else:
        make_var = lambda x: Variable(x, requires_grad=requires_grad)

    if isinstance(inputs, np.ndarray):
        return make_var(torch.from_numpy(inputs))
    elif isinstance(inputs, list):
        return [make_var(torch.from_numpy(el)) for el in inputs]
    elif isinstance(inputs, tuple):
        return [make_var(torch.from_numpy(el)) for el in inputs]
    elif isinstance(inputs, dict):
        return dict([(key, make_var(torch.from_numpy(inputs[key]))) for key in inputs])
    else:
        raise Exception("unknown input %s" % (type(inputs),))

def to_variables(inputs, volatile=False, requires_grad=True):
    return to_variable(inputs, volatile=volatile, requires_grad=requires_grad)

def to_cuda(inputs, gpu=Config.GPU):
    if gpu <= -1:
        return inputs
    else:
        if isinstance(inputs, Variable):
            return inputs.cuda(gpu)
        elif isinstance(inputs, list):
            return [el.cuda(gpu) for el in inputs]
        elif isinstance(inputs, tuple):
            return tuple([el.cuda(gpu) for el in inputs])
        elif isinstance(inputs, dict):
            return dict([(key, inputs[key].cuda(gpu)) for key in inputs])
        else:
            raise Exception("unknown input %s" % (type(inputs),))

def to_numpy(var):
    #if ia.is_numpy_array(var):
    if isinstance(var, (np.ndarray, np.generic)):
        return var
    else:
        return var.data.cpu().numpy()

def draw_heatmap_overlay(img, heatmap, alpha=0.5):
    #assert img.shape[0:2] == heatmap.shape[0:2]
    assert len(heatmap.shape) == 2 or (heatmap.ndim == 3 and heatmap.shape[2] == 1)
    assert img.dtype in [np.uint8, np.int32, np.int64]
    assert heatmap.dtype in [np.float32, np.float64]

    if heatmap.ndim == 3 and heatmap.shape[2] == 1:
        heatmap = np.squeeze(heatmap)

    if img.shape[0:2] != heatmap.shape[0:2]:
        heatmap_rs = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        heatmap_rs = ia.imresize_single_image(heatmap_rs[..., np.newaxis], img.shape[0:2], interpolation="nearest")
        heatmap = np.squeeze(heatmap_rs) / 255.0

    cmap = plt.get_cmap('jet')
    heatmap_cmapped = cmap(heatmap)
    #img_heatmaps_cmapped = img_heatmaps_cmapped[:, :, 0:3]
    heatmap_cmapped = np.delete(heatmap_cmapped, 3, 2)
    #heatmap_cmapped = np.clip(heatmap_cmapped * 255, 0, 255).astype(np.uint8)
    heatmap_cmapped = heatmap_cmapped * 255
    mix = (1-alpha) * img + alpha * heatmap_cmapped
    mix = np.clip(mix, 0, 255).astype(np.uint8)
    return mix

def create_2d_gaussian_old(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    unnormalized = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return unnormalized / np.max(unnormalized)

def create_2d_gaussian(size, sigma):
    sigma_x = sigma_y = sigma
    x = np.arange(-size, size, step=1)
    y = np.arange(-size, size, step=1)

    x, y = np.meshgrid(x, y)
    z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2))))
    z = z.reshape((size*2, size*2))
    z_norm = z / np.max(z)
    return z_norm
