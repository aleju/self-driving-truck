# code largely from
#   http://stackoverflow.com/questions/69645/take-a-screenshot-via-a-python-script-linux
from __future__ import print_function, division
import ctypes
import os
from PIL import Image
import numpy as np
import time
from scipy import misc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_NAME = "screenshot_c.so" # naming this just "screenshot.so" causes problem for "import screenshot" (seems to prefer the .so over .py?)
ABS_LIB_PATH = os.path.join(CURRENT_DIR, LIB_NAME)
SCREENSHOT_LIB = ctypes.CDLL(ABS_LIB_PATH)

def make_screenshot(x1, y1, x2, y2):
    #w, h = x1+x2, y1+y2
    w = x2 - x1
    h = y2 - y1
    size = w * h
    objlength = size * 3

    SCREENSHOT_LIB.getScreen.argtypes = []
    result = (ctypes.c_ubyte * objlength)()

    SCREENSHOT_LIB.getScreen(x1, y1, w, h, result)
    #return Image.frombuffer('RGB', (w, h), result, 'raw', 'RGB', 0, 1)
    img_flat = np.frombuffer(result, dtype=np.uint8)
    return img_flat.reshape((h, w, 3))

if __name__ == '__main__':
    start_time = time.time()
    im = make_screenshot(100, 100, 320, 320)
    im = np.array(im)
    end_time = time.time()
    print("Made screenshot in %.4fs" % (end_time - start_time,))
    print(im.ravel()[0:50], im.shape)
    misc.imshow(im)
