from __future__ import print_function, division
import screenshot
import numpy as np
import subprocess
import time
from pykeyboard import PyKeyboard

KEYBOARD = PyKeyboard()

def sendkeys(win_id, up=None, press=None, down=None):
    assert up is None or isinstance(up, (list, tuple, set))
    assert press is None or isinstance(press, (list, tuple, set))
    assert down is None or isinstance(down, (list, tuple, set))
    up = set(up) if up is not None else set()
    down = set(down) if down is not None else set()
    press = set(press) if press is not None else set()
    up = up - down
    press = press - down
    if len(up) > 0 or len(press) > 0 or len(down) > 0:
        process = subprocess.Popen(["xte"], stdin=subprocess.PIPE)
        commands = []
        for key in up:
            commands.append("keyup " + key)
        for key in press:
            commands.append("key " + key)
        for key in down:
            commands.append("keydown " + key)
        sequence = "\n".join(commands) + "\n"
        process.communicate(input=sequence)
        process.wait()

PYKB_KEYS_DOWN = set()
def sendkeys_pykb(up=None, tap=None, down=None):
    assert up is None or isinstance(up, (list, tuple, set))
    assert tap is None or isinstance(tap, (list, tuple, set))
    assert down is None or isinstance(down, (list, tuple, set))
    up = set(up) if up is not None else set()
    down = set(down) if down is not None else set()
    tap = set(tap) if tap is not None else set()
    #up = up - down
    tap = tap - down
    #print("up", up, "down", down, "tap", tap)
    #up = up.union(set(["a", "w", "s", "d"]))
    up = up.union(PYKB_KEYS_DOWN)

    #down = ["w"]
    if len(up) > 0:
        for key in up:
            #print("up", key)
            KEYBOARD.release_key(key)
    if len(tap) > 0:
        for key in tap:
            KEYBOARD.tap_key(key)
    if len(down) > 0:
        #KEYBOARD.press_keys(list(down))
        for key in down:
            #print("down", key)
            #pass
            #KEYBOARD.tap_key(key)
            #KEYBOARD.release_key(key)
            KEYBOARD.press_key(key)
    #time.sleep(0.1)

    PYKB_KEYS_DOWN.clear()
    for key in down:
        PYKB_KEYS_DOWN.add(key)

def xwininfo(win_id):
    process = subprocess.Popen(["xwininfo", "-id", str(win_id)], stdout=subprocess.PIPE)
    process.wait()
    assert process.returncode == 0
    lines = process.stdout.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line) > 0]
    return lines

def get_window_coordinates(win_id):
    lines = xwininfo(win_id)

    #for line in lines:
    #    print(line)

    x1 = None
    y1 = None
    h = None
    w = None
    for line in lines:
        line = line.lower()
        if line.startswith("absolute upper-left x:"):
            x1 = int(line[len("absolute upper-left x:"):].strip())
        elif line.startswith("absolute upper-left y:"):
            y1 = int(line[len("absolute upper-left y:"):].strip())
        elif line.startswith("width:"):
            w = int(line[len("width:"):].strip())
        elif line.startswith("height:"):
            h = int(line[len("height:"):].strip())
    assert x1 is not None
    assert y1 is not None
    assert h is not None
    assert w is not None
    return x1, y1, x1+w, y1+h

def find_window_ids(needle_name):
    assert len(needle_name) > 0
    process = subprocess.Popen(["xdotool", "search", "--name", needle_name], stdout=subprocess.PIPE)
    process.wait()
    result = process.stdout.readlines()
    #print("process.returncode", process.returncode)
    #print("lines", result)
    if process.returncode == 0:
        win_ids = [int(line.strip()) for line in result]
        return win_ids
    elif process.returncode == 1:
        if len(result) == 0:
            # xdotool exists, but window not found
            return []
        else:
            raise Exception("Received error return code when calling xdotool and non-zero output (xdotool not installed?): %s" % (result,))
    else:
        raise Exception("Unexpected returncode from xdotool: %s" % (process.returncode,))

def get_window_name(win_id):
    wininfo = xwininfo(win_id)
    pos = wininfo[0].find('"')
    assert pos > -1
    return wininfo[0][pos+1:-1]

def get_active_window_id():
    process = subprocess.Popen(["xdotool", "getactivewindow"], stdout=subprocess.PIPE)
    process.wait()
    assert process.returncode == 0
    result = process.stdout.readlines()
    win_ids = [int(line.strip()) for line in result]
    return win_ids[-1]

def activate_window(win_id):
    process = subprocess.Popen(["xdotool", "windowactivate", "--sync", str(win_id)])
    process.wait()
    assert process.returncode == 0
