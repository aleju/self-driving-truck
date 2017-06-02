from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib import replay_memory
from common import GridAnnotationWindow
import Tkinter

def main():
    print("Loading replay memory...")
    memory = replay_memory.ReplayMemory.create_instance_supervised()

    win = GridAnnotationWindow.create(
        memory,
        current_anno_attribute_name="crashables_grid",
        save_to_fp="annotations_crashables.pickle",
        every_nth_example=20
    )
    win.brush_size = 3
    win.autosave_every_nth = 100
    win.master.wm_title("Annotate crashables")

    Tkinter.mainloop()

if __name__ == "__main__":
    main()
