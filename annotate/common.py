from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib import replay_memory
from lib import util
import Tkinter
from PIL import Image, ImageTk
import numpy as np
import cPickle as pickle
import time
import imgaug as ia

def numpy_to_tk_image(image):
    image_pil = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image_pil)
    return image_tk

def load_annotations(fp):
    if os.path.isfile(fp):
        return pickle.load(open(fp, "r"))
    else:
        return None

def draw_normal_distribution(height, width, x, y, size):
    if 0 <= y < height and 0 <= x < width:
        pad_by = size * 10
        img = np.zeros((pad_by+height+pad_by, pad_by+width+pad_by), dtype=np.float32)
        #img = img.pad(img, ((20, 20), (20, 20)))
        #normal = util.create_2d_gaussian(size=size*2, fwhm=size)
        normal = util.create_2d_gaussian(size=size*4, sigma=size)
        #print(normal)
        normal_h, normal_w = normal.shape
        normal_hh, normal_wh = normal_h//2, normal_w//2
        #print("normal size", normal.shape)
        #print("img.shape", img.shape)
        #print("img[y-normal_hh:y+normal_hh, x-normal_wh:x+normal_wh]", img[y-normal_hh:y+normal_hh, x-normal_wh:x+normal_wh].shape)
        y1 = np.clip(y-normal_hh+pad_by, 0, img.shape[0]-1) #-(2*pad_by))
        y2 = np.clip(y+normal_hh+pad_by, 0, img.shape[0]-1) #-(2*pad_by))
        x1 = np.clip(x-normal_wh+pad_by, 0, img.shape[1]-1) #-(2*pad_by))
        x2 = np.clip(x+normal_wh+pad_by, 0, img.shape[1]-1) #-(2*pad_by))
        if x2 - x1 > 0 and y2 - y1 > 0:
            img[y1:y2, x1:x2] = normal
        return img[pad_by:-pad_by, pad_by:-pad_by]
    else:
        return np.zeros((height, width), dtype=np.float32)

class GridAnnotationWindow(object):
    def __init__(self, master, canvas, memory, current_state_idx, annotations, current_anno_attribute_name, save_to_fp, every_nth_example=10, zoom_factor=4):
        self.master = master
        self.canvas = canvas
        self.memory = memory
        self.current_state_idx = current_state_idx
        self.annotations = annotations if annotations is not None else dict()
        self.current_annotation = None
        self.background_label = None

        self.eraser = False
        self.dirty = False
        self.brush_size = 3
        self.last_autosave = 0
        self.heatmap_alpha = 0.3
        self.heatmap_alphas = (0.1, 0.3)
        self.current_anno_attribute_name = current_anno_attribute_name
        self.every_nth_example = every_nth_example
        self.zoom_factor = zoom_factor
        self.autosave_every_nth = 20
        self.save_to_fp = save_to_fp

        self.is_showing_directly_previous_state = False
        self.directly_previous_state = None
        self.current_state = None
        self.switch_to_state(self.current_state_idx, autosave=False)
        #self.current_state = memory.get_state_by_id(current_state_idx)

    @staticmethod
    def create(memory, current_anno_attribute_name, save_to_fp, every_nth_example=10, zoom_factor=4):
        print("Loading previous annotations...")
        annotations = load_annotations(save_to_fp)
        #is_annotated = dict([(str(annotation.idx), True) for annotation in annotations])

        current_state_idx = memory.id_min
        if annotations is not None:
            while current_state_idx < memory.id_max:
                key = str(current_state_idx)
                if key not in annotations or current_anno_attribute_name not in annotations[key]:
                    break
                current_state_idx += every_nth_example
        print("ID of first unannotated state: %d" % (current_state_idx,))

        master = Tkinter.Tk()
        state = memory.get_state_by_id(current_state_idx)
        canvas_height = state.screenshot_rs.shape[0] * zoom_factor
        canvas_width = state.screenshot_rs.shape[1] * zoom_factor
        print("canvas height, width:", canvas_height, canvas_width)
        canvas = Tkinter.Canvas(master, width=canvas_width, height=canvas_height)
        canvas.pack()
        canvas.focus_set()

        #y = int(canvas_height / 2)
        #w.create_line(0, y, canvas_width, y, fill="#476042")
        message = Tkinter.Label(master, text="Click to draw annotation. Press E to switch to eraser mode. Press S to save. Use Numpad +/- for brush size.")
        message.pack(side=Tkinter.BOTTOM)

        window_state = GridAnnotationWindow(
            master,
            canvas,
            memory,
            current_state_idx,
            annotations,
            current_anno_attribute_name,
            save_to_fp,
            every_nth_example,
            zoom_factor
        )

        #canvas.bind("<Button-1>", OnPaint(window_state))
        #master.bind("<Button-1>", lambda event: print(event))
        #master.bind("<Button-3>", lambda event: print("right", event))
        #master.bind("<ButtonPress-1>", lambda event: print("press", event))
        master.bind("<B1-Motion>", window_state.on_left_mouse_button)
        #master.bind("<ButtonRelease-1>", lambda event: print("release", event))
        master.bind("<B3-Motion>", window_state.on_right_mouse_button)
        canvas.bind("<e>", lambda event: window_state.toggle_eraser())
        canvas.bind("<s>", lambda event: window_state.save_annotations(force=True))
        canvas.bind("<w>", lambda event: window_state.toggle_heatmap())
        canvas.bind("<p>", lambda event: window_state.toggle_previous_screenshot())
        canvas.bind("<Left>", lambda event: window_state.previous_state(autosave=True))
        canvas.bind("<Right>", lambda event: window_state.next_state(autosave=True))
        canvas.bind("<KP_Add>", lambda event: window_state.increase_brush_size())
        canvas.bind("<KP_Subtract>", lambda event: window_state.decrease_brush_size())

        return window_state

    @property
    def grid(self):
        return self.current_annotation[self.current_anno_attribute_name]

    def toggle_eraser(self):
        self.eraser = not self.eraser
        print("Eraser set to %s" % (self.eraser,))

    def toggle_heatmap(self):
        pos = self.heatmap_alphas.index(self.heatmap_alpha)
        self.heatmap_alpha = self.heatmap_alphas[(pos+1) % len(self.heatmap_alphas)]

        self.set_canvas_background(self._generate_heatmap())

    def toggle_previous_screenshot(self):
        if self.directly_previous_state is not None:
            if self.is_showing_directly_previous_state:
                self.set_canvas_background(self._generate_heatmap())
            else:
                self.set_canvas_background(self.directly_previous_state.screenshot_rs)
            self.is_showing_directly_previous_state = not self.is_showing_directly_previous_state

    def increase_brush_size(self):
        self.brush_size = np.clip(self.brush_size+1, 1, 100)
        print("Increased brush size to %d" % (self.brush_size,))

    def decrease_brush_size(self):
        self.brush_size = np.clip(self.brush_size-1, 1, 100)
        print("Decreased brush size to %d" % (self.brush_size,))

    def previous_state(self, autosave):
        print("Switching to previous state...")
        self.current_state_idx -= self.every_nth_example
        assert self.current_state_idx >= self.memory.id_min, "Start of memory reached (%d vs %d)" % (self.current_state_idx, self.memory.id_min)
        self.switch_to_state(self.current_state_idx, autosave=autosave)

    def next_state(self, autosave):
        print("Switching to next state...")
        self.current_state_idx += self.every_nth_example
        assert self.current_state_idx <= self.memory.id_max, "End of memory reached (%d vs %d)" % (self.current_state_idx, self.memory.id_max)
        self.switch_to_state(self.current_state_idx, autosave=autosave)

    def switch_to_state(self, idx, autosave):
        print("Switching to state %d (autosave=%s)..." % (idx, str(autosave)))
        self.directly_previous_state = self.memory.get_state_by_id(idx-1)
        self.current_state = self.memory.get_state_by_id(idx)
        assert self.current_state is not None
        self.current_state_idx = idx

        if autosave:
            if (self.last_autosave+1) % self.autosave_every_nth == 0:
                # only autosaves if dirty flag is true, ie any example was changed
                self.save_annotations()
                self.last_autosave = 0
            else:
                self.last_autosave += 1
            print("last_autosave=", self.last_autosave)

        key = str(self.current_state_idx)
        if key in self.annotations:
            self.current_annotation = self.annotations[key]
        else:
            self.current_annotation = {
                "idx": self.current_state_idx,
                "from_datetime": self.current_state.from_datetime,
                "screenshot_rs": self.current_state.screenshot_rs,
            }
            self.annotations[key] = self.current_annotation

        annos_done = [key for key in self.current_annotation.keys() if key not in ["idx", "from_datetime", "screenshot_rs"]]
        print("Annotations added to this state: %s" % (", ".join(annos_done)))
        if self.current_anno_attribute_name not in annos_done:
            print("This state has not yet been annotated with '%s'." % (self.current_anno_attribute_name,))
            img = self.current_state.screenshot_rs
            empty_grid = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            self.current_annotation[self.current_anno_attribute_name] = empty_grid

        self.is_showing_directly_previous_state = False
        self.update_annotation_grid(self.grid, initial=True)

    def save_annotations(self, force=False):
        #print(self.annotations)
        if self.dirty or force:
            print("Saving...")
            with open(self.save_to_fp, "w") as f:
                pickle.dump(self.annotations, f, protocol=-1)
            self.dirty = False
            print("Finished saving.")
        else:
            print("Not saved (not marked dirty)")

    """
    def redraw_canvas(self):
        img = generate_canvas_image(self.current_state.screenshot_rs, self.grid)
        self.canvas.delete(Tkinter.ALL)
        self.set_canvas_background(self.canvas, img)
    """

    def update_annotation_grid(self, annotation_grid, initial=False):
        self.current_annotation[self.current_anno_attribute_name] = annotation_grid
        #self.redraw_canvas()
        #img = generate_canvas_image(self.current_state.screenshot_rs, annotation_grid)
        img_heatmap = self._generate_heatmap()
        self.set_canvas_background(img_heatmap)
        if not initial:
            self.dirty = True

    def set_canvas_background(self, image):
        if self.background_label is None:
            # initialize background image label (first call)
            #img = self.current_state.screenshot_rs
            #bg_img_tk = numpy_to_tk_image(np.zeros(img.shape))
            img_heatmap = self._generate_heatmap()
            img_heatmap_rs = ia.imresize_single_image(img_heatmap, (img_heatmap.shape[0]*self.zoom_factor, img_heatmap.shape[1]*self.zoom_factor), interpolation="nearest")
            bg_img_tk = numpy_to_tk_image(img_heatmap_rs)
            self.background_label = Tkinter.Label(self.canvas, image=bg_img_tk)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1, anchor=Tkinter.NW)
            self.background_label.image = bg_img_tk

        #print("image size", image.shape)
        #print("image height, width", image.to_array().shape)
        image_rs = ia.imresize_single_image(image, (image.shape[0]*self.zoom_factor, image.shape[1]*self.zoom_factor), interpolation="nearest")
        image_tk = numpy_to_tk_image(image_rs)
        self.background_label.configure(image=image_tk)
        self.background_label.image = image_tk

    def _generate_heatmap(self):
        return util.draw_heatmap_overlay(self.current_state.screenshot_rs, self.grid, alpha=self.heatmap_alpha)

    def on_left_mouse_button(self, event):
        #canvas = event.widget
        x = self.canvas.canvasx(event.x) / self.zoom_factor
        y = self.canvas.canvasy(event.y) / self.zoom_factor
        height, width = self.current_state.screenshot_rs.shape[0:2]
        #x = event.x
        #y = event.y
        #canvas.delete(Tkinter.ALL)

        grid = self.grid
        normal = draw_normal_distribution(height, width, int(x), int(y), self.brush_size)
        #normal = np.zeros_like(grid)
        #normal[int(y)-2:int(y)+2, int(x)-2:int(x)+2] = 1.0
        if not self.eraser:
            #grid = np.clip(grid + normal, 0, 1)
            grid = np.maximum(grid, normal)
        else:
            grid = grid - normal
        grid = np.clip(grid, 0, 1)
        self.update_annotation_grid(grid)
        #time.sleep(0.1)

    def on_right_mouse_button(self, event):
        x = self.canvas.canvasx(event.x) / self.zoom_factor
        y = self.canvas.canvasy(event.y) / self.zoom_factor
        height, width = self.current_state.screenshot_rs.shape[0:2]
        grid = self.grid
        normal = draw_normal_distribution(height, width, int(x), int(y), self.brush_size)
        grid = grid - normal
        grid = np.clip(grid, 0, 1)
        self.update_annotation_grid(grid)
