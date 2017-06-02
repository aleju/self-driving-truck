from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib import replay_memory
from common import load_annotations, numpy_to_tk_image
import Tkinter
from collections import OrderedDict
import imgaug as ia
import cPickle as pickle

class AttributeGroup(object):
    def __init__(self, name, name_shown, attributes=None, default_attribute=None):
        self.name = name
        self.name_shown = name_shown
        self.attributes = attributes if attributes is not None else []
        self.default_attribute = default_attribute

    def append(self, att):
        self.attributes.append(att)

    def get_by_name(self, name):
        atts = [att for att in self.attributes if att.name == name]
        return atts[0] if len(atts) > 0 else None

    def set_default_by_name(self, name):
        self.default_attribute = self.get_by_name(name)

class Attribute(object):
    def __init__(self, name, name_shown):
        self.name = name
        self.name_shown = name_shown

"""
ATTRIBUTE_GROUPS = [
    (
        "Road Type",
        "highway",
        OrderedDict([
            ("country_road", "Country Road"),
            ("highway", "Highway"),
            ("highway_entry_exit", "Highway entry/exit"),
            ("open_area", "Open Area / Parking Lot"),
            ("fuel_station", "Fuel Station"),
            ("hotel", "Hotel"),
            ("rest_area", "Rest Area"),
            ("city_road", "City Road"),
            ("toll_booth", "Toll Booth")
        ])
    )
]
"""

ATTRIBUTE_GROUP_ROAD_TYPE = AttributeGroup("road_type", "Road Type")
ATTRIBUTE_GROUP_ROAD_TYPE.append(Attribute("country_road", "Country Road"))
ATTRIBUTE_GROUP_ROAD_TYPE.append(Attribute("highway", "Highway"))
ATTRIBUTE_GROUP_ROAD_TYPE.append(Attribute("highway_entry_exit", "Highway entry/exit"))
ATTRIBUTE_GROUP_ROAD_TYPE.append(Attribute("open_area", "Open Area / Parking Lot"))
ATTRIBUTE_GROUP_ROAD_TYPE.append(Attribute("fuel_station", "Fuel Station"))
ATTRIBUTE_GROUP_ROAD_TYPE.append(Attribute("hotel", "Hotel"))
ATTRIBUTE_GROUP_ROAD_TYPE.append(Attribute("rest_area", "Rest Area"))
ATTRIBUTE_GROUP_ROAD_TYPE.append(Attribute("city_road", "City Road"))
ATTRIBUTE_GROUP_ROAD_TYPE.append(Attribute("toll_booth", "Toll Booth"))
ATTRIBUTE_GROUP_ROAD_TYPE.append(Attribute("other", "Other"))
ATTRIBUTE_GROUP_ROAD_TYPE.set_default_by_name("highway")

ATTRIBUTE_GROUP_INTERSECTION = AttributeGroup("intersection", "Intersection")
ATTRIBUTE_GROUP_INTERSECTION.append(Attribute("none", "None"))
ATTRIBUTE_GROUP_INTERSECTION.append(Attribute("t-left", "T (left -|)"))
ATTRIBUTE_GROUP_INTERSECTION.append(Attribute("t-right", "T (right |-)"))
ATTRIBUTE_GROUP_INTERSECTION.append(Attribute("t-frontal", "T (frontal -.-)"))
ATTRIBUTE_GROUP_INTERSECTION.append(Attribute("cross", "Cross"))
ATTRIBUTE_GROUP_INTERSECTION.append(Attribute("roundabout", "Roundabout"))
ATTRIBUTE_GROUP_INTERSECTION.append(Attribute("other", "Other"))
ATTRIBUTE_GROUP_INTERSECTION.set_default_by_name("none")

ATTRIBUTE_GROUP_DIRECTION = AttributeGroup("direction", "Direction")
ATTRIBUTE_GROUP_DIRECTION.append(Attribute("unidirection", "Unidirection"))
ATTRIBUTE_GROUP_DIRECTION.append(Attribute("bidirection", "Bidirection"))
ATTRIBUTE_GROUP_DIRECTION.append(Attribute("other", "Other"))
ATTRIBUTE_GROUP_DIRECTION.set_default_by_name("bidirection")

ATTRIBUTE_GROUP_LANE_COUNT = AttributeGroup("lane-count", "Lane Count (current dir.)")
ATTRIBUTE_GROUP_LANE_COUNT.append(Attribute("1", "1"))
ATTRIBUTE_GROUP_LANE_COUNT.append(Attribute("2", "2"))
ATTRIBUTE_GROUP_LANE_COUNT.append(Attribute("3", "3"))
ATTRIBUTE_GROUP_LANE_COUNT.append(Attribute("4+", "4+"))
ATTRIBUTE_GROUP_LANE_COUNT.append(Attribute("other", "Other"))
ATTRIBUTE_GROUP_LANE_COUNT.set_default_by_name("2")

ATTRIBUTE_GROUP_CURVE = AttributeGroup("curve", "Curve (current lane)")
ATTRIBUTE_GROUP_CURVE.append(Attribute("straight", "Straight"))
ATTRIBUTE_GROUP_CURVE.append(Attribute("left-slight", "Left (slight)"))
ATTRIBUTE_GROUP_CURVE.append(Attribute("left-medium", "Left (medium)"))
ATTRIBUTE_GROUP_CURVE.append(Attribute("left-strong", "Left (strong)"))
ATTRIBUTE_GROUP_CURVE.append(Attribute("right-slight", "Right (slight)"))
ATTRIBUTE_GROUP_CURVE.append(Attribute("right-medium", "Right (medium)"))
ATTRIBUTE_GROUP_CURVE.append(Attribute("right-strong", "Right (strong)"))
ATTRIBUTE_GROUP_CURVE.append(Attribute("other", "Other"))
ATTRIBUTE_GROUP_CURVE.set_default_by_name("straight")

ATTRIBUTE_GROUP_SPACE_FRONT = AttributeGroup("space-front", "Space (Front)")
ATTRIBUTE_GROUP_SPACE_FRONT.append(Attribute("plenty", "plenty (>3s)"))
ATTRIBUTE_GROUP_SPACE_FRONT.append(Attribute("some", "some (1-3s)"))
ATTRIBUTE_GROUP_SPACE_FRONT.append(Attribute("minimal", "minimal (<1s)"))
ATTRIBUTE_GROUP_SPACE_FRONT.append(Attribute("none", "none (crashing)"))
ATTRIBUTE_GROUP_SPACE_FRONT.set_default_by_name("plenty")

ATTRIBUTE_GROUP_SPACE_LEFT = AttributeGroup("space-left", "Space (Left)")
ATTRIBUTE_GROUP_SPACE_LEFT.append(Attribute("plenty", "plenty (good)"))
ATTRIBUTE_GROUP_SPACE_LEFT.append(Attribute("some", "some (meh)"))
ATTRIBUTE_GROUP_SPACE_LEFT.append(Attribute("minimal", "minimal (bad)"))
ATTRIBUTE_GROUP_SPACE_LEFT.append(Attribute("none", "none (crashing)"))
ATTRIBUTE_GROUP_SPACE_LEFT.set_default_by_name("plenty")

ATTRIBUTE_GROUP_SPACE_RIGHT = AttributeGroup("space-right", "Space (Right)")
ATTRIBUTE_GROUP_SPACE_RIGHT.append(Attribute("plenty", "plenty (good)"))
ATTRIBUTE_GROUP_SPACE_RIGHT.append(Attribute("some", "some (meh)"))
ATTRIBUTE_GROUP_SPACE_RIGHT.append(Attribute("minimal", "minimal (bad)"))
ATTRIBUTE_GROUP_SPACE_RIGHT.append(Attribute("none", "none (crashing)"))
ATTRIBUTE_GROUP_SPACE_RIGHT.set_default_by_name("plenty")

ATTRIBUTE_GROUP_OFFROAD = AttributeGroup("offroad", "Offroad")
ATTRIBUTE_GROUP_OFFROAD.append(Attribute("onroad", "Onroad"))
ATTRIBUTE_GROUP_OFFROAD.append(Attribute("slightly", "Slightly"))
ATTRIBUTE_GROUP_OFFROAD.append(Attribute("significantly", "Significantly"))
ATTRIBUTE_GROUP_OFFROAD.set_default_by_name("onroad")

ATTRIBUTE_GROUPS = [
    ATTRIBUTE_GROUP_ROAD_TYPE,
    ATTRIBUTE_GROUP_INTERSECTION,
    ATTRIBUTE_GROUP_DIRECTION,
    ATTRIBUTE_GROUP_LANE_COUNT,
    ATTRIBUTE_GROUP_CURVE,
    ATTRIBUTE_GROUP_SPACE_FRONT,
    ATTRIBUTE_GROUP_SPACE_LEFT,
    ATTRIBUTE_GROUP_SPACE_RIGHT,
    ATTRIBUTE_GROUP_OFFROAD
]

def main():
    print("Loading replay memory...")
    memory = replay_memory.ReplayMemory.create_instance_supervised()

    win = AttributesAnnotationWindow.create(
        memory,
        save_to_fp="annotations_attributes.pickle",
        every_nth_example=25
    )
    win.autosave_every_nth = 100
    win.master.wm_title("Annotate attributes")

    Tkinter.mainloop()

class AttributesAnnotationWindow(object):
    def __init__(self, master, canvas, memory, current_state_idx, annotations, save_to_fp, every_nth_example=10, zoom_factor=4):
        self.master = master
        self.canvas = canvas
        self.memory = memory
        self.current_state_idx = current_state_idx
        self.annotations = annotations if annotations is not None else dict()
        self.current_annotation = None
        self.background_label = None

        self.dirty = False
        self.last_autosave = 0
        self.every_nth_example = every_nth_example
        self.zoom_factor = zoom_factor
        self.autosave_every_nth = 20
        self.save_to_fp = save_to_fp

        self.is_showing_directly_previous_state = False
        self.directly_previous_state = None
        self.current_state = None
        self.att_group_to_variable = dict()
        #self.switch_to_state(self.current_state_idx, autosave=False)
        #self.current_state = memory.get_state_by_id(current_state_idx)

    @staticmethod
    def create(memory, save_to_fp, every_nth_example=10, zoom_factor=2):
        colcount = max([len(att_group.attributes) for att_group in ATTRIBUTE_GROUPS])

        print("Loading previous annotations...")
        annotations = load_annotations(save_to_fp)
        #is_annotated = dict([(str(annotation.idx), True) for annotation in annotations])

        current_state_idx = memory.id_min
        if annotations is not None:
            while current_state_idx < memory.id_max:
                key = str(current_state_idx)
                if key not in annotations:
                    break
                current_state_idx += every_nth_example
        print("ID of first unannotated state: %d" % (current_state_idx,))

        master = Tkinter.Tk()
        master.grid()
        state = memory.get_state_by_id(current_state_idx)
        canvas_height = state.screenshot_rs.shape[0] * zoom_factor
        canvas_width = state.screenshot_rs.shape[1] * zoom_factor
        print("canvas height, width:", canvas_height, canvas_width)
        canvas = Tkinter.Canvas(master, width=canvas_width, height=canvas_height)
        #canvas.pack()
        canvas.grid(row=0, column=0, columnspan=colcount)
        canvas.focus_set()

        #y = int(canvas_height / 2)
        #w.create_line(0, y, canvas_width, y, fill="#476042")
        message = Tkinter.Label(master, text="Press S to save.")
        #message.pack(side=Tkinter.BOTTOM)
        message.grid(row=1, column=0, columnspan=colcount)

        window_state = AttributesAnnotationWindow(
            master,
            canvas,
            memory,
            current_state_idx,
            annotations,
            save_to_fp,
            every_nth_example,
            zoom_factor
        )

        def build_lambda(att_group, att):
            return lambda: window_state.on_radio_click(att_group, att)

        for row_idx, att_group in enumerate(ATTRIBUTE_GROUPS):
            print(row_idx)
            var = Tkinter.StringVar()
            window_state.att_group_to_variable[att_group.name] = var
            var.set(att_group.default_attribute.name)
            lab = Tkinter.Label(master, text=att_group.name_shown)
            lab.grid(row=2+row_idx, column=0, sticky=Tkinter.W)
            #lab = Tkinter.Label(master, text=att_group.name_shown).pack(side=Tkinter.LEFT)
            #lab = Tkinter.Label(master, text=att_group.name_shown).pack(anchor=Tkinter.S)
            print("default:", att_group.default_attribute.name)
            for col_idx, att in enumerate(att_group.attributes):
                print("ns/n", att.name_shown, att.name)
                c = Tkinter.Radiobutton(
                    master, text=att.name_shown, variable=var,
                    value=att.name,
                    command=build_lambda(att_group, att)
                )
                #c.pack(side=Tkinter.LEFT)
                c.grid(row=2+row_idx, column=col_idx+1, sticky=Tkinter.W)
                print(row_idx, col_idx)

        canvas.bind("<s>", lambda event: window_state.save_annotations(force=True))
        canvas.bind("<p>", lambda event: window_state.toggle_previous_screenshot())
        canvas.bind("<Left>", lambda event: window_state.previous_state(autosave=True))
        canvas.bind("<Right>", lambda event: window_state.next_state(autosave=True))

        window_state.switch_to_state(window_state.current_state_idx, autosave=False)

        return window_state

    def on_radio_click(self, att_group, att):
        print("radio click", att_group.name, att.name)
        var = self.att_group_to_variable[att_group.name]
        var.set(att.name)
        self.current_annotation["attributes"][att_group.name] = att.name
        self.dirty = True

    #def update_annotations(self):


    def toggle_previous_screenshot(self):
        if self.directly_previous_state is not None:
            if self.is_showing_directly_previous_state:
                self.set_canvas_background(self._generate_heatmap())
            else:
                self.set_canvas_background(self.directly_previous_state.screenshot_rs)
            self.is_showing_directly_previous_state = not self.is_showing_directly_previous_state

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
            print("Annotation for state ", key, " available.")
            print("Attributes: ", self.annotations[key]["attributes"])
        else:
            print("No annotation yet for state ", key)
            last_annotation = self.current_annotation
            self.current_annotation = {
                "idx": self.current_state_idx,
                "from_datetime": self.current_state.from_datetime,
                "screenshot_rs": self.current_state.screenshot_rs,
                "attributes": dict()
            }
            for att_group in ATTRIBUTE_GROUPS:
                if last_annotation is not None:
                    self.current_annotation["attributes"][att_group.name] = last_annotation["attributes"][att_group.name]
                else:
                    self.current_annotation["attributes"][att_group.name] = att_group.default_attribute.name
            self.annotations[key] = self.current_annotation
            print("Initializing attributes to ", self.current_annotation["attributes"])

        # set variables to new annotation state
        for att_group_name in self.current_annotation["attributes"]:
            var = self.att_group_to_variable[att_group_name]
            val = self.current_annotation["attributes"][att_group_name]
            var.set(val)

        self.is_showing_directly_previous_state = False
        self.set_canvas_background(self._generate_heatmap())
        #self.update_annotation_grid(self.grid, initial=True)

    def save_annotations(self, force=False):
        #print(self.annotations)
        if self.dirty or force:
            print("Saving to %s..." % (self.save_to_fp,))
            with open(self.save_to_fp, "w") as f:
                pickle.dump(self.annotations, f, protocol=-1)
            self.dirty = False
            print("Finished saving.")
        else:
            print("Not saved (not marked dirty)")

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
        #return util.draw_heatmap_overlay(self.current_state.screenshot_rs, self.grid, alpha=self.heatmap_alpha)
        return self.current_state.screenshot_rs

if __name__ == "__main__":
    main()
