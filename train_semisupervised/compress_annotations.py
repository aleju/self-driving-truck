from __future__ import division, print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import train
from dataset import load_dataset_annotated

import cPickle as pickle
import gzip as gz
from lib import util

def main():
    examples = load_dataset_annotated()
    print("Loaded %d examples." % (len(examples),))

    """
    data = []
    for ex in examples:
        print("scr", type(ex.screenshot_rs_jpg))
        print("scrp", [type(scr) for scr in ex.previous_screenshots_rs_jpg])
        print(type(util.decompress_img(ex.screenshot_rs_jpg)))
        print(type(util.compress_to_jpg(util.decompress_img(ex.screenshot_rs_jpg))))
        data.append({
            "state_idx": ex.state_idx,
            "screenshot_rs_jpg": str(ex.screenshot_rs_jpg),
            "previous_screenshots_rs_jpg": [str(scr) for scr in ex.previous_screenshots_rs_jpg],
            "previous_multiaction_vec": ex.previous_multiaction_vec,
            "previous_multiaction_vecs": ex.previous_multiaction_vecs,
            "multiaction_vec": ex.multiaction_vec,
            "next_multiaction_vec": ex.next_multiaction_vec,
            "next_multiaction_vecs": ex.next_multiaction_vecs,
            "grids": ex.grids,
            "attributes": ex.attributes
        })
    """

    print("Writing to file...")
    with gz.open(train.ANNOTATIONS_COMPRESSED_FP, "wb") as f:
        pickle.dump(examples, f, protocol=-1)

if __name__ == "__main__":
    main()
