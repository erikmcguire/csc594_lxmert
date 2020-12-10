from scipy.special import softmax
from typing import Tuple
import numpy as np
import sys, cv2
import h5py

from vqa_paths import HAT_DATA

def debug(x):
    """Clumsy fn to step through script."""
    print(f'\n{x}\n')
    sys.exit()

def get_h_att() -> Tuple:
    """
    Load VQA 2.0 HLAT files.
    {"pre_attmap" : attention maps for question ids (ordered)}
    https://github.com/qiaott/HAN
    for question-image pairs:
        train_val: 658,111 attention maps
        test_dev:  107,394 attention maps
        test:      447,793 attention maps
    """
    trainval = h5py.File(f'{HAT_DATA}trainval2014_attention_maps.h5', 'r')
    test_dev = h5py.File(f'{HAT_DATA}test-dev2015_attention_maps.h5', 'r')
    test =         h5py.File(f'{HAT_DATA}test2015_attention_maps.h5', 'r')
    return {'trainval': trainval['pre_attmap'],
            'test-dev': test_dev['pre_attmap'],
            'test':         test['pre_attmap']}

def box_interpolation(bboxes, hm):
    """
    modified from:
        https://github.com/facebookresearch/mmf/issues/145
    Extract scalar human attentions within detected object regions.
    Re-normalize with cold softmax to get distinct peaks.
    """
    hmboxes = []
    for ix, bbox in enumerate(bboxes):
        # w: (0, 2); h: (1, 3)
        w1, h1, w2, h2 = bbox
        hweights = hm[int(h1):int(h2),
                      int(w1):int(w2)]
        hmboxes.append(hweights)
    # Weights were small, distributed across whole image,
    # so try to assign remaining mass proportionally
    # to each bounding box.
    hmboxes = np.array([hweights.sum()/hm.sum()
                        for hweights in hmboxes])
    # Cold temperature to get hardened, peakier distribution.
    hmboxes = softmax(hmboxes/0.1)
    return hmboxes

def human_interpolation(att, h: int, w: int):
    """
        Bilinear interpolation (cv2.INTER_LINEAR)
        of human attentions to given image size.
        cv2.resize (w, h) gives att.shape: (h, w)
    """
    att = cv2.resize(att, (w, h))
    return att

def get_hatt_boxed(boxes, img_h, img_w, h_atts) -> np.array:
    """
        Resize human attentions,
        extract via bounding boxes.
    """
    h_atts = human_interpolation(att=h_atts, h=img_h, w=img_w)
    return box_interpolation(bboxes=boxes, hm=h_atts)
