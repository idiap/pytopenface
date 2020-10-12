# coding=utf-8

# Copyright (c) 2018-2020 Idiap Research Institute, Martigny, Switzerland
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


COLORS = [
    (128,128,128),
    ( 30, 45,190),
    ( 30,110,240),
    ( 70,200,235),
    (110,185,160),
    (185,150, 20),
    ( 85, 60, 10)
]


def get_font():
    """
    """
    # Only linux
    filename = os.path.join(
        "usr", "share", "fonts", "truetype", "dejavu", "DejaVuSans.ttf")
    # if not os.path.isfile(filename):
    #     pass
    return ImageFont.truetype(filename, 32)


def load_via_pil(path):
    """Open image with PIL and convert it to cv2 format.

    This allows to load more image types that opencv does not support

    Args:
        path : Path to image file

    """
    img = Image.open(path).convert('RGB')
    # Transform to OpenCV, then convert RGB to BGR
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

def cvcolor2pilcolor(color):
    return (color[2], color[1], color[0])


def draw_reidentification_results(display, reid, bb):
    """Draw image from webcam

    Args:
        display  : Image to draw on
        reid     : A list of ReidentificationOutput elements
        bb       : A list [left, top, right, bottom]
    """
    FONT_FOR_NAMES = get_font()

    color = COLORS[0]

    if reid.identity >= 0:
        color = COLORS[1 + reid.identity % (len(COLORS)-1)]

    cv2.rectangle(display ,(bb[0],bb[1]), (bb[2],bb[3]), color, 3)

    # we use PIL for drawing the text, because OpenCV don't handle the UTF-8
    display_pil = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(display_pil)
    draw.text((bb[0], bb[1] - FONT_FOR_NAMES.getsize(reid.name)[1] - 3),
              reid.name, fill=cvcolor2pilcolor(color), font=FONT_FOR_NAMES)
    display[:,:,:] = cv2.cvtColor(np.array(display_pil), cv2.COLOR_RGB2BGR)


def draw_reidentification_candidates(results,
                                     size=(200,200),
                                     nb_to_show=5):
    """Draw in colum the input image and the candidates.

    Args:
        results     : A list of ReidentificationResult elements
        size        : Size of vignettes
        nb_to_show  : Nb of candidates to show

    Returns:
        a new image composed of the candidates

    """
    FONT_FOR_NAMES = get_font()

    black = np.zeros((size[0], size[1], 3), np.uint8)

    if nb_to_show < 1: return

    montage = None

    for reid in sorted(results, key=lambda k: k.position[0]):
        if reid.image is None: continue
        col = cv2.resize(reid.image, size)
        nb_added = 0
        for i in range(len(reid.distances)):
            im_with_score = reid.candidates[i].copy()
            cv2.putText(im_with_score,
                        "{:.2f}".format(reid.distances[i]),
                        (10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=(0,255,0), thickness=2)
            if col is None:
                col = im_with_score.copy()
            else:
                col = np.vstack((col, im_with_score))

            nb_added += 1

        # Fill with black cells
        for i in range(nb_added, nb_to_show):
            if col is None:
                col = black.copy()
            else:
                col = np.vstack((col, black))

        if montage is None:
            montage = col.copy()
        else:
            montage = np.hstack((montage, col))

    if montage is None:
        montage = black.copy()
        for i in range(0, nb_to_show):
            montage = np.vstack((montage, black))

    return montage
