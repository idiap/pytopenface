#!/usr/bin/env python
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

import io
import os
import sys
import cv2
import time
import logging
import argparse

from PIL import ImageFont

import numpy as np

from pytopenface import ReidentificationApplication


if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser \
      (add_help=False,
       formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--dlib-model", default="shape_predictor_68_face_landmarks.dat",
        help="Path to shape_predictor_68_face_landmarks.dat")
    parser.add_argument(
        "--face-book", type=str, default=os.path.join("data", "persons.txt"),
        help="Path to model of persons")
    parser.add_argument(
        "--source", type=str, default="0",
        help="Camera index for webcam or filename of image names")
    parser.add_argument(
        "-t", "--reid-threshold", type=float, default=0.7,
        help="Threshold below which match is performed for re-identification")
    parser.add_argument(
        "--enlarge", type=float, default=0.1,
        help="Factor to crop around a face for monitor display")
    parser.add_argument(
        "--record", type=str, default="",
        help="Directory where to record frames")
    parser.add_argument(
        "--show-closest", type=int, default=1,
        help="How many closest to show (nothing if 0)")
    parser.add_argument(
        "--use-cuda", type=int, default=1,
        help="Use cuda/GPU is available")
    parser.add_argument(
        "--verbose", type=int, default=20,
        help="Level of logging verbose (WARNING (30), INFO (20), DEBUG (10)")
    parser.add_argument(
        "--mirror", action="store_true",
        help="Mirror mode")

    try:
        opts = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(format="[%(name)s] %(message)s", level=opts.verbose)

    app = ReidentificationApplication(
        face_book=opts.face_book,
        dlib_model=opts.dlib_model,
        reid_threshold=opts.reid_threshold,
        source=opts.source,
        use_cuda=opts.use_cuda,
        enlarge=opts.enlarge,
        nb_to_show=opts.show_closest,
        record=opts.record,
        mirror=opts.mirror)
    app.run()
