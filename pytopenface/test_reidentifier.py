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

from __future__ import print_function

import os
import sys
import cv2
import argparse

import numpy as np
import torch
import dlib

from pyopenface import OpenfaceComputer

from pprint import pprint

def process_image(file_name):
    """
    """
    image = cv2.imread(file_name)
    cv2.imshow("Image", image)

    face = image[83:166,92:175].copy()
    cv2.imshow("Face", face)

    computer = OpenfaceComputer()
    res = computer.compute_aligned(face)
    print(res["features"])
    print(type(res["features"]))
    print(res["features"].shape)

    cv2.waitKey(0)



if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser \
             (add_help=False,
              formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-i", "--image",  type=str, help="Image")
    parser.add_argument("-l", "--list",   type=str, help="List of images")
    parser.add_argument("-D", "--dlib",   type=str, help="shape_predictor_68_face_landmarks.dat")
    parser.add_argument("-N", default=10, type=int, help="Max number")

    try:
        opts = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)

    openface_computer = OpenfaceComputer(opts.dlib)

    faces = []
    with open(opts.list) as fid:
        for i, line in enumerate(fid):
            file_name = line[:-1]
            print(i, file_name)
            image = cv2.imread(file_name)
            # face = image[77:173,86:182].copy()
            # face = image[83:166,92:175].copy()
            face = image.copy()
            # face = cv2.resize(face, (96, 96))
            f, debug = openface_computer.compute_on_image(face, visu=1)
            # f, visu = openface_computer.compute(face)
            cv2.imshow("Face"+str(i), debug["visu"])
            cv2.imshow("re"+str(i), debug["face"])
            cv2.waitKey(100)
            # f, _ = openface_computer.compute_aligned(face)
            d = { "face": face,
                  "path": file_name,
                  "features": f}
            faces.append(d)

            if i > opts.N: break

    # pprint(faces)
    n_faces = len(faces)
    features = np.zeros((n_faces, faces[0]["features"].shape[1]))

    for i, d in enumerate(faces):
        features[i] = d["features"]
        # cv2.imshow("Face", d["face"])
        # cv2.waitKey(0)
    # print(features.shape)
    # print(features)

    with open("output.html", "w") as fid:
        for i, d in enumerate(faces):
            current = features[i]
            diff = np.linalg.norm(features - current, axis=1)
            idx = np.argsort(diff)
            # print(idx)
            for j in range(min(15,n_faces)):
                fid.write('<img src="{}" title="{}" height="100" width="100">\n'. \
                          format(faces[idx[j]]["path"], diff[idx[j]]))
            fid.write("<br/>")

    cv2.waitKey(0)

    # if len(opts.image) > 0:
    #     process_image(opts.image)


    # a = np.eye(3)
    # print(a)

    # v = np.array([[1,2,3]])
    # print(v)

    # print(a.shape, v.shape)

    # b = a - v
    # print(b)

    # n = np.linalg.norm(b, axis=1)
    # print(n)
