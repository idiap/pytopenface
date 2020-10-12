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
import dlib
import time
import argparse
import numpy as np

from pyopenface import OpenfaceComputer


class Profiler(object):
    """Accumulate images and features for various identities"""

    def __init__(self, low_threshold=0.3, high_threshold=0.5):
        """
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.gallery = None
        self.images = []

    def add(self, image, row128):
        """Add 'row128' to the 'gallery' if the Euclidian distance between
           'row128' and all the other rows of 'self.gallery are <
           low_threshold.

        """
        if self.gallery is None:
            self.gallery = row128
            self.images.append(image)
            print("Add gallery")
        else:
            distances = np.linalg.norm(self.gallery - row128, axis=1)
            distances = np.sort(distances)
            if distances[0] > self.low_threshold:
                self.gallery = np.vstack((self.gallery, row128))
                self.images.append(image)
                print("gallery {} images {}".format(self.gallery.shape,
                                                    len(self.images)))


    def save(self, dirname, extension="png"):
        """Save the images"""
        if not os.path.exists(dirname): os.mkdir(dirname)

        for i, image in enumerate(self.images):
            name = "{:06d}.{}".format(i, extension)
            name = os.path.join(dirname, name)
            cv2.imwrite(name, image)


    def cluster(self, dirname, extension="png"):
        """Perform a clustering of the images to identify identities.

        The images are then saved in the directory 'dirname'
        alongside with a a text file woth following format:

        The clustering is performed by selecting randomly an image,
        and accumulate in the same cluster all images of distance <
        self.high_threshold. When no image are < self.high_threshold,
        a new one is randomly selected, etc. until all images have
        been assigned to as cluster.

        # A comment
        1 CHANGE-NAME
        /dirname/image1.png
        /dirname/image2.png
        /dirname/image3.png
        2 CHANGE-NAME
        /dirname/image4.png
        /dirname/image5.png
        /dirname/image6.png
        3 CHANGE
        ...

        """
        pool = self.gallery.copy()

        clusters = []
        n_images = self.gallery.shape[0]
        remaining = set(range(n_images))

        while len(remaining) > 0:
            # Pick a sample
            current = remaining.pop()
            is_included = 0

            for no_cluster, cluster in enumerate(clusters):
                for idx in cluster:
                    x1 = self.gallery[idx]
                    x2 = self.gallery[current]
                    distance = np.linalg.norm(x2 - x1)
                    # print("|{} - {}| = {}".format(current, idx, distance))
                    if distance < self.high_threshold:
                        cluster.append(current)
                        is_included = 1
                        # print("Add {} to {}".format(current, no_cluster))
                        break

                if is_included:
                    break

            if not is_included:
                clusters.append([current])

        # print(clusters)

        if not os.path.exists(dirname): os.mkdir(dirname)
        filename = os.path.join(dirname, "list.txt")
        with open(filename, "w") as fid:
            fid.write("# Gallery of persons\n")
            for no_cluster, cluster in enumerate(clusters):
                fid.write("{} set_name_here_in_one_word\n".format(no_cluster))
                prefix = os.path.join(dirname, "person_{}".format(no_cluster))
                for idx in cluster:
                    image_name = "{}-{}.{}".format(prefix, idx, extension)
                    cv2.imwrite(image_name, self.images[idx])
                    fid.write("{}\n".format(image_name))


if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlib",
                        type=str,
                        help="Path to shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--pwd",
                        type=str,
                        default="/tmp/profiler",
                        help="Directory where to save images (should ne absolute path)")
    parser.add_argument("--verbose",
                        type=int,
                        default=0,
                        help="Verbosity level")
    opts = parser.parse_args()

    detector = dlib.get_frontal_face_detector()

    computer = OpenfaceComputer(opts.dlib, useCuda=True)

    profiler = Profiler(low_threshold=0.3)

    # Open webcam
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Cannot open camera")
        sys.exit(1)

    while 1:
        _, frame = camera.read()

        detections = detector(frame)

        for d in detections:
            x0 = d.left()
            x1 = d.right()
            y0 = d.top()
            y1 = d.bottom()
            dx = 0.1 * (y1 - y0)
            dy = 0.1 * (x1 - x0)
            x0 = max(0, int(x0 - dx))
            x1 = min(int(x1 + dx), frame.shape[1]-1)
            y0 = max(0, int(y0 - dy))
            y1 = min(int(y1 + dy), frame.shape[0]-1)
            crop = frame[y0:y1-1,x0:x1-1] # No copy
            features, debug = computer.compute_on_image(crop, visu=1)
            if features is not None:
                profiler.add(crop, features)
            if "visu" in debug:
                cv2.imshow("crop", debug["visu"])

        # Display output
        display = frame.copy()
        for d in detections:
            bb = [d.left(), d.top(), d.left()+d.width(), d.top()+d.height()]
            cv2.rectangle(display, (bb[0],bb[1]), (bb[2],bb[3]), (255,0,0), 3)

        cv2.imshow("Camera", display)

        if cv2.waitKey(30) == 113: # 'q'
            cv2.destroyAllWindows()
            # profiler.save(opts.pwd)
            profiler.cluster(opts.pwd)
            break
