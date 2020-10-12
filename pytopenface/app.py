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
import logging

import cv2
import dlib

from .pyopenface import OpenfaceComputer
from .pyopenface import enlarge_dlib_bounding_box
from .reidentifier import Reidentifier
import pytopenface.utils as U

WINDOW_OFFSET_X = 700
WINDOW_OFFSET_Y = 100
CANDIDATES_WIN_NAME = "Candidates"
REID_WIN_NAME = "Reidentification"


class ReidentificationResult(object):
    """
    """
    def __init__(self):
        """Constructor

        Args:
            image      : Image on which reidentification is performed
            identity   : An integer
            name       : A string
            candidates : List of images that match (distance < reid_threshold)
            distances  : Corresponding distance from input image
            position   : Position of this cropped image in the full image

        """
        self.image        = None
        self.identity     = -1
        self.name         = ""
        self.candidates   = []
        self.distances    = []
        self.position    = (0, 0)

    def __repr__(self):
        return ('identity: {identity}, name: {name}, '
                + 'position: ({position_x}, {position_y}), '
                + 'distances: {distances}').format(
                        identity=self.identity,
                        name=self.name,
                        position_x=self.position[0],
                        position_y=self.position[1],
                        distances=self.distances)


class ReidentificationApplication(object):
    """Application that detects faces and reidentifies them from a face
    book of face images. The input image is taken from the webcam.

    """

    def __init__(self,
                 face_book,
                 dlib_model,
                 source=0,
                 reid_threshold=0.8,
                 use_cuda=True,
                 enlarge=0.2,
                 nb_to_show=2,
                 record="",
                 mirror=False):
        """Constructor

        Args:
            face_book       : Path to text file containing absolute
            dlib_model      : Path to shape_predictor_68_face_landmarks.dat
            source          : Camera ID for OpenCV or filename of image paths
            reid_threshold  : Path below which re-identification is performed
            use_cuda        : Use GPU if GPU found, otherwise CPU
            enlarge         : Factor to enlarge face to display profile
            nb_to_show      : Nb of candidates to show (0 means not shown)
            record          : Path to directory which to save results to
            mirror          : Mirror the input frame if true

        Example:
            'face_book' is the path to a file that can contain:

            name Name_of_person_1
            /absolute/path/to/images/person_1_1.png
            /absolute/path/to/images/person_1_2.jpg
            /absolute/path/to/images/person_1_3.jpg
            name Name_of_person_2
            /absolute/path/to/images/person_2_1.png
            name Name_of_person_3
            /absolute/path/to/images/person_3_1.png
            /absolute/path/to/images/person_3_2.png
            /absolute/path/to/images/person_3_3.png
            /absolute/path/to/images/person_3_4.png

        """
        self.source = 0
        self.index = -1
        self.filenames = []

        # If using directory of images
        if os.path.isfile(source):
            with open(source) as fp:
                for line in fp:
                    self.filenames.append(line.strip())
            logging.info("Loaded {} images".format(len(self.filenames)))
            self.index = 0
            self.source = -1
        else:
            # If using webcam
            self.source = int(source)

        self.mirror = mirror

        self.nb_to_show = nb_to_show
        self.show_size = (200, 200)

        self.record = record
        if len(record) > 0 and not os.path.exists(record):
            logging.info("Saving to {}".format(record))
            os.makedirs(record)

        self.reidentifier = Reidentifier(reid_threshold=reid_threshold)
        self.detector = dlib.get_frontal_face_detector()
        self.create_computer(dlib_model, use_cuda)

        self.id_to_name = {}

        self._load_face_book(face_book, enlarge)


    def create_computer(self, dlib_model, use_cuda):
        self.computer = OpenfaceComputer(dlib_model, useCuda=use_cuda)

    def _add_new_image_for_id(self, ID, image_path, enlarge = 0.2):
        """Add a new image for person with given ID

        Args:
            ID         :
            image_path : Absolute path to image
            enlarge    : Percentage to add before cropping around face

        """
        image = None
        if not os.path.exists(image_path):
            logging.warn("No file {}".format(image_path))
            return

        image = U.load_via_pil(image_path)

        if image is None:
            logging.warn("Problem with image {}".format(path))
            return

        # Detect face by up scaling once
        detections = self.detector(image, 1)

        if len(detections) == 0: return

        # Should only get one face, but if many, take the first one
        d = detections[0]

        x0, x1, y0, y1 = enlarge_dlib_bounding_box(d, image, enlarge)

        crop = image[y0:y1,x0:x1] # No copy

        output = self.computer.compute_on_image(crop, visu=1)

        if output.features is not None:
            self.reidentifier.add(output.features, ID, crop)


    def save_result(self, image):
        """Save input image to file name self.record / time_in_ns.png"""
        t = long(1000000*time.time())
        image_name = "{}.png".format(t)
        image_path = os.path.join(self.record, image_name)
        cv2.imwrite(image_path, image)


    def _load_face_book(self, face_book, enlarge):
        """Read file of faces and fill the reidentifier"""
        persons = {}
        with io.open(face_book, mode="r", encoding="utf-8") as fid:
            name = ""
            for line in fid:
                line = line.strip() # Remove newline
                if len(line) == 0: continue
                if line[0] == "#": continue

                tok = line.split(" ")
                if len(tok) > 1 and tok[0] == "name":
                    name = " ".join(tok[1:])
                else:
                    if len(name) == 0:
                        logging.error("Invalid file {}. Line '{}' should start with 'name'".format(face_book, line))
                        sys.exit(1)

                    if name not in persons:
                        persons[name] = []

                    logging.debug(u"Adding image for {} {}".format(name, line))
                    persons[name].append(line)
        logging.info("{} different identities loaded.".format(len(persons.keys())))

        # Read images of person and compute openface features
        for ID, (name, path_list) in enumerate(persons.items()):
            for path in path_list:
                self._add_new_image_for_id(ID, path, enlarge)
                self.id_to_name[ID] = name

        # The default identity when no math is -1
        self.id_to_name[-1] = "Unknown"


    def identity_crop_image(self, crop, x, y):
        """Identify who the person is from input image

        Args:
            crop : A face image
            x, y: where is the crop in the full image

        Returns:
            result : A ReidentificationResult element

        """
        result = ReidentificationResult()

        result.image = crop.copy()
        result.position = (x, y)

        features = self.computer.compute_on_image(crop, visu=1)

        if features.features is not None:
            reid = self.reidentifier.identify(features.features)
            result.identity = reid.ID

            # Iterate in increasing order of distance from input image
            for i, (_, distance, idx_in_gallery) in enumerate(reid.distances):
                if i >= self.nb_to_show: break
                result.candidates.append \
                    (cv2.resize(self.reidentifier.images[idx_in_gallery],
                                self.show_size))
                result.distances.append(distance)

        return result

    def plot_identity(self, detection, clean_frame, plotted_frame):
        x0, x1, y0, y1 = enlarge_dlib_bounding_box(detection, plotted_frame, 0.1)
        crop = clean_frame[y0:y1, x0:x1] # No copy

        reid = self.identity_crop_image(
                crop,
                detection.left() + int(detection.width() / 2),
                detection.top() + int(detection.height() / 2))
        reid.name = self.id_to_name[reid.identity]

        bb = [
                detection.left(),
                detection.top(),
                detection.left() + detection.width(),
                detection.top() + detection.height()
        ]
        U.draw_reidentification_results(plotted_frame, reid, bb)

        return reid

    def run(self):
        """Infinite loop to reidentify detected face in camera"""
        logging.debug("Using camera {}".format(self.source))

        cv2.namedWindow(CANDIDATES_WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(CANDIDATES_WIN_NAME, WINDOW_OFFSET_X, WINDOW_OFFSET_Y)
        cv2.namedWindow(REID_WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(REID_WIN_NAME, 10, WINDOW_OFFSET_Y)


        cam = None
        if self.source >= 0:
            cam = cv2.VideoCapture(self.source)
            if not cam.isOpened():
                logging.error("Cannot open camera {}".format(self.source))
                sys.exit(1)


        while 1:
            if self.source >= 0:
                _, frame = cam.read()
            else:
                frame = cv2.imread(self.filenames[self.index])
                self.index += 1
                self.index = self.index % len(self.filenames)

            if self.mirror:
                frame = cv2.flip(frame, 1)

            detections = self.detector(frame)

            display = frame.copy()

            if len(self.record) == 0:
                cv2.putText(display, "Press 'q' to quit", (5, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(0,0,255), thickness = 1)

            reid_results = []

            for d in detections:
                reid_results.append(self.plot_identity(d, frame, display))

            cv2.imshow(
                CANDIDATES_WIN_NAME,
                U.draw_reidentification_candidates(
                    reid_results, size=self.show_size,
                    nb_to_show=self.nb_to_show))

            if len(self.record) > 0:
                self.save_result(display)

            cv2.imshow(REID_WIN_NAME, display)

            if cv2.waitKey(30) == ord('q'):
                break
