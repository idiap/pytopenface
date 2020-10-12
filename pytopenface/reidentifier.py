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

import time
import numpy as np
import logging


class ReidentificationOutput(object):
    """Class to hold the output of Reidentifier.identify

    Members:
        votes     : Number of votes per ID
        distances : Distance
        ID        :
        time      : Time (in seconds) re-identification took

    Examples:

        After calling Reidentifier.identify(row_128), if 34 features
        of person 12 are at distance < reid_threshold, and 3 images of
        person 6 at distance < reid_threshold, then we have

            votes = { 6: 3, 12: 34 }

        and distances will be for example

            distances = [(12, 0.13, 5), ...]

        where the tuple is (ID, distance, index_in_gallery),
        meaning that the feature at position 5 in
        Reidentifier.features is at distance 0.13 from input
        image. One can access the correspondaing ID with
        Reidentifier.ids[5], the corresponding face with
        Reidentifier.images[5]. distances is sorted by inscreasing
        values of distance from input image.

    """
    def __init__(self):
        """Constructor"""
        self.votes = {}
        self.ID = -1
        self.distances = []
        self.time = 0


class Reidentifier(object):
    """Class holding a gallery of features and perform reidentification by
    comparing input to all features of the gallery.

    If the Euclidian distance between 2 features is < reid_threshold,
    the 2 features are considered as same ID.

    If the Euclidian distance between 2 features of of the same
    identity is < keep_threshold, only one of them is stored.

    """
    def __init__(self,
                 keep_threshold=0.1,
                 reid_threshold=0.35,
                 min_nb_features_per_id=20,
                 max_nb_features_per_id=200):
        """
        Args:
            keep_threshold: threshold below which a feature is not added
                            (too similar to others)
            reid_threshold: threshold below which two features are considered
                            of the same ID when reidentifying
            min_nb_features_per_id: number below which an ID is removed
                                    by clean()
            max_nb_features_per_id: number above which new features are
                                    not added
        """
        self.logger = logging.getLogger("reidentifier")

        self.ids      = None # Id corresponding to features
        self.features = None # Features
        self.images   = []   # Corresponding images for debugging

        self.keep_threshold = keep_threshold
        self.reid_threshold = reid_threshold
        self.min_nb_features_per_id = min_nb_features_per_id
        self.max_nb_features_per_id = max_nb_features_per_id


    def identify(self, row128):
        """Identify the input by comparing it to the gallery.

        Returns:
            votes: Number of votes for each identity with
                   distance < reid_threshold

        """
        output = ReidentificationOutput()

        if self.features is None: return output

        tic = time.time()
        distances = np.linalg.norm(self.features - row128, axis=1)
        sorted_indices = np.argsort(distances)

        for i, index_in_gallery in enumerate(sorted_indices):
            distance = distances[index_in_gallery]
            ID = self.ids[index_in_gallery]

            if distance < self.reid_threshold:
                if ID not in output.votes: output.votes[ID] = 0
                output.votes[ID] += 1
                output.distances.append((ID, distance, index_in_gallery))
            else:
                # Since sorted array, all remaining are farther
                break

        toc = time.time()
        output.time = toc - tic

        # print("[reidentifier] Reid {} took {}".format(self.features.shape,
        #                                               toc-tic))

        # Print for debug
        for i, index_in_gallery in enumerate(sorted_indices):
            distance = distances[index_in_gallery]
            ID = self.ids[index_in_gallery]
            self.logger.debug("ID {} at row {} at distance {}". \
                              format(ID, index_in_gallery, distance))
            if i > 15: break

        # output.ID will be the one with higher number of votes, or in
        # case of tie, the one with closest distance (element 0 from
        # votes, which may not be the one with highest votes... Maybe
        # fix that)
        maxi = -1
        ids = []
        for k, v in output.votes.items():
            if v == maxi:
                ids.append(k)
            elif v > maxi:
                ids = [k]
                maxi = v

        if len(ids) == 1:
            output.ID = ids[0]
        elif len(ids) > 1:
            output.ID = output.distances[0][0]

        return output


    def add_row(self, row128, ID, image=None):
        """Add a new feature vector (a row) to matrix 'gallery' with identity
           ID. No check is performed on the row.

        Args:
            ros128 : openface features of size 128
            ID     : corresponding ID

        """
        if self.features is None:
            self.features = row128
            self.ids = np.array([ID])
            if image is not None:
                self.images = [image]
        else:
            self.features = np.vstack((self.features, row128))
            self.ids = np.append(self.ids, ID)
            if image is not None:
                self.images.append(image)

        # if image is not None:
        #     if len(self.images) == len(seld.ids) - 1:
        #         self.images.append(image)
        #     else:
        #         self.logger.warning("Mismatch in gallery size")

        self.logger.debug("Person {} gallery {}". \
                          format(ID, self.features.shape))


    def add(self, row128, ID, image=None):
        """Add a new feature vector (a row) to matrix 'gallery' for identity
           ID.

        Args:
            ros128 : openface features of size 128
             ID    : corresponding ID

        """
        is_added = 0

        if self.features is None or self.features.size == 0:
            self.add_row(row128, ID, image)
            # self.features = row128
            # self.ids = np.array([ID])
            is_added = 1
            self.logger.debug("Adding features for {} (gallery {})". \
                              format(ID, self.features.shape))

        elif np.count_nonzero(self.ids == ID) >= self.max_nb_features_per_id:
            is_added = 0
            self.logger.debug("Enough features for {} (gallery {})". \
                              format(ID, self.features.shape))
        else:
            # If ID is too close from others, do not insert it
            distances = np.linalg.norm(self.features - row128, axis=1)
            sorted_indices = np.argsort(distances)
            # print(sorted_indices)
            # print(len(sorted_indices))
            distance = distances[sorted_indices[0]]

            if distance < self.keep_threshold:
                self.logger.debug("Not inserting person {} {:.3f} < {:.3f}". \
                                  format(ID, distance, self.keep_threshold))
            else:
                # Count how many features per ID
                # occurences = {}
                # for i in self.ids:
                #     if i not in occurences: occurences[i] = 0
                #     occurences[i] += 1
                self.add_row(row128, ID, image)
                is_added = 1
                # # if ID not in occurences:
                # self.features = np.vstack((self.features, row128))
                # self.ids = np.append(self.ids, ID)
                # self.logger.debug("Person {}: add, gallery {}". \
                #                   format(ID, self.features.shape))

        return is_added


    def merge(self, new_id, list_to_merge):
        """Set all the ids of 'list_to_merge' to the same value.

        The value used is the minimum of 'list_to_merge'.

        This function is meant for situations when 2 IDs are actually
        the same person.

        """
        self.logger.info("Merging {} into {}".format(list_to_merge, new_id))
        for theid in list_to_merge:
            self.ids[self.ids == theid] = new_id


    def get_available_ids(self):
        """Return the set of IDs which are currently in the gallery."""
        s = set([])
        if self.ids is not None:
            s = set(self.ids)
        return s


    def get_images_for_id(self, ID):
        """Return a list of images for the given ID.

        If the self.images is empty, it returns an empty list.

        """
        images = []

        n_images = len(self.images)

        if self.ids is not None and n_images > 0:
            n_ids = self.ids.shape[0]

            if n_images > 0 and n_images == n_ids:
                for i in range(n_images):
                    if self.ids[i] == ID:
                        images.append(self.images[i])

        return images


    def get_nb_features_per_id(self):
        """Return how many features are stored in the gallery per ID"""
        distrib = {}
        for i in self.ids:
            if i not in distrib:
                distrib[i] = 0
            distrib[i] += 1
        return distrib


    def print_status(self):
        """Print the number of elements in the gallery"""
        if self.ids is not None:
            self.logger.info("Gallery contains {} elements"
                             .format(len(self.ids)))
            self.logger.info("Nb of features per ID {}"
                             .format(self.get_nb_features_per_id()))

        else:
            self.logger.info("The gallery is empty")


    def remove_indices(self, indices):
        """Remove the elements of ids, features, and images with input
        indices.

        These are the indices in the list/arrays. No the IDs of the
        persons.

        Args:
            indices: A list with index to remove

        """
        self.logger.debug("IDs {}".format(self.ids.shape))
        self.logger.debug("Features {}".format(self.features.shape))
        self.logger.debug("Images {}".format(len(self.images)))
        self.logger.debug("Removing {} elements".format(len(indices)))

        self.ids = np.delete(self.ids, indices, 0)
        self.features = np.delete(self.features, indices, 0)
        self.images = [ self.images[i] for i in range(len(self.images))
                        if i not in indices]

        self.logger.debug("IDs {}".format(self.ids.shape))
        self.logger.debug("Features {}".format(self.features.shape))
        self.logger.debug("Images {}".format(len(self.images)))

    def clean(self):
        """Remove IDs which do not have enough features

        An ID is removed (features, ids, and images) when the number
        of features is lower than min_nb_features_per_id

        """
        if self.ids is None:
            return

        distrib = self.get_nb_features_per_id()

        ids_to_remove = [ ID for ID, count in distrib.items()
                          if count < self.min_nb_features_per_id ]
        self.logger.info("Removing IDs {}".format(ids_to_remove))

        indices_to_remove = [ i for i, idx in enumerate(self.ids)
                              if idx in ids_to_remove ]

        self.remove_indices(indices_to_remove)
