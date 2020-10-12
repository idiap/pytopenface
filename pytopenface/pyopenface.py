# coding=utf-8

# Copyright (c) 2018 Pete Tae-hoon Kim
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
import time

import cv2
import dlib
import numpy as np
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from collections import OrderedDict

# The point sometimes breaks... (Python 2/3?)
try:
    from .align_dlib import AlignDlib
except:
    try:
        from align_dlib import AlignDlib
    except:
        print("Cannot import AlignDlib")

import logging

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


def Conv2d(in_dim, out_dim, kernel, stride, padding):
    l = torch.nn.Conv2d(in_dim, out_dim, kernel, stride=stride, padding=padding)
    return l

def BatchNorm(dim):
    l = torch.nn.BatchNorm2d(dim)
    return l

def CrossMapLRN(size, alpha, beta, k=1.0, gpuDevice=0):
    n = nn.LocalResponseNorm(size, alpha, beta, k).cuda(gpuDevice)
    return n

def Linear(in_dim, out_dim):
    l = torch.nn.Linear(in_dim, out_dim)
    return l


class Inception(nn.Module):
    def __init__(self, inputSize, kernelSize, kernelStride, outputSize, reduceSize, pool, useBatchNorm, reduceStride=None, padding=True):
        super(Inception, self).__init__()
        #
        self.seq_list = []
        self.outputSize = outputSize

        #
        # 1x1 conv (reduce) -> 3x3 conv
        # 1x1 conv (reduce) -> 5x5 conv
        # ...
        for i in range(len(kernelSize)):
            od = OrderedDict()
            # 1x1 conv
            od['1_conv'] = Conv2d(inputSize, reduceSize[i], (1, 1), reduceStride[i] if reduceStride is not None else 1, (0,0))
            if useBatchNorm:
                od['2_bn'] = BatchNorm(reduceSize[i])
            od['3_relu'] = nn.ReLU()
            # nxn conv
            pad = int(numpy.floor(kernelSize[i] / 2)) if padding else 0
            od['4_conv'] = Conv2d(reduceSize[i], outputSize[i], kernelSize[i], kernelStride[i], pad)
            if useBatchNorm:
                od['5_bn'] = BatchNorm(outputSize[i])
            od['6_relu'] = nn.ReLU()
            #
            self.seq_list.append(nn.Sequential(od))

        ii = len(kernelSize)
        # pool -> 1x1 conv
        od = OrderedDict()
        od['1_pool'] = pool
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            od['2_conv'] = Conv2d(inputSize, reduceSize[i], (1,1), reduceStride[i] if reduceStride is not None else 1, (0,0))
            if useBatchNorm:
                od['3_bn'] = BatchNorm(reduceSize[i])
            od['4_relu'] = nn.ReLU()
        #
        self.seq_list.append(nn.Sequential(od))
        ii += 1

        # reduce: 1x1 conv (channel-wise pooling)
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            od = OrderedDict()
            od['1_conv'] = Conv2d(inputSize, reduceSize[i], (1,1), reduceStride[i] if reduceStride is not None else 1, (0,0))
            if useBatchNorm:
                od['2_bn'] = BatchNorm(reduceSize[i])
            od['3_relu'] = nn.ReLU()
            self.seq_list.append(nn.Sequential(od))

        self.seq_list = nn.ModuleList(self.seq_list)


    def forward(self, input):
        x = input

        ys = []
        target_size = None
        depth_dim = 0
        for seq in self.seq_list:
            #print(seq)
            #print(self.outputSize)
            #print('x_size:', x.size())
            y = seq(x)
            y_size = y.size()
            #print('y_size:', y_size)
            ys.append(y)
            #
            if target_size is None:
                target_size = [0] * len(y_size)
            #
            for i in range(len(target_size)):
                target_size[i] = max(target_size[i], y_size[i])
            depth_dim += y_size[1]

        target_size[1] = depth_dim
        #print('target_size:', target_size)

        for i in range(len(ys)):
            y_size = ys[i].size()
            pad_l = int((target_size[3] - y_size[3]) // 2)
            pad_t = int((target_size[2] - y_size[2]) // 2)
            pad_r = target_size[3] - y_size[3] - pad_l
            pad_b = target_size[2] - y_size[2] - pad_t
            ys[i] = F.pad(ys[i], (pad_l, pad_r, pad_t, pad_b))

        output = torch.cat(ys, 1)

        return output


class OpenFaceClassifier(nn.Module):
    """
    """
    def __init__(self, useCuda, gpuDevice=0):
        """
        """
        super(OpenFaceClassifier, self).__init__()

        self.useCuda = useCuda
        self.gpuDevice = gpuDevice

        self.layer1 = Conv2d(3, 64, (7,7), (2,2), (3,3))
        self.layer2 = BatchNorm(64)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        self.layer5 = CrossMapLRN(5, 0.0001, 0.75, gpuDevice=gpuDevice)
        self.layer6 = Conv2d(64, 64, (1,1), (1,1), (0,0))
        self.layer7 = BatchNorm(64)
        self.layer8 = nn.ReLU()
        self.layer9 = Conv2d(64, 192, (3,3), (1,1), (1,1))
        self.layer10 = BatchNorm(192)
        self.layer11 = nn.ReLU()
        self.layer12 = CrossMapLRN(5, 0.0001, 0.75, gpuDevice=gpuDevice)
        self.layer13 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        self.layer14 = Inception(192, (3,5), (1,1), (128,32), (96,16,32,64), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True)
        self.layer15 = Inception(256, (3,5), (1,1), (128,64), (96,32,64,64), nn.LPPool2d(2, (3,3), stride=(3,3)), True)
        self.layer16 = Inception(320, (3,5), (2,2), (256,64), (128,32,None,None), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True)
        self.layer17 = Inception(640, (3,5), (1,1), (192,64), (96,32,128,256), nn.LPPool2d(2, (3,3), stride=(3,3)), True)
        self.layer18 = Inception(640, (3,5), (2,2), (256,128), (160,64,None,None), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True)
        self.layer19 = Inception(1024, (3,), (1,), (384,), (96,96,256), nn.LPPool2d(2, (3,3), stride=(3,3)), True)
        self.layer21 = Inception(736, (3,), (1,), (384,), (96,96,256), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True)
        self.layer22 = nn.AvgPool2d((3,3), stride=(1,1), padding=(0,0))
        self.layer25 = Linear(736, 128)

        #
        self.resize1 = nn.UpsamplingNearest2d(scale_factor=3)
        self.resize2 = nn.AvgPool2d(4)

        #
        # self.eval()

        if useCuda > 0:
            self.cuda(gpuDevice)


    def forward(self, input):
        x = input

        # if self.useCuda and not x.data.is_cuda:
        #     x = x.cuda(self.gpuDevice)

        # if x.data.is_cuda and self.gpuDevice != 0:
        #     x = x.cuda(self.gpuDevice)

        if self.useCuda > 0:
            x = x.cuda(self.gpuDevice)

        if x.size()[-1] == 128:
            x = self.resize2(self.resize1(x))

        x = self.layer8(self.layer7(self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))))
        x = self.layer13(self.layer12(self.layer11(self.layer10(self.layer9(x)))))
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer21(x)
        x = self.layer22(x)
        x = x.view((-1, 736))

        x_736 = x

        x = self.layer25(x)
        x_norm = torch.sqrt(torch.sum(x**2, 1) + 1e-6)
        x = torch.div(x, x_norm.view(-1, 1).expand_as(x))

        return (x, x_736)


def prepare_open_face(model_file="",
                      useCuda=0,
                      gpuDevice=0,
                      useMultiGPU=False):
    """
    """
    if model_file == "":
        current_dir = os.path.dirname(os.path.realpath(__file__))
        current_dir = os.path.join(current_dir, "models")
        model_file = os.path.join(current_dir, "openface-nocuda.pth")

    model = OpenFaceClassifier(useCuda, gpuDevice)
    model.load_state_dict(torch.load(model_file))

    dir_path = os.path.dirname(os.path.realpath(__file__))

    if useMultiGPU:
        model = nn.DataParallel(model)

    model.eval()

    return model



def enlarge_bounding_box(x0, x1, y0, y1, image, percent=0.1):
    """Return the coordinates x0, x1, y0, y1 of an enlarged bounding box.

       If the enlarged dimensiosn go outside the image, coordinates
       are adapted.

       Args:
           dlib_bb: Bounding box from dlib
           image:   The image in which the bounding box is
           percent: How large to increase

       Returns:
           x0, x1, y0, y1

    """
    dx = percent * (x1 - x0)
    dy = percent * (y1 - y0)
    x0 = max(0, int(x0 - dx))
    x1 = min(int(x1 + dx), image.shape[1]-1)
    y0 = max(0, int(y0 - dy))
    y1 = min(int(y1 + dy), image.shape[0]-1)
    return x0, x1, y0, y1


def enlarge_dlib_bounding_box(dlib_bb, image, percent=0.1):
    """Calls enlarge_bounding_box on a dlib bounding box"""
    x0 = dlib_bb.left()
    x1 = dlib_bb.right()
    y0 = dlib_bb.top()
    y1 = dlib_bb.bottom()
    return enlarge_bounding_box(x0, x1, y0, y1, image, percent)


class OpenfaceOutput(object):
    """Class to hold the output of OpenfaceComputer.compute_on_image

    Members:
        features : Openface features of size 128
        face     : Aligned face on which features are computed
        visu     : Visualisation of the facial landmarks on the intput image

    """
    def __init__(self):
        """Constructor"""
        self.features = None
        self.face = None
        self.visu = None


class OpenfaceComputer(object):
    """
    """
    def __init__(self, lmks_name=None, useCuda=0):
        """
        """
        self.logger = logging.getLogger("openface")

        # Minimum size below which image is increased to apply face
        # detector in compute_on_image()
        self.min_size = 120

        if useCuda > 0:
            if not torch.cuda.is_available():
                self.logger.warn("Not using cuda")
                useCuda = 0

        self.openface = prepare_open_face(useCuda=useCuda)

        self.face_detector = dlib.get_frontal_face_detector()

        if lmks_name is not None:
            self.logger.info("Loading {}".format(lmks_name))
            self.aligner = AlignDlib(lmks_name)


    def compute_on_aligned_face(self, aligned_face):
        """Perform computation of features on a 96x96 face"""
        face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
        face = np.transpose(face, (2, 0, 1))
        face = face.astype(np.float32) / 255.0
        I_ = torch.from_numpy(face)
        I_ = Variable(I_, requires_grad=False)
        I_ = I_.unsqueeze(0)

        features = self.openface(I_)
        features = features[0]

        output = OpenfaceOutput()
        output.features = features.data.cpu().numpy()
        return output


    def compute_on_image(self, image, visu=0):
        """Compute the openface features on the provided image. Apply the dlib
        frontal face detector first, then the facial landmarks.

        If the input image is smaller than min_size, the detector is
        applied at multiple scales. If the face detection fails, it
        returns None, {}.

        Args:
            visu : Whether to draw results

        Returns:
            output : OpenfaceResult

        """
        detections = []
        if image.shape[0] < self.min_size or image.shape[1] < self.min_size:
            detections = self.face_detector(image, 1)
        else:
            detections = self.face_detector(image, 0)

        output = OpenfaceOutput()

        if len(detections) > 0:
            bb = detections[0]
            aligned_face, landmarks = self.aligner.align(96, image, bb)
            output = self.compute_on_aligned_face(aligned_face)

            if visu > 0:
                output.face = aligned_face.copy()
                display = image.copy()
                cv2.rectangle(display,
                              (int(bb.left()), int(bb.top())),
                              (int(bb.right()), int(bb.bottom())),
                              (255, 0, 0), 1)
                for (x,y) in landmarks:
                    cv2.circle(display, (x, y), 1, (0, 255, 0))
                output.visu = display

        return output


    # def compute_on_image2(self, image, bb=None, visu=0):
    #     """
    #     Computer features

    #     INPUTS

    #     bb (list): The bounding box [left, top, right, bottom]

    #     visu: Whether to draw results
    #     """
    #     debug = {} # The output dict

    #     if bb is None:
    #         detections = self.face_detector(image)
    #         if len(detections) > 0:
    #             # Take first detection when success
    #             dlib_bb = detections[0]
    #         else:
    #             # If no detection is made, take entire image
    #             bb = [0, 0, image.shape[0], image.shape[1]]
    #             dlib_bb = dlib.rectangle(bb[0], bb[1], bb[0]+bb[2]-1, bb[1]+bb[3]-1)
    #     else:
    #         if isinstance(bb, list):
    #             dlib_bb = dlib.rectangle(bb[0], bb[1], bb[0]+bb[2]-1, bb[1]+bb[3]-1)
    #         elif isinstance(bb, dlib.rectangle):
    #             dlib_bb = bb

    #     # Align face from landmarks
    #     aligned_face, landmarks = self.aligner.align(96, image, dlib_bb)

    #     if aligned_face is None:
    #         return debug

    #     if visu > 0:
    #         debug["face"] = aligned_face.copy()

    #     features = self.compute_on_aligned_face(aligned_face)

    #     if visu > 0:
    #         display = image.copy()
    #         for part in landmarks:
    #             cv2.circle(display, part, 1, (0,255,0))
    #         debug["visu"] = display

    #     return features, debug
