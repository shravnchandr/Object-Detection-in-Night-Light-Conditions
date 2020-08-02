from __future__ import division

import glob
import math
import os
import os.path as osp
import pickle as pkl
import random
import re
import shutil

import cv2
import dataloader
import model
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torchvision
from darknet import Darknet
from PIL import Image, ImageStat
from preprocess import inp_to_image, prep_image
from torch.autograd import Variable
from torchvision import transforms
from util import *


def lowlight(image, count, CUDA):

    data_lowlight = torch.from_numpy(image / 255.0).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)

    if CUDA:
        data_lowlight = data_lowlight.cuda().unsqueeze(0)
        DCE_net = model.enhance_net_nopool().cuda()
        DCE_net.load_state_dict(torch.load('cfg/Epoch99.pth'))
    else:
        data_lowlight = data_lowlight.unsqueeze(0)
        DCE_net = model.enhance_net_nopool()
        DCE_net.load_state_dict(torch.load(
            'cfg/Epoch99.pth', map_location=torch.device('cpu')))

    enhanced_image = DCE_net(data_lowlight)

    enhanced_image = torch.squeeze(enhanced_image)
    pil_image = transforms.ToPILImage()(enhanced_image)
    pil_image.save('test_data/temp/{}.jpg'.format(count))


def get_test_input(input_dim, CUDA):
    img = cv2.imread("data/dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_


def brightness(image):
    image = Image.fromarray(image)
    stat = ImageStat.Stat(image)
    r, g, b = stat.mean
    return math.sqrt(0.241 * (r**2) + 0.691 * (g**2) + 0.068 * (b**2))


if __name__ == '__main__':

    sec = 0
    fps = 12
    frameRate = 1 / fps
    hasFrames = True
    file_path = 'test_data/temp'
    pathOut = 'test_data/results/videos/'
    CUDA = torch.cuda.is_available()

    with torch.no_grad():
        filePath = 'test_data/videos'
        test_list = glob.glob(filePath + "/*")

        try:
            shutil.rmtree(file_path)
            os.mkdir(file_path)
        except:
            os.mkdir(file_path)

        for video in test_list:
            count = 0
            pathOut = pathOut + video.split('/')[-1]
            videoCap = cv2.VideoCapture(video)
            while hasFrames:
                videoCap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                hasFrames, image = videoCap.read()
                if hasFrames:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if brightness(image) < 50:
                        lowlight(image, count, CUDA)
                    else:
                        pil_image = Image.fromarray(image)
                        pil_image.save('test_data/temp/{}.jpg'.format(count))

                sec += frameRate
                sec = round(sec, 2)
                count += 1

    scales = "1,2,3"
    batch_size = 1
    confidence = 0.5
    nms_thesh = 0.4
    start = 0

    num_classes = 80
    classes = load_classes('data/coco.names')

    # Set up the neural network
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights("cfg/yolov3.weights")

    model.net_info["height"] = 416
    inp_dim = int(model.net_info["height"])

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    model.eval()

    # Detection phase
    imlist = [osp.join(osp.realpath('.'), file_path, img) for img in os.listdir(
        file_path) if os.path.splitext(img)[1] == '.jpg']

    batches = list(
        map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0

    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat(
            (im_batches[i * batch_size: min((i + 1) * batch_size, len(im_batches))])) for i in range(num_batches)]

    i = 0

    write = False
    model(get_test_input(inp_dim, CUDA), CUDA)

    objs = {}

    for batch in im_batches:
        if CUDA:
            batch = batch.cuda()

        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        prediction = write_results(
            prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        if type(prediction) == int:
            i += 1
            continue

        prediction[:, 0] += i * batch_size

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        i += 1

        if CUDA:
            torch.cuda.synchronize()

    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor *
                          im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor *
                          im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(
            output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(
            output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    colors = pkl.load(open("cfg/pallete", "rb"))

    def write(x, batches, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img

    list(map(lambda x: write(x, im_batches, orig_ims), output))

    det_names = pd.Series(imlist).apply(
        lambda x: "{}/{}".format('test_data/temp', x.split("/")[-1]))

    list(map(cv2.imwrite, det_names, orig_ims))

    frameArray = []
    files = [f for f in os.listdir(
        file_path) if osp.isfile(osp.join(file_path, f))]
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    for i in range(len(files)):
        filename = file_path + '/' + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        frameArray.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frameArray)):
        out.write(frameArray[i])

    out.release()

    shutil.rmtree(file_path)
    torch.cuda.empty_cache()
