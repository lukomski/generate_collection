import os
import sys
import argparse
import functools

from os import listdir
from os.path import isfile, join
import random
import numpy as np
import math
from itertools import chain, combinations

import multiprocessing
from itertools import repeat
import os
import time
import numpy
from typing import Union

# import my classes
from imageData import ImageData
from sack import Sack

def getValidImageDatas(folder_path):
    out_file_paths = []
    for file in listdir(folder_path):
        if not isfile(join(folder_path, file)):
            continue

        imageData = ImageData(str(folder_path), file[0:-4])
        extension = file[-3:]

        if extension != "jpg":
            continue

        if not os.path.exists(imageData.getImgFilePath()):
            print("WARNING: " + imageData.getImgFilePath() + " not not exists - omit")
            continue

        if not os.path.exists(imageData.getLabelFilePath()):
            print("WARNING: " + imageData.getLabelFilePath() + " not not exists - omit")
            continue

        labels = imageData.getListOfLabels()

        if len(labels) > 0:
            out_file_paths.append(imageData)

    print("out_file_paths = " + str(len(out_file_paths)))
    return out_file_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_with_files", type=str, default="../data/rozszerzone/v1/train/", help="path to folder with images and labels")
    parser.add_argument("--out_file", type=str, default="out.txt", help="file with paths selected by the program")
    parser.add_argument("--n_images", type=int, default=10, help="quantity of images to select")
    parser.add_argument("--class_id", type=int, default=3, help="class id to search images for")
    parser.add_argument("--n_classes", type=int, default=14, help="quantity of classes in dataset")
    parser.add_argument("--out_folder_with_selected_images", type=str, default="out", help="file with paths selected by the program")

    opt = parser.parse_args()
    print(opt)

    mypath = opt.folder_with_files
    if mypath[-1] != "/":
        mypath.append('/')
    

    # all images
    imageDataList = getValidImageDatas(mypath)
    sack = Sack(imageDataList)
 
    # filter images for only opt.class_id
    #selected_type = sack.getOnlyLabeled(opt.class_id) # max 573 for stop sign
    selected_type = sack.getLabeled(opt.class_id) # max 1215 for stop sign
    # order images to the biggest images of element
    selected_type.sort(reverse = True, key = lambda imageData : imageData.getMaxSize())
    print("n = ", len(selected_type))

    topN = selected_type[0:opt.n_images]
    s1 = Sack(topN)

    s1.copyImagesToFolder(opt.out_folder_with_selected_images)
    s1.saveToFile(opt.out_file, True)

main()