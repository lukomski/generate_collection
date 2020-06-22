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

def findBetterSubset(sack: Sack, collection: Sack, opt) -> Union[Sack, Sack, bool]:
    capacity = opt.expected_capacity
    ideal_vector = [capacity for i in range(opt.n_classes)]
    name = str(os.getpid())

    # if sack.checkDuplicates():
    #     print("findBetterSubset found duplicates in sack")

    number_of_getting_images1 = 11
    number_of_getting_images2 = 3#7

    it_range = opt.iter
    said_about_error_once = False
    
    for i in range(it_range): 
        before_diff = Sack.diffVectors(collection.toVector(opt.n_classes), ideal_vector)
        # print("bef collection vector = ", collection.toVector(opt.n_classes))
        # print("bef collection:", collection.imageDataList)
        # pop random images
        imageDataList1 = sack.popRandomImages(number_of_getting_images1)      
        imageDataList2 = collection.popRandomImages(number_of_getting_images2)

        # DEBUG
        # for imageData in imageDataList1:
        #     if imageData in sack.imageDataList:
        #         print("popped image", imageData, "is in sack still")
        #     if imageData in collection.imageDataList:
        #         print("popped image from sack is in collection")
        # for imageData in imageDataList2:
        #     if imageData in collection.imageDataList:
        #         print("popped image", imageData, "is in collection still")
        #     if imageData in sack.imageDataList:
        #         print("popped image", imageData, "is in sack")
        

        
        # calculate needs
        needed_vector = collection.getRemainedVector(opt.n_classes, capacity)
        # create subsack with combinations to check
        joinedList = imageDataList1 + imageDataList2
        sub_sack = Sack(joinedList)
        # if sub_sack.checkDuplicates():
        #     print("subsack has DUPLICATES")
        #     print("imageDataList1:", imageDataList1)
        #     print("imageDataList2:", imageDataList2)
        #     return sack, collection, True
        # fast-forward if we are faraway from destination
        gell_all_made = False
        
        if needed_vector[0] > capacity / 5:
            gell_all_made = True
            subset = joinedList

        else:
            subset = sub_sack.findBestFit(needed_vector)

        # return others images to sack
        # print("joinedList = ", joinedList)
        # print("subset = ", subset)
        restList = []
        for img in joinedList:
            if img not in subset:
                restList.append(img)
        # print("restList = ", restList)

        for img in restList:
            sack.addImage(img)

        if subset == []:
            print(name + ": Warning: subset = []")

        old_collection = collection
        #add images to collection
        for imageData in subset:
            if imageData in collection.imageDataList:
                print("Try to add image ", imageData, "which already is in collection")
            else:
                collection.addImage(imageData)
        
        after_diff =  Sack.diffVectors(collection.toVector(opt.n_classes), ideal_vector)
        
        if after_diff > before_diff:
            if not gell_all_made:
                if not said_about_error_once:
                    said_about_error_once = True # to display error info once
                    print("\n" + name + ": diff has exapanded - It shouldn't take place") 
                    return sack, collection, True
        elif not gell_all_made:
            print("\r", name + ": it[",i,"/",it_range,"]" + ": diff = " + str(Sack.diffVectors(collection.toVector(opt.n_classes), ideal_vector)), end='')  
        
    return sack, collection, False

def saveDataset(collection: Sack, use_absolute_path, filename):
    out_file = open(filename,"w")
    for img in collection.imageDataList:
        if use_absolute_path:
            out_file.write(img.getAbsoluteImgFilePath() + '\n')
        else:
            out_file.write(img.getImgFilePath() + '\n')
    out_file.close()
    print("changes saved to", filename, "file")

def findBestSubDataset(imageDataList: list, opt) -> Sack:
    class_number = opt.n_classes
    capacity = opt.expected_capacity
    print("PID =", os.getpid())
    name = str(os.getpid())
    sack = Sack(imageDataList)
    print(name + ": sack.toVector() = " + str(sack.toVector(opt.n_classes)))
    if (sack.checkDuplicates()) :
        print("Loaded sack is with duplicates")

    ideal_vector = []
    for label_id in range(class_number):
        ideal_vector.append(capacity)
    print(name + ": ideal_vector = " + str(ideal_vector))
   
    collection = None
    if opt.load_collection != "":
        collection = Sack.loadSackFromFile(opt.load_collection)
        for imageData in collection.imageDataList:
            sack.removeImage(imageData)
    else:
        collection = Sack([])

    collection_vector = collection.toVector(opt.n_classes)
    print(name,": collection.toVector() = ",collection_vector,"diff = " + str(Sack.diffVectors(collection.toVector(opt.n_classes), ideal_vector))) 
    
    pool = multiprocessing.Pool(processes=opt.n_cpu)
    for epoch in range(opt.epochs):

        # prepare data for pooling
        sacks = []
        collections = []
        opts = []
        for i in range(opt.n_cpu):
            sacks.append(sack)
            collections.append(collection)
            opts.append(opt)

        # check diff before calculations
        before_diff = Sack.diffVectors(collection.toVector(opt.n_classes), ideal_vector)

        # order calculations
        results =  pool.starmap(findBetterSubset, zip(sacks, collections, opts))

        print("") # new line
        # gather results
        for result in results:
            tmp_sack = result[0]
            tmp_collection = result[1]
            tmp_critical_error = result[2]
            if tmp_critical_error:
                print("CRITICAL ERROR")
                exit(0)
            tmp_diff = Sack.diffVectors(tmp_collection.toVector(opt.n_classes), ideal_vector)
            old_diff = Sack.diffVectors(collection.toVector(opt.n_classes), ideal_vector)
            if old_diff > tmp_diff:
                collection = tmp_collection
                sack = tmp_sack
                print("Better solution collection.toVector() = " + str(collection.toVector(opt.n_classes)) + " diff = " + str(Sack.diffVectors(collection.toVector(opt.n_classes), ideal_vector)))  

        print("Epoch ", epoch, " ended")
        saveDataset(collection, opt.use_absolute_path, opt.out_file)

        # check if everything is all right
        if collection.checkDuplicates():
            exit(0)

        if Sack.diffVectors(collection.toVector(opt.n_classes), ideal_vector) == 0:
            print("Search dataset success!!!")
            break


    return collection

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
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--folder_with_files", type=str, default="../data/rozszerzone/v1/train/", help="path to folder with images and labels")
    parser.add_argument("--out_file", type=str, default="out.txt", help="file with paths selected by the program")
    parser.add_argument("--override_out_file", type=bool, default=False, help="enable override output file")
    parser.add_argument("--use_absolute_path", type=bool, default=False, help="use absolute paths in out data")
    parser.add_argument("--expected_capacity", type=int, default=100, help="expected capacity of each class probe")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--iter", type=int, default=100, help="iterations in each epoch")
    parser.add_argument("--load_collection", type=str, default="", help="path to saved collection to begin from")
    parser.add_argument("--n_classes", type=int, default=14, help="quantity of classes in dataset")


    opt = parser.parse_args()
    print(opt)

    mypath = opt.folder_with_files
    if mypath[-1] != "/":
        mypath.append('/')

    # all images
    imageDataList = getValidImageDatas(mypath)
    sack = Sack(imageDataList)
    
    
    findBestSubDataset(imageDataList, opt)
    

main()