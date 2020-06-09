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

# generator subsets
def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1, 1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

class ImageData:
    def __init__(self, folder_path: str, image_name: str):
        self.folder_path = folder_path
        self.image_name = image_name
        self.listOfLabels = []
        self.readListOfLabels()

    def readListOfLabels(self):
        labels = []
        for line in open(self.getLabelFilePath()):
            words = line.split(' ')
            if len(words) > 0:
                labels.append(int(words[0]))
            else:
                print("Warning: " + self.image_name + " cannot get label from line")
        self.listOfLabels = labels

    def getImgFilePath(self) -> str:
        return self.folder_path + self.image_name + ".jpg"

    def getLabelFilePath(self) -> str:
        return self.folder_path + self.image_name + ".txt"

    def getAbsoluteImgFilePath(self) -> str:
        path = self.folder_path + self.image_name + ".jpg"
        return os.path.abspath(path)

    def getAbsoluteLabelFilePath(self) -> str:
        path = self.folder_path + self.image_name + ".txt"
        return os.path.abspath(path) 
    
    def getListOfLabels(self) -> list:
        return self.listOfLabels

    def toVector(self, class_number) -> list:
        v = []
        for class_id in range(class_number):
            count = self.getListOfLabels().count(class_id)
            v.append(count)
        return v

    def __repr__(self) -> str:
        return self.image_name

    def __ne__(self, other):
        return self.image_name == other.image_name

class Sack:
    def __init__(self, imageDataList: list, class_number: int, capacity: int):
        self.imageDataList = imageDataList
        self.class_number = class_number
        self.capacity = capacity

    def remove(self, imageData: ImageData):
        self.removeImages([imageData])
    
    def getRandom(self, n) -> list:
        if len(self.imageDataList) < n:
            return self.imageDataList
        selected = self.imageDataList.choices(list, k=n)
        return selected
    
    def toVector(self):
        v = []
        for class_id in range(self.class_number):
            v.append(0)
        
        for imageData in self.imageDataList:
            labels = imageData.getListOfLabels()
            for label_id in labels:
                if int(label_id) > len(v):
                    print("wrong label_id: " + label_id)

                v[int(label_id)] += 1
        return v
    
    def popRandomImages(self, n):
        selected = []
        if n > len(self.imageDataList):
            selected = self.imageDataList
        else:
            selected = random.choices(self.imageDataList, k=n)
        self.removeImages(selected)
        return selected

    def addImages(self, imageDataList: list):
        for imageData in imageDataList:
            self.imageDataList.append(imageData)
    
    def findBestFit(self, ideal_vector):
        the_smallest_diff = 999999999
        best_subset = []

        count_subsets = pow(2,len(self.imageDataList)) - 2
        i = 0
        for subset in powerset(self.imageDataList):
            i+=1
            sack = Sack(subset, self.class_number, self.capacity)
            diff = Sack.diffVectors(sack.toVector(), ideal_vector)
            if diff < the_smallest_diff:
                the_smallest_diff = diff
                best_subset = subset
        return best_subset

    def removeImages(self, imageDataList: list):
        v = []
        for s_img in self.imageDataList:
            still_in = True
            for o_img in imageDataList:
                if s_img.image_name == o_img.image_name:
                    still_in = False
                    break
            if still_in:
                v.append(s_img)
        self.imageDataList = v

    @staticmethod
    def diffVectors(v1: list, v2: list) -> int:
        np_v1 = np.array(v1)
        np_v2 = np.array(v2)

        v3 = np_v1 - np_v2
        v3 = np.absolute(v3)

        magnitude = np.sum(v3)
        return magnitude

    #TMP
    @staticmethod
    def diffVectorsWithDebug(v1: list, v2: list) -> int:
        np_v1 = np.array(v1)
        print("np_v1:",np_v1)
        np_v2 = np.array(v2)
        print("np_v2:",np_v2)

        v3 = np_v1 - np_v2
        print("v3:",v3)
        v3 = np.absolute(v3)
        print("abs v3:",v3)

        magnitude = np.sum(v3)
        print("magnitude",magnitude)
        return magnitude 
    
    #TMP
    def findBestFitWithDebug(self, ideal_vector, imageDataListTakenFromColl):
        the_smallest_diff = 999999999
        best_subset = []

        count_subsets = pow(2,len(self.imageDataList)) - 2
        print("count_subsets:", count_subsets)
        i = 0
        for subset in powerset(self.imageDataList):
            i+=1
            sack = Sack(subset, self.class_number, self.capacity)
            vec = sack.toVector()     
            if len(subset) == len(imageDataListTakenFromColl):
                print("subset = ", subset)
                the_same = True
                for img in imageDataListTakenFromColl:
                    if img in subset:
                        the_same = False
                if the_same:
                    print("FOUND ", subset)

                print("there is vec the same as from collection - OK")
            diff = Sack.diffVectors(sack.toVector(), ideal_vector)
            if diff < the_smallest_diff:
                the_smallest_diff = diff
                best_subset = subset
                print("better subset:", sack.toVector(), "diff",diff)
        return best_subset

class Collection(Sack):
    def getExpectedVector(self):
        return [self.capacity for i in range(self.class_number)]

    def getRemaindedVector(self):
        expected = self.getExpectedVector()
        vector = self.toVector()
        v = []
        for i in range(len(vector)):
            v.append(expected[i] - vector[i])
        return v

    @staticmethod
    def loadCollectionFromFile(filename, class_number, capacity):
        if not os.path.exists(filename):
            print("loadCollectionFromFile file", filename, "NOT exists")
            return Collection([], class_number, capacity)
        f = open(filename, "r")
        imageDataList = []
        for image_path in f.readlines():
            img_dir, img_name = os.path.split(os.path.abspath(image_path))
            img_dir += "/"
            img_name = img_name.split('.')[0]
            imageData = ImageData(img_dir,img_name)
            imageDataList.append(imageData)
        collection = Collection(imageDataList,class_number, capacity)
        print("loaded collection with",len(imageDataList),"images",collection.toVector())
        return collection

def findBetterSubset(sack: Sack, collection: Collection, opt) -> Union[Sack, Collection]:
    class_number = collection.class_number
    capacity = collection.capacity
    ideal_vector = [capacity for i in range(class_number)]
    name = str(os.getpid())

    number_of_getting_images1 = 11
    number_of_getting_images2 = 3#7

    it_range = opt.iter
    said_about_error_once = False
    for i in range(it_range): 
        before_diff = Sack.diffVectors(collection.toVector(), ideal_vector)
        # pop random images
        imageDataList1 = sack.popRandomImages(number_of_getting_images1)
        imageDataList2 = collection.popRandomImages(number_of_getting_images2)

        # calculate needs
        needed_vector = collection.getRemaindedVector()
        # create subsack with combinations to check
        joinedList = imageDataList1 + imageDataList2
        sub_sack = Sack(joinedList, class_number, capacity)
        # fast-forward if we are faraway from destination
        gell_all_made = False
        if needed_vector[0] > capacity / 5:
            gell_all_made = True
            subset = joinedList

        else:
            subset = sub_sack.findBestFit(needed_vector)

        # return others images to sack
        for img in subset:
            joinedList.remove(img)
        sack.addImages(joinedList)

        if subset == []:
            print(name + ": Warning: subset = []")

        old_collection = collection
        collection.addImages(subset)
        after_diff =  Sack.diffVectors(collection.toVector(), ideal_vector)

        if after_diff > before_diff:
            if not gell_all_made:
                if not said_about_error_once:
                    said_about_error_once = True # to display error info once
                    print("\n" + name + ": diff has exapanded - It shouldn't take place")
                    #TMP
                    # print("DEBUG")
                    # print("before_diff was ", before_diff, "after_diff is", after_diff)
                    # print("needed_vector: ", needed_vector)
                    # print("vector tagen from collection:",Sack(imageDataList2,class_number,capacity).toVector())
                    # sub_sack.findBestFitWithDebug(needed_vector, imageDataList2)
                    # return
            collection = old_collection     
        elif not gell_all_made:
            print("\r", name + ": it[",i,"/",it_range,"]" + ": diff = " + str(Sack.diffVectors(collection.toVector(), ideal_vector)), end='')  
        
    return sack, collection

def saveDataset(collection: Collection, use_absolute_path, filename):
    out_file = open(filename,"w")
    for img in collection.imageDataList:
        if use_absolute_path:
            out_file.write(img.getAbsoluteImgFilePath() + '\n')
        else:
            out_file.write(img.getImgFilePath() + '\n')
    out_file.close()
    print("changes saved to", filename, "file")

def findBestSubDataset(imageDataList: list, class_number: int, opt) -> Collection:
    capacity = opt.expected_capacity
    print("PID =", os.getpid())
    name = str(os.getpid())
    sack = Sack(imageDataList, class_number, capacity)
    print(name + ": sack.toVector() = " + str(sack.toVector()))

    ideal_vector = []
    for label_id in range(class_number):
        ideal_vector.append(capacity)
    print(name + ": ideal_vector = " + str(ideal_vector))

    collection = None
    if opt.load_collection != "":
        collection = Collection.loadCollectionFromFile(opt.load_collection, class_number, opt.expected_capacity)
        sack.removeImages(collection.imageDataList)
    else:
        collection = Collection([], class_number, capacity)

    collection_vector = collection.toVector()
    print(name,": collection.toVector() = ",collection_vector,"diff = " + str(Sack.diffVectors(collection.toVector(), ideal_vector))) 

    pool = multiprocessing.Pool(processes=opt.n_cpu)
    for epoch in range(opt.epochs):
        sacks = []
        collections = []
        opts = []
        for i in range(opt.n_cpu):
            sacks.append(sack)
            collections.append(collection)
            opts.append(opt)
            
        before_diff = Sack.diffVectors(collection.toVector(), ideal_vector)

        results =  pool.starmap(findBetterSubset, zip(sacks, collections, opts))
        print("") # new line
        for result in results:
            tmp_sack = result[0]
            tmp_collection = result[1]
            tmp_diff = Sack.diffVectors(tmp_collection.toVector(), ideal_vector)
            old_diff = Sack.diffVectors(collection.toVector(), ideal_vector)
            if old_diff > tmp_diff:
                collection = tmp_collection
                sack = tmp_sack
                print("Better solution collection.toVector() = " + str(collection.toVector()) + " diff = " + str(Sack.diffVectors(collection.toVector(), ideal_vector)))  



        print("Epoch ", epoch, " ended")
        saveDataset(collection, opt.use_absolute_path, opt.out_file)

        if Sack.diffVectors(collection.toVector(), ideal_vector) == 0:
            print("Search dataset success!!!")
            break


    return collection

         

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--folder_with_files", type=str, default="data/rozszerzone/v1/train/", help="path to folder with images and labels")
    parser.add_argument("--out_file", type=str, default="out.txt", help="file with paths selected by the program")
    parser.add_argument("--override_out_file", type=bool, default=False, help="enable override output file")
    parser.add_argument("--use_absolute_path", type=bool, default=False, help="use absolute paths in out data")
    parser.add_argument("--class_path", type=str, default="/home/jan/Dokumenty/Studia/Sem6/PD/Kurzynski/znaki.names", help="path to class label file")
    parser.add_argument("--expected_capacity", type=int, default=100, help="expected capacity of each class probe")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--iter", type=int, default=100, help="iterations in each epoch")
    parser.add_argument("--load_collection", type=str, default="", help="path to saved collection to begin from")


    opt = parser.parse_args()
    print(opt)

    mypath = opt.folder_with_files
    if mypath[-1] != "/":
        mypath.append('/')

    out_file_name = opt.out_file
    if not opt.override_out_file and os.path.exists(out_file_name):
        print("The output file exists. Script blocked NOT to override the file.\n")
        parser.print_help()
        exit(1)
    
    # count clases
    if not os.path.exists(opt.class_path):
        print("class path not exists")
        exit(1)
    class_number = sum(1 for line in open(opt.class_path))
    # end of count classes
   
    # create list with valid images
    out_file_paths = []
    for file in listdir(mypath):
        if not isfile(join(mypath, file)):
            continue

        imageData = ImageData(str(mypath), file[0:-4])

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
    imageDataList = out_file_paths
    # end of create list with valid images

    
    findBestSubDataset(imageDataList,class_number, opt)
    

main()