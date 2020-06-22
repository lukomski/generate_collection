from imageData import ImageData
import shutil # for copy files
import os # for check if file exists
import numpy as np # for diff vectors
import random # for random images from sack
from pathlib import Path

# generator subsets
def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1, 1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


class Sack:
    def __init__(self, imageDataList: list):
        self.imageDataList = imageDataList

    def addImage(self, imageData: ImageData):
        self.imageDataList.append(imageData)
    
    def removeImage(self, imageData: ImageData):
        v = []
        for s_img in self.imageDataList:
            if s_img.image_name != imageData.image_name:
                v.append(s_img)
        self.imageDataList = v
    def getLabeled(self, label_id):
        v = []
        for s_img in self.imageDataList:
            if label_id in s_img.getListOfLabels():
                v.append(s_img)
        return v 
    
    def getOnlyLabeled(self, label_id):
        v = []
        for s_img in self.imageDataList:
            correct_labeled = True
            for img_label_id in s_img.getListOfLabels():
                if img_label_id is not label_id:
                    correct_labeled = False
            if correct_labeled:
                v.append(s_img)
        return v
    
    def checkDuplicates(self):
        tmp = self.imageDataList[:]
        has_duplicates = False

        while len(tmp) > 0:
            # fill indexes
            idxes = [0]
            for i in range(1,len(tmp)):
                if tmp[i].image_name == tmp[0].image_name:
                    idxes.append(i)
            # remove already checked
            if len(idxes) > 1:
                print("image", tmp[0].image_name, "is on idxes = ", idxes)
                has_duplicates = True

            for idx in reversed(idxes):
                tmp.pop(idx)
        return has_duplicates

    @staticmethod
    def loadSackFromFile(filename: str):
        if not os.path.exists(filename):
            print("loadCollectionFromFile file", filename, "NOT exists")
            return Sack([])
        f = open(filename, "r")
        imageDataList = []
        for image_path in f.readlines():
            img_dir, img_name = os.path.split(os.path.abspath(image_path))
            img_dir += "/"
            img_name = img_name.split('.')[0]
            imageData = ImageData(img_dir,img_name)
            imageDataList.append(imageData)
        sack = Sack(imageDataList)
        print("loaded sack with", len(sack.imageDataList), "images")
        return sack

    def saveToFile(self, filename: str, use_absolute_path: bool):
        out_file = open(filename,"w")
        for img in self.imageDataList:
            if use_absolute_path:
                out_file.write(img.getAbsoluteImgFilePath() + '\n')
            else:
                out_file.write(img.getImgFilePath() + '\n')
        out_file.close()
        print("changes saved to", filename, "file")

    def copyImagesToFolder(self, folder_path: str) -> bool:     
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(folder_path):
            print("copyImagesToFolder:", folder_path, "not exists - abort")
            return False
        if not os.path.isdir(folder_path):
            print("copyImagesToFolder:", folder_path, "is not a folder - abort")
            return False
        if not len(os.listdir(folder_path)) == 0:
            print("copyImagesToFolder:", folder_path, "folder is NOT empty - abort")
            return False

        img_count = 0
        for imageData in self.imageDataList:
            original = imageData.getAbsoluteImgFilePath()
            target = folder_path + "/" + imageData.image_name + ".jpg"
            shutil.copyfile(original, target)
            img_count += 1
        print("copied", img_count, "to folder ", folder_path)

    def toVector(self, class_number):
        v = []
        for class_id in range(class_number):
            v.append(0)
        
        for imageData in self.imageDataList:
            labels = imageData.getListOfLabels()
            for label_id in labels:
                if int(label_id) > len(v):
                    print("wrong label_id: " + label_id)

                v[int(label_id)] += 1
        return v
    
    def getRemainedVector(self, class_number, expected_capactity):
        expected = np.full(class_number, expected_capactity)
        vector = self.toVector(class_number)
        v = []
        for i in range(len(vector)):
            v.append(expected[i] - vector[i])
        return v

    def popRandomImages(self, n: int) -> list:
        # selecte images
        selected = []
        if n > len(self.imageDataList):
            selected = self.imageDataList
        else:
            tmp = self.imageDataList[:]
            random.shuffle(tmp)
            selected = tmp[0:n]
            #selected = random.choices(self.imageDataList, k=n)
        # remove selected
        for imageData in selected:
            self.removeImage(imageData)
        return selected
    
    def findBestFit(self, ideal_vector):
        n_class = len(ideal_vector)
        the_smallest_diff = 999999999
        best_subset = []

        count_subsets = pow(2,len(self.imageDataList)) - 2
        
        i = 0
        for subset in powerset(self.imageDataList):
            i+=1
            sack = Sack(subset)
            diff = Sack.diffVectors(sack.toVector(n_class), ideal_vector)
            if diff < the_smallest_diff:
                the_smallest_diff = diff
                best_subset = subset
        return best_subset

    @staticmethod
    def diffVectors(v1: list, v2: list) -> int:
        np_v1 = np.array(v1)
        np_v2 = np.array(v2)

        v3 = np_v1 - np_v2
        v3 = np.absolute(v3)

        magnitude = np.sum(v3)
        return magnitude