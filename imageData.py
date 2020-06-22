import os

class ImageData:
    def __init__(self, folder_path: str, image_name: str):
        self.folder_path = folder_path
        self.image_name = image_name
        self.listOfLabels = []
        self.listOfSizes = []
        self.readListOfLabels()

    def readListOfLabels(self):
        labels = []
        sizes = []
        for line in open(self.getLabelFilePath()):
            words = line.split(' ')
            if len(words) >= 4:
                labels.append(int(words[0]))
                width = float(words[2])
                height = float(words[3])
                sizes.append(width * height)
            else:
                print("Warning: " + self.image_name + " cannot get label from line", len(words))
        self.listOfLabels = labels
        self.listOfSizes = sizes

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

    def __eq__(self, other):
        # for ex. imageData in imageDataList
        return self.image_name == other.image_name
        
    def getMaxSize(self):
        max = 0
        for size in self.listOfSizes:
            if max < size:
                max = size
        return max