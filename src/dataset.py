import json
import torch
from torch.utils import data
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np


def get_iCLEVR_data(root_folder, mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]

            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp

        return np.squeeze(img), np.squeeze(label)
    else:
        data = json.load(open(os.path.join(root_folder,'test.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]

            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp

        return None, np.squeeze(label)


class ICLEVRLoader(data.Dataset):
    def __init__(self, rootFolder, imgFolder, transform=None, mode='train'):
        self.rootFolder = rootFolder
        self.imgFolder = imgFolder
        self.mode = mode
        self.imgs, self.labels = get_iCLEVR_data(rootFolder, mode)
        self.trans = transform if transform is not None else A.Compose([A.Normalize(mean=[0.5, 0.5, 0.5], 
                                                                                    std=[0.5, 0.5, 0.5]),
                                                                        ToTensorV2()])
        self.numClasses = 24
        
        if self.mode == 'train':
            print("> Found %d images..." % (len(self.imgs)))
                
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.mode == "train":
            img = cv2.imread(os.path.join(self.imgFolder, self.imgs[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = torch.from_numpy(self.labels[i])
            return self.trans(image=img)["image"].float(), label.float(), self.GetLabelIndex(label)
        else:
            return torch.from_numpy(self.labels[i]).float()
    
    def GetLabelIndex(self, label):
        index = torch.ones(self.numClasses, dtype=torch.int64) * (-1)
        label = torch.arange(self.numClasses)[label.bool()]
        index[:label.size(0)] = label
        return index


if __name__ == '__main__':
    ds = ICLEVRLoader("../label", "../image", mode='train')
    print(ds[18][-1])