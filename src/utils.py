import cv2
import os
import glob
import numpy as np
import random

import torch
from torchvision.utils import make_grid, save_image


def SeedEverything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def GPUToNumpy(tensor, reduceDim=None, isSqueeze=True):
    if isSqueeze:
        if reduceDim is not None:
            return tensor.squeeze(reduceDim).cpu().detach().numpy().transpose(1, 2, 0)
        else:
            return tensor.squeeze(         ).cpu().detach().numpy().transpose(1, 2, 0)
    
    else:
        if len(tensor.shape) == 3:
            return tensor.cpu().detach().numpy().transpose(1, 2, 0)
        elif len(tensor.shape) == 4:
            return tensor.cpu().detach().numpy().transpose(0, 2, 3, 1)


def ResizeAndSave(srcFolder, dstFolder, size=(32, 32), ext="png"):
    if not os.path.isdir(dstFolder):
        os.mkdir(dstFolder)
    
    srcPaths = glob.glob(os.path.join(srcFolder, f"*.{ext}"))
    dstPaths = []
    for path in srcPaths:
        srcImg = cv2.imread(path)
        dstImg = cv2.resize(srcImg, size, interpolation=cv2.INTER_CUBIC)
        dstPath = os.path.join(dstFolder, os.path.split(path)[1])
        cv2.imwrite(dstPath, np.clip(dstImg, 0, 255))
        dstPaths.append(dstPath)
    
    return dstPaths


def Denormalize(x, mean=0.5, std=0.5):
    return x * std + mean


def GetRandomConditions(batchSize, dim, maxNum=3):
    nums = [random.sample(range(1, maxNum + 1), 1)[0] for _ in range(batchSize)]
    idxs = [random.sample(range(dim), nums[i]) for i in range(batchSize)]

    conds = np.zeros([batchSize, dim])
    for i in range(batchSize):
        conds[i, idxs[i]] = 1
    
    return conds


def SaveCheckPoint(path, epoch, gen, dis, genOpt, disOpt, scheduler=None):
    state = {
        'epoch': epoch,
        'gen': gen.state_dict(),
        'dis': dis.state_dict(),
        'genOpt': genOpt.state_dict(),
        'disOpt': disOpt.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else {}
    }
    torch.save(state, path)
    return state


def LoadModel(path, gen):
    state = torch.load(path)
    gen.load_state_dict(state['gen'])
    return gen


def MakeGrid(images, savePath):
    grid = make_grid(images, nrow=4, padding=1)
    save_image(grid, savePath)


if __name__ == '__main__':
    ResizeAndSave("../image/hr_images", "../image", (64, 64))
    # print(GetRandomConditions(4, 24, 3))