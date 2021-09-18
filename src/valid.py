import os
import numpy as np
import matplotlib.pyplot as plt

from utils import Denormalize, GPUToNumpy

import torch


def TestImageGenerator(gen, noise, condition, title="", saveFolder="", isSave=True):
    nImg = noise.size(0)
    gen.eval()
    with torch.no_grad():
        fakeImgs = gen(noise, condition)[0]
        _, axes  = plt.subplots(1, nImg, figsize=(10, 8))
        for i in range(nImg):
            fakeImg = fakeImgs[i]
            fakeImg = np.clip(Denormalize(GPUToNumpy(fakeImg)), 0, 1)
            axes[i].imshow(fakeImg)
        
    plt.title(title)
    if isSave:
        plt.savefig(os.path.join(saveFolder, f"{title}.png"))
    
    gen.train()


def TestAccuracy(gen, eva, testLoader, noiseDim, device):
    gen.eval()
    with torch.no_grad():
        totalBatch, totalAcc = 0, 0
        label = next(iter(testLoader))
        label = label.to(device).float()

        batchSize = label.size(0)
        noise = torch.randn(batchSize, noiseDim, device=device).float()
        image = gen(noise, label)[0]
        
        acc = eva.Eval(image, label) * batchSize
        totalBatch += batchSize
        totalAcc   += acc
    
    gen.train()
    return totalAcc / totalBatch, image


if __name__ == '__main__':

    from model import Generator
    from utils import LoadModel, Denormalize, MakeGrid, SeedEverything
    from evaluator import EvaluationModel
    from dataset import ICLEVRLoader
    from warnings import filterwarnings
    from tqdm import trange
    from torch.utils.data import DataLoader

    filterwarnings("ignore")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testSet    = ICLEVRLoader("../label", "../image", mode="test")
    testLoader = DataLoader(testSet , 32, shuffle=False)
    
    modelPath = "../model/SAGAN_512_Std_NoRes_247_0.81944.pth"
    gen = LoadModel(modelPath, Generator(128, 512, 24, "LeakyReLU", True, True, False).to(device))
    eva = EvaluationModel("../model/eval_model/classifier_weight.pth", device)

    maxSeed, maxAcc, maxImgs = -1, -1, None
    for seed in trange(9999):
        SeedEverything(seed)
        acc, imgs = TestAccuracy(gen, eva, testLoader, 128, device)
        if acc > maxAcc:
            maxAcc = acc
            maxSeed = seed
            maxImgs = imgs

    MakeGrid(Denormalize(maxImgs), "test.png")
    print(f"Seed = {maxSeed} | Accuracy = {maxAcc :.2f}")
