from model import Generator
from utils import LoadModel, Denormalize, MakeGrid, SeedEverything, Denormalize
from evaluator import EvaluationModel
from dataset import ICLEVRLoader
from valid import TestAccuracy

from warnings import filterwarnings

import torch
from torch.utils.data import DataLoader


filterwarnings("ignore")
SeedEverything(7049)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

testSet    = ICLEVRLoader("../label", "../image", mode="test")
testLoader = DataLoader(testSet , 32, shuffle=False)

modelPath = "../model/SAGAN_512_Std_NoRes_247_0.81944.pth"
gen = LoadModel(modelPath, Generator(128, 512, 24, "LeakyReLU", True, True, False).to(device))
eva = EvaluationModel("../model/eval_model/classifier_weight.pth", device)

acc, imgs = TestAccuracy(gen, eva, testLoader, 128, device)
print("Making grid image ...", end="")
MakeGrid(Denormalize(imgs), "test.png")
print("Done !")
print(f"Accuracy = {acc :.2f}")