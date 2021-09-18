import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from warnings import filterwarnings

from model import Generator, Discriminator, GeneratorLoss, DiscriminatorLoss
from dataset import ICLEVRLoader
from evaluator import EvaluationModel
from utils import SeedEverything, SaveCheckPoint, MakeGrid, Denormalize
from valid import TestAccuracy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


SeedEverything(87) 
filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


LABEL_FOLDER         = "../label"
IMAGE_FOLDER         = "../image"
EVAL_MODEL_PATH      = "../model/eval_model/classifier_weight.pth"
GEN_IMAGES_FOLDER    = "../plot"
MODEL_SAVE_FOLDER    = "../model"
      
IMAGE_SIZE           = 64
NUM_CLASS            = 24
NOISE_DIMENTION      = 128
BASE_CHANNEL         = 512
GEN_ACTIVATION       = "LeakyReLU"
DIS_ACTIVATION       = "LeakyReLU"
DIS_POOLING_TYPE     = "sum"
IS_USE_ATTENTION     = True
IS_GEN_USE_SN        = True
IS_DIS_USE_SN        = True
IS_BATCH_STD         = True
IS_SKIP_CONNECTION   = False
      
EPOCHS               = 1000
BATCH_SIZE           = 64
NUM_WORKER           = 8
GEN_LEARNING_RATE    = 0.0001
DIS_LEARNING_RATE    = 0.0004
ADAM_BETAS           = (0, 0.9)
IS_USE_AUXILIARY     = True
AUXILIARY_LOSS_TYPE  = "hinge"
GEN_AUXILIARY_WEIGHT = 1.

NUM_TEST_IMAGES      = 4
MAX_IMAGE_NUM_CLASS  = 3
EPOCHS_PER_TEST      = 1
IS_SAVE_GEN_IMAGES   = True


def IsInputIndexToLoss(lossType):
    if lossType in {}:
        return True
    
    return False


def Train():
    # Device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Datasets and dataloaders:
    transforms = A.Compose([
        A.HorizontalFlip(),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    trainSet = ICLEVRLoader(LABEL_FOLDER, IMAGE_FOLDER, transforms, mode="train")
    testSet  = ICLEVRLoader(LABEL_FOLDER, IMAGE_FOLDER, mode="test")
    trainLoader = DataLoader(trainSet, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
    testLoader  = DataLoader(testSet , BATCH_SIZE, shuffle=False)

    # Models:
    gen = Generator(NOISE_DIMENTION, BASE_CHANNEL, NUM_CLASS, GEN_ACTIVATION, IS_USE_ATTENTION, IS_GEN_USE_SN, IS_SKIP_CONNECTION).to(device)
    dis = Discriminator(BASE_CHANNEL, NUM_CLASS, DIS_ACTIVATION, IS_USE_ATTENTION, IS_DIS_USE_SN, IS_BATCH_STD, IS_USE_AUXILIARY, IS_SKIP_CONNECTION, DIS_POOLING_TYPE).to(device)
    eva = EvaluationModel(EVAL_MODEL_PATH, device)

    genLossFunc = GeneratorLoss(AUXILIARY_LOSS_TYPE).to(device)
    disLossFunc = DiscriminatorLoss(AUXILIARY_LOSS_TYPE).to(device)

    genOpt = optim.Adam(gen.parameters(), lr=GEN_LEARNING_RATE, betas=ADAM_BETAS)
    disOpt = optim.Adam(dis.parameters(), lr=DIS_LEARNING_RATE, betas=ADAM_BETAS)

    # Start training:
    isInputIdx = IsInputIndexToLoss(AUXILIARY_LOSS_TYPE)
    for epoch in range(1, EPOCHS + 1):
        totalBatch, totalGenLoss, totalDisLoss = 0, 0, 0
        for image, label, index in trainLoader:
            # Preprocess data:
            batchSize = image.size(0)
            label = label.to(device)
            index = index.to(device)

            # Train discriminator:
            noise   = torch.randn(batchSize, NOISE_DIMENTION, device=device)
            realImg = image.to(device)
            fakeImg = gen(noise, label)[0].detach()
            realOut = dis(realImg, label)
            fakeOut = dis(fakeImg, label)
            disLoss = disLossFunc(realOut[0], fakeOut[0], realOut[1], fakeOut[1], index if isInputIdx else label)

            disOpt.zero_grad()
            disLoss.backward()
            disOpt.step()

            # Train generator:
            noise   = torch.randn(batchSize, NOISE_DIMENTION, device=device)
            fakeImg = gen(noise  , label)[0]
            fakeOut = dis(fakeImg, label)
            genLoss = genLossFunc(fakeOut[0], fakeOut[1], index if isInputIdx else label, GEN_AUXILIARY_WEIGHT)

            genOpt.zero_grad()
            genLoss.backward()
            genOpt.step()

            # Print message:
            totalBatch   += batchSize
            totalDisLoss += disLoss.item()
            totalGenLoss += genLoss.item()
            avgGenLoss, avgDisLoss = totalGenLoss / totalBatch, totalDisLoss / totalBatch
            print(f"\r| Epoch {epoch}/{EPOCHS} | Batch {totalBatch}/{len(trainSet)} | GenLoss = {avgGenLoss :.8f} | DisLoss = {avgDisLoss :.8f}", end="")

        # Test generator:
        if epoch % EPOCHS_PER_TEST == 0:
            acc, imgs = TestAccuracy(gen, eva, testLoader, NOISE_DIMENTION, device)
            if IS_SAVE_GEN_IMAGES: 
                MakeGrid(Denormalize(imgs), os.path.join(GEN_IMAGES_FOLDER, f"Epoch {epoch}.png"))

            print(f" => Test Acc = {acc :.8f}")
        else:
            print("")
        
        # Save checkpoint:
        name = f"{'SAGAN' if IS_USE_ATTENTION else 'SNGAN'}_{BASE_CHANNEL}_{'Std' if IS_BATCH_STD else 'NoStd'}_{'Res' if IS_SKIP_CONNECTION else 'NoRes'}_{epoch}_{acc :.5f}.pth"
        SaveCheckPoint(os.path.join(MODEL_SAVE_FOLDER, name), epoch, gen, dis, genOpt, disOpt)


if __name__ == '__main__':
    
    Train()
    















