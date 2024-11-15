import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import HisToGene
from utils import *
from predict import model_predict, sr_predict
from dataset import ViT_HER2ST, ViT_SKIN, ViT_HEBCT_multilabel


def validation_test():
    fold = 5
    tag = '-htg_hebct_14_32_cv'

    model = HisToGene.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt",n_layers=8, n_genes=14, learning_rate=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ViT_HER2ST(train=False, sr=False, fold=fold)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4)



    # change this to testing, and calculate AUROC
    # We have a model loaded from a checkpoint

    # print("loading trainer")
    # trainer = pl.Trainer(max_epochs=1, accelerator="auto")  # gpus=0, ,
    # print("validating trainer")
    # trainer.validate(model=model, dataloaders=test_loader)


validation_test()

# calculating AOC should be very easy, just run the cell segmentation programs on the images.
# Read about cross validation and folding.

