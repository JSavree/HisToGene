import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import HisToGene
from utils import *
from predict import model_predict, sr_predict
from dataset import ViT_HER2ST, ViT_SKIN, ViT_HEBCT_multilabel


def train_model(fold):
    fold = fold
    tag = '-htg_hebct_14_32_cv'
    
    dataset = ViT_HEBCT_multilabel(train=True, fold=fold) # ViT_HER2ST
    train_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True) # , num_workers=4
    dataset_test = ViT_HEBCT_multilabel(train=True, val=True, sr=False, fold=fold)
    val_loader = DataLoader(dataset_test, batch_size=1, num_workers=2, shuffle=True)
    print("loading model")
    model = HisToGene(n_layers=8, n_genes=14, dim=1024, learning_rate=3e-5) #  785
    # fine-tuning from pre-trained model
    model = HisToGene.load_from_checkpoint("model/last_train_-htg_her2st_785_32_cv_5.ckpt",n_layers=8, n_genes=14, learning_rate=3e-5)
    print("loading trainer")
    trainer = pl.Trainer(max_epochs=100, accelerator="auto", check_val_every_n_epoch=5) # gpus=0, ,
    print("fitting trainer and validating")
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt")

    return


for i in range(20):
    train_model(fold=i)

    
