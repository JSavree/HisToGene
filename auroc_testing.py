import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAUROC
from torchmetrics.classification import BinaryAUROC
from utils import *
from vis_model import HisToGene
import warnings
from dataset import ViT_HER2ST, ViT_SKIN, ViT_HEBCT_multilabel
from tqdm import tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


MODEL_PATH = ''


def model_predict_AUROC(model, test_loader, adata=None, attention=True, device=torch.device('cpu')):
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, exp, center in tqdm(test_loader):

            patch, position = patch.to(device), position.to(device)

            pred = model(patch, position)

            torch.tensor()

            if preds is None:
                preds = pred.squeeze()
                ct = center
                gt = exp
            else:
                preds = torch.cat((preds, pred), dim=0)
                ct = torch.cat((ct, center), dim=0)
                gt = torch.cat((gt, exp), dim=0)

    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    adata = preds
    # adata = ann.AnnData(preds)
    # adata.obsm['spatial'] = ct
    #
    # adata_gt = ann.AnnData(gt)
    # adata_gt.obsm['spatial'] = ct

    return adata, ct, gt #, adata_gt


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_results = []
    for fold in range(20):
        # fold=30
        # tag = '-vit_skin_aug'
        # tag = '-cnn_her2st_785_32_cv'
        # tag = '-vit_her2st_785_32_cv'
        # tag = '-cnn_skin_134_cv'
        # tag = '-vit_skin_134_cv'
        tag = '-htg_hebct_14_32_cv'
        ds = 'HER2'
        # ds = 'Skin'
        fold = 0
        print('Loading model ...')
        # model = STModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = VitModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = STModel.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt")
        model = HisToGene.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt",n_layers=8, n_genes=14, learning_rate=3e-5)
        model = model.to(device)
        # model = torch.nn.DataParallel(model)
        print('Loading data ...')

        # g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        # g = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))

        # dataset = SKIN(train=False,ds=ds,fold=fold)
        dataset = ViT_HEBCT_multilabel(train=False,sr=True,fold=fold)
        # dataset = ViT_SKIN(train=False,mt=False,sr=False,fold=fold)
        # dataset = VitDataset(diameter=112,sr=True)

        test_loader = DataLoader(dataset, batch_size=1, num_workers=2)
        print('Making prediction ...')

        adata_pred, ct, gt = model_predict_AUROC(model, test_loader, attention=False) # , adata
        # print(adata_pred)
        # print(adata_pred.shape)
        # print(adata_pred[0])
        # print(adata_pred[:, 0])
        # print(ct.shape)
        # print(gt.shape)
        # print(gt[:, 0])
        # adata_pred = sr_predict(model,test_loader,attention=True)

        metric = BinaryAUROC(thresholds=None)
        # use this [[i for i in range(len(el)) if el[i]==1] for el in one_hot_1]
        # to convert from my one-hot labels back to index labels
        results = []
        for loop in np.arange(14):
            cell_type_i_preds = torch.Tensor(adata_pred[:, loop])
            cell_type_i_gt = torch.Tensor(gt[:, loop])
            result = metric(cell_type_i_preds, cell_type_i_gt)
            results.append(result.item())

        all_results.append(results)
        # these results will be the AUC scores for fold = 0. Now I repeat for like 20 folds?
    # all results is 20 x 14
    all_results = np.array(all_results)
    plotting_results = []
    for loop in np.arange(14):
        cell_i_result = all_results[:, loop]
        plotting_results.append(cell_i_result)

    # I want to plot the AUC scores according to cell type. E.g., cell type across the different models
    plt.figure()
    plt.violinplot(plotting_results, showmeans=False, showmedians=True)
    plt.xticks([y + 1 for y in range(len(plotting_results))],
               labels=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14'])
    plt.title("AUC violin plots: 20 folds")
    plt.savefig("20_folds_violin_plots.png")

    for loop in np.arange(14):
        results_to_plot = [plotting_results[loop]]
        plt.figure()
        plt.violinplot(results_to_plot, showmeans=False, showmedians=True)
        plt.xticks([1],
                   labels=['c'+str(loop)])
        plt.title("AUC violin plots: 20 folds")
        plt.savefig("celltype_"+str(loop)+"_20_folds_violin_plots.png")

    # adata_pred.var_names = g
    # print('Saving files ...')
    # adata_pred = comp_tsne_km(adata_pred,4)
    # # adata_pred = comp_umap(adata_pred)
    # print(fold)
    # print(adata_pred)
    #
    # adata_pred.write('processed/test_pred_'+ds+'_'+str(fold)+tag+'.h5ad')
    # adata_pred.write('processed/test_pred_sr_'+ds+'_'+str(fold)+tag+'.h5ad')

    # quit()

if __name__ == '__main__':
    main()

