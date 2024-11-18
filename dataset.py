import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import read_tiff
import numpy as np
import torchvision
import torchvision.transforms as transforms
import scanpy as sc
from utils import get_data
import os
import glob
import PIL
from PIL import Image
import pandas as pd
import scprep as scp
from PIL import ImageFile
import seaborn as sns
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

BCELL = ['CD19', 'CD79A', 'CD79B', 'MS4A1']
TUMOR = ['FASN']
CD4T = ['CD4']
CD8T = ['CD8A', 'CD8B']
DC = ['CLIC2', 'CLEC10A', 'CD1B', 'CD1A', 'CD1E']
MDC = ['LAMP3']
CMM = ['BRAF', 'KRAS']
IG = {'B_cell': BCELL, 'Tumor': TUMOR, 'CD4+T_cell': CD4T, 'CD8+T_cell': CD8T, 'Dendritic_cells': DC,
      'Mature_dendritic_cells': MDC, 'Cutaneous_Malignant_Melanoma': CMM}
MARKERS = []
for i in IG.values():
    MARKERS += i
LYM = {'B_cell': BCELL, 'CD4+T_cell': CD4T, 'CD8+T_cell': CD8T}


class ViT_HEBCT_multilabel(torch.utils.data.Dataset):

    def __init__(self, train=True, val=False, ds=None, sr=False, fold=0):  # root, df, transform,
        self.centers_dir = '/projects/illinois/vetmed/cb/kwang222/pythonDataAnalyze/spot_coordinates/filtered_spot_coordinates'
        self.labels_dir = '/projects/illinois/vetmed/cb/kwang222/pythonDataAnalyze/sample_binarized_labels/coord_binarized_labels'
        self.img_dir = '/projects/illinois/vetmed/cb/kwang222/pythonDataAnalyze/sample_images'
        self.new_images_dir = '/projects/illinois/vetmed/cb/kwang222/pythonDataAnalyze/new_sample_images'
        self.r = 224 // 4  # this is the image "radius" around the center

        # images are named HE_BTxxxxx_xx.jpg or HE_BCxxxxx_xx.jpg, so we want to remove the first 5 characters when
        # working with the names
        # img_names = os.listdir(self.img_dir)
        # img_names = [name.split(".", 1)[0] for name in img_names]  # get just the patient section of the names

        patient_names = os.listdir(self.new_images_dir)
        test_patient_names = [patient_names[fold]]
        train_patient_names = list(set(patient_names) - set(test_patient_names))
        validation_patient_names = []


        train_patient_names = train_patient_names[0:19]
        validation_patient_names = train_patient_names[19:22]

        test_img_names = []
        for patient in test_patient_names:
            files = os.listdir(self.new_images_dir + '/' + patient)
            for file in files:
                test_img_names.append(file.split(".", 1)[0])

        train_img_names = []
        for patient in train_patient_names:
            files = os.listdir(self.new_images_dir + '/' + patient)
            for file in files:
                train_img_names.append(file.split(".", 1)[0])

        validation_img_names = []
        for patient in validation_patient_names:
            files = os.listdir(self.new_images_dir + '/' + patient)
            for file in files:
                validation_img_names.append(file.split(".", 1)[0])

        # to get the patient images, I will just get the first one for now. Since it'll be easier.
        # So, I will divide the images by the patients. The image dict will get the name of the full name of the image?

        # The names for the patient are up to 10 characters. So, do [0:10] to get them

        # After getting the patient names, then construct another dict/array of just the actual image names.

        # sample_names = img_names  # [0:65]
        #
        # test_names = [sample_names[fold]]
        # train_names = list(set(sample_names) - set(test_names))

        # for testing
        if train:
            if val:
                img_names = validation_img_names
            else:
                img_names = train_img_names  # sample_names[0:30]
        else:
            if val:
                img_names = validation_img_names
            else:
                img_names = test_img_names  # sample_names[30:39]

        # img_names_test = img_names[:3]
        # samples = names[1:7]
        #
        # te_names = [samples[fold]]
        # tr_names = list(set(samples)-set(te_names))

        # I just need to set up the image and center dicts
        # load pillow images into torch tensors
        print('Loading imgs...')
        PIL.Image.MAX_IMAGE_PIXELS = None
        self.img_dict = {name: torch.Tensor(np.array(self.get_img(name))) for name in img_names}

        # set up a dictionary for the labels corresponding to each sample (which means each dictionary entry will
        # hold the labels for all the spots in a sample).
        # this dictionary will store the labels that correspond to the spots in a sample
        self.cells_dict = {name: self.get_labels(name).to_numpy().astype(np.float32) for name in img_names}

        # set up dictionary for the centers of the spots for each sample
        self.centers_dict = {name: np.floor(self.get_centers(name).to_numpy()).astype(int) for name in img_names}

        # set up dictionary for the locations of the spots for each sample
        self.locations_dict = {name: (self.get_centers(name).to_numpy()) / 160 for name in img_names}

        self.id2name = dict(enumerate(img_names))

    # this transforms one sample image into its respective patches for the spots
    def __getitem__(self, index):
        i = index
        img = self.img_dict[self.id2name[i]]
        img_perm = img.permute(1, 0, 2)
        cell_types = self.cells_dict[self.id2name[i]]
        centers = self.centers_dict[self.id2name[i]]
        locs = self.locations_dict[self.id2name[i]]
        positions = torch.LongTensor(locs)
        patch_dim = 3 * self.r * self.r * 4

        n_patches = len(centers)

        patches = torch.zeros((n_patches, patch_dim))
        cell_types_tensor = torch.Tensor(cell_types)

        for i in range(n_patches):
            center = centers[i]
            x, y = center
            patch = img_perm[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
            patches[i] = patch.flatten()

        return patches, positions, cell_types_tensor  # torch.Tensor(centers)

    def __len__(self):
        # temporary, will likely change this to the total number of spots across all samples?
        return len(self.cells_dict)

    def get_img(self, name):
        img_path = self.img_dir + "/" + name + ".jpg"
        print("loading image: " + name)
        img = Image.open(img_path)

        return img

    def get_centers(self, name):
        use_name = name[5:]
        path = self.centers_dir + "/" + "filtered_spots_BC" + use_name + ".csv"

        df = pd.read_csv(path, index_col=0)

        # convert dataframe to numpy array using .to_numpy() and access the individual centers using [i]
        # or just take the floor of the entire numpy array and change the type to int.

        return df

    def get_labels(self, name):
        use_name = name[5:]

        path = self.labels_dir + "/" + "filtered_BC" + use_name + "_binarized_union_data" + ".csv"

        df = pd.read_csv(path, index_col=0)

        return df


class STDataset(torch.utils.data.Dataset):
    """Some Information about STDataset"""

    def __init__(self, adata, img_path, diameter=177.5, train=True):
        super(STDataset, self).__init__()

        self.exp = adata.X.toarray()
        self.im = read_tiff(img_path)
        self.r = np.ceil(diameter / 2).astype(int)
        self.train = train
        # self.d_spot = self.d_spot if self.d_spot%2==0 else self.d_spot+1
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
        self.centers = adata.obsm['spatial']
        self.pos = adata.obsm['position_norm']

    def __getitem__(self, index):
        exp = self.exp[index]
        center = self.centers[index]
        x, y = center
        patch = self.im.crop((x - self.r, y - self.r, x + self.r, y + self.r))
        exp = torch.Tensor(exp)
        mask = exp != 0
        mask = mask.float()
        if self.train:
            patch = self.transforms(patch)
        pos = torch.Tensor(self.pos[index])
        return patch, pos, exp, mask

    def __len__(self):
        return len(self.centers)


class HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, fold=0):
        super(HER2ST, self).__init__()
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224 // 2
        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train

        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names[1:33]
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))
        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            names = te_names
        print('Loading imgs...')
        self.img_dict = {i: self.get_img(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}

        # self.gene_set = self.get_overlap(self.meta_dict,gene_list)
        # print(len(self.gene_set))
        # np.save('data/her_hvg',self.gene_set)
        # quit()
        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = 0
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]

        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        # if self.cls or self.train==False:

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x - self.r, y - self.r, x + self.r, y + self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)

        if self.train:
            return patch, loc, exp
        else:
            return patch, loc, exp, torch.Tensor(center)

    def __len__(self):
        return self.cumlen[-1]

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_lbl(self, name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)

        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        self.max_x = 0
        self.max_y = 0
        loc = meta[['x', 'y']].values
        self.max_x = max(self.max_x, loc[:, 0].max())
        self.max_y = max(self.max_y, loc[:, 1].max())
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)


class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, fold=0):
        super(ViT_HER2ST, self).__init__()

        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224 // 4

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.sr = sr

        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names[1:7]

        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1, 0, 2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y

            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat((positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
                    x += 56
                y += 56

            centers = centers[1:, :]
            positions = positions[1:, :]

            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            return patches, positions, centers

        else:
            n_patches = len(centers)

            patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)

            for i in range(n_patches):
                center = centers[i]
                print(im.shape)
                print(center)
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            if self.train:
                return patches, positions, exps
            else:
                return patches, positions, exps, torch.Tensor(centers)

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv.gz'
        df = pd.read_csv(path, sep='\t', index_col=0)

        return df

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_lbl(self, name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)

        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)


class ViT_SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, aug=False, norm=False, fold=0):
        super(ViT_SKIN, self).__init__()

        self.dir = '/ibex/scratch/pangm0a/spatial/data/GSE144240_RAW/'
        self.r = 224 // 4

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i + '_ST_' + j)
        test_names = ['P2_ST_rep2']

        # gene_list = list(np.load('data/skin_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/skin_hvg_cut_1000.npy', allow_pickle=True))
        # gene_list = list(np.load('figures/mse_2000-vit_skin_a.npy',allow_pickle=True))

        self.gene_list = gene_list

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.ToTensor()
        ])
        self.norm = norm

        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            # names = names
            # names = names[3:]
            # names = test_names
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = test_names
            names = te_names

        print('Loading imgs...')
        if self.aug:
            self.img_dict = {i: self.get_img(i) for i in names}
        else:
            self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {
                i: sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))) for
                i, m in self.meta_dict.items()}
        else:
            self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for
                             i, m in self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        if self.aug:
            im = self.transforms(im)
            # im = im.permute(1,2,0)
            im = im.permute(2, 1, 0)
        else:
            im = im.permute(1, 0, 2)
            # im = im

        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y

            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat((positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
                    x += 56
                y += 56

            centers = centers[1:, :]
            positions = positions[1:, :]

            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            return patches, positions, centers

        else:
            n_patches = len(centers)

            patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

                if self.train:
                    return patches, positions, exps
                else:
                    return patches, positions, exps, torch.Tensor(centers)

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        path = glob.glob(self.dir + '*' + name + '.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = glob.glob(self.dir + '*' + name + '_stdata.tsv')[0]
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        path = glob.glob(self.dir + '*spot*' + name + '.tsv')[0]
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'), how='inner')

        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)


class SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, fold=0):
        super(SKIN, self).__init__()

        self.dir = '/ibex/scratch/pangm0a/spatial/data/GSE144240_RAW/'
        self.r = 224 // 2

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i + '_ST_' + j)
        test_names = ['P2_ST_rep2']

        gene_list = list(np.load('data/skin_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list

        self.train = train
        self.sr = sr

        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            # names = names
            # names = names[3:]
            # names = test_names
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = test_names
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i: self.get_img(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = 0
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]

        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x - self.r, y - self.r, x + self.r, y + self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)

        if self.train:
            return patch, loc, exp
        else:
            return patch, loc, exp, torch.Tensor(center)

    def __len__(self):
        return self.cumlen[-1]

    def get_img(self, name):
        path = glob.glob(self.dir + '*' + name + '.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = glob.glob(self.dir + '*' + name + '_stdata.tsv')[0]
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        path = glob.glob(self.dir + '*spot*' + name + '.tsv')[0]
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'), how='inner')

        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)


if __name__ == '__main__':
    # dataset = VitDataset(diameter=112,sr=True)
    dataset = ViT_HER2ST(train=True, mt=False)
    # dataset = ViT_SKIN(train=True,mt=False,sr=False,aug=False)

    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    # print(dataset[0][3].shape)
    # print(dataset.max_x)
    # print(dataset.max_y)
    # print(len(dataset.gene_set))
    # np.save('data/her_g_list.npy',dataset.gene_set)
