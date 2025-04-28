import os
import random
import shutil

import torch
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score, classification_report, f1_score
from sklearn.model_selection import ParameterGrid
from termcolor import colored
import csv

from conv_layer import *
from submodules import *

def main():
    seeds = [3003, 300321, 399211]
    use_gpu = 9  # number of using gpus
    device = torch.device('cuda', index=use_gpu)
    for seed_no in tqdm(range(len(seeds))):
        names = ["rf", ""] # , , auto, mlp, add, cat
        seed = seeds[seed_no]
        test_hit = 10
        test_nonhit = 240
        train_hit = 80
        train_nonhit = 140

        sample_test_path = "../../data/origin/test_index_hit_" + str(test_hit) + "_nhit_" + str(test_nonhit) \
                           + "_seed_" + str(seed) + ".csv"
        sample_hit_path = "../../data/origin/index_hit_seed_" + str(seed) + ".csv"

        with open(sample_hit_path, "r") as f:
            reader = csv.reader(f)
            hit_index = [list(map(int, row)) for row in reader]
            hit_index = list(set(hit_index[0]))
        with open(sample_test_path, "r") as f:
            reader = csv.reader(f)
            test_index = [list(map(int, row)) for row in reader]
            test_index = list(set(test_index[0]))

        train_index_pth = "../../data/origin/train_index_hit_" + str(train_hit) + "_nhit_" + \
                              str(train_nonhit) + "_seed_" + str(seed) + ".csv"

        with open(train_index_pth, "r") as f:
            reader = csv.reader(f)
            train_index = [list(map(int, row)) for row in reader]
            train_index = list(set(train_index[0]))

        if names[0] == "mlp":
            sd = str(seed)
            ker = "16"
            optim_name = "SGD"
            init_rate = "0.005"
            num_lyr = "4"
            mlp_epoch = "00190"
            feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted_normed.csv"
            with open(feat_path, "r") as f:
                reader = csv.reader(f)
                global_feat_set_tmp = [row[2:] for row in reader]
                global_feat_set = global_feat_set_tmp[1:]
                global_feat_set = [list(map(float, row)) for row in global_feat_set]
                feat_numpy = np.array(global_feat_set)
                feat_tensor = torch.tensor(feat_numpy).float()
            num_graph, num_feat = torch.tensor(feat_numpy).size()
            in_channels = num_feat
            tmp_feat = torch.zeros(1, int(ker) + 2, dtype=torch.float)
            pretrained_model_path = "../../results/mlp_models/seeds_" + sd + "_" + ker + "/" + optim_name + "_lyr_" + num_lyr + "_drop_" \
                                    + "0.5_init_rate_" + init_rate + "/model" + mlp_epoch + ".pth"
            reduced_feat, mlp_pred_label = mlp_dim_reduce(feat_tensor, in_channels, int(ker), int(num_lyr),
                                                          2, 1, pretrained_model_path, 0.5, device, tmp_feat)
            save_pth = feat_path.replace("sorted",
                                         "reduced_" + optim_name + num_lyr + "0.5" + init_rate + mlp_epoch)
            with open(save_pth, "w") as f:
                writer = csv.writer(f)
                writer.writerows(reduced_feat.to('cpu').detach().numpy().copy())
            global_feat_path = save_pth
            if names[1] == "add":
                dataset_name = "original_mlp_add_node"
                cc_node_glob = Add_node_global_feat(global_feat_path)
            elif names[1] == "cat":
                dataset_name = "original_mlp_cat_first"
                cc_node_glob = Concat_node_global_feat(global_feat_path)
            else:
                print("component of names[1] is not suitable.")
                return 0
        elif names[0] == "auto":
            sd = str(seed)
            ker = "256"
            optim_name = "SGD"
            init_rate = "0.01"
            num_lyr = "3"
            auto_epoch = "00020"
            feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted_normed.csv"
            with open(feat_path, "r") as f:
                reader = csv.reader(f)
                global_feat_set_tmp = [row[2:] for row in reader]
                global_feat_set = global_feat_set_tmp[1:]
                global_feat_set = [list(map(float, row)) for row in global_feat_set]
                feat_numpy = np.array(global_feat_set)
                feat_tensor = torch.tensor(feat_numpy).float()
            num_graph, num_feat = torch.tensor(feat_numpy).size()
            in_channels = num_feat
            tmp_feat = torch.zeros(1, int(ker) + 2, dtype=torch.float)
            pretrained_model_path = "../../results/autoencoder_models/seeds_" + sd + "_" + ker + "/" + optim_name + "_lyr_" + num_lyr + "_drop_" \
                                    + "0.5_init_rate_" + init_rate + "/model" + auto_epoch + ".pth"
            reduced_feat = AE_dim_reduce(feat_tensor, in_channels, int(ker), int(num_lyr),
                                         pretrained_model_path, 0.5, device, tmp_feat)
            save_pth = feat_path.replace("sorted",
                                         "reduced_" + optim_name + num_lyr + "0.5" + init_rate + auto_epoch)
            with open(save_pth, "w") as f:
                writer = csv.writer(f)
                writer.writerows(reduced_feat.to('cpu').detach().numpy().copy())
            global_feat_path = save_pth
            if names[1] == "add":
                dataset_name = "original_auto_add_node"
                cc_node_glob = Add_node_global_feat(global_feat_path)
            elif names[1] == "cat":
                dataset_name = "original_auto_cat_first"
                cc_node_glob = Concat_node_global_feat(global_feat_path)
            else:
                print("component of names[1] is not suitable.")
                return 0
        elif names[0] == "":
            if names[1] == "add":
                dataset_name = "original_add_node"
                global_feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted_normed.csv"
                cc_node_glob = Add_node_global_feat(global_feat_path)
            elif names[1] == "cat":
                dataset_name = "original_cat_first"
                global_feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted_normed.csv"
                cc_node_glob = Concat_node_global_feat(global_feat_path)
            elif names[1] == "":
                dataset_name = 'original'
            else:
                print("component of names[1] is not suitable.")
                return 0
        elif names[0] == "rf":
            dataset_name = "original"
        else:
            print("component of names[0] is not suitable.")
            return 0
        if names[1] == "add" or names[1] == "cat":
            dataset = TUDataset(root='../../data/', name=dataset_name, use_node_attr=True,
                                pre_transform=cc_node_glob)
        elif names[1] == "":
            dataset = TUDataset(root='../../data/', name=dataset_name, use_node_attr=True)
        if names[0] == "rf":
            feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted_normed.csv"
            with open(feat_path, "r") as f:
                reader = csv.reader(f)
                global_feat_set_tmp = [row[2:] for row in reader]
                global_feat_set = global_feat_set_tmp[1:]
                global_feat_set = [list(map(float, row)) for row in global_feat_set]
                feat_numpy = np.array(global_feat_set)
                train_feat_numpy = feat_numpy[train_index]
                train_feat_tensor = torch.tensor(train_feat_numpy).float()
                test_feat_numpy = feat_numpy[test_index]
                test_feat_tensor = torch.tensor(test_feat_numpy).float()
            num_graph, num_feat = torch.tensor(feat_numpy).size()
            gt_tmp = np.zeros((num_graph, 1), dtype=np.int32)
            gt_tmp[hit_index] = 1
            train_gt = gt_tmp[train_index]
            val_gt = gt_tmp[test_index]
            train_gt = torch.tensor(train_gt)
            val_gt = torch.tensor(val_gt)
            in_channels = num_feat
        else:
            train_data = dataset[train_index]
            test_data = dataset[test_index]
            test_feat_tensor = train_data.x
            train_feat_tensor = test_data.x
            num_graph, num_feat = test_feat_tensor.size()
            train_gt = train_data.y
            val_gt = test_data.y
            in_channels = num_feat

            print(train_data.num_node_features)
            print(train_feat_tensor.size(), train_feat_tensor)
        grids = {"pre_lyr": [3, 4],
                 "optim_name": ["Adam", "SGD"],
                 "middle_channel": [16, 32, 64, 128, 256],
                 "output_channel": [2],
                 "init_rate": [0.01, 0.005, 0.001]}
        grid = ParameterGrid(grids)
        epochs = 500
        for pre_params in grid:
            train_feat_mlp(train_feat_tensor, test_feat_tensor, train_gt, val_gt, in_channels,
                           pre_params["middle_channel"], pre_params["pre_lyr"],
                           pre_params["output_channel"], 1, 0.5, epochs, pre_params["init_rate"],
                           device, seed, pre_params["optim_name"])
    return 0

if __name__ == "__main__":
    main()