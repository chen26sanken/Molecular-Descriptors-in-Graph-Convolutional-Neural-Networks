import os
import random
import shutil
import pprint

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

def train(model, train_loader, dropout, criterion, optimizer, epoch, use_gpu, edge_info=False, norm=True):
    device = torch.device('cuda', index=use_gpu)
    model.train()
    model = model.to(device)

    true_label_trn = []
    pred_label_trn = []
    training_loss = []

    for n, data in enumerate(train_loader):
        data = data.to(device)
        # print(data.x.size())
        if edge_info:
            if norm:
                out = model(data.x, data.edge_index, data.batch, dropout, data.edge_attr, device)
            else:
                out = model(data.x, data.edge_index, data.batch, dropout, data.edge_attr)
        elif not edge_info:
            out = model(data.x, data.edge_index, data.batch, dropout)
        else:
            print("Error! edge_info is bool!")
            return 0
        loss = criterion(out, data.y)

        check_flag = False
        if check_flag:
            if epoch == 0 and n == 0:
                with open('../../result/check.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch1"])
                    writer.writerow(out)
                    writer.writerow([loss.item()])
            elif epoch == 0 and n != 0:
                with open('../../result/check.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch1"])
                    writer.writerow(out)
                    writer.writerow([loss.item()])
            elif epoch == 21:
                with open('../../result/check.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch21"])
                    writer.writerow(out)
                    writer.writerow([loss.item()])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss = loss.item()
        training_loss.append(running_loss)
        true_label_trn += data.y
        pred_label_trn += out.argmax(dim=1)

    return training_loss, true_label_trn, pred_label_trn

def test(model, loader, dropout, criterion, use_gpu, edge_info=False, norm=True):
    device = torch.device('cuda', index=use_gpu)
    model.eval()
    model = model.to(device)

    true_label_test = []
    pred_label_test = []
    testing_loss = []
    correct = 0
    test_acc = 0

    with torch.no_grad():
        for n, data in enumerate(loader):
            data = data.to(device)
            if edge_info:
                if norm:
                    out = model(data.x, data.edge_index, data.batch, dropout, data.edge_attr, device)
                else:
                    out = model(data.x, data.edge_index, data.batch, dropout, data.edge_attr)
            elif not edge_info:
                out = model(data.x, data.edge_index, data.batch, dropout)
            else:
                print("Error! edge_info is bool!")
            test_acc += (torch.argmax(out, 1).flatten()==data.y).type(torch.float64).mean().item()
            pred = out.argmax(dim=1)
            correct += int((pred==data.y).sum())
            loss = criterion(out, data.y)

            running_loss = loss.item()
            testing_loss.append(running_loss)
            true_label_test += data.y
            pred_label_test += out.argmax(dim=1)

            if n == 0:
                tmp_out = out
            else:
                tmp_out = torch.cat((tmp_out, out), dim=0)

        f1_test = f1_score(torch.tensor(true_label_test).cpu(), torch.tensor(pred_label_test).cpu(),
                           average=None, zero_division=0)
        precision_test = precision_score(torch.tensor(true_label_test, device='cpu'),
                                         torch.tensor(pred_label_test, device='cpu'),
                                         average=None, zero_division=0)
        recall_test = recall_score(torch.tensor(true_label_test, device='cpu'),
                                   torch.tensor(pred_label_test, device='cpu'),
                                   average=None, zero_division=0)
        overall_report = classification_report(torch.tensor(true_label_test, device='cpu'),
                                               torch.tensor(pred_label_test, device='cpu'),
                                               zero_division=0)

        return correct, testing_loss, tmp_out, true_label_test, pred_label_test, \
               f1_test, precision_test, recall_test, overall_report

def save_score(list, true_trn, true_val, true, pred_trn, pred_val, pred):
    trn_f1 = f1_score(torch.tensor(true_trn).cpu(), torch.tensor(pred_trn).cpu(),
                          average=None, zero_division=0)
    trn_prec = precision_score(torch.tensor(true_trn, device='cpu'),
                                   torch.tensor(pred_trn).cpu(), average=None, zero_division=0)
    trn_rec = recall_score(torch.tensor(true_trn, device='cpu'),
                               torch.tensor(pred_trn).cpu(), average=None, zero_division=0)
    list.append(["trn", trn_f1[1], trn_prec[1], trn_rec[1]])
    val_f1 = f1_score(torch.tensor(true_val).cpu(), torch.tensor(pred_val).cpu(),
                          average=None, zero_division=0)
    val_prec = precision_score(torch.tensor(true_val, device='cpu'),
                                   torch.tensor(pred_val).cpu(), average=None, zero_division=0)
    val_rec = recall_score(torch.tensor(true_val, device='cpu'),
                               torch.tensor(pred_val).cpu(), average=None, zero_division=0)
    list.append(["val", val_f1[1], val_prec[1], val_rec[1]])
    test_f1 = f1_score(torch.tensor(true).cpu(), torch.tensor(pred).cpu(),
                           average=None, zero_division=0)
    test_prec = precision_score(torch.tensor(true, device='cpu'),
                                    torch.tensor(pred).cpu(), average=None, zero_division=0)
    test_rec = recall_score(torch.tensor(true, device='cpu'),
                                torch.tensor(pred).cpu(), average=None, zero_division=0)
    list.append(["test", test_f1[1], test_prec[1], test_rec[1]])

    return list

def Main():
    flag = "compare_mlp_gcn"  # check_dataset, check_data_structure, check_model, test, toy_test, learning_mlp, use_mlp,
    # learning_autoencoder, use_autoencoder, toy_node, toy_edge, compare_mlp_gcn, compare_ae_gcn, compare_ablation,
    # test_mlp, compare_drg_gcn, test_AE
    compare_flag = "ansamble"   # , _cat_first, _cat_after, _add_node, ansamble
    use_gpu = 1     # number of using gpus
    device = torch.device('cuda', index=use_gpu)
    epochs = 300
    n_epochs = epochs + 1  # for plot
    read_mlp_epoch = 800
    n_batches = 64
    test_hit = 10
    test_nonhit = 240
    train_hit = 83
    train_nonhit = 83
    ratio_for_training = 0.5
    # manual_seed = 728349
    manual_seed = 700000    # decide initial value for network
    # manual_seed = 540220
    # seeds = [3003, 300321, 399211]   # decide dataset
    name_dataset = "original"
    ae_name_dataset = 'mix' #79784_2nd_mini_5666_AlvaDesc, wo3D_79784_2nd_mini_5666_AlvaDesc
    lr_gamma = 0.3
    dropout = 0.5
    processing_step = 5
    middle_channels = 32
    out_channels = 2
    pre_layer = 3
    n_connected_layer = 1
    pre_epochs = 400
    init_rate = 0.005
    pre_dropout = 0.5
    # now "correct" is include file name(2022/10/25)
    model_name = "PDN"  # GCN, PDN, GAT, GATv2
    model_type1 = "_dense_add_skip"  # , _normal, _dense_add_skip, _dense_cat_skip, _res_add_skip, _res_cat_skip
    model_type2 = "_6bro"    # , _1bro(only res), _2bro, _3bro, _6bro, _9bro(only dense), _12bro(only dense pre batch max), _15bro(only dense pre batch max), _18bro(only dense)
    model_type3 = "_pre_batch_act"   # , _pre_batch_act, _original_act, _pre_relu_act, _post_layer_act, _post_norm_layer_act, _pre_layer_act, _post_batch_act, _post_morm_batch_act, _pre_layer_act
    model_type4 = "_max_test"    #_max_test, _mean_test, _sum_test, _set_aggr_test, _attn_aggr_test
    model_type5 = ""    # , _2lin, _3lin (only cat after), _2lin_v1, _2lin_v2, _3lin_v1, _3lin_v2, _3lin_v3
    model_type6 = ""   # , _aggr
    model_type7 = ""
    edge_info = True    # using edge_attr(>True) or not(>False)
    sampling_flag = True    #
    sampling_each_epochs_flag = False   #sample train(>True) or not(>False) in each epochs
    use_global_feat = " "    # , _concat_first, _concat_after, _add_node
    if ae_name_dataset == "":
        global_feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted_normed.csv"
    elif ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
        global_feat_path = "../../AE_data/shared_file_79784_molecules/79784_2nd_mini_5666_AlvaDesc_normed.csv"
    elif ae_name_dataset == "wo3D_79784_2nd_mini_5666_AlvaDesc":
        global_feat_path = "../../AE_data/shared_file_79784_molecules/wo3D_79784_2nd_mini_5666_AlvaDesc_normed.csv"
    save_pred = False
    save_pred_epoch = 490
    fst_molid_overlap = False    #!!!
    use_val = True  #!!!
    val_hit = 10
    val_nonhit = 240
    attr_flag = False   # If this is "True", node feature has node position

    param_grid = {'n_hidden_layers': [3],
                  'learning_rate': [0.001],
                  'kernels': [32]}
    grid = ParameterGrid(param_grid)
    # middle_channelss = [32]
    # init_rates = [0.01]
    # mlp_epochs = ["00200"]
    middle_channelss = []
    init_rates = []
    mlp_epochs = []
    ae_epochs = ["00002"]

    detect = np.zeros((2000, 1), dtype=int)

    for seed_no in tqdm(range(len(seeds))):
        if flag == "toy_test" or flag == "toy_node" or flag == "toy_edge":
            global_feat_path = "../../data/global_test/global_feat_diff_" + str(seeds[seed_no]) + ".csv"
        elif flag == "use_mlp" or flag == "compare_mlp_gcn":
            sd = str(seeds[seed_no])
            ker = str(middle_channelss[seed_no])
            optim_name = "SGD"
            init_rate = str(init_rates[seed_no])
            num_lyr = "3"
            mlp_epoch = mlp_epochs[seed_no]
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
            pretrained_model_path = "../../results/for_GCN_dataset/nonrapped/mlp_models/seeds_" + sd + "_" + ker + "/" + optim_name + "_lyr_" + num_lyr + "_drop_" \
                                    + str(pre_dropout) + "_init_rate_" + init_rate + "_sampled/model" + mlp_epoch + ".pth"
            reduced_feat, mlp_pred_test, mlp_pred_label = mlp_dim_reduce(feat_tensor, in_channels, int(ker), int(num_lyr),
                                                          out_channels, n_connected_layer,
                                                          pretrained_model_path, pre_dropout, device, tmp_feat)
            save_pth = feat_path.replace("sorted",
                                         "reduced_" + optim_name + num_lyr + str(dropout) + init_rate + mlp_epoch)
            with open(save_pth, "w") as f:
                writer = csv.writer(f)
                writer.writerows(reduced_feat.to('cpu').detach().numpy().copy())
            global_feat_path = save_pth
        elif flag == "use_autoencoder" or flag == "compare_ae_gcn":
            sd = str(300321)
            ker = str(middle_channelss[0])
            optim_name = "Ada"
            init_rate = "0.001"
            num_lyr = "4"
            auto_epoch = ae_epochs[0]
            if ae_name_dataset == "":
                feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted_normed.csv"
                f_p_name = ""
            elif ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                feat_path = "../../AE_data/shared_file_79784_molecules/79784_2nd_mini_5666_AlvaDesc_normed.csv"
                f_p_name = "79784_"
            elif ae_name_dataset == "wo3D_79784_2nd_mini_5666_AlvaDesc":
                feat_path = "../../AE_data/shared_file_79784_molecules/wo3D_79784_2nd_mini_5666_AlvaDesc_normed.csv"
                f_p_name = "wo3D_79784_"
            with open(feat_path, "r") as f:
                reader = csv.reader(f)
                global_feat_set_tmp = [row[2:] for row in reader]
                global_feat_set = global_feat_set_tmp[1:]
                global_feat_set = [list(map(float, row)) for row in global_feat_set]
                feat_numpy = np.array(global_feat_set)
                feat_tensor = torch.tensor(feat_numpy).float()
            num_graph, num_feat = torch.tensor(feat_numpy).size()
            in_channels = num_feat
            tmp_feat = torch.zeros(1, middle_channels + 2, dtype=torch.float)
            pretrained_model_path = "../../results/nonrapped/autoencoder_models/" + f_p_name + "seeds_" + sd + "_" + ker + "/" + optim_name + "_lyr_" + num_lyr + "_drop_" \
                                    + str(pre_dropout) + "_init_rate_" + init_rate + "/model" + auto_epoch + ".pth"
            reduced_feat = AE_dim_reduce(feat_tensor, in_channels, int(ker), int(num_lyr),
                                         pretrained_model_path, pre_dropout, device, tmp_feat)
            save_pth = feat_path.replace("normed",
                                         "normed_reduced" + optim_name + num_lyr + str(dropout) + init_rate + auto_epoch)
            with open(save_pth, "w") as f:
                writer = csv.writer(f)
                writer.writerows(reduced_feat.to('cpu').detach().numpy().copy())
            global_feat_path = save_pth
        elif flag == "test_AE":
            sd = str(1010)
            ker = str(middle_channelss[0])
            optim_name = "SGD"
            init_rate = "0.01"
            num_lyr = "4"
            auto_epoch = ae_epochs[0]
            feat_path_1 = "../../RFcompare/dataset/descriptor_feature_RF_sorted_normed.csv"
            feat_path_2 = "../../AE_data/shared_file_79784_molecules/79784_2nd_mini_5666_AlvaDesc_normed.csv"
            f_p_name = "79784_"
            with open(feat_path_1, "r") as f:
                reader = csv.reader(f)
                global_feat_set_tmp = [row[2:] for row in reader]
                global_feat_set = global_feat_set_tmp[1:]
                global_feat_set = [list(map(float, row)) for row in global_feat_set]
                feat_numpy = np.array(global_feat_set)
                feat_tensor = torch.tensor(feat_numpy).float()
            with open(feat_path_2, "r") as f:
                reader = csv.reader(f)
                global_feat_set_tmp = [row[2:] for row in reader]
                global_feat_set = global_feat_set_tmp[1:]
                global_feat_set = [list(map(float, row)) for row in global_feat_set]
                feat_numpy_2 = np.array(global_feat_set)
                feat_tensor_2 = torch.tensor(feat_numpy_2).float()
            num_graph_1, num_feat = torch.tensor(feat_numpy).size()
            num_graph_2, num_feat = torch.tensor(feat_numpy_2).size()
            in_channels = num_feat
            tmp_feat = torch.zeros(1, middle_channels + 2, dtype=torch.float)
            pretrained_model_path = "../../results/nonrapped/autoencoder_models/" + f_p_name + "seeds_" + sd + "_" + ker + "/" + optim_name + "_lyr_" + num_lyr + "_drop_" \
                                    + str(pre_dropout) + "_init_rate_" + init_rate + "/model" + auto_epoch + ".pth"
            print("train mix => main and sub")
            print("main:")
            reduced_feat = AE_test(feat_tensor, in_channels, int(ker), int(num_lyr),
                                         pretrained_model_path, pre_dropout, device, tmp_feat)
            print("sub:")
            reduced_feat = AE_test(feat_tensor_2, in_channels, int(ker), int(num_lyr),
                                   pretrained_model_path, pre_dropout, device, tmp_feat)
            return 0
        elif flag == "compare_mlp_gcn":
            sd = str(seeds[seed_no])
            ker = str(middle_channelss[seed_no])
            optim_name = "SGD"
            init_rate = str(init_rates[seed_no])
            num_lyr = "3"
            mlp_epoch = mlp_epochs[seed_no]
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
            pretrained_model_path = "../../results/for_GCN_dataset/nonrapped/mlp_models/seeds_" + sd + "_" + ker + "/" + optim_name + "_lyr_" + num_lyr + "_drop_" \
                                    + str(pre_dropout) + "_init_rate_" + init_rate + "_sampled/model" + mlp_epoch + ".pth"

        if use_global_feat != " ":
            name_dataset = 'original'
            if use_global_feat == "_concat_first":
                name_dataset = "concat_nf_gf_evl"
                # name_dataset = "dum_concat_nf_gf_evl"
                if flag == "test":
                    name_dataset = "original_cat_first"
                elif flag == "use_mlp":
                    name_dataset = "original_mlp_cat_first"
                elif flag == "use_autoencoder":
                    name_dataset = "original_auto_cat_first"
                cc_node_glob = Concat_node_global_feat(global_feat_path)
            elif use_global_feat == "_concat_after":
                model_type7 = use_global_feat
                name_dataset = "concat_na_gf_evl"
                # name_dataset = "dum_concat_na_gf_evl"
                if flag == "test":
                    name_dataset = "original_cat_after"
                elif flag == "use_mlp":
                    name_dataset = "original_mlp_cat_after"
                elif flag == "use_autoencoder":
                    name_dataset = "original_auto_cat_after"
            elif use_global_feat == "_add_node":
                name_dataset = "add_node_gf_evl"
                # name_dataset = "dum_add_node_gf_evl"
                if flag == "test":
                    name_dataset = "original_add_node"
                elif flag == "use_mlp":
                    name_dataset = "original_mlp_add_node"
                elif flag == "use_autoencoder":
                    name_dataset = "original_auto_add_node"
                cc_node_glob = Add_node_global_feat(global_feat_path)
            else:
                assert False, ("you choice variables that is cannot use! please check 'use_global_feat'!!")

            if attr_flag:
                name_dataset = name_dataset + "_attr"

            dir_path = "../../data/" + name_dataset + "/processed/"
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)

            if use_global_feat == "_concat_after":
                dataset = TUDataset(root='../../data/', name=name_dataset, use_node_attr=attr_flag)
            elif use_global_feat == "_concat_first" or use_global_feat == "_add_node":
                dataset = TUDataset(root='../../data/', name=name_dataset, use_node_attr=attr_flag,
                                    pre_transform=cc_node_glob)
            else:
                print("you are miss")
        else:
            name_dataset = 'original'
            if attr_flag:
                name_dataset = name_dataset + "_attr"
            if flag == "toy_node" or flag == "toy_edge":
                name_dataset = "edge_evl_1"
            if flag == "check_supernode":
                name_dataset = "supernode_evl"

            dir_path = "../../data/" + name_dataset + "/processed/"
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
            dataset = TUDataset(root='../../data/', name=name_dataset, use_node_attr=attr_flag)

        if compare_flag != "":
            if flag != "test" and flag != "test_mlp":
                if flag == "compare_mlp_gcn":
                    com_name = "mlp_"
                elif flag == "compare_ae_gcn":
                    com_name = "auto_"
                elif flag == "compare_drg_gcn":
                    com_name = ""
                if compare_flag == "ansamble":
                    name_dataset_glob_cat_first = "original_" + com_name + "cat_first"
                    cc_node_glob = Concat_node_global_feat(global_feat_path)
                    dir_path = "../../data/" + name_dataset_glob_cat_first + "/processed/"
                    if os.path.isdir(dir_path):
                        shutil.rmtree(dir_path)
                    dataset_glob_cat_first = TUDataset(root='../../data/', name=name_dataset_glob_cat_first, use_node_attr=attr_flag,
                                             pre_transform=cc_node_glob)
                    name_dataset_glob_cat_after = "original_" + com_name + "cat_after"
                    dir_path = "../../data/" + name_dataset_glob_cat_after + "/processed/"
                    if os.path.isdir(dir_path):
                        shutil.rmtree(dir_path)
                    dataset_glob_cat_after = TUDataset(root='../../data/', name=name_dataset_glob_cat_after, use_node_attr=attr_flag)
                    name_dataset_glob_add_node = "original_" + com_name + "add_node"
                    cc_node_glob = Add_node_global_feat(global_feat_path)
                    dir_path = "../../data/" + name_dataset_glob_add_node + "/processed/"
                    if os.path.isdir(dir_path):
                        shutil.rmtree(dir_path)
                    dataset_glob_add_node = TUDataset(root='../../data/', name=name_dataset_glob_add_node, use_node_attr=attr_flag,
                                             pre_transform=cc_node_glob)
                else:
                    if compare_flag == "_concat_first":
                        name_dataset_glob = "original_" + com_name + "_cat_first"
                        cc_node_glob = Concat_node_global_feat(global_feat_path)
                    elif compare_flag == "_concat_after":
                        name_dataset_glob = "original_" + com_name + "_cat_after"
                    elif compare_flag == "_add_node":
                        name_dataset_glob = "original_" + com_name + "_add_node"
                        cc_node_glob = Add_node_global_feat(global_feat_path)
                    dir_path = "../../data/" + name_dataset_glob + "/processed/"
                    if os.path.isdir(dir_path):
                        shutil.rmtree(dir_path)

                    if compare_flag == "_concat_after":
                        dataset_glob = TUDataset(root='../../data/', name=name_dataset_glob, use_node_attr=attr_flag)
                    elif compare_flag == "_concat_first" or compare_flag == "_add_node":
                        dataset_glob = TUDataset(root='../../data/', name=name_dataset_glob, use_node_attr=attr_flag,
                                            pre_transform=cc_node_glob)

        print('\n', f'Dataset: {dataset}', '\n')
        if flag == "check_dataset":
            print(f'Number of graphs: {len(dataset)}')
            print(f'Number of node features: {dataset.num_node_features}')
            print(f'Number of feature: {dataset.num_features}')
            print(f'Dimension of edges: {dataset.num_edge_features}')
            print(f'Number of classes: {dataset.num_classes}')
            return 0
        elif flag == "check_data_structure":
            data = dataset[0]
            print(data.y)
            print(f'Number of nodes: {data.num_nodes}')
            print(f'Number of edges: {data.num_edges}')
            print(f'Has isolated nodes: {data.has_isolated_nodes()}')
            print(f'Has self-loops: {data.has_self_loops()}')
            print(f'Is undirected: {data.is_undirected()}')
            print(f'Node position: {data.pos}')
            return 0

        with open('../../data/' + name_dataset + '/raw/' + name_dataset + '_graph_labels.txt') as fp:
            l_data = fp.read()
            l_data = l_data.split("\n")

        with open('../../data/origin/testing_sorted_graph_labels') as fp:
            t_data = fp.read()
            t_data = t_data.split(" ")
            print("******************************\n" + colored("num_testing_data:", "yellow")
                  + str(len(t_data) - 1))
        seed = seeds[seed_no]
        count_true = 0
        count_false = 0
        count_hit = 0
        count_nonhit = 0
        hit_index = []
        nonhit_index = []

        recall_results = []
        recall_best_results = []

        if flag == "toy_test":
            test_dataset = dataset[:4]
            train_dataset = dataset[4:]
        elif flag == "toy_node":
            test_dataset = dataset[2:4]
            train_dataset = dataset[6:]
        elif flag == "toy_edge":
            test_dataset = dataset[:2]
            train_dataset = dataset[4:6]
        elif flag == "check_supernode":
            test_dataset = dataset[2:]
            train_dataset = dataset[:2]
        else:
            for i in range(len(l_data)-1):
                data = dataset[i]
                if data.y == int(l_data[i]):
                    count_true += 1
                else:
                    count_false += 1
                if data.y == 1:
                    hit_index.append(i)
                    count_hit += 1
                else:
                    nonhit_index.append(i)
                    count_nonhit += 1
            print(colored("count_hit_inTest=", "green"), count_hit)
            print(colored("count_nonhit_inTest=", "green"), count_nonhit)

            random.seed(seed)
            if not fst_molid_overlap:
                non_overlaped_hit_index = []
                non_overlaped_nonhit_index = []
                index_path = "../../data/origin/mol_index.csv"
                with open(index_path, "r") as f:
                    reader = csv.reader(f)
                    non_overlaped_index = [row for row in reader]
                    non_overlaped_index = np.array(non_overlaped_index)
                for non_index in non_overlaped_index:
                    non_overlaped_index_1 = [r.split(",") for r in non_index]
                for non_index in non_overlaped_index_1:
                    non_index_1 = int(non_index[0])
                    nonhit_flag = True
                    for check_index in hit_index:
                        if non_index_1 == check_index:
                            non_overlaped_hit_index.append(non_index_1)
                            nonhit_flag = False
                            break
                    if nonhit_flag:
                        non_overlaped_nonhit_index.append(non_index_1)
                print(len(non_overlaped_hit_index), len(non_overlaped_nonhit_index))
                hit_index = non_overlaped_hit_index
                nonhit_index = non_overlaped_nonhit_index
            test_hit_index = random.sample(hit_index, test_hit)
            test_nonhit_index = random.sample(nonhit_index, test_nonhit)

            print("test_hit:", len(test_hit_index), ", test_nonhit:", len(test_nonhit_index))

            test_index = sorted(test_hit_index + test_nonhit_index)

            sample_test_path = "../../data/origin/test_index_hit_" + str(len(test_hit_index)) + "_nhit_" + \
                                str(len(test_nonhit_index)) + "_seed_" + str(seed) + ".csv"
            with open(sample_test_path, "w") as cs:
                writer = csv.writer(cs)
                writer.writerow(test_index)

            sample_hit_path = "../../data/origin/index_hit_seed_" + str(seed) + ".csv"
            with open(sample_hit_path, "w") as cs:
                writer = csv.writer(cs)
                writer.writerow(hit_index)

            if use_val:
                val_hit_index = random.sample(list(set(hit_index) - set(test_hit_index)), val_hit)
                val_nonhit_index = random.sample(list(set(nonhit_index) - set(test_nonhit_index)), val_nonhit)
                val_index = sorted(val_hit_index + val_nonhit_index)
                val_dataset = dataset[val_index]
                sample_val_pth = "../../data/origin/val_index_hit_" + str(len(val_hit_index)) + "_nhit_" + \
                                    str(len(val_nonhit_index)) + "_seed_" + str(seed) + ".csv"
                with open(sample_val_pth, "w") as cs:
                    writer = csv.writer(cs)
                    writer.writerow(val_index)
                test_hit_index = sorted(test_hit_index + val_hit_index)
                test_nonhit_index = sorted(test_nonhit_index + val_nonhit_index)

            #     test_hit_index, test_nonhit_indexを書き換えているので注意！


            if sampling_flag:
                train_hit_index = random.sample(list(set(hit_index) - set(test_hit_index)), train_hit)
                train_nonhit_index = random.sample(list(set(nonhit_index) - set(test_nonhit_index)), train_nonhit)
                train_index = sorted(train_hit_index + train_nonhit_index)
                train_dataset = dataset[train_index]
                sample_train_path = "../../data/origin/train_index_hit_" + str(len(train_hit_index)) + "_nhit_" + \
                                    str(len(train_nonhit_index)) + "_seed_" + str(seed) + ".csv"
                with open(sample_train_path, "w") as cs:
                    writer = csv.writer(cs)
                    writer.writerow(train_index)
            else:
                train_hit_index = list(set(hit_index) - set(test_hit_index))
                train_nonhit_index = list(set(nonhit_index) - set(test_nonhit_index))
                train_index = sorted(train_hit_index + train_nonhit_index)
                train_dataset = dataset[train_index]
                train_path = "../../data/origin/train_index_" + str(seed) + ".csv"
                with open(train_path, "w") as cs:
                    writer = csv.writer(cs)
                    writer.writerow(train_index)

            torch.manual_seed(manual_seed)

            test_dataset = dataset[test_index]
            print("train_hit:", len(train_hit_index), ", train_nonhit:", len(train_nonhit_index))

            pre_param_grid = {'optim': ["SGD"],
                          'learning_rate': [0.01],
                          'middle_channels': [32]}
            pre_grid = ParameterGrid(pre_param_grid)

            if flag == "learning_mlp":
                if use_val:
                    mlp_train_index = train_index
                    mlp_val_index = val_index
                    mlp_test_index = test_index
                else:
                    mlp_num_hit = 103 - test_hit - train_hit
                    mlp_num_nonhit = 1897 - test_nonhit - train_nonhit
                    mlp_num_val_nonhit = 304
                    mlp_num_trn_nonhit = 120
                    train_ratio = 0.8
                    mlp_train_hit_index = random.sample(sorted(list(set(hit_index) - set(test_hit_index) - set(train_hit_index))), int(mlp_num_hit*train_ratio))
                    mlp_train_nonhit_index = random.sample(sorted(list(set(nonhit_index) - set(test_nonhit_index) - set(train_nonhit_index))), int(mlp_num_trn_nonhit))
                    mlp_train_index = sorted(mlp_train_hit_index + mlp_train_nonhit_index)
                    mlp_val_index = sorted(list(set(hit_index) - set(test_hit_index) - set(train_hit_index) - set(mlp_train_hit_index))
                                            + random.sample(sorted(list(set(nonhit_index) - set(test_nonhit_index) - set(train_nonhit_index) - set(mlp_train_nonhit_index))), int(mlp_num_val_nonhit)))
                feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted_normed.csv"
                with open(feat_path, "r") as f:
                    reader = csv.reader(f)
                    global_feat_set_tmp = [row[2:] for row in reader]
                    global_feat_set = global_feat_set_tmp[1:]
                    global_feat_set = [list(map(float, row)) for row in global_feat_set]
                    feat_numpy = np.array(global_feat_set)
                    train_feat_numpy = feat_numpy[mlp_train_index]
                    train_feat_tensor = torch.tensor(train_feat_numpy).float()
                    val_feat_numpy = feat_numpy[mlp_val_index]
                    val_feat_tensor = torch.tensor(val_feat_numpy).float()
                    if use_val:
                        test_feat_numpy = feat_numpy[mlp_test_index]
                        test_feat_tensor = torch.tensor(test_feat_numpy).float()
                num_graph, num_feat = torch.tensor(feat_numpy).size()
                gt_tmp = np.zeros((num_graph, 1), dtype=np.int32)
                gt_tmp[hit_index] = 1
                train_gt = gt_tmp[mlp_train_index]
                val_gt = gt_tmp[mlp_val_index]
                if use_val:
                    test_gt = gt_tmp[mlp_test_index]
                train_gt = torch.tensor(train_gt)
                val_gt = torch.tensor(val_gt)
                if use_val:
                    test_gt = torch.tensor(test_gt)
                in_channels = num_feat
                tmp_acc = 0
                for pre_params in pre_grid:
                    if use_val:
                        tmp_saved_data = train_feat_mlp_val(train_feat_tensor, val_feat_tensor, test_feat_tensor, train_gt, val_gt, test_gt, in_channels, pre_params["middle_channels"], pre_layer,
                                                    out_channels, n_connected_layer, pre_dropout, pre_epochs, pre_params["learning_rate"], device, seed, pre_params["optim"], seeds[seed_no], train_hit_index, train_nonhit_index, feat_numpy, gt_tmp)
                    else:
                        tmp_saved_data = train_feat_mlp(train_feat_tensor, val_feat_tensor, train_gt, val_gt, in_channels,
                                                   pre_params["middle_channels"], pre_layer,
                                                   out_channels, n_connected_layer, pre_dropout, pre_epochs,
                                                   pre_params["learning_rate"], device, seed, pre_params["optim"])
                    print(tmp_saved_data)
                    if tmp_acc < tmp_saved_data[1]:
                        tmp_acc = tmp_saved_data[1]
                        saved_data = tmp_saved_data
                        saved_data.append(seed)
                        saved_data.append(pre_params["optim"])
                        saved_data.append(pre_params["middle_channels"])
                        saved_data.append(pre_params["learning_rate"])
                        mid = pre_params["middle_channels"]
                temp_models_saved_folder = "../../data/best_results/MLP/origindata_seed_" + str(seeds[seed_no]) + "_drop_" + str(dropout)
                file_path_tmp = temp_models_saved_folder + "/lyr_" + str(pre_layer) + "_chn_" + str(mid)
                os.makedirs(temp_models_saved_folder, exist_ok=True)
                if os.path.isfile(file_path_tmp + "_all_results.csv"):
                    with open(file_path_tmp + "_all_results.csv", "a") as f:
                        writer = csv.writer(f)
                        writer.writerow(saved_data)
                else:
                    with open(file_path_tmp + "_all_results.csv", "w") as f:
                        writer = csv.writer(f)
                        writer.writerow(saved_data)
            elif flag == "learning_autoencoder":
                train_ratio = 0.8
                if ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                    AEdata_size = 79748
                    AEdata_index = np.array(range(AEdata_size))
                    auto_train_index = random.sample(list(set(AEdata_index)), int(AEdata_size*train_ratio))
                    auto_val_index = sorted(list(set(AEdata_index) - set(auto_train_index)))
                    feat_path = "../../AE_data/shared_file_79784_molecules/79784_2nd_mini_5666_AlvaDesc_normed.csv"
                    f_s_name = "79784_"
                elif ae_name_dataset == "wo3D_79784_2nd_mini_5666_AlvaDesc":
                    AEdata_size = 79748
                    AEdata_index = np.array(range(AEdata_size))
                    auto_train_index = random.sample(list(set(AEdata_index)), int(AEdata_size*train_ratio))
                    auto_val_index = sorted(list(set(AEdata_index) - set(auto_train_index)))
                    feat_path = "../../AE_data/shared_file_79784_molecules/wo3D_79784_2nd_mini_5666_AlvaDesc_normed.csv"
                    f_s_name = "wo3D_79784_"
                elif ae_name_dataset == "":
                    auto_num_hit = 103 - test_hit - train_hit
                    auto_num_nonhit = 1897 - test_nonhit - train_nonhit
                    auto_num_val_nonhit = 304
                    auto_num_trn_nonhit = 120
                    auto_train_hit_index = random.sample(
                        sorted(list(set(hit_index) - set(test_hit_index) - set(train_hit_index))),
                        int(auto_num_hit * train_ratio))
                    auto_train_nonhit_index = random.sample(
                        sorted(list(set(nonhit_index) - set(test_nonhit_index) - set(train_nonhit_index))),
                        int(auto_num_trn_nonhit))
                    auto_train_index = sorted(auto_train_hit_index + auto_train_nonhit_index)
                    auto_val_index = sorted(
                        list(set(hit_index) - set(test_hit_index) - set(train_hit_index) - set(auto_train_hit_index))
                        + random.sample(sorted(list(set(nonhit_index) - set(test_nonhit_index) - set(train_nonhit_index) - set(
                            auto_train_nonhit_index))), int(auto_num_val_nonhit)))
                    feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted_normed.csv"
                    f_s_name = "original_"
                else:
                    AEdata_size = 79748
                    AEdata_index = np.array(range(AEdata_size))
                    auto_train_index = random.sample(list(set(AEdata_index)), int(AEdata_size * train_ratio))
                    auto_val_index = sorted(list(set(AEdata_index) - set(auto_train_index)))
                    feat_path = "../../AE_data/shared_file_79784_molecules/79784_2nd_mini_5666_AlvaDesc_normed.csv"
                    with open(feat_path, "r") as f:
                        reader = csv.reader(f)
                        global_feat_set_tmp = [row[2:] for row in reader]
                        global_feat_set = global_feat_set_tmp[1:]
                        global_feat_set = [list(map(float, row)) for row in global_feat_set]
                        feat_numpy = np.array(global_feat_set)
                        train_feat_numpy = feat_numpy[auto_train_index]
                        train_feat_tensor_79 = torch.tensor(train_feat_numpy).float()
                        val_feat_numpy = feat_numpy[auto_val_index]
                        val_feat_tensor_79 = torch.tensor(val_feat_numpy).float()
                    AEdata_size += 2000
                    auto_num_hit = 103
                    auto_num_nonhit = 1897
                    auto_train_hit_index = random.sample(sorted(list(set(hit_index))), int(auto_num_hit * train_ratio))
                    auto_train_nonhit_index = random.sample(sorted(list(set(nonhit_index))), int(auto_num_nonhit * train_ratio))
                    auto_train_index = sorted(auto_train_hit_index + auto_train_nonhit_index)
                    auto_val_index = sorted(
                        list(set(hit_index) - set(auto_train_hit_index)) + list(set(nonhit_index) - set(auto_train_nonhit_index)))
                    feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted_normed.csv"
                    f_s_name = "mix_"
                with open(feat_path, "r") as f:
                    reader = csv.reader(f)
                    global_feat_set_tmp = [row[2:] for row in reader]
                    global_feat_set = global_feat_set_tmp[1:]
                    global_feat_set = [list(map(float, row)) for row in global_feat_set]
                    feat_numpy = np.array(global_feat_set)
                    train_feat_numpy = feat_numpy[auto_train_index]
                    train_feat_tensor = torch.tensor(train_feat_numpy).float()
                    val_feat_numpy = feat_numpy[auto_val_index]
                    val_feat_tensor = torch.tensor(val_feat_numpy).float()
                train_feat_tensor = torch.cat((train_feat_tensor, train_feat_tensor_79))
                val_feat_tensor = torch.cat((val_feat_tensor, val_feat_tensor_79))
                num_graph, num_feat = torch.tensor(feat_numpy).size()
                in_channels = num_feat
                tmp_loss = 100000
                for pre_params in pre_grid:
                    tmp_saved_data = train_feat_autoencoder(train_feat_tensor, val_feat_tensor, in_channels, pre_params["middle_channels"], pre_layer,
                               pre_dropout, pre_epochs, pre_params["learning_rate"], device, seed, pre_params["optim"])
                    if tmp_loss > tmp_saved_data[1]:
                        tmp_loss = tmp_saved_data[1]
                        saved_data = tmp_saved_data
                        saved_data.append(seed)
                        saved_data.append(pre_params["optim"])
                        saved_data.append(pre_params["middle_channels"])
                        saved_data.append(pre_params["learning_rate"])
                temp_models_saved_folder = "../../data/best_results/AE_mix/" + f_s_name + "lyr_" + str(
                    pre_layer) + "_drop_" + str(dropout) + "_date_" + "0120"
                file_path_tmp = temp_models_saved_folder + "/"
                os.makedirs(temp_models_saved_folder, exist_ok=True)
                if os.path.isfile(file_path_tmp + "all_results.csv"):
                    with open(file_path_tmp + "all_results.csv", "a") as f:
                        writer = csv.writer(f)
                        writer.writerow(saved_data)
                else:
                    with open(file_path_tmp + "all_results.csv", "w") as f:
                        writer = csv.writer(f)
                        writer.writerow(saved_data)
                if len(seeds) > 2:
                    if seed_no == len(seeds) - 1:
                        return 0
                    else:
                        continue
                else:
                    return 0

        print("------- about dataset --------")
        print(f'Total number of graphs: {len(dataset)}\n'
              # + f'training: {len(train_dataset)}\n'
              + f'testing: {len(test_dataset)}')

        if flag == "compare_mlp_gcn" or flag == "compare_ae_gcn" or flag == "compare_drg_gcn":
            if compare_flag == "ansamble":
                test_dataset_glob_cat_first = dataset_glob_cat_first[test_index]
                test_glob_loader_cat_first = DataLoader(test_dataset_glob_cat_first, batch_size=n_batches, shuffle=False)
                train_dataset_glob_cat_first = dataset_glob_cat_first[train_index]
                train_glob_loader_cat_first = DataLoader(train_dataset_glob_cat_first, batch_size=n_batches, shuffle=False)
                val_dataset_glob_cat_first = dataset_glob_cat_first[val_index]
                val_glob_loader_cat_first = DataLoader(val_dataset_glob_cat_first, batch_size=n_batches, shuffle=False)
                test_dataset_glob_cat_after = dataset_glob_cat_after[test_index]
                test_glob_loader_cat_after = DataLoader(test_dataset_glob_cat_after, batch_size=n_batches, shuffle=False)
                train_dataset_glob_cat_after = dataset_glob_cat_after[train_index]
                train_glob_loader_cat_after = DataLoader(train_dataset_glob_cat_after, batch_size=n_batches, shuffle=False)
                val_dataset_glob_cat_after = dataset_glob_cat_after[val_index]
                val_glob_loader_cat_after = DataLoader(val_dataset_glob_cat_after, batch_size=n_batches, shuffle=False)
                test_dataset_glob_add_node = dataset_glob_add_node[test_index]
                test_glob_loader_add_node = DataLoader(test_dataset_glob_add_node, batch_size=n_batches, shuffle=False)
                train_dataset_glob_add_node = dataset_glob_add_node[train_index]
                train_glob_loader_add_node = DataLoader(train_dataset_glob_add_node, batch_size=n_batches, shuffle=False)
                val_dataset_glob_add_node = dataset_glob_add_node[val_index]
                val_glob_loader_add_node = DataLoader(val_dataset_glob_add_node, batch_size=n_batches, shuffle=False)
            elif compare_flag != "":
                test_dataset_glob = dataset_glob[test_index]
                test_glob_loader = DataLoader(test_dataset_glob, batch_size=n_batches, shuffle=False)
                train_dataset_glob = dataset_glob[train_index]
                train_glob_loader = DataLoader(train_dataset_glob, batch_size=n_batches, shuffle=False)
                val_dataset_glob = dataset_glob[val_index]
                val_glob_loader = DataLoader(val_dataset_glob, batch_size=n_batches, shuffle=False)

        test_loader = DataLoader(test_dataset, batch_size=n_batches, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=n_batches, shuffle=False)
        if use_val:
            val_loader = DataLoader(val_dataset, batch_size=n_batches, shuffle=False)


        if flag == "compare_mlp_gcn" or flag == "compare_ae_gcn" or flag == "compare_drg_gcn":
            pram_path = "../../data/best_results/GCN/origindata_seed_" + str(seeds[seed_no]) + "_drop_" + str(dropout) + "use_val/no_globall_results.csv"
            with open(pram_path, "r") as f:
                reader = csv.reader(f)
                prams = [s for s in reader]
                prams = prams[0]
                print(prams)
                gcn_epoch = int(prams[0])
                gcn_lyr = prams[6]
                gcn_ker = prams[7]
                gcn_late = prams[8]
            model = eval(model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6)(dataset,
                        out_channels=int(gcn_ker), n_hidden_layers=int(gcn_lyr))
            full_connect_channels = ""
            if model_type4 == "2lin":
                full_connect_channels = "full_chs" + str(int(params['kernels'])) + "_" + str(
                    int(params['kernels'] / 2)) + "_2_"
            if model_type4 == "_set":
                model_type4 = model_type4 + "_2_" + str(processing_step)
            main_info = model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6 \
                        + "_sample_" + str(sampling_flag) + "_v_late_" + full_connect_channels \
                        + name_dataset + "_test_hit_" + str(test_hit) \
                        + "_data_seed" + str(seed) \
                        + "_ratio" + str(ratio_for_training) + "_knl" \
                        + str(gcn_ker) + "_lyr" + str(gcn_lyr) \
                        + "initial_lr_" + str(gcn_late)
            if attr_flag:
                main_info = main_info + "_attr_True"
            if sampling_each_epochs_flag:
                main_info = main_info + "_each_sampled"
            elif ae_name_dataset == "wo3D_79784_2nd_mini_5666_AlvaDesc":
                main_info = main_info + "_wo3D_79784AE"
            if flag == "toy_node":
                main_info = main_info + "_toy_node"
            elif flag == "toy_edge":
                main_info = main_info + "_toy_edge"
            model_path = "../../result/" + main_info + "/model(" + main_info + ")/model%05d.pth" % (int(gcn_epoch)+30)
            model.load_state_dict(torch.load(model_path))
            if compare_flag != "":
                if compare_flag == "ansamble":
                    if flag == "compare_mlp_gcn":
                        glob_type = "mlp"
                        lin_index = 1
                    elif flag == "compare_ae_gcn":
                        glob_type = "auto"
                        if ae_name_dataset == "":
                            lin_index = 1
                        elif ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                            lin_index = 0
                    elif flag == "compare_drg_gcn":
                        glob_type = "dragon"
                        lin_index = 0
                    compare = "_catfirst"
                    if seed == 23230 and flag == "compare_mlp_gcn":
                        pram_glob_path = "../../data/best_results/GCN/origindata_seed_" + str(seed) \
                                         + "_drop_0.5use_val/mlp_catfirst__dense_add_skip_6bro_pre_batch_act_max_testall_results.csv"
                        lin_index = 0
                    else:
                        pram_glob_path = "../../data/best_results/GCN/origindata_seed_" + str(
                        seeds[seed_no]) + "_drop_" + str(
                        dropout) + "use_val/" + glob_type + compare + "_all_results.csv"
                    with open(pram_glob_path, "r") as f:
                        reader = csv.reader(f)
                        prams_glob = [s for s in reader]
                        if flag == "compare_ae_gcn" and len(prams_glob) < 2:
                            print("うーん，だめっぽ")
                        elif flag == "compare_ae_gcn" and len(prams_glob) > 2:
                            for j in range(len(prams_glob) - 1):
                                main_flag = False
                                alva_flag = False
                                data = prams_glob[j + 1]
                                epo = data[0]
                                lyr = data[6]
                                chn = data[7]
                                rate = data[8]
                                tmp_main_info = "PDN_dense_add_skip_6bro_pre_batch_act_max_test_sample_True_v_late_original_" \
                                            + "auto_cat_first_test_hit_10_data_seed" + str(
                                    seed) + "_ratio0.5_knl" + chn + "_lyr" \
                                            + lyr + "initial_lr_" + rate + "_use_autoencoder"
                                main_info_alva = tmp_main_info + "_79784AE_v2"
                                path = "../../result/" + tmp_main_info + "/model(" + tmp_main_info + ")/model%05d.pth" % int(
                                    epo)
                                path_alva = "../../result/" + main_info_alva + "/model(" + main_info_alva + ")/model%05d.pth" % int(
                                    epo)
                                if os.path.isfile(path):
                                    main_flag = True
                                if os.path.isfile(path_alva):
                                    alva_flag = True
                                if main_flag and alva_flag:
                                    print("困ったなぁ、、、", name, ":", seed, ":", j)
                                    return 0
                                elif alva_flag and ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                                    lin_index = j+1
                                elif main_flag and ae_name_dataset == "":
                                    lin_index = len(prams_glob)-1
                        prams_glob = prams_glob[lin_index]
                        print("cat_first:", prams_glob)
                        gcn_epoch_glob = int(prams_glob[0])
                        gcn_lyr_glob = prams_glob[6]
                        gcn_ker_glob = prams_glob[7]
                        gcn_late_glob = prams_glob[8]
                    model_glob_cat_first = eval(
                        model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6)(
                        dataset_glob_cat_first,
                        out_channels=int(gcn_ker_glob), n_hidden_layers=int(gcn_lyr_glob))
                    full_connect_channels = ""
                    if model_type4 == "2lin":
                        full_connect_channels = "full_chs" + str(int(params['kernels'])) + "_" + str(
                            int(params['kernels'] / 2)) + "_2_"
                    if model_type4 == "_set":
                        model_type4 = model_type4 + "_2_" + str(processing_step)
                    main_info_glob = model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6 \
                                     + "_sample_" + str(sampling_flag) + "_v_late_" + full_connect_channels \
                                     + name_dataset_glob_cat_first + "_test_hit_" + str(test_hit) \
                                     + "_data_seed" + str(seed) \
                                     + "_ratio" + str(ratio_for_training) + "_knl" \
                                     + str(gcn_ker_glob) + "_lyr" + str(gcn_lyr_glob) \
                                     + "initial_lr_" + str(gcn_late_glob)
                    if flag == "compare_mlp_gcn":
                        main_info_glob = main_info_glob + "_use_mlp"
                        if seed != 23230:
                            main_info_glob = main_info_glob + "_79784AE_v2"
                    elif flag == "compare_ae_gcn":
                        main_info_glob = main_info_glob + "_use_autoencoder"
                    if attr_flag:
                        main_info_glob = main_info_glob + "_attr_True"
                    if sampling_each_epochs_flag:
                        main_info_glob = main_info_glob + "_each_sampled"
                    if ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                        main_info_glob = main_info_glob + "_79784AE_v2"
                    elif ae_name_dataset == "wo3D_79784_2nd_mini_5666_AlvaDesc":
                        main_info_glob = main_info_golb + "_wo3D_79784AE"
                    if flag == "toy_node":
                        main_info_glob = main_info_glob + "_toy_node"
                    elif flag == "toy_edge":
                        main_info_glob = main_info_glob + "_toy_edge"
                    if seed == 23230 and flag == "compare_ae_gcn":
                        gcn_epoch_glob = gcn_epoch_glob-30
                    elif int(gcn_epoch_glob) >= 270:
                        gcn_epoch_glob = 270
                    model_glob_path = "../../result/" + main_info_glob + "/model(" + main_info_glob + ")/model%05d.pth" % (
                            int(gcn_epoch_glob) + 30)
                    print(main_info_glob)
                    model_glob_cat_first.load_state_dict(torch.load(model_glob_path))

                    compare = "_catafter"
                    if (seed == 4040 and flag == "compare_mlp_gcn") or (seed >= 11110 and flag == "compare_mlp_gcn"):
                        pram_glob_path = "../../data/best_results/GCN/origindata_seed_" + str(seed) \
                                         +"_drop_0.5use_val/mlp_catafter__dense_add_skip_6bro_pre_batch_act_max_test_concat_afterall_results.csv"
                        with open(pram_glob_path, "r") as f:
                            reader = csv.reader(f)
                            prams_glob = [s for s in reader]
                            prams_glob = prams_glob[0]
                            print("cat_after:", prams_glob)
                            gcn_epoch_glob = int(prams_glob[0])
                            gcn_lyr_glob = prams_glob[6]
                            gcn_ker_glob = prams_glob[7]
                            gcn_late_glob = prams_glob[8]
                    else:
                        pram_glob_path = "../../data/best_results/GCN/origindata_seed_" + str(
                            seeds[seed_no]) + "_drop_" + str(
                            dropout) + "use_val/" + glob_type + compare + "_all_results.csv"
                        with open(pram_glob_path, "r") as f:
                            reader = csv.reader(f)
                            prams_glob = [s for s in reader]
                            if flag == "compare_ae_gcn" and len(prams_glob) < 2:
                                print("うーん，だめっぽ")
                            elif flag == "compare_ae_gcn" and len(prams_glob) > 2:
                                for j in range(len(prams_glob)):
                                    main_flag = False
                                    alva_flag = False
                                    data = prams_glob[j]
                                    epo = data[0]
                                    lyr = data[6]
                                    chn = data[7]
                                    rate = data[8]
                                    tmp_main_info = "PDN_dense_add_skip_6bro_pre_batch_act_max_test_sample_True_v_late_original_" \
                                                    + "auto_cat_after_test_hit_10_data_seed" + str(
                                        seed) + "_ratio0.5_knl" + chn + "_lyr" \
                                                    + lyr + "initial_lr_" + rate + "_use_autoencoder"
                                    main_info_alva = tmp_main_info + "_79784AE_v2"
                                    path = "../../result/" + tmp_main_info + "/model(" + tmp_main_info + ")/model%05d.pth" % int(
                                        epo)
                                    path_alva = "../../result/" + main_info_alva + "/model(" + main_info_alva + ")/model%05d.pth" % int(
                                        epo)
                                    if os.path.isfile(path):
                                        main_flag = True
                                    if os.path.isfile(path_alva):
                                        alva_flag = True
                                    if main_flag and alva_flag:
                                        print("困ったなぁ、、、", name, ":", seed, ":", j)
                                        return 0
                                    elif alva_flag and ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                                        lin_index = j+1
                                        print("alva:", lin_index)
                                    elif main_flag and ae_name_dataset == "":
                                        lin_index = len(prams_glob)
                                        print("main:", lin_index)
                            elif flag == "compare_ae_gcn" and len(prams_glob) == 2:
                                if ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                                    lin_index = 1
                                elif ae_name_dataset == "":
                                    lin_index = 2
                            if glob_type == "mlp":
                                if len(prams_glob) == 3:
                                    lin_index = 1
                                elif len(prams_glob) < 3:
                                    print(len(prams_glob), "<< なぁぜなぁぜ？")
                                    return 0
                                else:
                                    for j in range(len(prams_glob) - 2):
                                        _2lin_flag = False
                                        _3lin_flag = False
                                        data = prams_glob[j + 1]
                                        epo = data[0]
                                        lyr = data[6]
                                        chn = data[7]
                                        rate = data[8]
                                        main_info_2lin = "PDN_dense_add_skip_6bro_pre_batch_act_max_test_2lin_sample_True_v_late_original_" \
                                                         + "mlp_cat_after_test_hit_10_data_seed" + str(seed) + "_ratio0.5_knl" + chn + "_lyr" \
                                                         + lyr + "initial_lr_" + rate + "_use_mlp"
                                        path_2lin = "../../result/" + main_info_2lin + "/model(" + main_info_2lin + ")/model%05d.pth" % int(
                                            epo)
                                        if os.path.isfile(path_2lin):
                                            print("あった")
                                            _2lin_flag = True
                                        else:
                                            print("なかった")
                                            lin_index = 1
                                        if _2lin_flag and _3lin_flag:
                                            print("困ったなぁ、、、", name, ":", seed, ":", j)
                                            return 0
                                        elif _2lin_flag:
                                            lin_index = j
                                            break
                            elif glob_type == "dragon":
                                lin_index = 1
                            prams_glob = prams_glob[lin_index-1]
                            # prams_glob =prams_glob[lin_index]
                            print("cat_after:", prams_glob)
                            gcn_epoch_glob = int(prams_glob[0])
                            gcn_lyr_glob = prams_glob[6]
                            gcn_ker_glob = prams_glob[7]
                            gcn_late_glob = prams_glob[8]
                    model_glob_cat_after = eval(
                        model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6 + "_concat_after")(
                        dataset_glob_cat_after, device=device,
                        out_channels=int(gcn_ker_glob), n_hidden_layers=int(gcn_lyr_glob), glob_feat_pth=global_feat_path)
                    full_connect_channels = ""
                    if model_type4 == "2lin":
                        full_connect_channels = "full_chs" + str(int(params['kernels'])) + "_" + str(
                            int(params['kernels'] / 2)) + "_2_"
                    if model_type4 == "_set":
                        model_type4 = model_type4 + "_2_" + str(processing_step)
                    main_info_glob = model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6 \
                                     + "_sample_" + str(sampling_flag) + "_v_late_" + full_connect_channels \
                                     + name_dataset_glob_cat_after + "_test_hit_" + str(test_hit) \
                                     + "_data_seed" + str(seed) \
                                     + "_ratio" + str(ratio_for_training) + "_knl" \
                                     + str(gcn_ker_glob) + "_lyr" + str(gcn_lyr_glob) \
                                     + "initial_lr_" + str(gcn_late_glob)
                    if flag == "compare_mlp_gcn":
                        main_info_glob = main_info_glob + "_use_mlp"
                    elif flag == "compare_ae_gcn":
                        main_info_glob = main_info_glob + "_use_autoencoder"
                    if attr_flag:
                        main_info_glob = main_info_glob + "_attr_True"
                    if sampling_each_epochs_flag:
                        main_info_glob = main_info_glob + "_each_sampled"
                    if ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                        main_info_glob = main_info_glob + "_79784AE_v2"
                    elif ae_name_dataset == "wo3D_79784_2nd_mini_5666_AlvaDesc":
                        main_info_glob = main_info_golb + "_wo3D_79784AE"
                    if flag == "toy_node":
                        main_info_glob = main_info_glob + "_toy_node"
                    elif flag == "toy_edge":
                        main_info_glob = main_info_glob + "_toy_edge"
                    if ((seed == 4040 or seed >= 11110) and flag == "compare_mlp_gcn") or flag == "compare_ae_gcn":
                        model_glob_path = "../../result/" + main_info_glob + "/model(" + main_info_glob + ")/model%05d.pth" % (
                                int(gcn_epoch_glob))
                    else:
                        model_glob_path = "../../result/" + main_info_glob + "/model(" + main_info_glob + ")/model%05d.pth" % (
                                int(gcn_epoch_glob) + 30)
                    print(main_info_glob)
                    model_glob_cat_after.load_state_dict(torch.load(model_glob_path))

                    compare = "_addnode"
                    pram_glob_path = "../../data/best_results/GCN/origindata_seed_" + str(
                        seeds[seed_no]) + "_drop_" + str(
                        dropout) + "use_val/" + glob_type + compare + "_all_results.csv"
                    if glob_type == "auto":
                        if ae_name_dataset == "":
                            lin_index = 1
                        elif ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                            lin_index = 0
                    elif glob_type == "mlp":
                        lin_index = 1
                    elif glob_type == "dragon":
                        lin_index = 0
                    with open(pram_glob_path, "r") as f:
                        reader = csv.reader(f)
                        prams_glob = [s for s in reader]
                        if flag == "compare_ae_gcn" and len(prams_glob) < 2:
                            print("うーん，だめっぽ")
                        elif flag == "compare_ae_gcn" and len(prams_glob) > 2:
                            for j in range(len(prams_glob) - 1):
                                main_flag = False
                                alva_flag = False
                                data = prams_glob[j + 1]
                                epo = data[0]
                                lyr = data[6]
                                chn = data[7]
                                rate = data[8]
                                tmp_main_info = "PDN_dense_add_skip_6bro_pre_batch_act_max_test_sample_True_v_late_original_" \
                                            + "auto_add_node_test_hit_10_data_seed" + str(
                                    seed) + "_ratio0.5_knl" + chn + "_lyr" \
                                            + lyr + "initial_lr_" + rate + "_use_autoencoder"
                                main_info_alva = tmp_main_info + "_79784AE_v2"
                                path = "../../result/" + tmp_main_info + "/model(" + tmp_main_info + ")/model%05d.pth" % int(
                                    epo)
                                path_alva = "../../result/" + main_info_alva + "/model(" + main_info_alva + ")/model%05d.pth" % int(
                                    epo)
                                if os.path.isfile(path):
                                    main_flag = True
                                if os.path.isfile(path_alva):
                                    alva_flag = True
                                if main_flag and alva_flag:
                                    print("困ったなぁ、、、", name, ":", seed, ":", j)
                                    return 0
                                elif alva_flag and ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                                    lin_index = j+1
                                elif main_flag and ae_name_dataset == "":
                                    lin_index = len(prams_glob)-1
                        prams_glob = prams_glob[lin_index]
                        print("add_node:", prams_glob)
                        gcn_epoch_glob = int(prams_glob[0])
                        gcn_lyr_glob = prams_glob[6]
                        gcn_ker_glob = prams_glob[7]
                        gcn_late_glob = prams_glob[8]
                    model_glob_add_node = eval(
                        model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6)(
                        dataset_glob_add_node,
                        out_channels=int(gcn_ker_glob), n_hidden_layers=int(gcn_lyr_glob))
                    full_connect_channels = ""
                    if model_type4 == "2lin":
                        full_connect_channels = "full_chs" + str(int(params['kernels'])) + "_" + str(
                            int(params['kernels'] / 2)) + "_2_"
                    if model_type4 == "_set":
                        model_type4 = model_type4 + "_2_" + str(processing_step)
                    main_info_glob = model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6 \
                                     + "_sample_" + str(sampling_flag) + "_v_late_" + full_connect_channels \
                                     + name_dataset_glob_add_node + "_test_hit_" + str(test_hit) \
                                     + "_data_seed" + str(seed) \
                                     + "_ratio" + str(ratio_for_training) + "_knl" \
                                     + str(gcn_ker_glob) + "_lyr" + str(gcn_lyr_glob) \
                                     + "initial_lr_" + str(gcn_late_glob)
                    if flag == "compare_mlp_gcn":
                        main_info_glob = main_info_glob + "_use_mlp_79784AE_v2"
                    elif flag == "compare_ae_gcn":
                        main_info_glob = main_info_glob + "_use_autoencoder"
                    if attr_flag:
                        main_info_glob = main_info_glob + "_attr_True"
                    if sampling_each_epochs_flag:
                        main_info_glob = main_info_glob + "_each_sampled"
                    if ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                        main_info_glob = main_info_glob + "_79784AE_v2"
                    elif ae_name_dataset == "wo3D_79784_2nd_mini_5666_AlvaDesc":
                        main_info_glob = main_info_golb + "_wo3D_79784AE"
                    if flag == "toy_node":
                        main_info_glob = main_info_glob + "_toy_node"
                    elif flag == "toy_edge":
                        main_info_glob = main_info_glob + "_toy_edge"
                    if flag == "compare_ae_gcn":
                        gcn_epoch_glob = gcn_epoch_glob - 30
                    if int(gcn_epoch_glob) >= 270:
                        gcn_epoch_glob = 270
                    model_glob_path = "../../result/" + main_info_glob + "/model(" + main_info_glob + ")/model%05d.pth" % (
                            int(gcn_epoch_glob) + 30)
                    print(main_info_glob)
                    model_glob_add_node.load_state_dict(torch.load(model_glob_path))

                else:
                    if compare_flag == "_add_node":
                        compare = "_addnode"
                    elif compare_flag == "_concat_first":
                        compare = "_catfirst"
                    elif compare_flag == "_concat_after":
                        compare = "_catafter"
                    pram_glob_path = "../../data/best_results/GCN/origindata_seed_" + str(seeds[seed_no]) + "_drop_" + str(
                        dropout) + "use_val/mlp" + compare + "_all_results.csv"
                    with open(pram_glob_path, "r") as f:
                        reader = csv.reader(f)
                        prams_glob = [s for s in reader]
                        prams_glob = prams_glob[len(prams_glob)-1]
                        print("_add_node:", prams_glob)
                        gcn_epoch_glob = int(prams_glob[0])
                        gcn_lyr_glob = prams_glob[6]
                        gcn_ker_glob = prams_glob[7]
                        gcn_late_glob = prams_glob[8]
                    model_glob = eval(
                        model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6)(
                        dataset_glob,
                        out_channels=int(gcn_ker_glob), n_hidden_layers=int(gcn_lyr_glob))
                    full_connect_channels = ""
                    if model_type4 == "2lin":
                        full_connect_channels = "full_chs" + str(int(params['kernels'])) + "_" + str(
                            int(params['kernels'] / 2)) + "_2_"
                    if model_type4 == "_set":
                        model_type4 = model_type4 + "_2_" + str(processing_step)
                    main_info_glob = model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6 \
                                + "_sample_" + str(sampling_flag) + "_v_late_" + full_connect_channels \
                                + name_dataset_glob + "_test_hit_" + str(test_hit) \
                                + "_data_seed" + str(seed) \
                                + "_ratio" + str(ratio_for_training) + "_knl" \
                                + str(gcn_ker_glob) + "_lyr" + str(gcn_lyr_glob) \
                                + "initial_lr_" + str(gcn_late_glob) + "_use_mlp_79784AE_v2"
                    if attr_flag:
                        main_info_glob = main_info_glob + "_attr_True"
                    if sampling_each_epochs_flag:
                        main_info_glob = main_info_glob + "_each_sampled"
                    if ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                        main_info_glob = main_info_glob + "_79784AE_v2"
                    elif ae_name_dataset == "wo3D_79784_2nd_mini_5666_AlvaDesc":
                        main_info_glob = main_info_golb + "_wo3D_79784AE"
                    if flag == "toy_node":
                        main_info_glob = main_info_glob + "_toy_node"
                    elif flag == "toy_edge":
                        main_info_glob = main_info_glob + "_toy_edge"
                    model_glob_path = "../../result/" + main_info_glob + "/model(" + main_info_glob + ")/model%05d.pth" % (
                                int(gcn_epoch_glob) + 30)
                    model_glob.load_state_dict(torch.load(model_glob_path))
            criterion = torch.nn.CrossEntropyLoss()
            train_acc, training_loss, out_trn, true_trn, pred_trn, f1_trn, precision_trn, \
            recall_trn, report_sum_trn = test(model, train_loader, dropout, criterion, use_gpu, edge_info)
            if use_val:
                val_acc, val_loss, out_val, true_val, pred_val, f1_val, precision_val, \
                recall_val, report_sum_val = test(model, val_loader, dropout, criterion, use_gpu, edge_info)
            test_acc, testing_loss, out, true, pred, f1_test, precision_test, \
            recall_test, report_sum_test = test(model, test_loader, dropout, criterion, use_gpu, edge_info)
            if compare_flag == "ansamble":
                train_acc_glob, training_loss_glob, out_trn_glob_cat_first, true_trn_glob, pred_trn_glob_cat_first, f1_trn_glob, precision_trn_glob, \
                recall_trn_glob, report_sum_trn_glob = test(model_glob_cat_first, train_glob_loader_cat_first, dropout, criterion, use_gpu,
                                                            edge_info)
                if use_val:
                    val_acc_glob, val_loss_glob, out_val_glob_cat_first, true_val_glob, pred_val_glob_cat_first, f1_val_glob, precision_val_glob, \
                    recall_val_glob, report_sum_val_glob = test(model_glob_cat_first, val_glob_loader_cat_first, dropout, criterion,
                                                                use_gpu, edge_info)
                test_acc_glob, testing_loss_glob, out_glob_cat_first, true_glob, pred_glob_cat_first, f1_test_glob, precision_test_glob, \
                recall_test_glob, report_sum_test_glob = test(model_glob_cat_first, test_glob_loader_cat_first, dropout, criterion, use_gpu,
                                                              edge_info)

                train_acc_glob, training_loss_glob, out_trn_glob_cat_after, true_trn_glob, pred_trn_glob_cat_after, f1_trn_glob, precision_trn_glob, \
                recall_trn_glob, report_sum_trn_glob = test(model_glob_cat_after, train_glob_loader_cat_after, dropout,
                                                            criterion, use_gpu,
                                                            edge_info)
                if use_val:
                    val_acc_glob, val_loss_glob, out_val_glob_cat_after, true_val_glob, pred_val_glob_cat_after, f1_val_glob, precision_val_glob, \
                    recall_val_glob, report_sum_val_glob = test(model_glob_cat_after, val_glob_loader_cat_after,
                                                                dropout, criterion,
                                                                use_gpu, edge_info)
                test_acc_glob, testing_loss_glob, out_glob_cat_after, true_glob, pred_glob_cat_after, f1_test_glob, precision_test_glob, \
                recall_test_glob, report_sum_test_glob = test(model_glob_cat_after, test_glob_loader_cat_after, dropout,
                                                              criterion, use_gpu,
                                                              edge_info)

                train_acc_glob, training_loss_glob, out_trn_glob_add_node, true_trn_glob, pred_trn_glob_add_node, f1_trn_glob, precision_trn_glob, \
                recall_trn_glob, report_sum_trn_glob = test(model_glob_add_node, train_glob_loader_add_node, dropout,
                                                            criterion, use_gpu,
                                                            edge_info)
                if use_val:
                    val_acc_glob, val_loss_glob, out_val_glob_add_node, true_val_glob, pred_val_glob_add_node, f1_val_glob, precision_val_glob, \
                    recall_val_glob, report_sum_val_glob = test(model_glob_add_node, val_glob_loader_add_node,
                                                                dropout, criterion,
                                                                use_gpu, edge_info)
                test_acc_glob, testing_loss_glob, out_glob_add_node, true_glob, pred_glob_add_node, f1_test_glob, precision_test_glob, \
                recall_test_glob, report_sum_test_glob = test(model_glob_add_node, test_glob_loader_add_node, dropout,
                                                              criterion, use_gpu,
                                                              edge_info)
            elif compare_flag != "":
                train_acc_glob, training_loss_glob, out_trn_glob, true_trn_glob, pred_trn_glob, f1_trn_glob, precision_trn_glob, \
                recall_trn_glob, report_sum_trn_glob = test(model_glob, train_glob_loader, dropout, criterion, use_gpu, edge_info)
                if use_val:
                    val_acc_glob, val_loss_glob, out_val_glob, true_val_glob, pred_val_glob, f1_val_glob, precision_val_glob, \
                    recall_val_glob, report_sum_val_glob = test(model_glob, val_glob_loader, dropout, criterion, use_gpu, edge_info)
                test_acc_glob, testing_loss_glob, out_glob, true_glob, pred_glob, f1_test_glob, precision_test_glob, \
                recall_test_glob, report_sum_test_glob = test(model_glob, test_glob_loader, dropout, criterion, use_gpu, edge_info)
            if flag == "compare_mlp_gcn":
                test_feat_tensor = feat_tensor[test_index]
                val_feat_tensor = feat_tensor[val_index]
                train_feat_tensor = feat_tensor[train_index]
                *_, mlp_pred_test, mlp_label_test = mlp_dim_reduce(test_feat_tensor, in_channels, int(ker), pre_layer,
                                                   out_channels, n_connected_layer, pretrained_model_path, pre_dropout,
                                                   device, tmp_feat)
                *_, mlp_pred_val, mlp_label_val = mlp_dim_reduce(val_feat_tensor, in_channels, int(ker), pre_layer,
                                                  out_channels, n_connected_layer, pretrained_model_path,
                                                  pre_dropout,
                                                  device, tmp_feat)
                *_, mlp_pred_trn, mlp_label_trn = mlp_dim_reduce(train_feat_tensor, in_channels, int(ker), pre_layer,
                                                    out_channels, n_connected_layer, pretrained_model_path, pre_dropout,
                                                    device, tmp_feat)
                mlp_f1 = ["", "f1", "prec", "rec"]
                mlp_trn_f1 = f1_score(torch.tensor(true_trn).cpu(), torch.tensor(mlp_label_trn).cpu(),
                                      average=None, zero_division=0)
                mlp_trn_prec = precision_score(torch.tensor(true_trn, device='cpu'),
                                            torch.tensor(mlp_label_trn).cpu(), average=None, zero_division=0)
                mlp_trn_rec = recall_score(torch.tensor(true_trn, device='cpu'),
                                        torch.tensor(mlp_label_trn).cpu(), average=None, zero_division=0)
                mlp_f1.append(["mlp_trn", mlp_trn_f1[1], mlp_trn_prec[1], mlp_trn_rec[1]])
                mlp_val_f1 = f1_score(torch.tensor(true_val).cpu(), torch.tensor(mlp_label_val).cpu(),
                                      average=None, zero_division=0)
                mlp_val_prec = precision_score(torch.tensor(true_val, device='cpu'),
                                               torch.tensor(mlp_label_val).cpu(), average=None, zero_division=0)
                mlp_val_rec = recall_score(torch.tensor(true_val, device='cpu'),
                                           torch.tensor(mlp_label_val).cpu(), average=None, zero_division=0)
                mlp_f1.append(["mlp_val", mlp_val_f1[1], mlp_val_prec[1], mlp_val_rec[1]])
                mlp_test_f1 = f1_score(torch.tensor(true).cpu(), torch.tensor(mlp_label_test).cpu(),
                                       average=None, zero_division=0)
                mlp_test_prec = precision_score(torch.tensor(true, device='cpu'),
                                               torch.tensor(mlp_label_test).cpu(), average=None, zero_division=0)
                mlp_test_rec = recall_score(torch.tensor(true, device='cpu'),
                                           torch.tensor(mlp_label_test).cpu(), average=None, zero_division=0)
                mlp_f1.append(["mlp_test", mlp_test_f1[1], mlp_test_prec[1], mlp_test_rec[1]])

                sum_f1 = ["", "f1", "prec", "rec"]
                avg_trn = (out_trn + mlp_pred_trn) / 2.0
                avg_val = (out_val + mlp_pred_val) / 2.0
                avg_test = (out + mlp_pred_test) / 2.0
                avg_label_trn = avg_trn.argmax(dim=1)
                avg_trn_f1 = f1_score(torch.tensor(true_trn).cpu(), avg_label_trn.detach().cpu(),
                                   average=None, zero_division=0)
                avg_trn_prec = precision_score(torch.tensor(true_trn, device='cpu'),
                                               avg_label_trn.detach().cpu(), average=None, zero_division=0)
                avg_trn_rec = recall_score(torch.tensor(true_trn, device='cpu'),
                                           avg_label_trn.detach().cpu(), average=None, zero_division=0)
                sum_f1.append(["avg_trn", avg_trn_f1[1], avg_trn_prec[1], avg_trn_rec[1]])
                avg_label_val = avg_val.argmax(dim=1)
                avg_val_f1 = f1_score(torch.tensor(true_val).cpu(), avg_label_val.detach().cpu(),
                                   average=None, zero_division=0)
                avg_val_prec = precision_score(torch.tensor(true_val, device='cpu'),
                                               avg_label_val.detach().cpu(), average=None, zero_division=0)
                avg_val_rec = recall_score(torch.tensor(true_val, device='cpu'),
                                           avg_label_val.detach().cpu(), average=None, zero_division=0)
                sum_f1.append(["avg_val", avg_val_f1[1], avg_val_prec[1], avg_val_rec[1]])
                avg_label_test = avg_test.argmax(dim=1)
                avg_test_f1 = f1_score(torch.tensor(true).cpu(), avg_label_test.detach().cpu(),
                                   average=None, zero_division=0)
                avg_test_prec = precision_score(torch.tensor(true, device='cpu'),
                                               avg_label_test.detach().cpu(), average=None, zero_division=0)
                avg_test_rec = recall_score(torch.tensor(true, device='cpu'),
                                           avg_label_test.detach().cpu(), average=None, zero_division=0)
                sum_f1.append(["avg_test", avg_test_f1[1], avg_test_prec[1], avg_test_rec[1]])

                xor_trn = torch.logical_xor(torch.tensor(pred_trn).cpu(), torch.tensor(mlp_label_trn).cpu())
                xor_val = torch.logical_xor(torch.tensor(pred_val).cpu(), torch.tensor(mlp_label_val).cpu())
                xor_test = torch.logical_xor(torch.tensor(pred).cpu(), torch.tensor(mlp_label_test).cpu())
            save_pth = "../../data/compare_" + glob_type + "_gcn/seed_" + str(seed) + "/"
            if not os.path.isdir(save_pth):
                os.makedirs(save_pth)
            if compare_flag != "":
                if compare_flag == "ansamble":
                    cat_a = [["", "f1", "prec", "rec"]]
                    cat_f = [["", "f1", "prec", "rec"]]
                    add = [["", "f1", "prec", "rec"]]
                    if flag == "compare_mlp_gcn":
                        ens_5 = [["", "f1", "prec", "rec"]]
                        ens_3 = [["", "f1", "prec", "rec"]]
                        ens_3_glob = [["", "f1", "prec", "rec"]]
                        avg_5 = [["", "f1", "prec", "rec"]]
                        avg_3_glob = [["", "f1", "prec", "rec"]]
                        or_5 = [["", "f1", "prec", "rec"]]
                        or_3 = [["", "f1", "prec", "rec"]]
                        or_3_glob = [["", "f1", "prec", "rec"]]
                        or_2 = [["", "f1", "prec", "rec"]]
                        or_2_gcn_add = [["", "f1", "prec", "rec"]]
                        or_2_mlp_add = [["", "f1", "prec", "rec"]]
                        or_2_gcn_cat_a = [["", "f1", "prec", "rec"]]
                        or_2_mlp_cat_a = [["", "f1", "prec", "rec"]]
                        or_2_gcn_cat_f = [["", "f1", "prec", "rec"]]
                        or_2_mlp_cat_f = [["", "f1", "prec", "rec"]]
                        or_2_add_cat_a = [["", "f1", "prec", "rec"]]
                        or_2_add_cat_f = [["", "f1", "prec", "rec"]]
                        or_2_cat_a_f = [["", "f1", "prec", "rec"]]
                        trn = torch.tensor(pred_trn_glob_cat_after).cpu()+torch.tensor(pred_trn_glob_cat_first).cpu()+torch.tensor(pred_trn_glob_add_node)+torch.tensor(mlp_label_trn).cpu()+torch.tensor(pred_trn).cpu()
                        val = torch.tensor(pred_val_glob_cat_after).cpu() + torch.tensor(
                            pred_val_glob_cat_first).cpu() + torch.tensor(pred_val_glob_add_node) + torch.tensor(
                            mlp_label_val).cpu() + torch.tensor(pred_val).cpu()
                        tst = torch.tensor(pred_glob_cat_after).cpu() + torch.tensor(
                            pred_glob_cat_first).cpu() + torch.tensor(pred_glob_add_node) + torch.tensor(
                            mlp_label_test).cpu() + torch.tensor(pred).cpu()
                        threshold = torch.Tensor([3])
                        trn_out = (trn >= threshold)
                        val_out = (val >= threshold)
                        test_out = (tst >= threshold)
                        threshold_or = torch.Tensor([1])
                        trn_out_or = (trn >= threshold_or)
                        val_out_or = (val >= threshold_or)
                        test_out_or = (tst >= threshold_or)

                        trn_3 = torch.tensor(pred_trn_glob_add_node) + torch.tensor(
                            mlp_label_trn).cpu() + torch.tensor(pred_trn).cpu()
                        val_3 = torch.tensor(pred_val_glob_add_node) + torch.tensor(
                            mlp_label_val).cpu() + torch.tensor(pred_val).cpu()
                        tst_3 = torch.tensor(pred_glob_add_node) + torch.tensor(
                            mlp_label_test).cpu() + torch.tensor(pred).cpu()
                        threshold_3 = torch.Tensor([2])
                        trn_out_3 = (trn_3 >= threshold_3)
                        val_out_3 = (val_3 >= threshold_3)
                        test_out_3 = (tst_3 >= threshold_3)
                        trn_out_3_or = (trn_3 >= threshold_or)
                        val_out_3_or = (val_3 >= threshold_or)
                        test_out_3_or = (tst_3 >= threshold_or)

                        trn_2 = torch.tensor(mlp_label_trn).cpu() + torch.tensor(pred_trn).cpu()
                        val_2 = torch.tensor(mlp_label_val).cpu() + torch.tensor(pred_val).cpu()
                        tst_2 = torch.tensor(mlp_label_test).cpu() + torch.tensor(pred).cpu()
                        trn_out_2_or = (trn_2 >= threshold_or)
                        val_out_2_or = (val_2 >= threshold_or)
                        test_out_2_or = (tst_2 >= threshold_or)

                        avg_5_trn = (out_trn + mlp_pred_trn + out_trn_glob_cat_first + out_trn_glob_cat_after + out_trn_glob_add_node) / 5.0
                        avg_5_val = (out_val + mlp_pred_val + out_val_glob_cat_first + out_val_glob_cat_after + out_val_glob_add_node) / 5.0
                        avg_5_test = (out + mlp_pred_test + out_glob_cat_first + out_glob_cat_after + out_glob_add_node) / 5.0
                        avg_label_5_trn = avg_5_trn.argmax(dim=1)
                        avg_label_5_val = avg_5_val.argmax(dim=1)
                        avg_label_5_tst = avg_5_test.argmax(dim=1)
                        avg_5 = save_score(avg_5, true_trn, true_val, true, avg_label_5_trn, avg_label_5_val, avg_label_5_tst)
                        with open(save_pth + "_avg_5_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(avg_5)

                        avg_3_trn = (out_trn_glob_cat_first + out_trn_glob_cat_after + out_trn_glob_add_node) / 3.0
                        avg_3_val = (out_val_glob_cat_first + out_val_glob_cat_after + out_val_glob_add_node) / 3.0
                        avg_3_test = (out_glob_cat_first + out_glob_cat_after + out_glob_add_node) / 3.0
                        avg_label_3_trn = avg_3_trn.argmax(dim=1)
                        avg_label_3_val = avg_3_val.argmax(dim=1)
                        avg_label_3_tst = avg_3_test.argmax(dim=1)
                        avg_3_glob = save_score(avg_3_glob, true_trn, true_val, true, avg_label_3_trn, avg_label_3_val, avg_label_3_tst)
                        with open(save_pth + "_avg_3_glob_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(avg_3_glob)

                        trn_g = torch.tensor(pred_trn_glob_cat_after).cpu() + torch.tensor(
                            pred_trn_glob_cat_first).cpu() + torch.tensor(pred_trn_glob_add_node)
                        val_g = torch.tensor(pred_val_glob_cat_after).cpu() + torch.tensor(
                            pred_val_glob_cat_first).cpu() + torch.tensor(pred_val_glob_add_node)
                        tst_g = torch.tensor(pred_glob_cat_after).cpu() + torch.tensor(
                            pred_glob_cat_first).cpu() + torch.tensor(pred_glob_add_node)
                        trn = (trn_g >= threshold_or)
                        val = (val_g >= threshold_or)
                        tst = (tst_g >= threshold_or)
                        or_3_glob = save_score(or_3_glob, true_trn, true_val, true, trn, val, tst)
                        with open(save_pth + "_or_3_glob_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_3_glob)

                        trn = (trn_g >= threshold_3)
                        val = (val_g >= threshold_3)
                        tst = (tst_g >= threshold_3)
                        ens_3_glob = save_score(ens_3_glob, true_trn, true_val, true, trn, val, tst)
                        with open(save_pth + "_ens_3_glob_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(ens_3_glob)

                        trn = torch.tensor(pred_trn_glob_add_node) + torch.tensor(pred_trn).cpu()
                        val = torch.tensor(pred_val_glob_add_node) + torch.tensor(pred_val).cpu()
                        tst = torch.tensor(pred_glob_add_node) + torch.tensor(pred).cpu()
                        trn = (trn >= threshold_or)
                        val = (val >= threshold_or)
                        tst = (tst >= threshold_or)
                        or_2_gcn_add = save_score(or_2_gcn_add, true_trn, true_val, true, trn, val, tst)
                        with open(save_pth + "_or_2_gcn_add_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_2_gcn_add)

                        trn = torch.tensor(pred_trn_glob_add_node) + torch.tensor(mlp_label_trn).cpu()
                        val = torch.tensor(pred_val_glob_add_node) + torch.tensor(mlp_label_val).cpu()
                        tst = torch.tensor(pred_glob_add_node) + torch.tensor(mlp_label_test).cpu()
                        trn = (trn >= threshold_or)
                        val = (val >= threshold_or)
                        tst = (tst >= threshold_or)
                        or_2_mlp_add = save_score(or_2_mlp_add, true_trn, true_val, true, trn, val, tst)
                        with open(save_pth + "_or_2_mlp_add_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_2_mlp_add)

                        trn = torch.tensor(pred_trn_glob_cat_after).cpu() + torch.tensor(pred_trn).cpu()
                        val = torch.tensor(pred_val_glob_cat_after).cpu() + torch.tensor(pred_val).cpu()
                        tst = torch.tensor(pred_glob_cat_after).cpu() + torch.tensor(pred).cpu()
                        trn = (trn >= threshold_or)
                        val = (val >= threshold_or)
                        tst = (tst >= threshold_or)
                        or_2_gcn_cat_a = save_score(or_2_gcn_cat_a, true_trn, true_val, true, trn, val, tst)
                        with open(save_pth + "_or_2_gcn_cat_a_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_2_gcn_cat_a)

                        trn = torch.tensor(pred_trn_glob_cat_after).cpu() + torch.tensor(mlp_label_trn).cpu()
                        val = torch.tensor(pred_val_glob_cat_after).cpu() + torch.tensor(mlp_label_val).cpu()
                        tst = torch.tensor(pred_glob_cat_after).cpu() + torch.tensor(mlp_label_test).cpu()
                        trn = (trn >= threshold_or)
                        val = (val >= threshold_or)
                        tst = (tst >= threshold_or)
                        or_2_mlp_cat_a = save_score(or_2_mlp_cat_a, true_trn, true_val, true, trn, val, tst)
                        with open(save_pth + "_or_2_mlp_cat_a_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_2_mlp_cat_a)

                        trn = torch.tensor(pred_trn_glob_cat_first).cpu() + torch.tensor(pred_trn).cpu()
                        val = torch.tensor(pred_val_glob_cat_first).cpu() + torch.tensor(pred_val).cpu()
                        tst = torch.tensor(pred_glob_cat_first).cpu() + torch.tensor(pred).cpu()
                        trn = (trn >= threshold_or)
                        val = (val >= threshold_or)
                        tst = (tst >= threshold_or)
                        or_2_gcn_cat_f = save_score(or_2_gcn_cat_f, true_trn, true_val, true, trn, val, tst)
                        with open(save_pth + "_or_2_gcn_cat_f_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_2_gcn_cat_f)

                        trn = torch.tensor(pred_trn_glob_cat_first).cpu() + torch.tensor(mlp_label_trn).cpu()
                        val = torch.tensor(pred_val_glob_cat_first).cpu() + torch.tensor(mlp_label_val).cpu()
                        tst = torch.tensor(pred_glob_cat_first).cpu() + torch.tensor(mlp_label_test).cpu()
                        trn = (trn >= threshold_or)
                        val = (val >= threshold_or)
                        tst = (tst >= threshold_or)
                        or_2_mlp_cat_f = save_score(or_2_mlp_cat_f, true_trn, true_val, true, trn, val, tst)
                        with open(save_pth + "_or_2_mlp_cat_f_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_2_mlp_cat_f)

                        trn = torch.tensor(pred_trn_glob_cat_after).cpu() + torch.tensor(pred_trn_glob_add_node)
                        val = torch.tensor(pred_val_glob_cat_after).cpu() + torch.tensor(pred_val_glob_add_node)
                        tst = torch.tensor(pred_glob_cat_after).cpu() + torch.tensor(pred_glob_add_node)
                        trn = (trn >= threshold_or)
                        val = (val >= threshold_or)
                        tst = (tst >= threshold_or)
                        or_2_add_cat_a = save_score(or_2_add_cat_a, true_trn, true_val, true, trn, val, tst)
                        with open(save_pth + "_or_2_add_cat_a_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_2_add_cat_a)

                        trn = torch.tensor(pred_trn_glob_cat_first).cpu() + torch.tensor(pred_trn_glob_add_node)
                        val = torch.tensor(pred_val_glob_cat_first).cpu() + torch.tensor(pred_val_glob_add_node)
                        tst = torch.tensor(pred_glob_cat_first).cpu() + torch.tensor(pred_glob_add_node)
                        trn = (trn >= threshold_or)
                        val = (val >= threshold_or)
                        tst = (tst >= threshold_or)
                        or_2_add_cat_f = save_score(or_2_add_cat_f, true_trn, true_val, true, trn, val, tst)
                        with open(save_pth + "_or_2_add_cat_f_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_2_add_cat_f)

                        trn = torch.tensor(pred_trn_glob_cat_after).cpu() + torch.tensor(pred_trn_glob_cat_first).cpu()
                        val = torch.tensor(pred_val_glob_cat_after).cpu() + torch.tensor(pred_val_glob_cat_first).cpu()
                        tst = torch.tensor(pred_glob_cat_after).cpu() + torch.tensor(pred_glob_cat_first).cpu()
                        trn = (trn >= threshold_or)
                        val = (val >= threshold_or)
                        tst = (tst >= threshold_or)
                        or_2_cat_a_f = save_score(or_2_cat_a_f, true_trn, true_val, true, trn, val, tst)
                        with open(save_pth + "_or_2_cat_a_f_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_2_cat_a_f)

                        ens_5 = save_score(ens_5, true_trn, true_val, true, trn_out, val_out, test_out)
                        ens_3 = save_score(ens_3, true_trn, true_val, true, trn_out_3, val_out_3, test_out_3)
                        or_5 = save_score(or_5, true_trn, true_val, true, trn_out_or, val_out_or, test_out_or)
                        or_3 = save_score(or_3, true_trn, true_val, true, trn_out_3_or, val_out_3_or, test_out_3_or)
                        or_2 = save_score(or_2, true_trn, true_val, true, trn_out_2_or, val_out_2_or, test_out_2_or)
                        with open(save_pth + "_ens_5_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(ens_5)
                        with open(save_pth + "_ens_3_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(ens_3)
                        with open(save_pth + "_or_5_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_5)
                        with open(save_pth + "_or_3_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_3)
                        with open(save_pth + "_or_2_score.csv", "w") as f:
                            writer = csv.writer(f)
                            writer.writerows(or_2)

                        test_ens = np.full(3982, 2)
                        print(test_ens)
                        print(test_ens[test_index])
                        print(test_out_or)
                        test_ens[test_index] = test_out_or
                        pres_pth_test = "../../data/compare_mlp_gcn/sum/suitei.csv"
                        if seed == 1010:
                            gt_tmp = np.zeros(3982, dtype=np.int32)
                            gt_tmp[hit_index] = 1
                            feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted.csv"
                            with open(feat_path, "r") as f:
                                reader = csv.reader(f)
                                global_feat_set_tmp = [row[1] for row in reader]
                                sorted_mol_name = global_feat_set_tmp[1:]
                                sorted_mol_name = np.array(sorted_mol_name)
                            with open(pres_pth_test, "w") as f:
                                writer = csv.writer(f)
                                test_label = ["seed"] + ["mol_ID"] + list(sorted_mol_name)
                                writer.writerow(test_label)
                                writer.writerow(["-"] + list(gt_tmp))
                                writer.writerow([str(seed)] + list(test_ens))
                        else:
                            with open(pres_pth_test, "a") as f:
                                writer = csv.writer(f)
                                writer.writerow([seed] + test_ens)

                    cat_a = save_score(cat_a, true_trn, true_val, true, pred_trn_glob_cat_after, pred_val_glob_cat_after, pred_glob_cat_after)
                    cat_f = save_score(cat_f, true_trn, true_val, true, pred_trn_glob_cat_first, pred_val_glob_cat_first, pred_glob_cat_first)
                    add = save_score(add, true_trn, true_val, true, pred_trn_glob_add_node, pred_val_glob_add_node, pred_glob_add_node)

                    result_name = ""
                    if flag == "compare_ae_gcn":
                        if ae_name_dataset == "":
                            result_name = "_main"
                        elif ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                            result_name = "_79784_AE"
                    with open(save_pth + result_name + "_cat_a_score.csv", "w") as f:
                        writer = csv.writer(f)
                        writer.writerows(cat_a)
                    with open(save_pth + result_name + "_cat_f_score.csv", "w") as f:
                        writer = csv.writer(f)
                        writer.writerows(cat_f)
                    with open(save_pth + result_name + "_add_score.csv", "w") as f:
                        writer = csv.writer(f)
                        writer.writerows(add)

                else:
                    xor_trn_mlp_glob = torch.logical_xor(torch.tensor(pred_trn_glob).cpu(), torch.tensor(mlp_label_trn).cpu())
                    xor_val_mlp_glob = torch.logical_xor(torch.tensor(pred_val_glob).cpu(), torch.tensor(mlp_label_val).cpu())
                    xor_test_mlp_glob = torch.logical_xor(torch.tensor(pred_glob).cpu(), torch.tensor(mlp_label_test).cpu())

                    xor_trn_gcn_glob = torch.logical_xor(torch.tensor(pred_trn_glob).cpu(),
                                                         torch.tensor(pred_trn).cpu())
                    xor_val_gcn_glob = torch.logical_xor(torch.tensor(pred_val_glob).cpu(),
                                                         torch.tensor(pred_val).cpu())
                    xor_test_gcn_glob = torch.logical_xor(torch.tensor(pred_glob).cpu(), torch.tensor(pred).cpu())

            if flag == "compare_mlp_gcn":
                with open(save_pth + "f1-score.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(sum_f1)

                with open(save_pth + "mlp_f1-score.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(mlp_f1)

            label_dict = ('0', '1')
            label_dict2 = ('nonhit', 'hit')
            if compare_flag == "ansamble":
                if flag == "compare_mlp_gcn":
                    plot_confusion_matrix(torch.tensor(true).cpu(),
                                          test_out.detach().cpu(),
                                          classes=label_dict2, results_saved_folder=save_pth,
                                          epoch=gcn_epoch,
                                          trn_or_test="ensemble for test", cmap="Blues")

                    plot_confusion_matrix(torch.tensor(true_val).cpu(),
                                          val_out.detach().cpu(),
                                          classes=label_dict2, results_saved_folder=save_pth,
                                          epoch=gcn_epoch, trn_or_test="ensemble for validation",
                                          title="ensemble for validation", cmap="Oranges")

                    plot_confusion_matrix(torch.tensor(true_trn).cpu(),
                                          trn_out.detach().cpu(),
                                          classes=label_dict2, results_saved_folder=save_pth,
                                          epoch=gcn_epoch, trn_or_test="ensemble for training",
                                          title="ensemble for training", cmap="Greens")

                    plot_confusion_matrix(torch.tensor(true).cpu(),
                                          test_out_3.detach().cpu(),
                                          classes=label_dict2, results_saved_folder=save_pth,
                                          epoch=gcn_epoch,
                                          trn_or_test="ensemble(3) for test", cmap="Blues")

                    plot_confusion_matrix(torch.tensor(true_val).cpu(),
                                          val_out_3.detach().cpu(),
                                          classes=label_dict2, results_saved_folder=save_pth,
                                          epoch=gcn_epoch, trn_or_test="ensemble(3) for validation",
                                          title="ensemble(3) for validation", cmap="Oranges")

                    plot_confusion_matrix(torch.tensor(true_trn).cpu(),
                                          trn_out_3.detach().cpu(),
                                          classes=label_dict2, results_saved_folder=save_pth,
                                          epoch=gcn_epoch, trn_or_test="ensemble(3) for training",
                                          title="ensemble(3) for training", cmap="Greens")
                    continue
            plot_confusion_matrix(torch.tensor(true).cpu(),
                                  torch.tensor(pred).cpu(),
                                  classes=label_dict2, results_saved_folder=save_pth,
                                  epoch=gcn_epoch,
                                  trn_or_test="gcn for test", cmap="Blues")

            plot_confusion_matrix(torch.tensor(true_val).cpu(),
                                  torch.tensor(pred_val).cpu(),
                                  classes=label_dict2, results_saved_folder=save_pth,
                                  epoch=gcn_epoch, trn_or_test="gcn for validation",
                                  title="gcn for validation", cmap="Oranges")

            plot_confusion_matrix(torch.tensor(true_trn).cpu(),
                                  torch.tensor(pred_trn).cpu(),
                                  classes=label_dict2, results_saved_folder=save_pth,
                                  epoch=gcn_epoch, trn_or_test="gcn for training",
                                  title="gcn for training", cmap="Greens")

            if flag == "compare_mlp_gcn":
                plot_confusion_matrix(torch.tensor(true).cpu(),
                                      torch.tensor(mlp_label_test).cpu(),
                                      classes=label_dict2, results_saved_folder=save_pth,
                                      epoch=gcn_epoch,
                                      trn_or_test="mlp for test", cmap="Blues")

                plot_confusion_matrix(torch.tensor(true_val).cpu(),
                                      torch.tensor(mlp_label_val).cpu(),
                                      classes=label_dict2, results_saved_folder=save_pth,
                                      epoch=gcn_epoch, trn_or_test="mlp for validation",
                                      title="mlp for validation", cmap="Oranges")

                plot_confusion_matrix(torch.tensor(true_trn).cpu(),
                                      torch.tensor(mlp_label_trn).cpu(),
                                      classes=label_dict2, results_saved_folder=save_pth,
                                      epoch=gcn_epoch, trn_or_test="mlp for training",
                                      title="mlp for training", cmap="Greens")

                plot_confusion_matrix(torch.tensor(true).cpu(),
                                      xor_test.detach().cpu(),
                                      classes=label_dict, results_saved_folder=save_pth,
                                      epoch=gcn_epoch,
                                      trn_or_test="xor mlp gcn for test", cmap="Blues", cmp=True)

                plot_confusion_matrix(torch.tensor(true_val).cpu(),
                                      xor_val.detach().cpu(),
                                      classes=label_dict, results_saved_folder=save_pth,
                                      epoch=gcn_epoch, trn_or_test="xor mlp gcn for validation",
                                      title="xor mlp gcn for validation", cmap="Oranges", cmp=True)

                plot_confusion_matrix(torch.tensor(true_trn).cpu(),
                                      xor_trn.detach().cpu(),
                                      classes=label_dict, results_saved_folder=save_pth,
                                      epoch=gcn_epoch, trn_or_test="xor mlp gcn for training",
                                      title="xor mlp gcn for training", cmap="Greens", cmp=True)

                plot_confusion_matrix(torch.tensor(true).cpu(),
                                      avg_label_test.detach().cpu(),
                                      classes=label_dict2, results_saved_folder=save_pth,
                                      epoch=gcn_epoch,
                                      trn_or_test="avg mlp gcn for test", cmap="Blues")

                plot_confusion_matrix(torch.tensor(true_val).cpu(),
                                      avg_label_val.detach().cpu(),
                                      classes=label_dict2, results_saved_folder=save_pth,
                                      epoch=gcn_epoch, trn_or_test="avg mlp gcn for validation",
                                      title="avg mlp gcn for validation", cmap="Oranges")

                plot_confusion_matrix(torch.tensor(true_trn).cpu(),
                                      avg_label_trn.detach().cpu(),
                                      classes=label_dict2, results_saved_folder=save_pth,
                                      epoch=gcn_epoch, trn_or_test="avg mlp gcn for training",
                                      title="avg mlp gcn for training", cmap="Greens")

            if compare_flag != "":
                if flag == "compare_mlp_gcn":
                    plot_confusion_matrix(torch.tensor(true).cpu(),
                                          xor_test_mlp_glob.detach().cpu(),
                                          classes=label_dict, results_saved_folder=save_pth,
                                          epoch=gcn_epoch,
                                          trn_or_test="xor mlp glob for test", cmap="Blues", cmp=True)

                    plot_confusion_matrix(torch.tensor(true_val).cpu(),
                                          xor_val_mlp_glob.detach().cpu(),
                                          classes=label_dict, results_saved_folder=save_pth,
                                          epoch=gcn_epoch, trn_or_test="xor mlp glob for validation",
                                          title="xor mlp glob for validation", cmap="Oranges", cmp=True)

                    plot_confusion_matrix(torch.tensor(true_trn).cpu(),
                                          xor_trn_mlp_glob.detach().cpu(),
                                          classes=label_dict, results_saved_folder=save_pth,
                                          epoch=gcn_epoch, trn_or_test="xor mlp glob for training",
                                          title="xor mlp glob for training", cmap="Greens", cmp=True)

                    plot_confusion_matrix(torch.tensor(true).cpu(),
                                          xor_test_gcn_glob.detach().cpu(),
                                          classes=label_dict, results_saved_folder=save_pth,
                                          epoch=gcn_epoch,
                                          trn_or_test="xor gcn glob for test", cmap="Blues", cmp=True)

                    plot_confusion_matrix(torch.tensor(true_val).cpu(),
                                          xor_val_gcn_glob.detach().cpu(),
                                          classes=label_dict, results_saved_folder=save_pth,
                                          epoch=gcn_epoch, trn_or_test="xor gcn glob for validation",
                                          title="xor gcn glob for validation", cmap="Oranges", cmp=True)

                    plot_confusion_matrix(torch.tensor(true_trn).cpu(),
                                          xor_trn_gcn_glob.detach().cpu(),
                                          classes=label_dict, results_saved_folder=save_pth,
                                          epoch=gcn_epoch, trn_or_test="xor gcn glob for training",
                                          title="xor gcn glob for training", cmap="Greens", cmp=True)
            # return 0
        elif flag == "test" and compare_flag == "ansamble":
            if model_type1 == "_dense_add_skip":
                add_tmp = "no_glob_" + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6 + model_type7
            else:
                add_tmp = "no_glob" + model_type1
            pram_path = "../../data/best_results/GCN/origindata_seed_" + str(seeds[seed_no]) + "_drop_" + str(
                dropout) + "use_val/" + add_tmp + "all_results.csv"
            with open(pram_path, "r") as f:
                reader = csv.reader(f)
                prams = [s for s in reader]
                for j in range(len(prams)):
                    if model_type1 == "_dense_cat_skip" or model_type1 == "_res_add_skip":
                        if j != len(prams)-1:
                            continue
                    pram_2 = prams[j].copy()
                    print(pram_2)
                    if model_type1 == "_dense_cat_skip" or model_type1 == "_res_add_skip":
                        if int(pram_2[0]) >= 270:
                            gcn_epoch = 300
                        else:
                            gcn_epoch = int(pram_2[0]) + 30
                    else:
                        gcn_epoch = int(pram_2[0])
                    gcn_lyr = pram_2[6]
                    gcn_ker = pram_2[7]
                    gcn_late = pram_2[8]
                    model = eval(
                        model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6)(dataset,
                                                                                                                out_channels=int(
                                                                                                                    gcn_ker),
                                                                                                                n_hidden_layers=int(
                                                                                                                    gcn_lyr))
                    full_connect_channels = ""
                    if model_type4 == "2lin":
                        full_connect_channels = "full_chs" + str(int(params['kernels'])) + "_" + str(
                            int(params['kernels'] / 2)) + "_2_"
                    if model_type4 == "_set":
                        model_type4 = model_type4 + "_2_" + str(processing_step)
                    main_info = model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6 \
                                + "_sample_" + str(sampling_flag) + "_v_late_" + full_connect_channels \
                                + name_dataset + "_test_hit_" + str(test_hit) \
                                + "_data_seed" + str(seed) \
                                + "_ratio" + str(ratio_for_training) + "_knl" \
                                + str(gcn_ker) + "_lyr" + str(gcn_lyr) \
                                + "initial_lr_" + str(gcn_late)
                    if attr_flag:
                        main_info = main_info + "_attr_True"
                    if sampling_each_epochs_flag:
                        main_info = main_info + "_each_sampled"
                    elif ae_name_dataset == "wo3D_79784_2nd_mini_5666_AlvaDesc":
                        main_info = main_info + "_wo3D_79784AE"
                    if flag == "toy_node":
                        main_info = main_info + "_toy_node"
                    elif flag == "toy_edge":
                        main_info = main_info + "_toy_edge"
                    if model_type1 == "_dense_cat_skip":
                        main_info = main_info + "_79784AE_v2"
                    model_path = "../../result/" + main_info + "/model(" + main_info + ")/model%05d.pth" % (int(gcn_epoch))
                    model.load_state_dict(torch.load(model_path))
                    criterion = torch.nn.CrossEntropyLoss()
                    train_acc, training_loss, out_trn, true_trn, pred_trn, f1_trn, precision_trn, \
                    recall_trn, report_sum_trn = test(model, train_loader, dropout, criterion, use_gpu, edge_info)
                    if use_val:
                        val_acc, val_loss, out_val, true_val, pred_val, f1_val, precision_val, \
                        recall_val, report_sum_val = test(model, val_loader, dropout, criterion, use_gpu, edge_info)
                    test_acc, testing_loss, out, true, pred, f1_test, precision_test, \
                    recall_test, report_sum_test = test(model, test_loader, dropout, criterion, use_gpu, edge_info)
                    save_pth = "../../data/compare_gcn/seed_" + str(seed) + "/"
                    if not os.path.isdir(save_pth):
                        os.makedirs(save_pth)
                    gcn_n = [["", "f1", "prec", "rec"]]
                    gcn_n = save_score(gcn_n, true_trn, true_val, true, pred_trn, pred_val, pred)
                    print(report_sum_trn, report_sum_trn[148:152])
                    print(report_sum_val, report_sum_val[148:152])
                    print(report_sum_test, report_sum_test[148:152])
                    print(gcn_n)
        elif flag == "test_mlp" and compare_flag == "ansamble":
            sd = str(seeds[seed_no])
            ker = str(middle_channels)
            num_lyr = str(pre_layer)
            csv_path = "../../data/best_results/MLP/origindata_seed_" + str(
                sd) + "_drop_0.5/lyr_" + num_lyr + "_chn_" + ker + "_all_results.csv"
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                prams_glob = [s for s in reader]
                prams_glob = prams_glob[len(prams_glob) - 1]
                print("mlp:", prams_glob)
                mlp_epoch_glob = int(prams_glob[0])
                mlp_seed_glob = prams_glob[5]
                if mlp_seed_glob != sd:
                    mlp_epoch_glob = 399
                mlp_ker_glob = prams_glob[7]
                mlp_late_glob = prams_glob[8]
            optim_name = "SGD"
            feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted_normed.csv"
            with open(feat_path, "r") as f:
                reader = csv.reader(f)
                global_feat_set_tmp = [row[2:] for row in reader]
                global_feat_set = global_feat_set_tmp[1:]
                # global_feat_set = [[s.replace("na", "0") for s in row] for row in global_feat_set]
                global_feat_set = [list(map(float, row)) for row in global_feat_set]
                feat_numpy = np.array(global_feat_set)
                feat_tensor = torch.tensor(feat_numpy).float()
            num_graph, num_feat = torch.tensor(feat_numpy).size()
            gt_tmp = np.zeros((num_graph, 1), dtype=np.int32)
            gt_tmp[hit_index] = 1
            train_gt = gt_tmp[train_index]
            val_gt = gt_tmp[val_index]
            test_gt = gt_tmp[test_index]
            train_gt = torch.tensor(train_gt)
            val_gt = torch.tensor(val_gt)
            test_gt = torch.tensor(test_gt)
            in_channels = num_feat
            tmp_feat = torch.zeros(1, int(ker) + 2, dtype=torch.float)
            pretrained_model_path = "../../results/for_GCN_dataset/nonrapped/mlp_models/compare/seeds_" + sd + "_" + ker + "/" + optim_name + "_lyr_" + num_lyr + "_drop_" \
                                    + str(pre_dropout) + "_init_rate_" + mlp_late_glob + "_sampled/model%05d.pth" % (mlp_epoch_glob)
            test_feat_tensor = feat_tensor[test_index]
            val_feat_tensor = feat_tensor[val_index]
            train_feat_tensor = feat_tensor[train_index]
            *_, mlp_pred_test, mlp_label_test = mlp_dim_reduce(test_feat_tensor, in_channels, int(ker), pre_layer,
                                                               out_channels, n_connected_layer, pretrained_model_path,
                                                               pre_dropout,
                                                               device, tmp_feat)
            *_, mlp_pred_val, mlp_label_val = mlp_dim_reduce(val_feat_tensor, in_channels, int(ker), pre_layer,
                                                             out_channels, n_connected_layer, pretrained_model_path,
                                                             pre_dropout,
                                                             device, tmp_feat)
            *_, mlp_pred_trn, mlp_label_trn = mlp_dim_reduce(train_feat_tensor, in_channels, int(ker), pre_layer,
                                                             out_channels, n_connected_layer, pretrained_model_path,
                                                             pre_dropout,
                                                             device, tmp_feat)
            mlp_f1 = [["", "f1", "prec", "rec"]]
            mlp_trn_f1 = f1_score(train_gt.cpu(), torch.tensor(mlp_label_trn).cpu(),
                                  average=None, zero_division=0)
            mlp_trn_prec = precision_score(train_gt.cpu(),
                                           torch.tensor(mlp_label_trn).cpu(), average=None, zero_division=0)
            mlp_trn_rec = recall_score(train_gt.cpu(),
                                       torch.tensor(mlp_label_trn).cpu(), average=None, zero_division=0)
            mlp_f1.append(["mlp_trn", mlp_trn_f1[1], mlp_trn_prec[1], mlp_trn_rec[1]])
            mlp_val_f1 = f1_score(val_gt.cpu(), torch.tensor(mlp_label_val).cpu(),
                                  average=None, zero_division=0)
            mlp_val_prec = precision_score(val_gt.cpu(),
                                           torch.tensor(mlp_label_val).cpu(), average=None, zero_division=0)
            mlp_val_rec = recall_score(val_gt.cpu(),
                                       torch.tensor(mlp_label_val).cpu(), average=None, zero_division=0)
            mlp_f1.append(["mlp_val", mlp_val_f1[1], mlp_val_prec[1], mlp_val_rec[1]])
            mlp_test_f1 = f1_score(test_gt.cpu(), torch.tensor(mlp_label_test).cpu(),
                                   average=None, zero_division=0)
            mlp_test_prec = precision_score(test_gt.cpu(),
                                            torch.tensor(mlp_label_test).cpu(), average=None, zero_division=0)
            mlp_test_rec = recall_score(test_gt.cpu(),
                                        torch.tensor(mlp_label_test).cpu(), average=None, zero_division=0)
            mlp_f1.append(["mlp_test", mlp_test_f1[1], mlp_test_prec[1], mlp_test_rec[1]])

            save_pth = "../../data/compare_mlp/seed_" + str(seed) + "/"
            if not os.path.isdir(save_pth):
                os.makedirs(save_pth)
            with open(save_pth + "lyr_" + num_lyr + "_chn_" + ker + "-score.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(mlp_f1)

        elif flag != "learning_mlp" or flag != "learning_autoencoder":
            tmp_acc = 0
            for params in grid:
                if flag == "check_supernode":
                    model = eval(
                        model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6+"_supernode")(
                        dataset, out_channels=params['kernels'],
                        n_hidden_layers=params['n_hidden_layers'])
                elif model_name == "GCN":
                    model = GCN(dataset, hidden_channels=params['kernels'], n_hidden_layers=params['n_hidden_layers'])
                elif model_name == "PDN":
                    if model_type7 == "_concat_after":
                        model = eval(model_name+model_type1+model_type2+model_type3+model_type4+model_type5+model_type6
                                     +model_type7)(dataset, device=device, out_channels=params['kernels'],
                                                   n_hidden_layers=params['n_hidden_layers'], glob_feat_pth=global_feat_path)
                    else:
                        model = eval(
                            model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6)(
                            dataset, out_channels=params['kernels'],
                            n_hidden_layers=params['n_hidden_layers'])
                elif model_name == "GAT":
                    model = GAT(dataset, out_channels=params['kernels'], n_hidden_layers=params['n_hidden_layers'])
                elif model_name == "GATv2":
                    model = GATv2(dataset, out_channels=params['kernels'], n_hidden_layers=params['n_hidden_layers'])
                else:
                    print("The model you specified does not exist. This process is finished.")
                    return 0
                print('model:', model)
                print('--- current para in grid search ---')
                print('parameter_grid', params)
                if flag == "check_model":
                    return 0

                criterion = torch.nn.CrossEntropyLoss()
                # optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])
                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
                scheduler = MultiStepLR(optimizer, milestones=[20, 200, 800, 2000, 5000], gamma=0.5)
                # scheduler = ExponentialLR(optimizer, gamma=0.9)

                max_recall_hit = 0
                max_recall_nonhit = 0

                sum_training_loss = []
                sum_testing_loss = []
                sum_val_loss = []
                sum_training_acc = []
                sum_testing_acc = []
                sum_val_acc = []
                sum_learning_rate = []

                full_connect_channels = ""
                if model_type4 == "2lin":
                    full_connect_channels = "full_chs" + str(int(params['kernels'])) + "_" + str(int(params['kernels']/2)) + "_2_"
                if model_type4 == "_set":
                    model_type4 = model_type4 + "_2_" + str(processing_step)
                main_info = model_name + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6 \
                            + "_sample_" + str(sampling_flag) + "_v_late_" + full_connect_channels \
                            + name_dataset + "_test_hit_" + str(test_hit) \
                            + "_data_seed" + str(seed) \
                            + "_ratio" + str(ratio_for_training) + "_knl" \
                            + str(params['kernels']) + "_lyr" + str(params['n_hidden_layers']) \
                            + "initial_lr_" + str(params['learning_rate'])
                if flag != "test":
                    main_info = main_info + "_" + flag
                if attr_flag:
                    main_info = main_info + "_attr_True"
                if sampling_each_epochs_flag:
                    main_info = main_info + "_each_sampled"
                if ae_name_dataset == "79784_2nd_mini_5666_AlvaDesc":
                    main_info = main_info + "_79784AE_v2"
                elif ae_name_dataset == "wo3D_79784_2nd_mini_5666_AlvaDesc":
                    main_info = main_info + "_wo3D_79784AE"
                if flag == "toy_node":
                    main_info = main_info + "_toy_node"
                elif flag == "toy_edge":
                    main_info = main_info + "_toy_edge"
                temp_models_saved_folder = "../../result/" + main_info + "/model(" + main_info + ")"
                results_saved_folder = "../../result/" + main_info + "/results(" + main_info + ")"

                os.makedirs(temp_models_saved_folder, exist_ok=True)
                os.makedirs(results_saved_folder, exist_ok=True)

                pdf_path = "../../result/" + main_info + "/"
                random.seed(seeds[seed_no])

                tmp_saved_data = []
                saved_epoch = 0
                saved_val_acc_of_epoch = 0
                step_count = 0

                for epoch in tqdm(range(n_epochs)):
                    # for each epoch train-data is sampling
                    if sampling_each_epochs_flag:
                        train_hit_index = random.sample(list(set(hit_index) - set(test_hit_index)), train_hit)
                        train_nonhit_index = random.sample(list(set(nonhit_index)-set(test_nonhit_index)), train_nonhit)
                        train_index = sorted(train_hit_index + train_nonhit_index)
                        train_dataset = dataset[train_index]
                        train_loader = DataLoader(train_dataset, batch_size=n_batches, shuffle=False)

                    train(model, train_loader, dropout, criterion, optimizer, epoch, use_gpu, edge_info)
                    train_acc, training_loss, *_, true_trn, pred_trn, f1_trn, precision_trn, \
                    recall_trn, report_sum_trn = test(model, train_loader, dropout, criterion, use_gpu, edge_info)
                    if use_val:
                        val_acc, val_loss, *_, true_val, pred_val, f1_val, precision_val, \
                        recall_val, report_sum_val = test(model, val_loader, dropout, criterion, use_gpu, edge_info)
                    else:
                        precision_val = [0, 0]
                        recall_val = [0, 0]
                    scheduler.step()
                    test_acc, testing_loss, *_, true, pred, f1_test, precision_test, \
                    recall_test, report_sum_test = test(model, test_loader, dropout, criterion, use_gpu, edge_info)

                    if epoch == 0:
                        able_learn_flag = False
                        saved_epoch = epoch
                        if precision_val[1] == 0 and recall_val[1] == 0:
                            saved_val_acc_of_epoch = 0
                        else:
                            saved_val_acc_of_epoch = 4 * precision_val[1] * recall_val[1] / (recall_val[1] + 2 * precision_val[1])
                    elif epoch == n_epochs-1:
                        if precision_val[1] == 0 and recall_val[1] == 0:
                            tmp_val_acc = 0
                        else:
                            tmp_val_acc = 4 * precision_val[1] * recall_val[1] / (recall_val[1] + 2 * precision_val[1])
                        able_learn_flag = True
                        saved_val_acc_of_epoch = tmp_val_acc
                        saved_epoch = epoch
                        step_count = 30
                        saved_pred = pred
                        saved_pred_trn = pred_trn
                        saved_report_sum_test = report_sum_test
                        saved_report_sum_trn = report_sum_trn
                        save_model = model.state_dict().copy()
                        if use_val:
                            saved_pred_val = pred_val
                            saved_report_sum_val = report_sum_val
                            saved_sum_training_acc, saved_sum_val_acc, saved_sum_testing_acc, saved_sum_training_loss, saved_sum_val_loss, saved_sum_testing_loss, \
                            saved_sum_learning_rate = acc_loss_collection_with_val(sum_training_loss, sum_val_loss, sum_testing_loss, sum_training_acc,
                                                            sum_val_acc, sum_testing_acc, sum_learning_rate, train_acc,
                                                            train_dataset, val_acc, val_dataset,
                                                            test_acc, test_dataset, training_loss, val_loss, testing_loss,
                                                            scheduler)
                        else:
                            saved_sum_training_acc, saved_sum_testing_acc, saved_sum_training_loss, saved_sum_testing_loss,\
                            saved_sum_learning_rate = acc_loss_collection(sum_training_loss, sum_testing_loss,
                                                                          sum_training_acc,
                                                                          sum_testing_acc, sum_learning_rate, train_acc,
                                                                          train_dataset,
                                                                          test_acc, test_dataset, training_loss, testing_loss,
                                                                          scheduler)
                    else:
                        if precision_val[1] == 0 and recall_val[1] == 0:
                            tmp_val_acc = 0
                        else:
                            tmp_val_acc = 4 * precision_val[1] * recall_val[1] / (recall_val[1] + 2 * precision_val[1])
                        if tmp_val_acc > saved_val_acc_of_epoch and recall_val[1] > 0.5:
                            print("\n\nあと30epoch!\n\n")
                            able_learn_flag = True
                            saved_val_acc_of_epoch = tmp_val_acc
                            saved_epoch = epoch
                            step_count = 0
                            saved_pred = pred
                            saved_pred_trn = pred_trn
                            saved_report_sum_test = report_sum_test
                            saved_report_sum_trn = report_sum_trn
                            save_model = model.state_dict().copy()
                            if use_val:
                                saved_pred_val = pred_val
                                saved_report_sum_val = report_sum_val
                                saved_sum_training_acc, saved_sum_val_acc, saved_sum_testing_acc, saved_sum_training_loss, saved_sum_val_loss, saved_sum_testing_loss, \
                                saved_sum_learning_rate = acc_loss_collection_with_val(sum_training_loss, sum_val_loss, sum_testing_loss, sum_training_acc,
                                                                sum_val_acc, sum_testing_acc, sum_learning_rate, train_acc,
                                                                train_dataset, val_acc, val_dataset,
                                                                test_acc, test_dataset, training_loss, val_loss, testing_loss,
                                                                scheduler)
                            else:
                                saved_sum_training_acc, saved_sum_testing_acc, saved_sum_training_loss, saved_sum_testing_loss, \
                                saved_sum_learning_rate = acc_loss_collection(sum_training_loss, sum_testing_loss,
                                                                              sum_training_acc,
                                                                              sum_testing_acc, sum_learning_rate, train_acc,
                                                                              train_feat,
                                                                              test_acc, val_feat, training_loss, testing_loss,
                                                                              scheduler)
                        else:
                            step_count += 1

                    if use_val:
                        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
                        sum_training_acc, sum_val_acc, sum_testing_acc, sum_training_loss, sum_val_loss, sum_testing_loss, \
                        sum_learning_rate = acc_loss_collection_with_val(sum_training_loss, sum_val_loss, sum_testing_loss, sum_training_acc,
                                                                sum_val_acc, sum_testing_acc, sum_learning_rate, train_acc,
                                                                train_dataset, val_acc, val_dataset,
                                                                test_acc, test_dataset, training_loss, val_loss, testing_loss,
                                                                scheduler)
                    else:
                        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
                        sum_training_acc, sum_testing_acc, sum_training_loss, sum_testing_loss, \
                        sum_learning_rate = acc_loss_collection(sum_training_loss, sum_testing_loss, sum_training_acc,
                                                                sum_testing_acc, sum_learning_rate, train_acc, train_dataset,
                                                                test_acc, test_dataset, training_loss, testing_loss, scheduler)

                    label_dict = ('non-hit', 'hit')
                    if flag == "toy_test":
                        label_dict = ("0", "1", "2", "3")
                    elif flag == "toy_node" or flag == "toy_edge":
                        label_dict = ("0", "1")
                    if epoch != 0 and epoch % 10 == 0 and save_pred == False:
                        # torch.save(model.state_dict(), temp_models_saved_folder + '/model%05d.pth' % epoch)

                        with PdfPages(pdf_path + ".pdf") as pdf:
                            if use_val:
                                output_loss_with_val(epoch, sum_training_loss, sum_val_loss, sum_testing_loss, sum_training_acc, sum_val_acc, sum_testing_acc,
                                            sum_learning_rate, results_saved_folder)
                                additional_info = "parameters\n" + str(params) + "\nTesting  ---------------------------\n" \
                                                  + report_sum_test + "\nValidation  ---------------------------\n" + report_sum_val\
                                                  + "\nTraining  ---------------------------\n" + report_sum_trn
                                pdf.attach_note(additional_info, positionRect=[10, 500, 10, 10])
                                pdf.savefig()

                                y_pred = pred
                                y_true = true
                                plot_confusion_matrix(torch.tensor(y_true, device='cpu'),
                                                      torch.tensor(y_pred, device='cpu'),
                                                      classes=label_dict, results_saved_folder=results_saved_folder,
                                                      epoch=epoch,
                                                      trn_or_test="Test", cmap="Blues")
                                pdf.savefig()

                                y_pred = pred_val
                                y_true = true_val
                                plot_confusion_matrix(torch.tensor(y_true, device='cpu'),
                                                      torch.tensor(y_pred, device='cpu'),
                                                      classes=label_dict, results_saved_folder=results_saved_folder,
                                                      epoch=epoch, trn_or_test="Validation",
                                                      title="cfm on validation", cmap="Oranges")
                                pdf.savefig()

                                y_pred = pred_trn
                                # print(y_pred)
                                y_true = true_trn
                                # print(y_true)
                                plot_confusion_matrix(torch.tensor(y_true, device='cpu'),
                                                      torch.tensor(y_pred, device='cpu'),
                                                      classes=label_dict, results_saved_folder=results_saved_folder,
                                                      epoch=epoch,
                                                      title="cfm on training", cmap="Greens")
                                pdf.savefig()

                                print("Testing\n", report_sum_test)
                                print("Training\n", report_sum_trn)
                                print("Validation\n", report_sum_val)
                            else:
                                output_loss(epoch, sum_training_loss, sum_testing_loss, sum_training_acc, sum_testing_acc, sum_learning_rate, results_saved_folder)
                                additional_info = "parameters\n" + str(params) + "\nTesting  ---------------------------\n" \
                                                  + report_sum_test + "\nTraining  ---------------------------\n" + report_sum_trn
                                pdf.attach_note(additional_info, positionRect=[10, 500, 10, 10])
                                pdf.savefig()

                                y_pred = pred
                                y_true = true
                                # print(y_pred, y_true)
                                plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                                      classes=label_dict, results_saved_folder=results_saved_folder, epoch=epoch,
                                                      trn_or_test="Test", cmap="Blues")
                                pdf.savefig()

                                y_pred = pred_trn
                                # print(y_pred)
                                y_true = true_trn
                                # print(y_true)
                                plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                                      classes=label_dict, results_saved_folder=results_saved_folder, epoch=epoch,
                                                      title="cfm on training", cmap="Greens")
                                pdf.savefig()

                                print("Testing\n", report_sum_test)
                                print("Training\n", report_sum_trn)
                    elif epoch == 0 and flag == "use_mlp" and save_pred == False:
                        test_feat_tensor = feat_tensor[test_index]
                        val_feat_tensor = feat_tensor[val_index]
                        train_feat_tensor = feat_tensor[train_index]
                        *_, mlp_pred_test = mlp_dim_reduce(test_feat_tensor, in_channels, int(ker), pre_layer,
                                                       out_channels, n_connected_layer, pretrained_model_path, pre_dropout,
                                                       device, tmp_feat)
                        *_, mlp_pred_val = mlp_dim_reduce(val_feat_tensor, in_channels, int(ker), pre_layer,
                                                           out_channels, n_connected_layer, pretrained_model_path,
                                                           pre_dropout,
                                                           device, tmp_feat)
                        *_, mlp_pred_train = mlp_dim_reduce(train_feat_tensor, in_channels, int(ker), pre_layer,
                                                        out_channels, n_connected_layer, pretrained_model_path, pre_dropout,
                                                        device, tmp_feat)
                        with PdfPages(pdf_path + "_mlps.pdf") as pdf:
                            y_pred = mlp_pred_test
                            y_true = true
                            plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                                  classes=label_dict, results_saved_folder=results_saved_folder,
                                                  epoch=epoch,
                                                  trn_or_test="Test", cmap="Blues")
                            pdf.savefig()

                            y_pred = mlp_pred_val
                            y_true = true_val
                            plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                                  classes=label_dict, results_saved_folder=results_saved_folder,
                                                  epoch=epoch,
                                                  trn_or_test="Validation", cmap="Oranges")
                            pdf.savefig()

                            y_pred = mlp_pred_train
                            # print(y_pred)
                            y_true = true_trn
                            # print(y_true)
                            plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                                  classes=label_dict, results_saved_folder=results_saved_folder,
                                                  epoch=epoch,
                                                  title="cfm on training", cmap="Greens")
                            pdf.savefig()
                    if save_pred and epoch == save_pred_epoch:
                        if use_global_feat == " ":
                            for_words = ["add"]
                            for words in for_words:
                                print("Gooood")
                                pres_pth_test = "../../results/pred_" + words + "_test_0711.csv"
                                pres_pth_train = "../../results/pred_" + words + "_train_0711.csv"
                                pred_type = "no_glob"
                                true = ["GT"] + [t.item() for t in true]
                                true_trn = ["GT"] + [t.item() for t in true_trn]
                                pred = [pred_type] + [p.item() for p in pred]
                                pred_trn = [pred_type] + [pt.item() for pt in pred_trn]
                                if os.path.isfile(pres_pth_test):
                                    with open(pres_pth_test, "a") as f:
                                        writer = csv.writer(f)
                                        writer.writerow(pred)
                                    with open(pres_pth_train, "a") as f:
                                        writer = csv.writer(f)
                                        writer.writerow(pred_trn)
                                else:
                                    feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted.csv"
                                    with open(feat_path, "r") as f:
                                        reader = csv.reader(f)
                                        global_feat_set_tmp = [row[1] for row in reader]
                                        sorted_mol_name = global_feat_set_tmp[1:]
                                        sorted_mol_name = np.array(sorted_mol_name)
                                    with open(pres_pth_test, "w") as f:
                                        writer = csv.writer(f)
                                        test_label = ["mol_ID"] + list(sorted_mol_name[test_index])
                                        writer.writerow(test_label)
                                        writer.writerow(true)
                                        writer.writerow(pred)
                                    with open(pres_pth_train, "w") as f:
                                        writer = csv.writer(f)
                                        train_label = ["mol_ID"] + list(sorted_mol_name[train_index])
                                        writer.writerow(train_label)
                                        writer.writerow(true_trn)
                                        writer.writerow(pred_trn)
                        else:
                            if use_global_feat == "_concat_first":
                                gf_type = "cat"
                            elif use_global_feat == "_add_node":
                                gf_type = "add"
                            pres_pth_test = "../../results/pred_" + gf_type + "_test_0711.csv"
                            pres_pth_train = "../../results/pred_" + gf_type + "_train_0711.csv"
                            if flag == "test":
                                if use_global_feat == " ":
                                    pred_type = "no_glob"
                                else:
                                    pred_type = "raw_glob"
                            elif flag == "use_mlp":
                                pred_type = "mlp_glob"
                            elif flag == "use_autoencoder":
                                pred_type = "ae_glob"
                            true = ["GT"] + [t.item() for t in true]
                            true_trn = ["GT"] + [t.item() for t in true_trn]
                            pred = [pred_type] + [p.item() for p in pred]
                            pred_trn = [pred_type] + [pt.item() for pt in pred_trn]
                            if os.path.isfile(pres_pth_test):
                                with open(pres_pth_test, "a") as f:
                                    writer = csv.writer(f)
                                    writer.writerow(pred)
                                with open(pres_pth_train, "a") as f:
                                    writer = csv.writer(f)
                                    writer.writerow(pred_trn)
                            else:
                                feat_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted.csv"
                                with open(feat_path, "r") as f:
                                    reader = csv.reader(f)
                                    global_feat_set_tmp = [row[1] for row in reader]
                                    sorted_mol_name = global_feat_set_tmp[1:]
                                    sorted_mol_name = np.array(sorted_mol_name)
                                with open(pres_pth_test, "w") as f:
                                    writer = csv.writer(f)
                                    test_label = ["mol_ID"] + list(sorted_mol_name[test_index])
                                    writer.writerow(test_label)
                                    writer.writerow(true)
                                    writer.writerow(pred)
                                with open(pres_pth_train, "w") as f:
                                    writer = csv.writer(f)
                                    train_label = ["mol_ID"] + list(sorted_mol_name[train_index])
                                    writer.writerow(train_label)
                                    writer.writerow(true_trn)
                                    writer.writerow(pred_trn)
                        return 0
                    if step_count == 30 and able_learn_flag:
                        torch.save(save_model, temp_models_saved_folder + '/model%05d.pth' % saved_epoch)

                        with PdfPages(pdf_path + ".pdf") as pdf:
                            if use_val:
                                output_loss_with_val(saved_epoch, saved_sum_training_loss, saved_sum_val_loss, saved_sum_testing_loss, saved_sum_training_acc, saved_sum_val_acc, saved_sum_testing_acc,
                                            saved_sum_learning_rate, results_saved_folder)
                                additional_info = "parameters\n" + str(params) + "\nTesting  ---------------------------\n" \
                                                  + saved_report_sum_test + "\nValidation  ---------------------------\n" + saved_report_sum_val\
                                                  + "\nTraining  ---------------------------\n" + saved_report_sum_trn
                                pdf.attach_note(additional_info, positionRect=[10, 500, 10, 10])
                                pdf.savefig()

                                y_pred = saved_pred
                                y_true = true
                                plot_confusion_matrix(torch.tensor(y_true, device='cpu'),
                                                      torch.tensor(y_pred, device='cpu'),
                                                      classes=label_dict, results_saved_folder=results_saved_folder,
                                                      epoch=saved_epoch,
                                                      trn_or_test="Test", cmap="Blues")
                                pdf.savefig()

                                y_pred = saved_pred_val
                                y_true = true_val
                                plot_confusion_matrix(torch.tensor(y_true, device='cpu'),
                                                      torch.tensor(y_pred, device='cpu'),
                                                      classes=label_dict, results_saved_folder=results_saved_folder,
                                                      epoch=saved_epoch, trn_or_test="Validation",
                                                      title="cfm on validation", cmap="Oranges")
                                pdf.savefig()

                                y_pred = saved_pred_trn
                                # print(y_pred)
                                y_true = true_trn
                                # print(y_true)
                                plot_confusion_matrix(torch.tensor(y_true, device='cpu'),
                                                      torch.tensor(y_pred, device='cpu'),
                                                      classes=label_dict, results_saved_folder=results_saved_folder,
                                                      epoch=saved_epoch,
                                                      title="cfm on training", cmap="Greens")
                                pdf.savefig()

                                print("Testing\n", saved_report_sum_test)
                                print("Training\n", saved_report_sum_trn)
                                print("Validation\n", saved_report_sum_val)
                            else:
                                output_loss(saved_epoch, saved_sum_training_loss, saved_sum_testing_loss, saved_sum_training_acc, saved_sum_testing_acc, saved_sum_learning_rate, results_saved_folder)
                                additional_info = "parameters\n" + str(params) + "\nTesting  ---------------------------\n" \
                                                  + saved_report_sum_test + "\nTraining  ---------------------------\n" + saved_report_sum_trn
                                pdf.attach_note(additional_info, positionRect=[10, 500, 10, 10])
                                pdf.savefig()

                                y_pred = saved_pred
                                y_true = true
                                plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                                      classes=label_dict, results_saved_folder=results_saved_folder, epoch=saved_epoch,
                                                      trn_or_test="Test", cmap="Blues")
                                pdf.savefig()

                                y_pred = saved_pred_trn
                                # print(y_pred)
                                y_true = true_trn
                                # print(y_true)
                                plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                                      classes=label_dict, results_saved_folder=results_saved_folder, epoch=saved_epoch,
                                                      title="cfm on training", cmap="Greens")
                                pdf.savefig()

                                print("Testing\n", saved_report_sum_test)
                                print("Training\n", saved_report_sum_trn)
                        tmp_saved_data.append(saved_epoch)
                        tmp_saved_data.append(saved_val_acc_of_epoch)
                        tmp_saved_data.append(saved_report_sum_trn[148:152])
                        if use_val:
                            tmp_saved_data.append(saved_report_sum_val[148:152])
                        tmp_saved_data.append(saved_report_sum_test[148:152])
                        break
                with open(pdf_path + "/recall_lyr_" + str(params['n_hidden_layers']) + "_rate_" + \
                          str(params['learning_rate']) + "_ker_" + str(params['kernels']) + ".csv", "w") as cs:
                    writer = csv.writer(cs)
                    writer.writerow(recall_results)
                with open(pdf_path + "/recall_best_lyr_" + str(params['n_hidden_layers']) + "_rate_" + \
                          str(params['learning_rate']) + "_ker_" + str(params['kernels']) + ".csv", "w") as cs:
                    writer = csv.writer(cs)
                    writer.writerow(recall_best_results)
                if tmp_acc < tmp_saved_data[1]:
                    tmp_acc = tmp_saved_data[1]
                    saved_data = tmp_saved_data
                    saved_data.append(seed)
                    saved_data.append(params["n_hidden_layers"])
                    saved_data.append(params["kernels"])
                    saved_data.append(params["learning_rate"])
                elif tmp_acc == 0 and tmp_saved_data[1] == 0:
                    tmp_acc = tmp_saved_data[1]
                    saved_data = tmp_saved_data
                    saved_data.append(seed)
                    saved_data.append(params["n_hidden_layers"])
                    saved_data.append(params["kernels"])
                    saved_data.append(params["learning_rate"])

            if use_val:
                temp_models_saved_folder = "../../data/best_results/GCN/origindata_seed_" + str(saved_data[5]) + "_drop_" + str(
                dropout) + "use_val"
            else:
                temp_models_saved_folder = "../../data/best_results/GCN/origindata_seed_" + str(saved_data[4]) + "_drop_" + str(
                    dropout)
                # use_global_feat = " "  # , _concat_first, _concat_after, _add_node
            if flag == "test":
                add_tmp = "dragon_"
                if use_global_feat == " ":
                    add_tmp = "no_glob_"
            elif flag == "use_mlp":
                add_tmp = "mlp_"
            elif flag == "use_autoencoder":
                add_tmp = "auto_"
            else:
                add_tmp = "toy"
            if use_global_feat == "_concat_after":
                add_tmp = add_tmp + "catafter_"
            elif use_global_feat == "_concat_first":
                add_tmp = add_tmp + "catfirst_"
            elif use_global_feat == "_add_node":
                add_tmp = add_tmp + "addnode_"
            add_tmp = add_tmp + model_type1 + model_type2 + model_type3 + model_type4 + model_type5 + model_type6 + model_type7
            file_path_tmp = temp_models_saved_folder + "/" + add_tmp
            os.makedirs(temp_models_saved_folder, exist_ok=True)
            if os.path.isfile(file_path_tmp + "all_results.csv"):
                with open(file_path_tmp + "all_results.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(saved_data)
            else:
                with open(file_path_tmp + "all_results.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(saved_data)


if __name__ == '__main__':
    Main()