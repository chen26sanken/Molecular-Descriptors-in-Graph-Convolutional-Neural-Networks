import fnmatch
import math
import os
# import cv2
import numpy as np
from pathlib import Path
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import csv
import torch
import torch.nn as nn
# from torch_scatter import scatter
from submodules import *
# from rdkit import Chem
# from rdkit.Chem import Draw, AllChem
import pathlib
from pdf2image import convert_from_path
import sys
import pyocr
import pyocr.builders
import pandas as pd
from scipy import stats

def rewrite_sdf(path):
    with open(path, 'r', errors='ignore') as f:
        src = f.read()
        src2 = src.split('\n')[0]
        src3 = src2.split('-')
        src4 = str(int(src3[0]) + 500)
        re_src = src4 + "-" + src3[1] + "\n"
        src = src.replace(src2+"\n", re_src)
    with open(path, 'w', errors='ignore') as f:
        f.write(src)

#check hit and nonhit information
def Main():
    path = '../../data/origin/hits_ref'
    with open(path, 'r') as f:
        a = f.read()
        a = a.split('\n')
        a = a[:len(a)-1]
        print(len(a), "\n")
        # print(a, len(a))

    path2 = '../../data/augmented_data/nonhits_sampling'
    with open(path2, 'r') as f:
        b = f.read()
        b = b.split('\n')
        b = b[:len(b)-1]
        # print(len(b), "\n")
        # print(b, len(b))

    path3 = '../../data/augmented_data/hits_ref'
    with open(path3, 'r') as f:
        c = f.read()
        c = c.split('\n')
        c = c[:len(c)-1]
        # print(len(c), "\n")
        # print(c, len(c))

    # print('check1')
    path = '../../data/augmented_data/mol_data'
    # print(path)
    datas = sorted(os.listdir(path))
    # print(datas)
    # print('check2')
    a_2 = []
    b_2 = []
    count = 0
    h_count = 0
    h_count_2 = 0
    h_rm = 0
    h_rm_2 = 0
    n_count = 0
    n_count_2 = 0
    n_rm = 0
    n_rm_2 = 0
    h = np.zeros((len(a), 2), dtype=str)
    h = [[str(0) for i in range(2)] for j in range(len(a))]
    # print('check3')
    j = 0
    for i, data in enumerate(datas):
        # print(i)
        # print(j)
        if data.split("_")[0] in a:
            if j == 0:
                h[j][0] = data.split("_")[0]
                print(data.split("_")[0], h[j][0])
                h[j][1] = str(1)
                j += 1
            else:
                if data.split("_")[0] == h[j-1][0]:
                    h[j-1][1] = str(int(h[j-1][1]) + 1)
                else:
                    # print(data, h[j-1, 0])
                    j += 1
                    h[j-1][0] = data.split("_")[0]
                    h[j-1][1] = str(1)
            if not fnmatch.fnmatch(data, "*_*"):
                h_count += 1
                a_2.append(data.split("-")[0])
            else:
                h_rm += 1
                a_2.append(data.split("_")[0])
        elif data.split("-")[0] in b:
            if not fnmatch.fnmatch(data, '*_*'):
                n_count += 1
                b_2.append(data.split("-")[0])
            else:
                n_rm += 1
                b_2.append(data.split("_")[0])
        elif data.split("-")[0] in c:
            h_count_2 += 1
        else:
            n_count_2 += 1

    # print(a_2)
    # print(b_2)
    print('In test ==> hit:', h_count, ', hit_remove:', h_rm, ', nonhit:', n_count, ', non_remove:', n_rm)
    print("In trian==> hit:", h_count_2, ', nonhit:', n_count_2)

    print('test ratio:', h_count/n_count, ', train ratio:', h_count_2/n_count_2)
    print(h)
    print(j)
    with open("../../data/hit_num.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(h)

#count hit mols under No.n
def Main1_2():
    path = '../../data/check_edgeattr3/txt_nonhits/'
    files = os.listdir(path)
    files = sorted(files)
    all_filr = []
    count = 0

    for n, file in enumerate(files):
        file_num = int(file.split(".")[0].split("e")[2])
        all_filr.append(file_num)
        count += 1
        # if file_num < 3250:
        #     continue
        # elif file_num < 7500:
        #     count += 1
        # else:
        #     break
    print(all_filr)
    print("counts:", count)

#check data
def Main2():
    path = '../../data/dCAG1817_aug/raw/dCAG1817_aug_node_attributes.txt'
    path2 = '../../data/dCAG1817_aug/raw/dCAG1817_aug_node_attributes.txt'

    count = 0
    with open(path, 'r') as f:
        data = f.read().split('\n')
        for i, line in enumerate(data):
            pos = line.split(', ')
            count += 1
            # print(len(pos))
            for j in range(len(pos)):
                if pos[j] == '-3.5675.6594':
                    num = i
                    print(i, pos)
        print(count)

    # with open

#change color
def Main3():
    path = "../../result/10to1dCAG1817_aug_SGD_MultiStepLR obj_seed728349_ratio0.7_drp0.5_knl128_lyr4initial_lr_0.02/results(10to1dCAG1817_aug_SGD_MultiStepLR obj_seed728349_ratio0.7_drp0.5_knl128_lyr4initial_lr_0.02)/"
    names = ["Train_cfm00300.png", "Test_cfm00300.png"]
    re_names = ["Train_cfm00300_2.png", "Test_cfm00300_2.png"]

    for i, (name, re_name) in enumerate(zip(names, re_names)):
        img = cv2.imread(path+name)
        b, g, r = cv2.split(img)
        if i == 0:
            img2 = cv2.merge((r, b, g))
        else:
            img2 = cv2.merge((r, g, b))
        # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GBR)
        cv2.imwrite(path+re_name, img2)

#change (a, b) to (a, b), (b, a) in _A.txt
def Main4():
    path = '../../data/dCAG1817_aug2/'
    path2 = '../../data/dCAG1817_aug/'
    name3 = 'dCAG1817_aug_A.txt'
    name = 'dCAG1817_aug2_A.txt'
    name2 = 'dCAG1817_aug2_edge_labels.txt'
    rename = 'test_A.txt'
    rename3 = 'test2_A.txt'
    rename2 = 'test_edge_labels.txt'
    b = []
    redata = []
    redata2 = []
    with open(path2+name3, 'r') as f:
        with open(path+name2, 'r') as fp:
            data = f.read().split('\n')
            data2 = fp.read().split('\n')
            print(len(data), len(data2))
            for i, (line, bond) in enumerate(zip(data, data2)):
                if i != len(data)-1:
                    # print(i, line)
                    sub = line.split(', ')
                    b.append([int(sub[0]), int(sub[1])])
                    redata.append([int(sub[0]), int(sub[1])])
                    redata.append([int(sub[1]), int(sub[0])])
                    redata2.append(int(bond))
                    redata2.append(int(bond))
                if i == 10:
                    break
    print(len(redata), len(redata2))
    print(data[:10])
    print(redata[:20])
    with open(path+rename, 'w') as f:
        a = np.array(b)
        # print(a)
        np.savetxt(f, a, fmt='%i', delimiter=', ')
    with open(path+rename3, 'w') as f:
        a = np.array(redata)
        np.savetxt(f, a, fmt='%i', delimiter=', ')
    print('A.txt is completed.')
    # with open(path+name2, 'w') as f:
    #     a = np.array(redata2)
    #     # print(a)
    #     np.savetxt(f, a, fmt='%i')
    #     # for line in redata:
    #     #     a = np.array(line)
    #     #     print(a)
    #     #     np.savetxt(f, a, fmt='%i', delimiter=', ')
    # print('code is success.')

#create fake data
def Main5():
    path = '../../data/check_edgeattr/fake2/'
    files = sorted(os.listdir(path))

    for n, file in enumerate(files):
        rewrite_sdf(path + file)
        file_num, a_num = file.split(".")[0].split("-")
        file_num = int(file_num) + 500
        os.rename(path+file, path+str(file_num)+"-"+a_num+".txt")
        # if n == 0:
        #     break

# remove data we don't need
def Main6():
    dataname = "edge_evl_1"
    path = "../../data/" + dataname + "/raw/"
    files = os.listdir(path)
    name_list = ["_A", "_edge_labels", "_graph_indicator", "_graph_labels", "_node_attributes", "_node_labels"]
    name_list = ["_node_attributes"]
    edge_list = [1, 133, 265, 397]
    node_list = [1, 121, 241, 361]
    graph_list = [1, 13, 25, 37]

    for name in name_list:
        re_data = []
        with open(path+dataname+name+".txt", "r") as f:
            data = f.read().split("\n")
        if name == "_graph_labels":
            for num in graph_list:
                re_data.append(int(data[num-1]))
        elif name == "_A":
            for n, num in enumerate(edge_list):
                for i in range(0, 6):
                    split_data = data[num+i-1].split(", ")
                    re_data.append([int(split_data[0])-114*int(n), int(split_data[1])-114*int(n)])
        elif name == "_edge_labels":
            for num in edge_list:
                for i in range(0, 6):
                    re_data.append(int(data[num+i-1]))
        elif name == "_node_attributes":
            for num in node_list:
                for i in range(0, 6):
                    split_data = data[num+i-1].split(", ")
                    print(split_data)
                    re_data.append([float(split_data[0]), float(split_data[1]), float(split_data[2])])
        elif name == "_graph_indicator":
            for n, num in enumerate(node_list):
                for i in range(0, 6):
                    re_data.append(int(data[num+i-1])-11*int(n))
        else:
            for num in node_list:
                for i in range(0, 6):
                    re_data.append(int(data[num+i-1]))
        if name == "_node_attributes":
            with open(path + dataname + name + ".txt", 'w') as f:
                a = np.array(re_data)
                np.savetxt(f, a, fmt='%1.4f', delimiter=', ')
        else:
            with open(path+dataname+name+".txt", 'w') as f:
                a = np.array(re_data)
                np.savetxt(f, a, fmt='%i', delimiter=', ')

# change from number(a) to one-hot(...,0,0,a,0,0,...)
def Main7():
    dataname = "edge_evl_1"
    path = "../../data/" + dataname + "/raw/"
    name = dataname + "_edge_labels.txt"

    with open(path+name, "r") as f:
        data = f.read().split("\n")

    re_data = []

    for i in range(len(data)-1):
        tmp = np.zeros(3, dtype=int)
        tmp[int(data[i])] = 1
        re_data.append(tmp)

    with open(path+name, "w") as f:
        a = np.array(re_data)
        np.savetxt(f, a, fmt='%i', delimiter=', ')

# print csv file
def Main8():
    with open("../../data/check_sampling.csv", "r") as f:
        a = f.read().split("\n")
        b = np.array(a)
        print(b)

def Main9():
    sum = []
    a = [0]*(10)
    b = [0]*(10)
    c = [0]*(10)
    a_input = [1, 1, 2, 3, 4, 6]
    b_input = [1, 2, 4, 4, 4, 5, 7]
    # c_input = [1, 3, 3, 4]
    print(a, b)
    a[:len(a_input)] = a_input
    b[:len(b_input)] = b_input
    # c[:len(c_input)] = c_input
    print(a, b)
    a = np.reshape(a, [-1, 1])
    b = np.reshape(b, [-1, 1])
    print(a)
    sum.append(a)
    sum.append(b)
    # sum.append(c)
    print(sum)
    enc = OneHotEncoder(categories=[[1,2,3,4,5,6,7,8,9,10]], handle_unknown='ignore', sparse=False)
    # one_hot_feature_matrices = enc.fit_transform(sum)
    # print(one_hot_feature_matrices.shape)
    # print(one_hot_feature_matrices)
    one_hot_a = enc.fit_transform(a)
    print(one_hot_a)

def Main10():
    src = torch.randn(10, 6, 64)
    index = torch.tensor([0, 1, 0, 1, 2, 1])

    size = int(index.max().item() + 1)
    dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    # Broadcasting in the first and last dim.
    out = scatter(src, index, dim=-2, dim_size=dim_size, reduce="sum")
    out2 = scatter(src, index, dim=-2, dim_size=size, reduce="add")
    # out3 = scatter(src, index, dim=0, dim_size=size, reduce="add")

    print("sum", out)
    print("add", out2)
    # print("add_2", out3)

    if torch.max(out-out2) == 0:
        print("same matrix")

def Main11():
    datas = "../../AE_data/shared_file_79784_molecules/wo3D_79784_2nd_mini_5666_AlvaDesc.txt"  # datasはテキストファイルの場所
    # 保存するCSVファイルの場所
    file_csv = datas.replace("txt", "csv")

    # テキストファイルを開く
    with open(datas) as rf:
        # 書き込むＣＳＶファイルを開く
        with open(file_csv, "w") as wf:
            # テキストを１行ずつ読み込む
            # テキストの１行を要素としたlistになる
            readfile = rf.readlines()

            for n, read_text in enumerate(readfile):
                # listに分割
                read_text = read_text.split()
                if n == 1:
                    print(read_text[:10])
                # csvに書き込む
                writer = csv.writer(wf, delimiter=',')
                writer.writerow(read_text)

    with open(file_csv, "r") as f:
        print(f.read().split("\n")[1][:10])

def Main12():
    pre_sort = "../../RFcompare/dataset/descriptor_feature_RF.csv"
    post_sort = "../../RFcompare/dataset/descriptor_feature_RF_sorted.csv"

    sort_info = "../../data/origin/mol_sort.csv"
    with open(pre_sort) as f:
        reader = csv.reader(f)
        mol = [row for row in reader]
    with open(sort_info) as f:
        reader = csv.reader(f)
        info = [row[0].split(".")[0] for row in reader]
    sorted = []

    with open(post_sort, "w") as cs:
        writer = csv.writer(cs)
        writer.writerow(mol[0][:])
        # length = mol[0].split(',')
        length = mol[0]
        print(len(length))
        for name in info:
            # print(name)
            for i, row in enumerate(mol):
                if row[1] == name:
                    writer.writerow(row)

def Main13():
    seeds = [30000]
    tmp_path = "../../results_for_share/GCNs_test_results_seed_"
    tmp_ex = ".csv"
    for seed in seeds:
        path = tmp_path + str(seed) + tmp_ex
        with open(path, "r") as f:
            reader = csv.reader(f)
            # print(reader)
            datas = [row for row in reader]
            tmp_data = [row for row in reader]
            for n, row in enumerate(datas):
                if n == 0:
                    print(row)
                    # print(tmp_data[1])
                    continue
                else:
                    datas[n] = [r.split(",")[0].split("(")[1] for r in row]
            print(datas)
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerows(datas)

def Main14():
    seeds = [30000, 50000, 90000] + [10000*i+i for i in range(1, 11)] + [1000*i+i for i in range(11, 16)]+ [1000*i+i for i in range(21, 25)]
    seeds.sort()
    type = ["test"]
    save_data = [["seed_for_split", "c_hit_both", "c_hit_only_gcn", "c_hit_only_rf", "m_hit_both", "c_nonhit_both", "c_nonhit_only_gcn", "c_nonhit_only_rf", "m_nonhit_both"]]
    for seed in seeds:
        gcn_path = "../../results_for_share/GCNs_test_results_seed_" + str(seed) + ".csv"
        rf_path = "../../results_for_share/RF_test_results_seed_" + str(seed) + ".csv"
        with open(gcn_path, "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]
            index = data[0]
            gt_data = data[1]
            gcn_data = data[2]
        with open(rf_path, "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]
            rf_data = data[2]
        count_collect_hit_only_gcn = 0
        count_collect_hit_only_rf = 0
        count_collect_hit_both = 0
        count_miss_hit_both = 0
        count_collect_nonhit_only_gcn = 0
        count_collect_nonhit_only_rf = 0
        count_collect_nonhit_both = 0
        count_miss_nonhit_both = 0
        for gt, gcn, rf in zip(gt_data, gcn_data, rf_data):
            # print(gt, gcn, int(float(rf)))
            rf = str(int(float(rf)))
            if gt == "0":
                if gcn == "0" and rf == "0":
                    count_collect_nonhit_both += 1
                elif gcn == "1" and rf == "1":
                    count_miss_nonhit_both += 1
                elif gcn == "0" and rf == "1":
                    count_collect_nonhit_only_gcn += 1
                else:
                    count_collect_nonhit_only_rf += 1
            else:
                if gcn == "0" and rf == "0":
                    count_miss_hit_both += 1
                elif gcn == "1" and rf == "1":
                    count_collect_hit_both += 1
                elif gcn == "0" and rf == "1":
                    count_collect_hit_only_rf += 1
                else:
                    count_collect_hit_only_gcn += 1
        save_data.append([str(seed), str(count_collect_hit_both), str(count_collect_hit_only_gcn), \
                          str(count_collect_hit_only_rf), str(count_miss_hit_both), str(count_collect_nonhit_both), \
                          str(count_collect_nonhit_only_gcn), str(count_collect_nonhit_only_rf), str(count_miss_nonhit_both)])
        print("finish seed is" + str(seed))
    with open("../../results_for_share/count_of_test_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(save_data)

def Main15():
    inpath = "../../data/concat_nf_gf_evl/raw/concat_nf_gf_evl_A.txt"
    outpath = "../../data/concat_nf_gf_evl/raw/concat_nf_gf_evl_A_2.txt"

    with open(inpath, 'r') as inf:
        data = inf.read().split('\n')
        out = data.copy()
        for i, line in enumerate(data):
            pos = line.split(', ')
            # print(pos)
            if pos != [""]:
                pos = [int(pos[0])+24, int(pos[1])+24]
                outdata = str(pos[0]) + ", " + str(pos[1])
                out.append(outdata)
    with open(inpath, "w") as f:
        for s in out:
            f.write("%s\n" % s)

def make_global_feat():
    # path = "../../data/global_test/global_feat_"
    rng = np.random.default_rng(123456)

    feat_lengths = [2, 16, 32, 64, 128, 5666]
    for f_l in feat_lengths:
        path = "../../data/global_test/global_feat_diff_"
        feat_length = f_l
        path = path + str(feat_length) + ".csv"
        data_label = np.ones((1, feat_length))
        graph_label = np.array([["No.", "Name"], [1, "1-1"], [2, "1-2"],
                         [3, "1-3"], [4, "1-4"], [5, "1-1"], [6, "1-2"],
                         [7, "1-3"], [8, "1-4"]])
        raw_data = rng.random((4, feat_length))
        repeat_data = np.concatenate([data_label, raw_data, raw_data])
        data = np.concatenate([graph_label, repeat_data], axis=1)
        print(data.shape)



        # data = np.array([["No.", "Name", "ele1", "ele2"], [1, "1-1", 0, 1], [2, "1-2", 0, 1],
        #                  [3, "1-3", 0, 1], [4, "1-4", 0, 1], [5, "1-1", 0, 1], [6, "1-2", 0, 1],
        #                  [7, "1-3", 0, 1], [8, "1-4", 0, 1]])

        with open(path, "w") as cs:
            writer = csv.writer(cs)
            writer.writerows(data)

def test_relu():
    a = torch.tensor([[1, 2, -1], [1, 0, 3]])
    print(a)
    b = a.relu()
    print("\nuse x.relu()\n", b)
    m = nn.ReLU()
    c = m(a)
    print("\nuse nn.ReLU()\n", c)

def normalize_01():
    inpath = "../../AE_data/shared_file_79784_molecules/79784_2nd_mini_5666_AlvaDesc.csv"
    outpath = "../../AE_data/shared_file_79784_molecules/79784_2nd_mini_5666_AlvaDesc_normed.csv"

    with open(inpath, "r") as f:
        reader = csv.reader(f)
        global_feat_set_tmp = [row[2:] for row in reader]
        global_feat_set = global_feat_set_tmp[1:]
        global_feat_set = [[s.replace("na", "0") for s in row] for row in global_feat_set]
        global_feat_set = [list(map(float, row)) for row in global_feat_set]
        feat_numpy = np.array(global_feat_set)
        feat_tensor = torch.tensor(feat_numpy).float()
    list_2d = feat_tensor.tolist()
    proc = preprocessing.MinMaxScaler()
    norm_list = proc.fit_transform(list_2d)
    norm_tensor = torch.tensor(norm_list).float()
    name_tmp = torch.zeros(norm_tensor.size()[0], 2, dtype=torch.float)
    reduce_feat = torch.cat((name_tmp, norm_tensor), dim=1)
    tmp_feat = torch.zeros(1, reduce_feat.size()[1], dtype=torch.float)
    data_tmp = torch.cat((tmp_feat, reduce_feat), dim=0)
    # norm_tensor = torch.reshape(norm_tensor, (-1, 1, norm_tensor.size()[1]))
    with open(outpath, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_tmp.to('cpu').detach().numpy().copy())

def DrawMols():
    path = "../../data/origin/raw/"
    save_path = "../../data/origin/img/"
    os.makedirs(save_path, exist_ok=True)
    files = os.listdir(path)
    for file in files:
        suppl = Chem.SDMolSupplier(path + file)
        filename = file.replace(".sdf", "_2.png")
        filename_2 = file.replace(".sdf", "_2.svg")
        AllChem.Compute2DCoords(suppl[0])
        # m = Chem.AddHs(suppl[0])
        # AllChem.EmbedMolecule(m)
        # AllChem.MMFFOptimizeMolecule(m)
        # m2 = Chem.RemoveHs(m)
        Draw.MolToFile(suppl[0], save_path+filename)
        Draw.MolToFile(suppl[0], save_path + filename_2)

def sdftosmiles():
    mol_path = "../../data/origin/raw/"
    files = os.listdir(mol_path)
    path = "../../results/"
    date = "0711"
    con_type = ["add", "cat"]
    pre_type = ["test", "train"]
    for con in con_type:
        for pre in pre_type:
            csv_name = "pred_" + con + "_" + pre + "_" + date + ".csv"
            with open(path+csv_name, "r") as f:
                reader = csv.reader(f)


def MakeIndex():
    inpath = "../../RFcompare/dataset/descriptor_feature_RF_sorted.csv"
    outpath = "../../data/origin/mol_index.csv"

    with open(inpath, "r") as f:
        reader = csv.reader(f)
        global_feat_set_tmp = [row[1] for row in reader]
        sorted_mol_name = global_feat_set_tmp[1:]
        sorted_mol_name = np.array(sorted_mol_name)

    index_list = []
    mol_list = []
    mol_list_2 = []
    for n, name in enumerate(sorted_mol_name):
        print(name)
        f_name, s_name = name.split("-")
        if s_name == "1":
            index_list.append(n)
            mol_list.append(name)
            mol_list_2.append(name+"\n")

    with open(outpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(index_list)
        # writer.writerow(mol_list)

    with open("../../data/origin/mol_index.txt", "w") as f:
        f.writelines(mol_list_2)

def Check2ndID():
    all_path = "../../RFcompare/dataset/descriptor_feature_RF_sorted.csv"
    hit_path = "../../data/origin/hits_ref"

    hit_index = []

    with open(hit_path, "r") as f:
        hits = f.read()
        hits = hits.split("\n")
        for hit in hits:
            # print(hit)
            if hit == "":
                break
            overrap_flag = False
            fst, snd = hit.split("-")
            for index in hit_index:
                if fst == index:
                    overrap_flag = True
            if not overrap_flag:
                hit_index.append(fst)

        with open(all_path, "r") as f:
            reader = csv.reader(f)
            global_feat_set_tmp = [row[1] for row in reader]
            sorted_mol_name = global_feat_set_tmp[1:]
            sorted_mol_name = np.array(sorted_mol_name)

        for name in sorted_mol_name:
            exist_flag = False
            f_name = name.split("-")[0]
            for index in hit_index:
                if f_name == index:
                    for h_idx in hits:
                        if h_idx == name:
                            exist_flag = True
                    if not exist_flag:
                        print("おわた")
                    else:
                        print("いけた")
        print("よかた")
        print(len(hit_index), hit_index)

def checkshape_of_alvadexc():
    path = "../../AE_data/shared_file_79784_molecules/79784_2nd_mini_5666_AlvaDesc.txt"

    with open(path, 'r') as f:
        data = f.read().split('\n')
        count = 0
        for i, line in enumerate(data):
            pos = line.split('\t')
            count += 1
            if i==0:
                print(pos)
                print(len(pos))
            # if i == 0:
            #     for j in range(len(pos)):
            #         print(pos)
        print(count)

def correspo_alva_drag():
    alva_path = "../../AE_data/shared_file_79784_molecules/79784_2nd_mini_5666_AlvaDesc.txt"

def results_sum():
    seeds = [i*1010 for i in range(1, 51)]
    # names = ["dragon_addnode_all", "dragon_catafter_all", "dragon_catfirst_all", "mlp_addnode_all", "mlp_catafter_all", "mlp_catfirst_all", "no_globall"]
    # names = ["mlp_catafter_all"]
    # names = ["no_glob__dense_add_skip_1bro_pre_batch_act_max_testall", "no_glob__dense_add_skip_2bro_pre_batch_act_max_testall",
    #          "no_glob__dense_add_skip_3bro_pre_batch_act_max_testall", "no_glob__dense_add_skip_4bro_pre_batch_act_max_testall",
    #          "no_glob__dense_add_skip_5bro_pre_batch_act_max_testall", "no_glob__dense_add_skip_6bro_pre_batch_act_max_testall",
    #          "no_glob__dense_add_skip_6bro_original_act_max_testall", "no_glob__dense_add_skip_6bro_pre_batch_act_mean_testall",
    #          "no_glob__dense_add_skip_6bro_pre_relu_act_max_testall", "no_glob__dense_add_skip_6bro_pre_batch_act_sum_testall",
    #          "no_glob__dense_add_skip_6bro_pre_batch_act_attn_aggr_testall", "no_glob__dense_add_skip_6bro_pre_batch_act_max_test_2lin_v1all",
    #          "no_glob__dense_add_skip_6bro_pre_batch_act_max_test_2lin_v2all", "no_glob__dense_add_skip_6bro_pre_batch_act_set_aggr_testall",
    #          "no_glob__dense_add_skip_6bro_pre_batch_act_max_test_3lin_v1all", "no_glob__dense_add_skip_6bro_pre_batch_act_max_test_3lin_v2all",
    #          "no_glob__dense_add_skip_6bro_pre_batch_act_max_test_3lin_v3all", "auto_catfirst_all",
    #          "auto_catafter_all", "auto_addnode_all", "mlp_catafter_all"]

    names = ["no_glob_dense_cat_skipall", "no_glob_res_add_skipall"]
    for name in names:
        save_name = ""
        if name == "no_glob__dense_add_skip_6bro_pre_batch_act_max_testall":
            for i in range(1, 7):
                save_name = ""
                save_name = name + "_lyr_" + str(i)
                save_data = np.array(("trn", "val", "test"))
                for seed in seeds:
                    csv_path = "../../data/best_results/GCN/origindata_seed_" + str(
                        seed) + "_drop_0.5use_val/" + name + "_results.csv"
                    with open(csv_path, "r") as f:
                        reader = csv.reader(f)
                        tmp_data = [r for r in reader]
                        for j in range(1, 7):
                            data = tmp_data[j-1]
                            lyr = data[6]
                            if lyr == str(i):
                                num = j
                        tmp_data = tmp_data[num-1]
                        f1_data = np.array(tmp_data[2:5])
                        # for i in range(6,9):
                        #     epo_data.append(tmp_data[i])
                    save_data = np.concatenate((save_data, f1_data))
                save_path = "../../data/summary_results/" + save_name + ".csv"
                save_data = np.reshape(save_data, (3, -1), order="F")
                save_data = save_data.T
                with open(save_path, "w") as f:
                    writer = csv.writer(f)
                    # writer.writerow(save_data)
                    writer.writerows(save_data)
        elif name.split("_")[0] == "auto":
            save_data = np.array(("trn", "val", "test"))
            save_data_main = np.array(("trn", "val", "test"))
            for seed in seeds:
                csv_path = "../../data/best_results/GCN/origindata_seed_" + str(
                    seed) + "_drop_0.5use_val/" + name + "_results.csv"
                with open(csv_path, "r") as f:
                    reader = csv.reader(f)
                    tmp_data = [r for r in reader]
                    if len(tmp_data) == 2:
                        tmp_data_main = tmp_data[1].copy()
                        tmp_data = tmp_data[0]
                        f1_data = np.array(tmp_data[2:5])
                        f1_data_main = np.array(tmp_data_main[2:5])
                    elif len(tmp_data) < 2:
                        print(len(tmp_data), "<< なぁぜなぁぜ？")
                    else:
                        tmp_data_main = tmp_data[len(tmp_data)-1].copy()
                        f1_data_main = np.array(tmp_data_main[2:5])
                        tmp_data_alva = tmp_data[0].copy()
                        for j in range(len(tmp_data)-2):
                            main_flag = False
                            alva_flag = False
                            data = tmp_data[j+1]
                            epo = data[0]
                            lyr = data[6]
                            chn = data[7]
                            rate = data[8]
                            if name == "auto_catfirst_all":
                                type = "auto_cat_first_"
                            elif name == "auto_catafter_all":
                                type = "auto_cat_after_"
                            else:
                                type = "auto_add_node_"
                            main_info = "PDN_dense_add_skip_6bro_pre_batch_act_max_test_sample_True_v_late_original_"\
                                        + type + "test_hit_10_data_seed" + str(seed) + "_ratio0.5_knl" + chn + "_lyr"\
                                        + lyr + "initial_lr_" + rate + "_use_autoencoder"
                            main_info_alva = main_info + "_79784AE_v2"
                            path = "../../result/" + main_info + "/model(" + main_info + ")/model%05d.pth" % int(epo)
                            path_alva = "../../result/" + main_info_alva + "/model(" + main_info_alva + ")/model%05d.pth" % int(epo)
                            if os.path.isfile(path):
                                main_flag = True
                            if os.path.isfile(path_alva):
                                alva_flag = True
                            if main_flag and alva_flag:
                                print("困ったなぁ、、、", name, ":", seed, ":", j)
                                return  0
                            elif alva_flag:
                                tmp_data_alva = data.copy()
                        f1_data = np.array(tmp_data_alva[2:5])
                    # for i in range(6,9):
                    #     epo_data.append(tmp_data[i])
                save_data = np.concatenate((save_data, f1_data))
                save_data_main = np.concatenate((save_data_main, f1_data_main))
            save_path = "../../data/summary_results/" + name + "_79784.csv"
            save_main_path = "../../data/summary_results/" + name + "_main.csv"
            save_data = np.reshape(save_data, (3, -1), order="F")
            save_data_main = np.reshape(save_data_main, (3, -1), order="F")
            save_data = save_data.T
            save_data_main = save_data_main.T
            with open(save_path, "w") as f:
                writer = csv.writer(f)
                # writer.writerow(save_data)
                writer.writerows(save_data)
            with open(save_main_path, "w") as f:
                writer = csv.writer(f)
                writer.writerows(save_data_main)
        elif name == "mlp_catafter_all":
            save_data_2lin = np.array(("trn", "val", "test"))
            save_data_3lin = np.array(("trn", "val", "test"))
            for seed in seeds:
                csv_path = "../../data/best_results/GCN/origindata_seed_" + str(
                    seed) + "_drop_0.5use_val/" + name + "_results.csv"
                with open(csv_path, "r") as f:
                    reader = csv.reader(f)
                    tmp_data = [r for r in reader]
                    if len(tmp_data) == 3:
                        tmp_data_2lin = tmp_data[1].copy()
                        tmp_data_3lin = tmp_data[2].copy()
                        f1_data_2lin = np.array(tmp_data_2lin[2:5])
                        f1_data_3lin = np.array(tmp_data_3lin[2:5])
                    elif len(tmp_data) < 3:
                        print(len(tmp_data), "<< なぁぜなぁぜ？")
                    else:
                        tmp_data_3lin = tmp_data[len(tmp_data)-1].copy()
                        f1_data_3lin = np.array(tmp_data_3lin[2:5])
                        tmp_data_2lin = tmp_data[1].copy()
                        for j in range(len(tmp_data) - 3):
                            _2lin_flag = False
                            _3lin_flag = False
                            data = tmp_data[j + 2]
                            epo = data[0]
                            lyr = data[6]
                            chn = data[7]
                            rate = data[8]
                            main_info_2lin = "PDN_dense_skip_add_6bro_pre_batch_act_max_test_2lin_sample_True_v_late_original_" \
                                        + "mlp_cat_after_test_hit_10_data_seed" + str(seed) + "_ratio0.5_knl" + chn + "_lyr" \
                                        + lyr + "initial_lr_" + rate + "_use_mlp"
                            main_info_3lin = "PDN_dense_skip_add_6bro_pre_batch_act_max_test_3lin_sample_True_v_late_original_" \
                                        + "mlp_cat_after_test_hit_10_data_seed" + str(seed) + "_ratio0.5_knl" + chn + "_lyr" \
                                        + lyr + "initial_lr_" + rate + "_use_mlp"
                            path_2lin = "../../result/" + main_info_2lin + "/model(" + main_info_2lin + ")/model%05d.pth" % int(epo)
                            path_3lin = "../../result/" + main_info_3lin + "/model(" + main_info_3lin + ")/model%05d.pth" % int(
                                epo)
                            if os.path.isfile(path_2lin):
                                _2lin_flag = True
                            if os.path.isfile(path_3lin):
                                _3lin_flag = True
                            if _2lin_flag and _3lin_flag:
                                print("困ったなぁ、、、", name, ":", seed, ":", j)
                                return 0
                            elif _2lin_flag:
                                tmp_data_2lin = data.copy()
                        f1_data_2lin = np.array(tmp_data_2lin[2:5])
                    # for i in range(6,9):
                    #     epo_data.append(tmp_data[i])
                save_data_2lin = np.concatenate((save_data_2lin, f1_data_2lin))
                save_data_3lin = np.concatenate((save_data_3lin, f1_data_3lin))
            save_2lin_path = "../../data/summary_results/" + name + "_2lin.csv"
            save_3lin_path = "../../data/summary_results/" + name + "_3lin.csv"
            save_data_2lin = np.reshape(save_data_2lin, (3, -1), order="F")
            save_data_3lin = np.reshape(save_data_3lin, (3, -1), order="F")
            save_data_2lin = save_data_2lin.T
            save_data_3lin = save_data_3lin.T
            with open(save_2lin_path, "w") as f:
                writer = csv.writer(f)
                # writer.writerow(save_data)
                writer.writerows(save_data_2lin)
            with open(save_3lin_path, "w") as f:
                writer = csv.writer(f)
                writer.writerows(save_data_3lin)
        else:
            save_data = np.array(("trn", "val", "test"))
            for seed in seeds:
                # csv_path = "../../data/best_results/GCN/origindata_seed_" + str(seed) + "_drop_0.5use_val/" + name + "_results.csv"
                csv_path = "../../data/best_results/MLP/origindata_seed_" + str(seed) + "_drop_0.5/lyr_" + name + "_chn_" + name2 + "_all_results.csv"
                with open(csv_path, "r") as f:
                    reader = csv.reader(f)
                    tmp_data = [r for r in reader]
                    tmp_data = tmp_data[len(tmp_data)-1]
                    f1_data = np.array(tmp_data[2:5])
                    # for i in range(6,9):
                    #     epo_data.append(tmp_data[i])
                save_data = np.concatenate((save_data, f1_data))
            # save_path = "../../data/summary_results/" + name + ".csv"
            save_path = "../../data/summary_results/MLP/"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            save_path = save_path + "lyr_" + name + "_chn_" + name2 + ".csv"
            save_data = np.reshape(save_data, (3, -1), order="F")
            save_data = save_data.T
            with open(save_path, "w") as f:
                writer = csv.writer(f)
                # writer.writerow(save_data)
                writer.writerows(save_data)

def results_sum_from_mlp():
    seeds = [1010*i for i in range(1, 51)]
    # names = ["cat_f_score", "cat_a_score", "add_score", "ens_3_score", "ens_5_score"]
    names = ["mlp_f1-score"]

    for name in names:
        save_path = "../../data/compare_mlp_gcn/sum/"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        data = [["", "f1", "prec", "rec", "", "f1", "prec", "rec", "", "f1", "prec", "rec"]]
        for n, seed in enumerate(seeds):
            path = "../../data/compare_mlp_gcn/seed_" + str(seed) + "/"
            # num_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            with open(path + name + ".csv", "r") as f:
                reader = csv.reader(f)
                sada = [r for r in reader]
                tmp_data = np.array(sada[4:7]).reshape((1, -1))
                if n == 0:
                    num_data = np.array(sada[4:7])[:, 1:].reshape((1, -1)).astype(float)
                else:
                    num_data = np.concatenate((num_data, np.array(sada[4:7])[:, 1:].reshape((1, -1)).astype(float)))
                data = np.concatenate((data, tmp_data))
        pd_data = pd.DataFrame(num_data)
        tmp_save_025 = np.array(([["0.25", pd_data[:][0].quantile(0.25), pd_data[:][1].quantile(0.25), pd_data[:][2].quantile(0.25),
                              "0.25", pd_data[:][3].quantile(0.25), pd_data[:][4].quantile(0.25), pd_data[:][5].quantile(0.25),
                              "0.25", pd_data[:][6].quantile(0.25), pd_data[:][7].quantile(0.25), pd_data[:][8].quantile(0.25)]]))
        tmp_save_05 = np.array(
            ([["0.50", pd_data[:][0].quantile(0.5), pd_data[:][1].quantile(0.5), pd_data[:][2].quantile(0.5),
              "0.50", pd_data[:][3].quantile(0.5), pd_data[:][4].quantile(0.5), pd_data[:][5].quantile(0.5),
              "0.50", pd_data[:][6].quantile(0.5), pd_data[:][7].quantile(0.5), pd_data[:][8].quantile(0.5)]]))
        tmp_save_075 = np.array(
            ([["0.75", pd_data[:][0].quantile(0.75), pd_data[:][1].quantile(0.75), pd_data[:][2].quantile(0.75),
              "0.75", pd_data[:][3].quantile(0.75), pd_data[:][4].quantile(0.75), pd_data[:][5].quantile(0.75),
              "0.75", pd_data[:][6].quantile(0.75), pd_data[:][7].quantile(0.75), pd_data[:][8].quantile(0.75)]]))
        tmp_save_avg = np.array(
            ([["avg", pd_data[:][0].mean(), pd_data[:][1].mean(), pd_data[:][2].mean(),
               "avg", pd_data[:][3].mean(), pd_data[:][4].mean(), pd_data[:][5].mean(),
               "avg", pd_data[:][6].mean(), pd_data[:][7].mean(), pd_data[:][8].mean()]]))
        data = np.concatenate((data, tmp_save_025))
        data = np.concatenate((data, tmp_save_05))
        data = np.concatenate((data, tmp_save_075))
        data = np.concatenate((data, tmp_save_avg))
        with open(save_path + name + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)

def results_sum_from_gcn():
    seeds = [1010*i for i in range(1, 51)]
    names = ["ens_3_glob_score", "avg_5_score", "avg_3_glob_score"]

    for name in names:
        save_path = "../../data/compare_mlp_gcn/sum/"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        data = [["", "f1", "prec", "rec", "", "f1", "prec", "rec", "", "f1", "prec", "rec"]]
        for n, seed in enumerate(seeds):
            path = "../../data/compare_mlp_gcn/seed_" + str(seed) + "/"
            # num_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            with open(path + "_" + name + ".csv", "r") as f:
                reader = csv.reader(f)
                sada = [r for r in reader]
                tmp_data = np.array(sada[1:4]).reshape((1, -1))
                if n == 0:
                    num_data = np.array(sada[1:4])[:, 1:].reshape((1, -1)).astype(float)
                else:
                    num_data = np.concatenate((num_data, np.array(sada[1:4])[:, 1:].reshape((1, -1)).astype(float)))
                data = np.concatenate((data, tmp_data))
        pd_data = pd.DataFrame(num_data)
        tmp_save_025 = np.array(([["0.25", pd_data[:][0].quantile(0.25), pd_data[:][1].quantile(0.25), pd_data[:][2].quantile(0.25),
                              "0.25", pd_data[:][3].quantile(0.25), pd_data[:][4].quantile(0.25), pd_data[:][5].quantile(0.25),
                              "0.25", pd_data[:][6].quantile(0.25), pd_data[:][7].quantile(0.25), pd_data[:][8].quantile(0.25)]]))
        tmp_save_05 = np.array(
            ([["0.50", pd_data[:][0].quantile(0.5), pd_data[:][1].quantile(0.5), pd_data[:][2].quantile(0.5),
              "0.50", pd_data[:][3].quantile(0.5), pd_data[:][4].quantile(0.5), pd_data[:][5].quantile(0.5),
              "0.50", pd_data[:][6].quantile(0.5), pd_data[:][7].quantile(0.5), pd_data[:][8].quantile(0.5)]]))
        tmp_save_075 = np.array(
            ([["0.75", pd_data[:][0].quantile(0.75), pd_data[:][1].quantile(0.75), pd_data[:][2].quantile(0.75),
              "0.75", pd_data[:][3].quantile(0.75), pd_data[:][4].quantile(0.75), pd_data[:][5].quantile(0.75),
              "0.75", pd_data[:][6].quantile(0.75), pd_data[:][7].quantile(0.75), pd_data[:][8].quantile(0.75)]]))
        tmp_save_avg = np.array(
            ([["avg", pd_data[:][0].mean(), pd_data[:][1].mean(), pd_data[:][2].mean(),
               "avg", pd_data[:][3].mean(), pd_data[:][4].mean(), pd_data[:][5].mean(),
               "avg", pd_data[:][6].mean(), pd_data[:][7].mean(), pd_data[:][8].mean()]]))
        data = np.concatenate((data, tmp_save_025))
        data = np.concatenate((data, tmp_save_05))
        data = np.concatenate((data, tmp_save_075))
        data = np.concatenate((data, tmp_save_avg))
        with open(save_path + name + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)

def t_test():
    tmp_path = "../../data/summary_results/"
    names = [f for f in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, f))]
    types = ["trn", "val", "test"]
    alpha = 0.05
    z_alpha = 1.96
    save_path = tmp_path + "t_test/"
    for name in names:
        for name_2 in names:
            name = name.split(".")[0]
            name_2 = name_2.split(".")[0]
            save_data_side = np.array(("", "data_name_1", "0.25", "0.5", "0.75", "avg", "data_nane_2", "0.25", "0.5", "0.75", "avg", "t-value", "p-value", "t-result", "Z-result"))
            if name != name_2:
                data_1 = pd.read_csv(tmp_path+name+".csv")
                data_2 = pd.read_csv(tmp_path+name_2+".csv")
                for type in types:
                    s_data = data_1[type]
                    t_data = data_2[type]
                    s_n = len(s_data)
                    t_n = len(t_data)
                    s_mean = s_data.mean()
                    t_mean = t_data.mean()
                    s_var = stats.tvar(s_data)
                    t_var = stats.tvar(t_data)
                    z = (s_mean - t_mean) / np.sqrt(s_var/(s_n-1) + t_var/(t_n-1))
                    if math.fabs(z) > z_alpha:
                        print("zでは棄却")
                        score = "res in ZZ"
                    else:
                        score = "non res in ZZ"
                    stat, p = stats.ttest_ind(s_data, t_data, alternative="two-sided", equal_var=False)
                    if p < alpha:
                        # print("両側棄却")
                        res = "there are res"
                    else:
                        print(name, "=", name_2, ":両側採択")
                        res = "there are not res"
                    tmp_save = np.array(([type, name, s_data.quantile(0.25), s_data.quantile(0.5), s_data.quantile(0.75), s_mean, name_2, t_data.quantile(0.25), t_data.quantile(0.5), t_data.quantile(0.75), t_mean, stat, p, res, score]))
                    save_data_side = np.concatenate((save_data_side, tmp_save))
                # save_data_less = np.reshape(save_data_less, (12, -1), order="F").T
                save_data_side = np.reshape(save_data_side, (15, -1), order="F").T
                # with open(save_path + name + "_syonari_" + name_2 + ".csv", "w") as f:
                #     writer = csv.writer(f)
                #     writer.writerows(save_data_less)
                with open(save_path + name + "!=" + name_2 + ".csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(save_data_side)

def t_test_gcn():
    tmp_path = "../../data/compare_mlp_gcn/"
    path = tmp_path + "sum/"
    names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # print(names)
    # return 0
    # names = ["dragon_addnode_all", "dragon_catafter_all", "dragon_catfirst_all", "mlp_addnode_all", "mlp_catafter_all", "mlp_catfirst_all", "no_globall"]
    types = ["trn", "val", "test"]
    types_2 = [["f1", "prec", "rec"], ["f1.1", "prec.1", "rec.1"], ["f1.2", "prec.2", "rec.2"]]
    alpha = 0.05
    z_alpha = 1.96
    save_path = tmp_path + "t_test/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for name in names:
        for name_2 in names:
            name = name.split(".")[0]
            name_2 = name_2.split(".")[0]
            save_data_side = np.array((["type1", "type2", "data_name_1", "0.25", "0.5", "0.75", "avg", "var", "data_nane_2", "0.25", "0.5", "0.75", "avg", "var", "t-value", "p-value", "t-result", "Z-result"]))
            if name != name_2:
                data_1 = pd.read_csv(path+name+".csv")
                data_2 = pd.read_csv(path+name_2+".csv")
                for type, type_2s in zip(types, types_2):
                    for type_2 in type_2s:
                        s_data = data_1[:50][type_2]
                        t_data = data_2[:50][type_2]
                        s_n = len(s_data)
                        t_n = len(t_data)
                        s_mean = s_data.mean()
                        t_mean = t_data.mean()
                        s_var = stats.tvar(s_data)
                        t_var = stats.tvar(t_data)
                        z = (s_mean - t_mean) / np.sqrt(s_var/(s_n-1) + t_var/(t_n-1))
                        if math.fabs(z) > z_alpha:
                            # print("zでは棄却")
                            score = "res in ZZ"
                        else:
                            score = "non res in ZZ"
                        stat, p = stats.ttest_ind(s_data, t_data, alternative="two-sided", equal_var=False)
                        if p < alpha:
                            # print("両側棄却")
                            res = "there are res"
                        else:
                            # print(name, "=", name_2, ":", type, ":", type_2, ":両側採択")
                            res = "there are not res"
                        tmp_save = np.array(([type, type_2.split(".")[0], name, s_data.quantile(0.25), s_data.quantile(0.5), s_data.quantile(0.75), s_mean, s_var, name_2, t_data.quantile(0.25), t_data.quantile(0.5), t_data.quantile(0.75), t_mean, t_var, stat, p, res, score]))
                        # print(save_data_side)
                        # print(tmp_save)
                        save_data_side = np.concatenate((save_data_side, tmp_save))
                save_data_side = np.reshape(save_data_side, (18, -1), order="F").T
                with open(save_path + name + "_" + name_2 + ".csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(save_data_side)

def t_test_mlp():
    tmp_path = "../../data/compare_mlp/"
    path = tmp_path + "sum/"
    names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    types = ["trn", "val", "test"]
    types_2 = [["f1", "prec", "rec"], ["f1.1", "prec.1", "rec.1"], ["f1.2", "prec.2", "rec.2"]]
    alpha = 0.05
    z_alpha = 1.96
    save_path = tmp_path + "t_test/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for name in names:
        for name_2 in names:
            name = name.split(".")[0]
            name_2 = name_2.split(".")[0]
            save_data_side = np.array((["type1", "type2", "data_name_1", "0.25", "0.5", "0.75", "avg", "var", "data_nane_2", "0.25", "0.5", "0.75", "avg", "var", "t-value", "p-value", "t-result", "Z-result"]))
            if name != name_2:
                data_1 = pd.read_csv(path+name+".csv")
                data_2 = pd.read_csv(path+name_2+".csv")
                for type, type_2s in zip(types, types_2):
                    for type_2 in type_2s:
                        s_data = data_1[:50][type_2]
                        t_data = data_2[:50][type_2]
                        s_n = len(s_data)
                        t_n = len(t_data)
                        s_mean = s_data.mean()
                        t_mean = t_data.mean()
                        s_var = stats.tvar(s_data)
                        t_var = stats.tvar(t_data)
                        z = (s_mean - t_mean) / np.sqrt(s_var/(s_n-1) + t_var/(t_n-1))
                        if math.fabs(z) > z_alpha:
                            # print("zでは棄却")
                            score = "res in ZZ"
                        else:
                            score = "non res in ZZ"
                        stat, p = stats.ttest_ind(s_data, t_data, alternative="two-sided", equal_var=False)
                        if p < alpha:
                            # print("両側棄却")
                            res = "there are res"
                        else:
                            # print(name, "=", name_2, ":", type, ":", type_2, ":両側採択")
                            res = "there are not res"
                        tmp_save = np.array(([type, type_2.split(".")[0], name, s_data.quantile(0.25), s_data.quantile(0.5), s_data.quantile(0.75), s_mean, s_var, name_2, t_data.quantile(0.25), t_data.quantile(0.5), t_data.quantile(0.75), t_mean, t_var, stat, p, res, score]))
                        # print(save_data_side)
                        # print(tmp_save)
                        save_data_side = np.concatenate((save_data_side, tmp_save))
                save_data_side = np.reshape(save_data_side, (18, -1), order="F").T
                with open(save_path + name + "_" + name_2 + ".csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(save_data_side)

def exe_ocr(imgs):
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)  # 引数1は終了ステータスで１を返す

    tool = tools[0]

    df = pd.DataFrame()
    index = 0
    # 画像の数（ページ数）ごとの処理
    for i, img in enumerate(imgs):

        # OCR実行。WordBoxBuilderでは座標も取得可能
        word_boxes = tool.image_to_string(
            img,
            lang="eng",
            builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6)
        )

        # 文字の塊ごとに分けてデータフレームに変換
        for box in word_boxes:
            df_page = pd.DataFrame({
                "x_start": box.position[0][0],
                "x_end": box.position[1][0],
                "y_start": box.position[0][1],
                "y_end": box.position[1][1],
                "text": box.content,
                "page": i + 1}, index=[index]
            )

            df = df.append(df_page)
            index += 1

        print(word_boxes)

    return df

if __name__ == '__main__':
    results_sum_from_gcn()
    t_test_gcn()