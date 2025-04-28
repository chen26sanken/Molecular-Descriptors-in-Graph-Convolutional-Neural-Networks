import random

import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import Data
import os
import numpy as np
from termcolor import colored
from tqdm import tqdm
import csv


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src, dtype=dtype).squeeze()

    return src


def parse_txt_array2(src, sep=None, start=0, end=None, dtype=None, device=None, random_flag=0):
    # src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    n_src = []
    for line in src:
        line = line.split(sep)[start:end]
        if end - start == 1:
            edge_f = np.zeros(3, dtype=float)
            if random_flag==0:
                edge_f[int(line[0])-1] = 1.0
            else:
                random_edge = random.randint(0, 2)
                edge_f[random_edge] = 1.0
            n_src.append([edge_f[0], edge_f[1], edge_f[2]])
            n_src.append([edge_f[0], edge_f[1], edge_f[2]])
        else:
            n_src.append([float(line[0]), float(line[1])])
            n_src.append([float(line[1]), float(line[0])])
    n_src = torch.tensor(n_src, dtype=dtype).squeeze()
    return n_src


def splitting(src):
    item = src.split()[0]
    num_atoms = int(item[:3])
    num_bonds = int(item[3:])

    return num_atoms, num_bonds


def parse_txt_array3(src, name, sep=None, start=0, end=None, dtype=None, device=None, random_flag=0):
    n_src = []
    for line in src:
        if len(line.split()) == 6:
            line = line.split(sep)[start:end]
            if name == "labels":
                edge_f = np.zeros(3, dtype=float)
                if random_flag == 0:
                    edge_f[int(line[0])-1] = 1.0
                else:
                    random_edge = random.randint(0, 2)
                    edge_f[random_edge] = 1.0
                n_src.append([edge_f[0], edge_f[1], edge_f[2]])
                n_src.append([edge_f[0], edge_f[1], edge_f[2]])
            elif name == "attr":
                n_src.append(float(line[0]))
            elif name == "index":
                n_src.append([float(line[0]), float(line[1])])
                n_src.append([float(line[1]), float(line[0])])
            else:
                n_src.append([float(line[0]), float(line[1])])
        else:
            if name == "labels":
                edge_f = np.zeros(3, dtype=float)
                if random_flag == 0:
                    line = line.split()[1]
                    edge_f[int(line)-1] = 1.0
                else:
                    random_edge = random.randint(0, 2)
                    edge_f[random_edge] = 1.0
                n_src.append([edge_f[0], edge_f[1], edge_f[2]])
                n_src.append([edge_f[0], edge_f[1], edge_f[2]])
            elif name == "attr":
                line = line.split()[1]
                n_src.append(float(line))
            else:
                line = line.split()[0]
                if int(line[:3]) < int(line[3:]):
                    if name == "index":
                        n_src.append([float(line[:3]), float(line[3:])])
                        n_src.append([float(line[3:]), float(line[:3])])
                    else:
                        n_src.append([float(line[:3]), float(line[3:])])
                else:
                    if name == "index":
                        n_src.append([float(line[:2]), float(line[2:])])
                        n_src.append([float(line[2:]), float(line[:2])])
                    else:
                        n_src.append([float(line[:2]), float(line[2:])])
    n_src = torch.tensor(n_src, dtype=dtype).squeeze()
    return n_src


def parse_sdf(src, elems, random_flag):
    flag = 0
    name = src.split('\n')[0].split("-")[0]
    src = src.split('\n')[3:]

    num_atoms, num_bonds = [int(item) for item in src[0].split()[:2]]
    if num_bonds == 0:
        flag = 1
        num_atoms, num_bonds = splitting(src[0])
    atom_block = src[1:num_atoms + 1]
    pos = parse_txt_array(atom_block, end=3)
    x = torch.tensor([elems[item.split()[3]] for item in atom_block])
    x = F.one_hot(x, num_classes=len(elems))

    bond_block = src[1 + num_atoms:1 + num_atoms + num_bonds]
    if flag == 0:
        row, col = parse_txt_array(bond_block, end=2, dtype=torch.long).t() - 1
    else:
        row, col = parse_txt_array3(bond_block, name="a", end=2, dtype=torch.long).t() - 1
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    # edge_index2 = parse_txt_array(bond_block, end=2)
    if flag == 0:
        edge_index2 = parse_txt_array2(bond_block, end=2)
        # edge_labels = parse_txt_array(bond_block, start=2, end=3)
        edge_labels = parse_txt_array2(bond_block, start=2, end=3, random_flag=random_flag)
        edge_attr = parse_txt_array(bond_block, start=2, end=3) - 1
    else:
        edge_index2 = parse_txt_array3(bond_block, 'index', end=2)
        edge_labels = parse_txt_array3(bond_block, 'labels', start=2, end=3, random_flag=random_flag)
        edge_attr = parse_txt_array3(bond_block, 'attr', start=2, end=3) - 1
    node_labels = torch.tensor([elems[item.split()[3]] for item in atom_block])

    # edge_attr = parse_txt_array(bond_block, start=2, end=3) - 1
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_atoms, num_atoms)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos,
                edge_index2=edge_index2, edge_labels=edge_labels, num_atoms=num_atoms, node_labels=node_labels)

    return data


def read_sdf(path, list, random_flag):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return parse_sdf(f.read(), list, random_flag)


def Main():
    testing_dataset = "../../data/origin/testing_dataset"
    training_dataset = "../../data/origin/training_dataset"

    name_of_generated_dataset = "../../data/origin_2/raw"

    elems = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'P': 7, 'Br': 8}

    files = os.listdir(testing_dataset)
    if '.DS_Store.txt' in files:
        files.remove('.DS_Store.txt')
    files = sorted(files)

    files_trn = os.listdir(training_dataset)
    if '.DS_Store.txt' in files_trn:
        files_trn.remove('.DS_Store.txt')
    files_trn = sorted(files_trn)

    mol = []
    mol_trn = []
    os.makedirs(name_of_generated_dataset, exist_ok=True)
    name = name_of_generated_dataset
    A_file = name + '_A.txt'
    edge_labels_file = name + '_edge_labels.txt'
    node_labels_file = name + '_node_labels.txt'
    graph_indicator_file = name + '_graph_indicator.txt'
    node_attributes_file = name + '_node_attributes.txt'

    list = [A_file, edge_labels_file, node_labels_file, graph_indicator_file, node_attributes_file]
    for p in list:
        if os.path.exists(p):
            os.remove(p)

    mol_id = 0
    global_atom_id = 0

    random_flag = 0 # 0: original edge feature, 1: random edge feature

    with open("../../data/origin/mol_sort.csv", "w") as cs:
        writer = csv.writer(cs)
        for i, file in enumerate(tqdm(files)):
            # print(colored("current_files_testing", "blue"), file)
            if not os.path.isdir(file):  # return a boolean value
                print(colored("current_files_testing", "blue"), file)
                curr_file = read_sdf(testing_dataset + '/{}'.format(file), elems, random_flag)
                writer.writerow([file])
                mol.append(file)

                # print(curr_file)
                # print(curr_file.x)
                # print(curr_file.pos)
                # print(curr_file.edge_labels)
                # print(curr_file.node_labels)

                # DNA_A
                adjacency = curr_file.edge_index2.numpy() + global_atom_id
                with open(A_file, 'a') as handle:
                    np.savetxt(handle, adjacency, fmt='%i', delimiter=', ')

                # DNA_edge_labels
                flatten_edge_labels = curr_file.edge_labels.reshape(-1, 3)
                with open(edge_labels_file, 'a') as handle:
                    np.savetxt(handle, flatten_edge_labels.numpy(), fmt='%i', delimiter=', ')

                # DNA_node_labels
                flatten_node_labels = curr_file.node_labels.reshape(-1, 1)
                with open(node_labels_file, 'a') as handle:
                    np.savetxt(handle, flatten_node_labels.numpy(), fmt='%i')
                # print(flatten_node_labels)

                # new added DNA_node_attributes
                position = curr_file.pos
                with open(node_attributes_file, 'a') as handle:
                    np.savetxt(handle, position.numpy(), fmt='%1.4f', delimiter=', ')

                # DNA_graph_indicator
                n_atoms = np.array([curr_file.num_atoms])
                DNA_graph_indicator = np.repeat(mol_id, n_atoms).reshape(-1, 1)
                DNA_graph_indicator_1 = DNA_graph_indicator + 1
                with open(graph_indicator_file, 'a') as handle:
                    np.savetxt(handle, DNA_graph_indicator_1, fmt='%0i')

                global_atom_id += n_atoms
                mol_id += 1
            print("Current working directory: {0}".format(os.getcwd()))
        # print(mol_id, global_atom_id)

        for i, file in enumerate(tqdm(files_trn)):
            # print(colored("current_files_training", "green"), file)
            if not os.path.isdir(file):  # with the isdir function,
                curr_file = read_sdf(training_dataset + '/{}'.format(file), elems, random_flag)
                writer.writerow([file])
                mol.append(file)

                # DNA_A
                adjacency = curr_file.edge_index2.numpy() + global_atom_id
                with open(A_file, 'a') as handle:
                    np.savetxt(handle, adjacency, fmt='%i', delimiter=', ')

                # DNA_edge_labels
                flatten_edge_labels = curr_file.edge_labels.reshape(-1, 3)
                with open(edge_labels_file, 'a') as handle:
                    np.savetxt(handle, flatten_edge_labels.numpy(), fmt='%i', delimiter=', ')

                # DNA_node_labels
                flatten_node_labels = curr_file.node_labels.reshape(-1, 1)
                with open(node_labels_file, 'a') as handle:
                    np.savetxt(handle, flatten_node_labels.numpy(), fmt='%i')

                # new added DNA_node_attributes
                position = curr_file.pos
                with open(node_attributes_file, 'a') as handle:
                    np.savetxt(handle, position.numpy(), fmt='%1.4f', delimiter=', ')

                # DNA_graph_indicator
                n_atoms = np.array([curr_file.num_atoms])
                DNA_graph_indicator = np.repeat(mol_id, n_atoms).reshape(-1, 1)
                DNA_graph_indicator_1 = DNA_graph_indicator + 1
                with open(graph_indicator_file, 'a') as handle:
                    np.savetxt(handle, DNA_graph_indicator_1, fmt='%0i')

                global_atom_id += n_atoms
                mol_id += 1
            print("Current working directory_trn: {0}".format(os.getcwd()))
        print(mol_id, global_atom_id)


if __name__ == '__main__':
    Main()