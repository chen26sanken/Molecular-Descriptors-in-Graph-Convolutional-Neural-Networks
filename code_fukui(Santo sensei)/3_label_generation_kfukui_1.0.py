import array
import os
from pathlib import Path
import numpy as np
import torch
from numpy.ma import count
from termcolor import colored

def read_grab(read_path, out_name, slice_name=False):
    new_list = []
    out_file = '../../data/check_edgeattr4/' + out_name
    # out_file = out_name
    if os.path.exists(out_file):
        os.remove(out_file)

    path = '../../data/check_edgeattr4/' + read_path
    # path = read_path
    os.chdir(path)
    if os.path.exists('.DS_Store.txt'):
        os.remove('.DS_Store.txt')

    if slice_name:
        for i in os.listdir():
            new_list.append(i[8:12])
        print(colored("sliced name", "yellow"), new_list[0:4])
    else:
        for i in os.listdir():
            new_list.append(i)

        new_list = sorted(new_list)
        if '.DS_Store' in new_list:
            new_list.remove('.DS_Store')

    out_file = '../../check_edgeattr4/' + out_name
    with open(out_file, 'w') as textfile:
        for element in new_list:
            textfile.write(element + "\n")

def graph_label(hit_list, unlabeled_mls, out_labels):
    with open(hit_list, 'r') as my_file:
        raw_hit_index = [item for item in my_file.read().split()]
        hit_index = [ele.lstrip('0') for ele in raw_hit_index]

    c = 0
    h = 0
    labels = []
    with open('../../check_edgeattr4/' + unlabeled_mls) as f:
        for line in f:
            line = line.split(".")[0]
            # print(line)
            if line in hit_index:
                labels.append(1)
                h += 1
                c += 1
            else:
                labels.append(0)
                c += 1
        with open("../../check_edgeattr4/" + out_labels, "w") as textfile:
            for element in labels:
                textfile.write(str(element) + " ")
    print(colored("samples_count=", "yellow"), colored(str(c), "yellow"))
    print(colored("hits=", "green"), colored(str(h), "yellow"))
    print(colored("------------------", "yellow"))
    os.chdir('../')

def Main():
    known_hit = "../../check_edgeattr4/hits_ref"

    augmented_data = 'testing_dataset'
    augmented_molecule_name = 'testing_sorted_name'
    read_grab(augmented_data, augmented_molecule_name)

    print("Current working directory_trn: {0}".format(os.getcwd()))
    graph_label(known_hit, augmented_molecule_name, 'testing_sorted_graph_labels')

    augmented_data = 'training_dataset'
    augmented_molecule_name = 'training_sorted_name'
    read_grab(augmented_data, augmented_molecule_name)
    graph_label(known_hit, augmented_molecule_name, 'training_sorted_graph_labels')

    with open('../check_edgeattr4/testing_sorted_graph_labels') as fp:
        data = fp.read()
        data_cal = data.split(" ")
        data_cal = np.delete(data_cal, len(data_cal)-1)
        hit_in_testing = list(data_cal).count("1")
        print(colored(hit_in_testing, "cyan"), colored("hits_in_testing:", "cyan"),
              "ratio:", hit_in_testing/len(data_cal))

    with open('../check_edgeattr4/training_sorted_graph_labels') as fp:
        data2 = fp.read()
        data_cal_trn = data2.split(" ")
        data_cal_trn = np.delete(data_cal_trn, len(data_cal_trn) - 1)
        hit_in_training = list(data_cal_trn).count("1")
        print(colored(hit_in_training, "cyan"), colored("hits_in_training:", "cyan"),
              "ratio:", hit_in_training / len(data_cal_trn))

    data += data2
    data = data.split(" ")
    data = np.delete(data, len(data)-1)
    # print(data_cal, data_cal_trn)
    data3 = np.concatenate([data_cal, data_cal_trn], axis=0)
    print(colored("(testing)", "yellow"), colored(len(data_cal), "yellow"), "+",
          colored("(training)", "yellow"), colored(len(data_cal_trn), "yellow"), "=",
          colored("(total data_size)", "yellow"), colored(len(data), "yellow"),
          " or ", colored(len(data3), "yellow"))
    data = np.array(data).reshape(len(data), 1)
    data3 = np.array(data3).reshape(len(data3), 1)
    print(len(data))

    with open('../../data/check_edgeattr4/true/raw/true_graph_labels.txt', 'w') as handle:
        np.savetxt(handle, data, fmt="%s")
    with open('../../data/check_edgeattr4/false/raw/false_graph_labels.txt', 'w') as handle:
        np.savetxt(handle, data3, fmt="%s")

if __name__ == "__main__":
    Main()