import fnmatch
import os
import random
from pathlib import Path
from termcolor import colored

def remove_files(folder, list):
    """
    This is a function to remove the overlap .txt files from the entire folder
    -------------------------------
    Folder: the original folder you want to remove the txt files
    list: the reference list for removing the txt files
    """
    rm_count = 0
    for filename in os.listdir(folder):
        part_filename = filename.split("-")[0]
        if part_filename in list:
            os.remove(folder + filename)
            rm_count += 1
    print(colored(rm_count, "green"),
          colored("samples are removed from training dataset!", "green", "on_grey", attrs=["bold"]))

def remove_trn_files(folder, list):
    # for original dataset
    rm_count = 0
    for filename in os.listdir(folder):
        part_filename = filename.split(".")[0]
        if part_filename in list:
            os.remove(folder + filename)
            rm_count += 1
        elif fnmatch.fnmatch(filename, "*_*"):
            os.remove(folder + filename)
            rm_count += 1
    print(colored(rm_count, "green"),
          colored("samples are removed from training dataset!", "green", "on_grey", attrs=["bold"]))

def remove_test_files(folder, list):
    # for original dataset
    rm_count = 0
    for filename in os.listdir(folder):
        part_filename = filename.split(".")[0]
        if part_filename not in list:
            os.remove(folder + filename)
            rm_count += 1
        elif fnmatch.fnmatch(filename, "*_*"):
            os.remove(folder + filename)
            rm_count += 1
    print("===================================")
    print(colored(str(rm_count), "blue"),
          colored("samples are removed from testing dataset!", "blue", "on_grey", attrs=["bold"]),
          colored("*include .DS.store file", "grey"))

def remove_aug_files(folder, list):
    """
    This is a function to remove the augmented .txt files from the entire folder
    -------------------------------
    Folder: the original folder you want to remove the augmented txt files
    """
    rm_aug_count = remove_files_not_in(folder, list)
    for filename in os.listdir(folder):
        if fnmatch.fnmatch(filename, "*_*"):
            os.remove(folder + filename)
            rm_aug_count += 1
    print("===================================")
    print(colored(str(rm_aug_count), "blue"),
          colored("samples are removed from testing dataset!", "blue", "on_grey", attrs=["bold"]),
          colored("*include .DS.store file", "grey"))

def remove_files_not_in(folder, list):
    """
    This is a function to remove the overlap .txt files from the entire folder
    -------------------------------
    Folder: the original folder you want to remove the txt files
    a_list: the reference list for removing the txt files
    if "not_in=True", the txt files not in "a_list" will be removed
    """
    rm_aug_count = 0
    for filename in os.listdir(folder):
        part_filename = filename.split("-")[0]
        if part_filename not in list:
            os.remove(folder + filename)
            rm_aug_count += 1

    return rm_aug_count

def read_sdf(path):
    with open(path, 'r', errors='ignore') as f:
        return parse_sdf2(f.read())

def parse_sdf(src):
    src = src.split('-')[0:1]
    hit_name = src[0]
    return hit_name

def parse_sdf2(src):
    src = src.split('\n')[0:1]
    hit_name = src[0]
    return hit_name

def Main():
    os.chdir('../../data/check_edgeattr3/')
    # ---  replace suffix from .sdf to txt ---
    # path = ['testing_dataset', '../training_dataset', '../txt_hits', '../txt_nonhits']
    # path = ['txt']
    # for p_name in path:
    #     os.chdir(p_name)
    #     for file in os.listdir():
    #         p = Path(file)
    #         p.rename(p.with_suffix('.txt'))
    # os.chdir('../')

    # -------- sampling info ------------------------
    ratio_training = 0.7
    total_number = 3717  # the total number of molecules
    num_hit = 160  # originally we have 125 hit
    ratio_hit_nonhit = 5 / 22  #1:30

    num_test_hit = int(num_hit * (1 - ratio_training))
    num_test_nonhit = int(1 / ratio_hit_nonhit * num_test_hit)

    # ----------- sampling for hit and nonhit ---------------
    f_names = ['txt_hits', 'txt_nonhits']
    t_names = ['hits_ref', 'a']
    s_names = ['hits_sampling', 'nonhits_sampling']
    seeds = [400, 100]
    nums = [num_test_hit, num_test_nonhit]
    for f, t, s, seed, num in zip(f_names, t_names, s_names, seeds, nums):
        files = os.listdir(f)
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        ref = []
        with open(t, "w") as txtfile:
            for file in files:
                if not os.path.isdir(file):
                    curr_file = read_sdf(f + '/{}'.format(file))
                    name = curr_file
                    ref.append(name)
                    if f == 'txt_hits':
                        txtfile.write(name + "\n")
        random.seed(seed)
        sampling = random.sample(ref, num)
        if f == 'txt_hits':
            combine_sampling_for_test = sampling
        else:
            combine_sampling_for_test += sampling

        with open(s, "w") as txtfile:
            for element in sampling:
                txtfile.write(element + "\n")

    # -------- creating the training and testing dataset ------------------------
    # require two original folder that include all augmented txt files
    testing_dataset = "testing_dataset/"  # the augmented whole dataset
    training_dataset = "training_dataset/"  # the same with above

    # creating the testing dataset
    # remove_aug_files(testing_dataset, combine_sampling_for_test)  # remove the augmented data and create the final testing dataset
    remove_test_files(testing_dataset, combine_sampling_for_test)  #for not augmentation

    # creating the training dataset
    # remove_files(training_dataset, combine_sampling_for_test)  # creat a training dataset
    remove_trn_files(training_dataset, combine_sampling_for_test)  # for not augmentation

if __name__ == '__main__':
    Main()