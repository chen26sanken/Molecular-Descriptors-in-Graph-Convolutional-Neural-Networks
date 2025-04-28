import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score, classification_report, f1_score
from .mlps import *
from .auto_encoder import *
from .plot import plot_confusion_matrix
from .loss import *

def min_max_norm(tensor_2d):
    list_2d = tensor_2d.tolist()
    proc = preprocessing.MinMaxScaler()
    norm_list = proc.fit_transform(list_2d)
    norm_tensor = torch.tensor(norm_list).float()
    norm_tensor = torch.reshape(norm_tensor, (-1, 1, norm_tensor.size()[1]))
    return norm_tensor

def train_feat_mlp(train_feat, val_feat, train_gt, val_gt, in_channels, middle_channels, n_pre_layer, out_channels,
                   n_connected_layer, dropout, epochs, init_rate, device, seed, optim,
                   sam_seed=None, train_hit_index=None, train_nonhit_index=None, feat_numpy=None, gt_tmp=None):
    sample_name = ""
    saved_data = []
    saved_epoch = 0
    saved_val_acc_of_epoch = 0
    step_count = 0
    if sam_seed != None:
        train_nonhit = 83
        random.seed(sam_seed)
        sample_name = "_sampled"
    train_feat = torch.reshape(train_feat, (-1, 1, train_feat.size()[1]))
    val_feat = torch.reshape(val_feat, (-1, 1, val_feat.size()[1]))

    print("train:", train_feat.size(), ", ", train_gt.size())
    print("val:", val_feat.size(), ", ", val_gt.size())


    model = mlp(in_channels, middle_channels, out_channels, n_pre_layer, n_connected_layer)
    print('mlp_model:', model)
    criterion = torch.nn.CrossEntropyLoss()
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=init_rate)
    elif optim == "Adam":
        # optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        optimizer = torch.optim.Adam(model.parameters(), lr=init_rate)
    scheduler = MultiStepLR(optimizer, milestones=[20, 200, 800, 2000, 5000], gamma=0.5)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    temp_models_saved_folder = "../../results/for_GCN_dataset/nonrapped/mlp_models/" + "seeds_" + str(seed) + "_" + str(middle_channels) + "/" + str(optimizer)[:3] + "_lyr_"\
                               + str(n_pre_layer) + "_drop_" + str(dropout) + "_init_rate_" + str(init_rate) + sample_name
    results_saved_folder = "../../results/for_GCN_dataset/nonrapped/mlp_models/" + "seeds_" + str(seed) + "_" + str(middle_channels) + "/" + str(optimizer)[:3] + "_lyr_" \
                           + str(n_pre_layer) + "_drop_" + str(dropout) + "_init_rate_" + str(init_rate) + sample_name
    pdf_path_tmp = temp_models_saved_folder + "/"

    os.makedirs(temp_models_saved_folder, exist_ok=True)
    os.makedirs(results_saved_folder, exist_ok=True)

    sum_training_loss = []
    sum_testing_loss = []
    sum_training_acc = []
    sum_testing_acc = []
    sum_learning_rate = []

    for epoch in tqdm(range(epochs)):
        if sam_seed != None:
            if epoch == 0:
                print("can sampled!!")
            sampled_train_nonhit_index = random.sample(train_nonhit_index, train_nonhit)
            train_index = sorted(train_hit_index + sampled_train_nonhit_index)
            train_data_numpy = feat_numpy[train_index]
            train_gt_numpy = gt_tmp[train_index]
            train_feat = torch.tensor(train_data_numpy).float()
            train_feat = torch.reshape(train_feat, (-1, 1, train_feat.size()[1]))
            train_gt = torch.tensor(train_gt_numpy)
        train_mlps(model, train_feat, train_gt, dropout, criterion, optimizer, device)
        train_acc, training_loss, *_, true_trn, pred_trn, f1_trn, precision_trn, \
        recall_trn, report_sum_trn = eval_mlps(model, train_feat, train_gt, dropout, criterion, device)
        scheduler.step()
        test_acc, testing_loss, *_, true, pred, f1_test, precision_test, \
        recall_test, report_sum_test = eval_mlps(model, val_feat, val_gt, dropout, criterion, device)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        if epoch == 0:
            saved_epoch = epoch
            saved_val_acc_of_epoch = 4*precision_test*recall_test/(recall_test + 2*precision_test)
        else:
            tmp_val_acc = 4*precision_test*recall_test/(recall_test + 2*precision_test)
            if tmp_val_acc > saved_val_acc_of_epoch:
                saved_val_acc_of_epoch = tmp_val_acc
                saved_epoch = epoch
                step_count = 0
                saved_pred = pred
                saved_pred_trn = pred_trn
                saved_report_sum_test = report_sum_test
                saved_report_sum_trn = report_sum_trn
                saved_sum_training_acc, saved_sum_testing_acc, saved_sum_training_loss, saved_sum_testing_loss, \
                saved_sum_learning_rate = acc_loss_collection(sum_training_loss, sum_testing_loss, sum_training_acc,
                                                sum_testing_acc, sum_learning_rate, train_acc, train_feat,
                                                test_acc, val_feat, training_loss, testing_loss, scheduler)
            else:
                step_count += 1
        sum_training_acc, sum_testing_acc, sum_training_loss, sum_testing_loss, \
        sum_learning_rate = acc_loss_collection(sum_training_loss, sum_testing_loss, sum_training_acc,
                                                sum_testing_acc, sum_learning_rate, train_acc, train_feat,
                                                test_acc, val_feat, training_loss, testing_loss, scheduler)

        label_dict = ('non-hit', 'hit')
        if epoch != 0 and epoch % 10 == 0:
            torch.save(model.state_dict(), temp_models_saved_folder + '/model%05d.pth' % epoch)

            pdf_path = pdf_path_tmp + "epoch_" + str(epoch)
            with PdfPages(pdf_path + ".pdf") as pdf:
                output_loss_for_mlp(epoch, sum_training_loss, sum_testing_loss, sum_training_acc, sum_testing_acc,
                            sum_learning_rate, results_saved_folder)
                additional_info = "parameters\n\nTesting  ---------------------------\n" \
                                  + report_sum_test + "\nTraining  ---------------------------\n" + report_sum_trn
                pdf.attach_note(additional_info, positionRect=[10, 500, 10, 10])
                pdf.savefig()

                y_pred = pred
                y_true = true
                plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                      classes=label_dict, results_saved_folder=results_saved_folder, epoch=epoch,
                                      trn_or_test="Validation", cmap="Oranges")
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
        if step_count == 20:
            torch.save(model.state_dict(), temp_models_saved_folder + '/model%05d.pth' % saved_epoch)

            pdf_path = pdf_path_tmp + "epoch_" + str(saved_epoch)
            with PdfPages(pdf_path + ".pdf") as pdf:
                output_loss_for_mlp(saved_epoch, saved_sum_training_loss, saved_sum_testing_loss, saved_sum_training_acc, saved_sum_testing_acc,
                                    saved_sum_learning_rate, results_saved_folder)
                additional_info = "parameters\n\nTesting  ---------------------------\n" \
                                  + saved_report_sum_test + "\nTraining  ---------------------------\n" + saved_report_sum_trn
                pdf.attach_note(additional_info, positionRect=[10, 500, 10, 10])
                pdf.savefig()

                y_pred = saved_pred
                y_true = true
                plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                      classes=label_dict, results_saved_folder=results_saved_folder, epoch=saved_epoch,
                                      trn_or_test="Validation", cmap="Oranges")
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
            saved_data.append(saved_epoch)
            saved_data.append(saved_val_acc_of_epoch)
            saved_data.append(saved_report_sum_trn["1"]["f1-score"])
            saved_data.append(saved_report_sum_test["1"]["f1-score"])
            return saved_data

def train_feat_mlp_val(train_feat, val_feat, test_feat, train_gt, val_gt, test_gt, in_channels, middle_channels, n_pre_layer, out_channels,
                   n_connected_layer, dropout, epochs, init_rate, device, seed, optim,
                   sam_seed=None, train_hit_index=None, train_nonhit_index=None, feat_numpy=None, gt_tmp=None):
    sample_name = ""
    saved_data = []
    saved_epoch = 0
    saved_val_acc_of_epoch = 0
    step_count = 0
    if sam_seed != None:
        train_nonhit = 83
        random.seed(sam_seed)
        sample_name = "_sampled"
    train_feat = torch.reshape(train_feat, (-1, 1, train_feat.size()[1]))
    val_feat = torch.reshape(val_feat, (-1, 1, val_feat.size()[1]))
    test_feat = torch.reshape(test_feat, (-1, 1, test_feat.size()[1]))

    print("train:", train_feat.size(), ", ", train_gt.size())
    print("val:", val_feat.size(), ", ", val_gt.size())
    print("test:", test_feat.size(), ",", test_gt.size())


    model = mlp(in_channels, middle_channels, out_channels, n_pre_layer, n_connected_layer)
    print('mlp_model:', model)
    criterion = torch.nn.CrossEntropyLoss()
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=init_rate)
    elif optim == "Adam":
        # optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        optimizer = torch.optim.Adam(model.parameters(), lr=init_rate)
    scheduler = MultiStepLR(optimizer, milestones=[20, 200, 800, 2000, 5000], gamma=0.5)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    temp_models_saved_folder = "../../results/for_GCN_dataset/nonrapped/mlp_models/compare/" + "seeds_" + str(seed) + "_" + str(middle_channels) + "/" + str(optimizer)[:3] + "_lyr_"\
                               + str(n_pre_layer) + "_drop_" + str(dropout) + "_init_rate_" + str(init_rate) + sample_name
    results_saved_folder = "../../results/for_GCN_dataset/nonrapped/mlp_models/compare/" + "seeds_" + str(seed) + "_" + str(middle_channels) + "/" + str(optimizer)[:3] + "_lyr_" \
                           + str(n_pre_layer) + "_drop_" + str(dropout) + "_init_rate_" + str(init_rate) + sample_name
    pdf_path_tmp = temp_models_saved_folder + "/"

    os.makedirs(temp_models_saved_folder, exist_ok=True)
    os.makedirs(results_saved_folder, exist_ok=True)

    sum_training_loss = []
    sum_val_loss = []
    sum_testing_loss = []
    sum_training_acc = []
    sum_val_acc = []
    sum_testing_acc = []
    sum_learning_rate = []

    for epoch in tqdm(range(epochs)):
        if sam_seed != None:
            if epoch == 0:
                print("can sampled!!")
            sampled_train_nonhit_index = random.sample(train_nonhit_index, train_nonhit)
            train_index = sorted(train_hit_index + sampled_train_nonhit_index)
            train_data_numpy = feat_numpy[train_index]
            train_gt_numpy = gt_tmp[train_index]
            train_feat = torch.tensor(train_data_numpy).float()
            train_feat = torch.reshape(train_feat, (-1, 1, train_feat.size()[1]))
            train_gt = torch.tensor(train_gt_numpy)
        train_mlps(model, train_feat, train_gt, dropout, criterion, optimizer, device)
        train_acc, training_loss, *_, true_trn, pred_trn, f1_trn, precision_trn, \
        recall_trn, report_sum_trn = eval_mlps(model, train_feat, train_gt, dropout, criterion, device)
        scheduler.step()
        val_acc, val_loss, *_, true_val, pred_val, f1_val, precision_val, \
        recall_val, report_sum_val = eval_mlps(model, val_feat, val_gt, dropout, criterion, device)
        test_acc, testing_loss, *_, true, pred, f1_test, precision_test, \
        recall_test, report_sum_test = eval_mlps(model, test_feat, test_gt, dropout, criterion, device)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

        if epoch == 0:
            able_learn_flag = False
            saved_epoch = epoch
            if precision_val[1] == 0 and recall_val[1] == 0:
                saved_val_acc_of_epoch = 0
            else:
                saved_val_acc_of_epoch = 4*precision_val[1]*recall_val[1]/(recall_val[1] + 2*precision_val[1])
        elif epoch == epochs - 1:
            if precision_val[1] == 0 and recall_val[1] == 0:
                tmp_val_acc = 0
            else:
                tmp_val_acc = 4 * precision_val[1] * recall_val[1] / (recall_val[1] + 2 * precision_val[1])
            able_learn_flag = True
            saved_val_acc_of_epoch = tmp_val_acc
            saved_epoch = epoch
            step_count = 20
            saved_pred = pred
            saved_pred_trn = pred_trn
            saved_pred_val = pred_val
            saved_report_sum_test = report_sum_test
            saved_report_sum_trn = report_sum_trn
            saved_report_sum_val = report_sum_val
            save_model = model.state_dict().copy()
            saved_sum_training_acc, saved_sum_val_acc, saved_sum_testing_acc, saved_sum_training_loss, saved_sum_val_loss, saved_sum_testing_loss, \
            saved_sum_learning_rate = acc_loss_collection_with_val(sum_training_loss, sum_val_loss,
                                                                   sum_testing_loss, sum_training_acc,
                                                                   sum_val_acc, sum_testing_acc, sum_learning_rate,
                                                                   train_acc, train_feat,
                                                                   val_acc, val_feat, test_acc, test_feat,
                                                                   training_loss, val_loss, testing_loss, scheduler)
        else:
            if precision_val[1] == 0 and recall_val[1] == 0:
                tmp_val_acc = 0
            else:
                tmp_val_acc = 4*precision_val[1]*recall_val[1]/(recall_val[1] + 2*precision_val[1])
            if tmp_val_acc > saved_val_acc_of_epoch:
                print("\n\nあと20epoch!\n\n")
                able_learn_flag = True
                saved_val_acc_of_epoch = tmp_val_acc
                saved_epoch = epoch
                step_count = 0
                saved_pred = pred
                saved_pred_trn = pred_trn
                saved_pred_val = pred_val
                saved_report_sum_test = report_sum_test
                saved_report_sum_trn = report_sum_trn
                saved_report_sum_val = report_sum_val
                save_model = model.state_dict().copy()
                saved_sum_training_acc, saved_sum_val_acc, saved_sum_testing_acc, saved_sum_training_loss, saved_sum_val_loss, saved_sum_testing_loss, \
                saved_sum_learning_rate = acc_loss_collection_with_val(sum_training_loss, sum_val_loss, sum_testing_loss, sum_training_acc,
                                                sum_val_acc, sum_testing_acc, sum_learning_rate, train_acc, train_feat,
                                                val_acc, val_feat, test_acc, test_feat, training_loss, val_loss, testing_loss, scheduler)
            else:
                step_count += 1
        sum_training_acc, sum_val_acc, sum_testing_acc, sum_training_loss, sum_val_loss, sum_testing_loss, \
        sum_learning_rate = acc_loss_collection_with_val(sum_training_loss, sum_val_loss, sum_testing_loss,
                                                               sum_training_acc,
                                                               sum_val_acc, sum_testing_acc, sum_learning_rate,
                                                               train_acc, train_feat,
                                                               val_acc, val_feat, test_acc, test_feat, training_loss,
                                                               val_loss, testing_loss, scheduler)

        label_dict = ('non-hit', 'hit')
        if epoch != 0 and epoch % 10 == 0:
            torch.save(model.state_dict(), temp_models_saved_folder + '/model%05d.pth' % epoch)

            pdf_path = pdf_path_tmp + "epoch_" + str(epoch)
            with PdfPages(pdf_path + ".pdf") as pdf:
                output_loss_with_val(epoch, sum_training_loss, sum_val_loss, sum_testing_loss, sum_training_acc,
                                     sum_val_acc, sum_testing_acc, sum_learning_rate, results_saved_folder)
                # additional_info = "parameters\n\nTesting  ---------------------------\n" \
                #                   + report_sum_test + "\nTraining  ---------------------------\n" + report_sum_trn
                # pdf.attach_note(additional_info, positionRect=[10, 500, 10, 10])
                # pdf.savefig()

                y_pred = pred
                y_true = true
                plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                      classes=label_dict, results_saved_folder=results_saved_folder, epoch=epoch,
                                      trn_or_test="Validation", cmap="Oranges")
                # pdf.savefig()

                y_pred = pred_trn
                # print(y_pred)
                y_true = true_trn
                # print(y_true)
                plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                      classes=label_dict, results_saved_folder=results_saved_folder, epoch=epoch,
                                      title="cfm on training", cmap="Greens")
                # pdf.savefig()

                print("Testing\n", report_sum_test)
                print("Training\n", report_sum_trn)
                print("Validation\n", report_sum_val)
        if step_count == 20 and able_learn_flag:
            torch.save(save_model, temp_models_saved_folder + '/model%05d.pth' % saved_epoch)

            pdf_path = pdf_path_tmp + "epoch_" + str(saved_epoch)
            with PdfPages(pdf_path + ".pdf") as pdf:
                output_loss_with_val(saved_epoch, saved_sum_training_loss, saved_sum_val_loss, saved_sum_testing_loss, saved_sum_training_acc, saved_sum_val_acc, saved_sum_testing_acc,
                                    saved_sum_learning_rate, results_saved_folder)
                additional_info = "parameters\n\nTesting  ---------------------------\n" \
                                  + saved_report_sum_test + "\nTraining  ---------------------------\n" + saved_report_sum_trn
                # pdf.attach_note(additional_info, positionRect=[10, 500, 10, 10])
                # pdf.savefig()

                y_pred = saved_pred
                y_true = true
                plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                      classes=label_dict, results_saved_folder=results_saved_folder, epoch=saved_epoch,
                                      trn_or_test="Validation", cmap="Blues")
                # pdf.savefig()

                y_pred = saved_pred_trn
                # print(y_pred)
                y_true = true_trn
                # print(y_true)
                plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                      classes=label_dict, results_saved_folder=results_saved_folder, epoch=saved_epoch,
                                      title="cfm on training", cmap="Greens")
                # pdf.savefig()

                y_pred = saved_pred_val
                y_true = true_val
                # print(y_true)
                plot_confusion_matrix(torch.tensor(y_true, device='cpu'), torch.tensor(y_pred, device='cpu'),
                                      classes=label_dict, results_saved_folder=results_saved_folder, epoch=saved_epoch,
                                      title="cfm on validation", cmap="Oranges")

                print("Testing\n", saved_report_sum_test)
                print("Training\n", saved_report_sum_trn)
                print("Validation\n", saved_report_sum_val)
            saved_data.append(saved_epoch)
            saved_data.append(saved_val_acc_of_epoch)
            # print(saved_report_sum_trn[148:152])
            saved_data.append(saved_report_sum_trn[148:152])
            saved_data.append(saved_report_sum_val[148:152])
            saved_data.append(saved_report_sum_test[148:152])
            return saved_data

def train_feat_autoencoder(train_feat, val_feat, in_channels, middle_channels, n_layer, dropout, epochs, init_rate, device, seed, optim):
    saved_data = []
    saved_epoch = 0
    saved_loss_epoch = 100000
    step_count = 0

    print("train:", train_feat.size())
    print("val:", val_feat.size())

    model = autoencoder(in_channels, middle_channels, n_layer)
    print('auto-encoder_model:', model)
    criterion = torch.nn.MSELoss()
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=init_rate)
    elif optim == "Adam":
        # optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        optimizer = torch.optim.Adam(model.parameters(), lr=init_rate)
    scheduler = MultiStepLR(optimizer, milestones=[20, 200, 800, 2000, 5000], gamma=0.5)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    temp_models_saved_folder = "../../results/nonrapped/autoencoder_models/" + "79784_seeds_" + str(seed) + "_" + str(middle_channels) + "/" + str(optimizer)[:3] \
                               + "_lyr_" + str(n_layer) + "_drop_" + str(dropout) + "_init_rate_" + str(init_rate)
    results_saved_folder = "../../results/nonrapped/autoencoder_models/" + "79784_seeds_" + str(seed) + "_" + str(middle_channels) + "/" + str(optimizer)[:3] \
                           + "_lyr_" + str(n_layer) + "_drop_" + str(dropout) + "_init_rate_" + str(init_rate)
    pdf_path_tmp = temp_models_saved_folder + "/"

    os.makedirs(temp_models_saved_folder, exist_ok=True)
    os.makedirs(results_saved_folder, exist_ok=True)

    sum_training_loss = []
    sum_testing_loss = []
    sum_learning_rate = []

    for epoch in tqdm(range(epochs)):
        train_autoencoder(model, train_feat, dropout, criterion, optimizer, device)
        training_loss = eval_autoencoder(model, train_feat, dropout, criterion, device)
        scheduler.step()
        testing_loss = eval_autoencoder(model, val_feat, dropout, criterion, device)
        print(f'Epoch: {epoch:03d}, Train Loss: {(sum(training_loss)/len(training_loss)):.4f}, Test Loss: {(sum(testing_loss)/len(testing_loss)):.4f}')

        sum_training_loss, sum_testing_loss, \
        sum_learning_rate = acc_loss_collection_for_auto(sum_training_loss, sum_testing_loss, sum_learning_rate,
                                                         training_loss, testing_loss, scheduler)

        if epoch == 0:
            able_learn_flag = False
            saved_epoch = epoch
            saved_loss_epoch = sum(testing_loss) / len(testing_loss)
            step_count = 0
            saved_sum_training_loss, saved_sum_testing_loss, \
            save_sum_learning_rate = sum_training_loss, sum_testing_loss, sum_learning_rate
            save_model = model.state_dict().copy()
        elif epoch == epochs-1:
            step_count = 20
            able_learn_flag = True
        else:
            if sum(testing_loss) / len(testing_loss) < saved_loss_epoch:
                print("\n\nあと20epoch!\n\n")
                able_learn_flag = True
                saved_loss_epoch = sum(testing_loss) / len(testing_loss)
                saved_epoch = epoch
                step_count = 0
                saved_sum_training_loss, saved_sum_testing_loss, \
                save_sum_learning_rate = sum_training_loss, sum_testing_loss, sum_learning_rate
                save_model = model.state_dict().copy()
            else:
                step_count += 1

        if epoch != 0 and epoch % 10 == 0:
            torch.save(model.state_dict(), temp_models_saved_folder + '/model%05d.pth' % epoch)

            pdf_path = pdf_path_tmp + "epoch_" + str(epoch)
            with PdfPages(pdf_path + ".pdf") as pdf:
                output_loss_for_auto(epoch, sum_training_loss, sum_testing_loss,
                            sum_learning_rate, results_saved_folder)
                pdf.savefig()
        if step_count >= 20 and able_learn_flag:
            torch.save(save_model, temp_models_saved_folder + '/model%05d.pth' % saved_epoch)

            pdf_path = pdf_path_tmp + "epoch_" + str(saved_epoch)
            with PdfPages(pdf_path + ".pdf") as pdf:
                output_loss_for_auto(saved_epoch, saved_sum_training_loss, saved_sum_testing_loss,
                                     save_sum_learning_rate, results_saved_folder)
                pdf.savefig()
            saved_data.append(saved_epoch)
            saved_data.append(saved_loss_epoch)
            return saved_data

def mlp_dim_reduce(feat_tensor, in_channels, middle_channels, n_pre_layer, out_channels, n_connected_layer,
               pretrained_model, dropout, device, tmp_tensor):
    feat_tensor = torch.reshape(feat_tensor, (-1, 1, feat_tensor.size()[1]))
    model = mlp(in_channels, middle_channels, out_channels, n_pre_layer, n_connected_layer)
    model.load_state_dict(torch.load(pretrained_model))

    reduced_feat, pred_test, pred_label = test_mlps(model, feat_tensor, dropout, device, tmp_tensor)

    return reduced_feat, pred_test, pred_label

def AE_dim_reduce(feat_tensor, in_channels, middle_channels, n_pre_layer,
               pretrained_model, dropout, device, tmp_tensor):
    feat_tensor = torch.reshape(feat_tensor, (-1, 1, feat_tensor.size()[1]))
    model = autoencoder(in_channels, middle_channels, n_pre_layer)
    model.load_state_dict(torch.load(pretrained_model))

    reduced_feat = test_autoencoder(model, feat_tensor, dropout, device, tmp_tensor)

    return reduced_feat

def AE_test(feat_tensor, in_channels, middle_channels, n_pre_layer,
               pretrained_model, dropout, device, tmp_tensor):
    feat_tensor = torch.reshape(feat_tensor, (-1, 1, feat_tensor.size()[1]))
    model = autoencoder(in_channels, middle_channels, n_pre_layer)
    model.load_state_dict(torch.load(pretrained_model))

    criterion = torch.nn.MSELoss()

    testing_loss = eval_autoencoder(model, feat_tensor, dropout, criterion, device)
    print(f'Test Loss: {(sum(testing_loss)/len(testing_loss)):.4f}')


    return testing_loss