import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def output_loss(epoch, sum_training_loss, sum_testing_loss, sum_training_acc, sum_testing_acc, sum_learning_rate, results_saved_folder):
    plt.rcParams['figure.figsize'] = (8, 7)
    sns.set(font_scale=1)
    fig, ax = plt.subplots(3, 1)

    ax[0].set_title('Loss Curve (g:training, b:testing)', fontweight='bold', fontsize=10)
    ax[0].plot(sum_training_loss, '-o', color='g', markersize=3, linewidth=2, alpha=0.5)
    ax[0].plot(sum_testing_loss, '-o', color='b', markersize=3, linewidth=2, alpha=0.5)
    ax[0].set_xlabel('Epochs', fontsize=10)
    ax[0].set_ylabel('Loss', fontsize=10)

    ax[1].set_title('Acc Curve (g:training, b:testing)', fontweight='bold', fontsize=10)
    ax[1].plot(sum_training_acc, '-o', color='g', markersize=3, linewidth=2, alpha=0.5)
    ax[1].plot(sum_testing_acc, '-o', color='b', markersize=3, linewidth=2, alpha=0.5)
    ax[1].set_xlabel("Epochs", fontsize=10)
    ax[1].set_ylabel("Accuracy", fontsize=10)

    ax[2].set_title('learning rate (lr)', fontweight='bold', fontsize=10)
    ax[2].plot(sum_learning_rate, '-', color='m', alpha=0.8, markersize=4, linewidth=2)
    ax[2].set_xlabel('Epochs', fontsize=10)

    plt.tight_layout()
    plt.savefig(results_saved_folder + '/ls_acc_lr_curve%05d.png' % epoch)

    return plt

def output_loss_with_val(epoch, sum_training_loss, sum_val_loss, sum_testing_loss, sum_training_acc, sum_val_acc, sum_testing_acc, sum_learning_rate, results_saved_folder):
    plt.rcParams['figure.figsize'] = (8, 7)
    sns.set(font_scale=1)
    fig, ax = plt.subplots(3, 1)

    ax[0].set_title('Loss Curve (g:training, o:validation, b:testing)', fontweight='bold', fontsize=10)
    ax[0].plot(sum_training_loss, '-o', color='g', markersize=3, linewidth=2, alpha=0.5)
    ax[0].plot(sum_testing_loss, '-o', color='b', markersize=3, linewidth=2, alpha=0.5)
    ax[0].plot(sum_val_loss, '-o', color='r', markersize=3, linewidth=2, alpha=0.5)
    ax[0].set_xlabel('Epochs', fontsize=10)
    ax[0].set_ylabel('Loss', fontsize=10)

    ax[1].set_title('Acc Curve (g:training, o:validation, b:testing)', fontweight='bold', fontsize=10)
    ax[1].plot(sum_training_acc, '-o', color='g', markersize=3, linewidth=2, alpha=0.5)
    ax[1].plot(sum_testing_acc, '-o', color='b', markersize=3, linewidth=2, alpha=0.5)
    ax[1].plot(sum_val_acc, '-o', color='r', markersize=3, linewidth=2, alpha=0.5)
    ax[1].set_xlabel("Epochs", fontsize=10)
    ax[1].set_ylabel("Accuracy", fontsize=10)

    ax[2].set_title('learning rate (lr)', fontweight='bold', fontsize=10)
    ax[2].plot(sum_learning_rate, '-', color='m', alpha=0.8, markersize=4, linewidth=2)
    ax[2].set_xlabel('Epochs', fontsize=10)

    plt.tight_layout()
    plt.savefig(results_saved_folder + '/ls_acc_lr_curve%05d.png' % epoch)

    return plt

def output_loss_for_mlp(epoch, sum_training_loss, sum_val_loss, sum_training_acc, sum_val_acc, sum_learning_rate, results_saved_folder):
    plt.rcParams['figure.figsize'] = (8, 7)
    sns.set(font_scale=1)
    fig, ax = plt.subplots(3, 1)

    ax[0].set_title('Loss Curve (g:training, o:validation)', fontweight='bold', fontsize=10)
    ax[0].plot(sum_training_loss, '-o', color='g', markersize=3, linewidth=2, alpha=0.5)
    ax[0].plot(sum_val_loss, '-o', color='r', markersize=3, linewidth=2, alpha=0.5)
    ax[0].set_xlabel('Epochs', fontsize=10)
    ax[0].set_ylabel('Loss', fontsize=10)

    ax[1].set_title('Acc Curve (g:training, o:validation)', fontweight='bold', fontsize=10)
    ax[1].plot(sum_training_acc, '-o', color='g', markersize=3, linewidth=2, alpha=0.5)
    ax[1].plot(sum_val_acc, '-o', color='r', markersize=3, linewidth=2, alpha=0.5)
    ax[1].set_xlabel("Epochs", fontsize=10)
    ax[1].set_ylabel("Accuracy", fontsize=10)

    ax[2].set_title('learning rate (lr)', fontweight='bold', fontsize=10)
    ax[2].plot(sum_learning_rate, '-', color='m', alpha=0.8, markersize=4, linewidth=2)
    ax[2].set_xlabel('Epochs', fontsize=10)

    plt.tight_layout()
    plt.savefig(results_saved_folder + '/ls_acc_lr_curve%05d.png' % epoch)

    return plt

def output_loss_for_auto(epoch, sum_training_loss, sum_testing_loss, sum_learning_rate, results_saved_folder):
    plt.rcParams['figure.figsize'] = (8, 7)
    sns.set(font_scale=1)
    fig, ax = plt.subplots(2, 1)

    ax[0].set_title('Loss Curve (g:training, b:testing)', fontweight='bold', fontsize=10)
    ax[0].plot(sum_training_loss, '-o', color='g', markersize=3, linewidth=2, alpha=0.5)
    ax[0].plot(sum_testing_loss, '-o', color='b', markersize=3, linewidth=2, alpha=0.5)
    ax[0].set_xlabel('Epochs', fontsize=10)
    ax[0].set_ylabel('Loss', fontsize=10)

    ax[1].set_title('learning rate (lr)', fontweight='bold', fontsize=10)
    ax[1].plot(sum_learning_rate, '-', color='m', alpha=0.8, markersize=4, linewidth=2)
    ax[1].set_xlabel('Epochs', fontsize=10)

    plt.tight_layout()
    plt.savefig(results_saved_folder + '/ls_lr_curve%05d.png' % epoch)

    return plt

def acc_loss_collection(sum_training_loss, sum_testing_loss, sum_training_acc,
                       sum_testing_acc, sum_learning_rate, train_acc, train_dataset,
                       test_acc, test_dataset, training_loss, testing_loss, scheduler):
    sum_training_acc.append(train_acc / len(train_dataset))
    sum_testing_acc.append(test_acc / len(test_dataset))
    avg_training_loss = sum(training_loss) / len(training_loss)
    sum_training_loss.append(avg_training_loss)
    avg_testing_loss = sum(testing_loss) / len(testing_loss)
    sum_testing_loss.append(avg_testing_loss)
    sum_learning_rate.append(scheduler.get_last_lr()[0])

    return sum_training_acc, sum_testing_acc, sum_training_loss, sum_testing_loss, sum_learning_rate

def acc_loss_collection_with_val(sum_training_loss, sum_val_loss, sum_testing_loss, sum_training_acc,
                       sum_val_acc, sum_testing_acc, sum_learning_rate, train_acc, train_dataset,
                       val_acc, val_dataset, test_acc, test_dataset, training_loss, val_loss, testing_loss, scheduler):
    sum_training_acc.append(train_acc / len(train_dataset))
    sum_testing_acc.append(test_acc / len(test_dataset))
    sum_val_acc.append(val_acc / len(val_dataset))
    avg_training_loss = sum(training_loss) / len(training_loss)
    sum_training_loss.append(avg_training_loss)
    avg_testing_loss = sum(testing_loss) / len(testing_loss)
    sum_testing_loss.append(avg_testing_loss)
    avg_val_loss = sum(val_loss) / len(val_loss)
    sum_val_loss.append(avg_val_loss)
    sum_learning_rate.append(scheduler.get_last_lr()[0])

    return sum_training_acc, sum_val_acc, sum_testing_acc, sum_training_loss, sum_val_loss, sum_testing_loss, sum_learning_rate

def acc_loss_collection_for_auto(sum_training_loss, sum_testing_loss, sum_learning_rate, training_loss, testing_loss, scheduler):
    avg_training_loss = sum(training_loss) / len(training_loss)
    sum_training_loss.append(avg_training_loss)
    avg_testing_loss = sum(testing_loss) / len(testing_loss)
    sum_testing_loss.append(avg_testing_loss)
    sum_learning_rate.append(scheduler.get_last_lr()[0])

    return sum_training_loss, sum_testing_loss, sum_learning_rate