import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, classes, results_saved_folder, epoch,
                          normalize=None, title=None, cmap=plt.cm.Reds, trn_or_test="Train", cmp=False):
    cm = confusion_matrix(y_true, y_pred)
    # print(cm.shape, cm)
    fig_cm, ax_cm = plt.subplots(figsize=(22, 15))
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.grid(False)

    ax_cm.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[1]),
              xticklabels=classes, yticklabels=classes, title=title,
              ylabel='True label', xlabel='Predicted label')
    ax_cm.set_title('Confusion matrix', fontsize=45)
    if cmp:
        ax_cm.set_xlabel('XOR both MLP and GCN', fontsize=45)
    else:
        ax_cm.set_xlabel('Predicted label', fontsize=45)
    ax_cm.set_ylabel('True label', fontsize=45)

    plt.setp(ax_cm.get_xticklabels(), horizontalalignment='center',
             rotation_mode='anchor', fontsize=35)
    plt.setp(ax_cm.get_yticklabels(), rotation=90, horizontalalignment='center',
             verticalalignment='baseline', rotation_mode='anchor', fontsize=35)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.grid(False)

    sns.set(font_scale=7)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                       verticalalignment='center', color="white" if cm[i, j] > thresh else "black")
    fig_cm.tight_layout()
    plt.savefig(results_saved_folder + '/' + trn_or_test + '_cfm%05d.png' % epoch)
    return ax_cm