import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    """
    Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
    """
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def _one_sample_positive_class_precisions(scores, truth):
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)

    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)

    retrieved_classes = np.argsort(scores)[::-1]

    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits

def lwlrap(truth, scores):
    # https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/198418
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = _one_sample_positive_class_precisions(scores[sample_num, :], truth[sample_num, :])
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = precision_at_hits

    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))

    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    return per_class_lwlrap, weight_per_class


def visual_train_result(log)->None:
    """
    take log file after train, *.txt
    save image or plot
    """
    names = ['times', 'folds', 'epoch', 'lr', 'auc_val','auc_train', 'LRAPS', 'lwlrap', 'train_loss', 'val_loss']
    df = pd.read_csv(os.path.join('../logs/', log), header = None, names = names)
    for col in names[1:]:
        df[col] = df[col].str.split(':', n = 1,expand =True)[1].astype('float')
        
    data, title, _ = os.path.basename(log).split('.')

    # plot
    plt.figure(figsize= (15, 5))
    plt.plot(df.epoch, df.auc_train, '-o', label = 'Train AUC', color = '#11ad77')
    plt.plot(df.epoch, df.auc_val, '-o', label = 'Val AUC', color = '#780b0b')
    x = np.argmax(df.auc_val)
    y = np.max(df.auc_val)
    plt.scatter(x, y, s = 200, color = '#780b0b')
    plt.text(x-0.20, y-0.08, 'max_auc\n%.3f' % y, size = 14)
    plt.ylabel('AUC',size=14)
    plt.xlabel('Epoch',size=14)
    plt.legend(loc=2)
    plt2 = plt.gca().twinx()
    plt2.plot(df.epoch, df.train_loss, '-o', label = 'Train_loss', color = '#8cc2af')
    plt2.plot(df.epoch, df.val_loss, '-o', label = 'Val_loss', color = '#db1819')
    x = np.argmin(df.val_loss)
    y = np.min(df.val_loss)
    plt2.scatter(x, y, s = 200, color = '#db1819')
    plt2.text(x-0.20, y-0.10, 'min_loss\n%.3f' % y, size = 14)
    plt.ylabel('Loss',size=14)
    plt.legend(loc=3)
    plt.title(title.replace('log_', ''), size = 18)
    plt.savefig(os.path.join('../logs/', f'{data}_{title}.png')) # before show


def rand_window(data: np.array, version = None)->np.array:
    """
    idea we make image 10 sec size
    after cut random by 6 sec

    v1 image shape 10 sec is 384*563 --> cut image 384*376  
    v2                       260*844 --> cut image 260*563
   
    """
    assert version, 'add version v1(384) or v2(260)'
    if version == 'v1':
        current_len = 563 # ~ 10 sec
        cut_len = 376 # ~ 6 sec   
    else:
        current_len = 844
        cut_len = 563    
 
    start = np.random.randint(0, current_len - cut_len)
    len_img = (start + cut_len) - start
    assert len_img == cut_len, f'error len {start}, {start + cut_len}'
    return data[:, start: start + cut_len]  
