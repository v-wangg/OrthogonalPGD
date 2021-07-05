import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import sys
import itertools
import math
import numpy as np
import pickle
import random
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve, auc

import attack
import trapdoor.defense
import sid.defense
import dla.defense
import trapdoor.defense

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EasyDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

class CIFAR10:
    def __init__(self, seed = 43):
        (train_data, train_labels),(self.test_data, self.test_labels) = tf.keras.datasets.cifar10.load_data()
        train_data = train_data/255.
        self.test_data = self.test_data/255.

        VALIDATION_SIZE = 5000

        np.random.seed(seed)
        shuffled_indices = np.arange(len(train_data))
        np.random.shuffle(shuffled_indices)
        train_data = train_data[shuffled_indices]
        train_labels = train_labels[shuffled_indices]

        shuffled_indices = np.arange(len(self.test_data))
        np.random.shuffle(shuffled_indices)
        self.test_data = self.test_data[shuffled_indices].transpose((0,3,1,2))
        self.test_labels = self.test_labels[shuffled_indices].flatten()

        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :].transpose((0,3,1,2))
        self.validation_labels = train_labels[:VALIDATION_SIZE].flatten()
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :].transpose((0,3,1,2))
        self.train_labels = train_labels[VALIDATION_SIZE:].flatten()

from sklearn.metrics import roc_auc_score, roc_curve, auc


def score(num_images, orig, advx_final, detector):
    is_adversarial_label = np.concatenate((np.zeros(num_images), np.ones(num_images))).reshape(-1, 1)
    original_and_adversarial_images = torch.cat((orig.clone(), advx_final.clone())).float()

    test_fprs = []
    test_tprs = []


    adv_scores = detector(original_and_adversarial_images.to(device), "correct").detach().cpu().numpy()

    phi_range = [-np.inf] + list(np.sort(adv_scores)) + [np.inf]
    
    threshold_5 = np.sort(adv_scores[:num_images])[num_images-num_images//20-1]
    threshold_10 = np.sort(adv_scores[:num_images])[num_images-num_images//10-1]
    threshold_50 = np.sort(adv_scores[:num_images])[num_images-num_images//2-1]

    tpr_5 = np.mean(adv_scores[num_images:]>threshold_5)
    tpr_10 = np.mean(adv_scores[num_images:]>threshold_10)
    tpr_50 = np.mean(adv_scores[num_images:]>threshold_50)

    fpr_5 = np.mean(adv_scores[:num_images]>threshold_5)
    fpr_10 = np.mean(adv_scores[:num_images]>threshold_10)
    fpr_50 = np.mean(adv_scores[:num_images]>threshold_50)
    
    for phi in phi_range:
        is_adversarial_pred = 1 * (adv_scores > phi)
        is_adversarial_pred = is_adversarial_pred.reshape(-1, 1)
        
        TP_count = np.sum(is_adversarial_pred*is_adversarial_label)
        TN_count = np.sum((1-is_adversarial_pred)*(1-is_adversarial_label))
        FP_count = np.sum((is_adversarial_pred)*(1-is_adversarial_label))
        FN_count = np.sum((1-is_adversarial_pred)*(is_adversarial_label))

        assert TP_count + TN_count + FP_count + FN_count == num_images*2
        tpr = TP_count/(TP_count+FN_count) if TP_count+FN_count != 0 else 0
        fpr = FP_count/(FP_count+TN_count) if FP_count+TN_count != 0 else 0

        test_tprs.append(tpr)
        test_fprs.append(fpr)
    return tpr_5, tpr_10, tpr_50, auc(test_fprs, test_tprs), adv_scores

def run_experiment(num_images, loader, offset=0, **attack_args):
    model, detector = loader()
    
    pgd = attack.PGD(model, detector, **attack_args)
    images = torch.tensor(data.test_data[offset:offset+num_images], dtype=torch.float32)
    labels = torch.tensor(data.test_labels[offset:offset+num_images], dtype=torch.int64)

    advx = pgd.attack(images.clone(), labels)

    if 'target' in attack_args and attack_args['target'] is not None:
        attack_succeeded = (model(advx.to(device)).argmax(1).cpu()==attack_args['target'])
    else:
        attack_succeeded = (model(advx.to(device)).argmax(1).cpu()!=labels)

    sr = torch.mean(attack_succeeded.float())

    tpr5, tpr10, tpr50, auc, scores = score(len(images), images, advx, detector)
    return EasyDict(success=list(attack_succeeded.numpy()),
                    score=list(scores),
                    sr=sr.item(),
                    tpr5=tpr5,
                    tpr10=tpr10,
                    tpr50=tpr50)

N = 100
data = CIFAR10()

offset = 0
d = {'use_projection': True, 'eps': 0.03, 'alpha': .001, 'steps': 1000,
     'projection_norm': 'linf'
}
out = run_experiment(N, dla.defense.load_model,
                     offset=0,
                     classifier_loss=torch.nn.CrossEntropyLoss(),
                     detector_loss=None,
                     target=None,
                     **d)

print(out)

