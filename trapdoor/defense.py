import torch
import math
import numpy as np
import pickle
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CIFAR10:
    def __init__(self, seed = 43):
        import tensorflow as tf
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


def injection_func(mask, pattern, adv_img):
    if len(adv_img.shape) == 4:
        return mask.transpose((0,3,1,2)) * pattern.transpose((0,3,1,2)) + (1 - mask.transpose((0,3,1,2))) * adv_img
    else:
        return mask.transpose((2,0,1)) * pattern.transpose((2,0,1)) + (1 - mask.transpose((2,0,1))) * adv_img
    
def mask_pattern_func(y_target, pattern_dict):
    mask, pattern = random.choice(pattern_dict[y_target])
    mask = np.copy(mask)
    return mask, pattern
    
def infect_X(img, tgt, num_classes, pattern_dict):
    mask, pattern = mask_pattern_func(tgt, pattern_dict)
    raw_img = np.copy(img)
    adv_img = np.copy(raw_img)

    adv_img = injection_func(mask, pattern, adv_img)
    return adv_img, None


def build_neuron_signature(model, X, Y, y_target, pattern_dict):
    num_classes = 10
    X_adv = np.array(
        [infect_X(img, y_target, pattern_dict=pattern_dict, num_classes=num_classes)[0] for img in np.copy(X)])
    BS = 512
    X_neuron_adv = np.concatenate([model(X_adv[i:i+BS], upto=-3) for i in range(0,len(X_adv),BS)])
    X_neuron_adv = np.mean(X_neuron_adv, axis=0)
    sig = X_neuron_adv
    return sig


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        class Transpose(torch.nn.Module):
            def forward(self, x):
                return x.permute((0, 2, 3, 1))

        self.layers = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32, eps=.000),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32, eps=.000),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64, eps=.000),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64, eps=.000),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128, eps=.000),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128, eps=.000),
            torch.nn.MaxPool2d(2, 2),

            Transpose(),
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024, eps=.000),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512, eps=.000),
            torch.nn.Linear(512, 10),
            ])

    def __call__(self, x, training=False, upto=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        extra = []
        for i,layer in enumerate(self.layers[:upto] if upto is not None else self.layers):
            #print(layer, x)                                                                                                                                                                
            x = layer(x)
        return x

def run_detect(model, x, adv_sig, random=None):
    X_neuron_adv = model(x, upto=-3)

    if random is None: random="correct"
    filter_ratio = .1
    if random == "fast":
        n_mask = torch.rand(X_neuron_adv.shape[1]) < filter_ratio
        X_neuron_adv = X_neuron_adv * n_mask.to(device)
    elif random == "correct":
        number_neuron = X_neuron_adv.shape[1]
        number_keep = int(number_neuron * filter_ratio)
        n_mask = np.array([1] * number_keep + [0] * (number_neuron - number_keep))
        n_mask = np.array(n_mask)
        np.random.shuffle(n_mask)
        #print(n_mask)
        X_neuron_adv = X_neuron_adv * torch.tensor(n_mask).to(device)
    else:
        raise
        
    adv_scores = torch_sim(X_neuron_adv, adv_sig)
    return adv_scores

def torch_sim(X_neuron, adv_sig):
    nb_sample = X_neuron.shape[0]

    dotted = torch.matmul(X_neuron, adv_sig.reshape((512, 1))).flatten()
    dotted /= (X_neuron**2).sum(axis=1)**.5
    dotted /= (adv_sig**2).sum()**.5

    return dotted

def load_model():
    RES = pickle.load(open("trapdoor/torch_cifar_res.p","rb"))
    target_ls = RES['target_ls']
    pattern_dict = RES['pattern_dict']
    
    model = TorchModel()
    model.load_state_dict(torch.load('trapdoor/torch_cifar_model.h5'))
    model = model.eval().to(device)

    #"""
    data = CIFAR10()
    signature = build_neuron_signature(lambda x,upto=None: model(torch.tensor(x, dtype=torch.float32).to(device),upto=upto).cpu().detach().numpy(),
                                       data.train_data, data.train_labels, 3, pattern_dict)
    np.save("next_signature.npy", signature)
    #"""
    signature = np.load("next_signature.npy")
    signature = torch.tensor(signature).to(device)

    def detector(x, how=None):
        return run_detect(model, x, signature, how)
    
    return model, detector
