from __future__ import print_function, absolute_import, division
import os
import torch
import numpy as np
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler
from common.datasets import (Denormalise,PACS,PACSMultiple,
                             OfficeHome,OfficeHomeMultiple,
                             VLCS,VLCSMultiple,
                             TerraIncognita,TerraIncognitaMultiple,
                             DomainNet,DomainNetMultiple,
                             Fundus,FundusMultiple)
from models.alexnet import alexnet
from models.resnet import resnet18,resnet34,resnet50,resnet101,resnet152
from models.resnext import resnext29
from models.wideresnet import WideResNet
from common.utils import *
from common.utils import (
    fix_all_seed,
    write_log,
    adam,
    sgd,
    compute_accuracy,
    entropy_loss,
    Averager,
)

import torchvision.transforms as transforms
import torch.nn as nn
import kornia
import random
from common.contrastive import SupConLoss
import torchvision
import math
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from config import (PACS_DATA_FOLDER,OfficeHome_DATA_FOLDER,
                    VLCS_DATA_FOLDER,TerraIncognita_DATA_FOLDER,
                    DomainNet_DATA_FOLDER,Fundus_DATA_FOLDER)

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import _LRScheduler
def bn_eval(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

class LEAwareSGD(SGD):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, le_weight=0.1):
        super(LEAwareSGD, self).__init__(params, lr, momentum, dampening,
                                         weight_decay, nesterov)
        self.le_weight = le_weight
        self.current_le = None

    def step(self, closure=None, current_le=None, pre_le=0):
        loss = None
        if closure is not None:
            loss = closure()
        # update LE
        if current_le is not None:
            self.current_le = current_le

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            # Calculate the change in LE
            if self.current_le is not None and pre_le != 0:
                delta_le = self.current_le - pre_le
                # Dynamically adjust the learning rate
                if delta_le > 0:
                    le_factor = math.exp(-self.le_weight * delta_le)
                else:
                    le_factor = 1
            else:
                le_factor = 1.0
            adjusted_lr = lr * le_factor

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Add L2 regularization
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(d_p, alpha=-adjusted_lr)

        return loss


class inv_lr_scheduler(_LRScheduler):
    def __init__(self, optimizer, alpha, beta, total_epoch, last_epoch=-1):
        self.alpha = alpha
        self.beta = beta
        self.total_epoch = total_epoch
        super(inv_lr_scheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr
            * ((1 + self.alpha * self.last_epoch / self.total_epoch) ** (-self.beta))
            for base_lr in self.base_lrs
        ]


class DataPool:
    def __init__(self, pool_size):
        self.data = [[]] * pool_size
        self.pool_size = pool_size
        self.count = 0
        self.num = 0

    def add(self, x):
        if self.count < self.pool_size:
            self.data[self.count] = x
            self.count += 1
        else:
            self.count = 0
            self.data[self.count] = x
            self.count += 1
        if self.num < self.pool_size:
            self.num += 1

    def get(self, num=-1):
        if self.num == 0:
            return []
        if num < 0:
            return self.data[0: self.num]
        else:
            num = min(num, self.num)
            indexes = list(range(self.num))
            random.shuffle(indexes)
            sel_indexes = indexes[0:num]
            return [self.data[i] for i in sel_indexes]


def hsv_aug(x, hsv):
    rgb2hsv = kornia.color.RgbToHsv()
    hsv2rgb = kornia.color.HsvToRgb()
    B = x.shape[0]
    hsv_img = rgb2hsv(x) + hsv.view(B, 3, 1, 1)
    rgb_img = hsv2rgb(hsv_img)
    return torch.clamp(rgb_img, -10, 10)


def rotate_aug(x, angle):
    rgb_img = kornia.geometry.transform.rotate(x, torch.clamp(angle, 0.01, 1) * 360)
    return rgb_img


def translate_aug(x, trans):
    h = x.shape[-1] * 0.1
    rgb_img = kornia.geometry.transform.translate(x, torch.clamp(trans, -1, 1) * h)
    return rgb_img


def invert_aug(x, max_val):
    x = torch.clamp(max_val, 0.5, 1.0).view(len(max_val), 1, 1, 1) - x
    return x


def shear_aug(x, val):
    x = kornia.geometry.transform.shear(x, val)
    return x


def contrast_aug(x, con):
    rgb_img = kornia.enhance.adjust_contrast(x, torch.clamp(con, 0.1, 1.9))
    return rgb_img


def sharpness_aug(x, factor):
    x = kornia.enhance.sharpness(x, torch.clamp(factor, 0, 1))
    return x


def scale_aug(x, factor):
    factor = factor.view(len(factor), 1)
    x = kornia.geometry.transform.scale(x, torch.clamp(factor, 0.5, 2.0))
    return x


def solarize_aug(x, factor):
    x = kornia.enhance.solarize(x, additions=torch.clamp(factor, -0.499, 0.499))
    return x


def equalize_aug(x, factor):
    ex = kornia.enhance.equalize(torch.clamp(x, 0.001, 1.0))
    return ex.detach() + x - x.detach()


def posterize_aug(x, factor):
    bits = torch.randint(0, 9, size=(len(x),)).to(x.device)
    nx = kornia.enhance.posterize(x, bits)
    return nx.detach() + x - x.detach()


def cutout(img_batch, num_holes, hole_size, fill_value=0):
    img_batch = img_batch.clone()
    B = len(img_batch)
    height, width = img_batch.shape[-2:]
    masks = torch.zeros_like(img_batch)
    for _n in range(num_holes):
        if height == hole_size:
            y1 = torch.tensor([0])
        else:
            y1 = torch.randint(0, height - hole_size, (1,))
        if width == hole_size:
            x1 = torch.tensor([0])
        else:
            x1 = torch.randint(0, width - hole_size, (1,))
        y2 = y1 + hole_size
        x2 = x1 + hole_size
        masks[:, :, y1:y2, x1:x2] = 1.0
    img_batch = (1.0 - masks) * img_batch + masks * fill_value.view(B, 3, 1, 1)
    return img_batch


def cutout_fixed_num_holes(x, factor, num_holes=8, image_shape=(84, 84)):
    height, width = image_shape
    min_size = min(height, width)
    hole_size = max(int(min_size * 0.2), 1)
    return cutout(x, num_holes=num_holes, hole_size=hole_size, fill_value=factor)


class SemanticAugment(nn.Module):
    def __init__(self, batch_size, op_tuples, op_label):
        super(SemanticAugment, self).__init__()
        self.ops = [op[0] for op in op_tuples]
        self.op_label = op_label
        params = []
        for tup in op_tuples:
            min_val = tup[1][0]
            max_val = tup[1][1]
            num = tup[2]
            init_val = torch.rand(batch_size, num) * (max_val - min_val) + min_val
            init_val = init_val.squeeze(1)
            params.append(torch.nn.Parameter(init_val))
        self.params = nn.ParameterList(params)

    def forward(self, x):
        for i, op in enumerate(self.ops):
            x = torch.clamp(op(x, self.params[i]), 0, 1)
        return x


class Counter:
    def __init__(self):
        self.v = 0
        self.c = 0

    def add(self, x):
        self.v += x
        self.c += 1

    def avg(self):
        if self.c == 0:
            return 0
        return self.v / self.c


class SemanticPerturbation:
    semantics_list = np.array(
        [
            (hsv_aug, (-1, 1), 3),
            (rotate_aug, (0.01, 1), 1),
            (translate_aug, (-1, 1), 2),
            (invert_aug, (0.5, 1), 1),
            (shear_aug, (-0.3, 0.3), 2),
            (contrast_aug, (0.1, 1.9), 1),
            (sharpness_aug, (0, 1), 1),
            (solarize_aug, (-0.5, 0.5), 1),
            (scale_aug, (0.5, 2.0), 1),
            (equalize_aug, (0, 0), 1),
            (posterize_aug, (0, 0), 1),
            (cutout_fixed_num_holes, (0, 1), 3),
        ],
        dtype=object,
    )

    def __init__(self, sel_index=None, max_len=3):
        if sel_index is None:
            self.semantic_aug_list = self.semantics_list
        else:
            self.semantic_aug_list = self.semantics_list[sel_index]
        num = len(self.semantic_aug_list)
        aug_comb = [[c] for c in range(num)]
        num_arr = [num]
        if self.semantic_aug_list[0][0] == hsv_aug:
            op_set = set(list(range(1, num)))
        else:
            op_set = set(list(range(0, num)))
        prev = aug_comb
        for _ in range(2, max_len + 1):
            curr = []
            for comb in prev:
                curr.extend([comb + [c] for c in list(op_set - set(comb))])
            prev = curr
            aug_comb.extend(curr)
            num_arr.append(len(curr))
        probs = []
        for n in num_arr:
            probs.extend([1 / max_len / n] * n)
        self.probs = np.array(probs)
        self.ops = aug_comb

    def sample(self, batch_size):
        op_label = np.random.choice(len(self.ops), p=self.probs / self.probs.sum())
        ops = self.ops[op_label]
        op_tuples = [self.semantic_aug_list[o] for o in ops]
        return SemanticAugment(batch_size, op_tuples, op_label)


class ModelBaseline(object):
    def __init__(self, flags):
        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)

    def get_images(
            self, images, labels, save_path, shuffle=False, sel_indexes=None, nsamples=10
    ):
        class_dict = {}
        for i, l in enumerate(labels):
            if class_dict.get(l, None) is None:
                class_dict[l] = [images[i]]
            else:
                class_dict[l].append(images[i])
        num_classes = len(class_dict)
        total_num_per_class = np.array([len(class_dict[i]) for i in class_dict]).min()
        nsamples = min(nsamples, total_num_per_class)
        indexes = list(range(total_num_per_class))
        if shuffle:
            random.shuffle(indexes)
        if sel_indexes is None:
            sel_indexes = np.array(indexes[0:nsamples])
        else:
            assert nsamples >= len(sel_indexes), "sel_indexes too long"
        data_matrix = []
        keys = sorted(list(class_dict.keys()))
        for c in keys:
            data_matrix.append(np.array(class_dict[c])[sel_indexes])
        data_matrix = np.concatenate(data_matrix, axis=0)
        self.vis_image(data_matrix, nsamples, save_path)
        return sel_indexes

    def vis_image(self, data, max_per_row=10, save_path="./"):
        num = len(data)
        nrow = int(np.ceil(num / max_per_row))

        fig, ax = plt.subplots(figsize=(max_per_row, nrow))
        demo = []
        for i in range(len(data)):
            demo.append(torch.tensor(data[i]))
        demo = torch.stack(demo)

        grid_img = torchvision.utils.make_grid(demo, nrow=max_per_row)
        grid_img = grid_img.permute(1, 2, 0).detach().cpu().numpy()
        ax.imshow(grid_img, interpolation="nearest")
        ax.axis("off")
        fig.savefig(save_path)
        plt.close(fig)

    def spectral_init(self, model):
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                scale = np.sqrt(2 / (layer.weight.size(0) + layer.weight.size(1)))
                layer.weight.data *= scale
        return model

    def setup(self, flags):
        torch.backends.cudnn.deterministic = flags.deterministic
        print("torch.backends.cudnn.deterministic:", torch.backends.cudnn.deterministic)
        fix_all_seed(flags.seed)

        if flags.model == "resnet18":
            self.network = resnet18(
                pretrained=True,
                num_classes=flags.num_classes,
                contrastive=flags.train_mode,
            ).cuda()
            self.network = self.spectral_init(self.network)

        if flags.model == "resnet34":
            self.network = resnet34(
                pretrained=True,
                num_classes=flags.num_classes,
                contrastive=flags.train_mode,
            ).cuda()
            self.network = self.spectral_init(self.network)

        if flags.model == "resnet50":
            self.network = resnet50(
                pretrained=True,
                num_classes=flags.num_classes,
                contrastive=flags.train_mode,
            ).cuda()
            self.network = self.spectral_init(self.network)

        if flags.model == "resnet101":
            self.network = resnet101(
                pretrained=True,
                num_classes=flags.num_classes,
                contrastive=flags.train_mode,
            ).cuda()
            self.network = self.spectral_init(self.network)

        if flags.model == "resnet152":
            self.network = resnet152(
                pretrained=True,
                num_classes=flags.num_classes,
                contrastive=flags.train_mode,
            ).cuda()
            self.network = self.spectral_init(self.network)

        if flags.model == "alexnet":
            self.network = alexnet(
                pretrained=True,
                num_classes=flags.num_classes,
                contrastive=flags.train_mode,
            ).cuda()
            self.network = self.spectral_init(self.network)

        if flags.model == "resnext29":
            self.network = resnext29(
                num_classes=flags.num_classes,
                contrastive=flags.train_mode,
            ).cuda()
            self.network = self.spectral_init(self.network)

        if flags.model == "WideResNet":
            self.network = WideResNet(
                depth=28,
                num_classes=flags.num_classes,
                contrastive=flags.train_mode,
            ).cuda()
            self.network = self.spectral_init(self.network)

        print("flags:", flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)
        flag_str = (
                "--------Parameters--------\n"
                + "\n".join(["{}={}".format(k, flags.__dict__[k]) for k in flags.__dict__])
                + "\n--------------------"
        )
        print("flags:", flag_str)
        flags_log = os.path.join(flags.logs, "flags_log.txt")
        write_log(flag_str, flags_log)

    def setup_path(self, flags):
        if flags.dataset == "PACS":
            root_folder = PACS_DATA_FOLDER
            dataset_names = ["art_painting", "cartoon", "photo", "sketch"]

            seen_index = flags.seen_index
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(224, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            if not os.path.exists(flags.logs):
                os.makedirs(flags.logs)

            if type(seen_index) == list:
                names = [dataset_names[i] for i in seen_index]
                self.train_name = "+".join(names)
                self.train_dataset = PACSMultiple(root_folder, names, "train")
                self.val_dataset = PACSMultiple(root_folder, names, "val")
                self.test_loaders = []
                for index, name in enumerate(dataset_names):
                    if index not in seen_index:
                        dataset = PACS(root_folder, name, "test")
                        loader = DataLoader(
                            dataset,
                            batch_size=flags.batch_size,
                            shuffle=False,
                            num_workers=flags.num_workers,
                            pin_memory=False,
                        )
                        self.test_loaders.append((name, loader))

            else:
                self.train_dataset = PACS(
                    root_folder, dataset_names[seen_index], "train", ratio=flags.ratio
                )
                if flags.algorithm == "ERM":
                    self.train_dataset.transform = self.train_transform
                self.val_dataset = PACS(root_folder, dataset_names[seen_index], "val")
                self.test_loaders = []
                self.train_name = dataset_names[seen_index]
                for index, name in enumerate(dataset_names):
                    if index != seen_index:
                        dataset = PACS(root_folder, name, "test")
                        loader = DataLoader(
                            dataset,
                            batch_size=flags.batch_size,
                            shuffle=False,
                            num_workers=flags.num_workers,
                            pin_memory=False,
                        )

                        self.test_loaders.append((name, loader))

        if flags.dataset == "OfficeHome":
            seen_index = flags.seen_index
            dataset_names = ["art","clipart","product","realworld",]
            root_folder = OfficeHome_DATA_FOLDER

            if not os.path.exists(flags.logs):
                os.makedirs(flags.logs)

            if type(seen_index) == list:
                names = [dataset_names[i] for i in seen_index]
                self.train_name = "+".join(names)
                self.train_dataset = OfficeHomeMultiple(names, root_folder=root_folder, split="train")
                self.val_dataset = OfficeHomeMultiple(names, root_folder=root_folder, split="test")
                self.test_loaders = []
                for index, name in enumerate(dataset_names):
                    if index not in seen_index:
                        dataset = OfficeHome(name, split="test")
                        loader = DataLoader(
                            dataset,
                            batch_size=flags.batch_size,
                            shuffle=False,
                            num_workers=flags.num_workers,
                            pin_memory=False,

                        )
                        self.test_loaders.append((name, loader))

            else:

                self.train_name = dataset_names[seen_index]

                self.train_dataset = OfficeHome(dataset_names[seen_index], root_folder=root_folder, split="train",
                                                    ratio=flags.ratio)
                self.val_dataset = OfficeHome(dataset_names[seen_index], root_folder=root_folder, split="test")
                self.test_loaders = []
                for index, name in enumerate(dataset_names):
                    if index != seen_index:
                        dataset = OfficeHome(name, split="test")
                        loader = DataLoader(
                            dataset,
                            batch_size=flags.batch_size,
                            num_workers=flags.num_workers,
                            shuffle=False,
                        )
                        self.test_loaders.append((name, loader))

        if flags.dataset == "VLCS":
            seen_index = flags.seen_index
            dataset_names = [
                "caltech101",
                "labelme",
                "sun09",
                "voc2007",
            ]
            root_folder = VLCS_DATA_FOLDER

            if not os.path.exists(flags.logs):
                os.makedirs(flags.logs)

            if type(seen_index) == list:
                names = [dataset_names[i] for i in seen_index]
                self.train_name = "+".join(names)
                self.train_dataset = VLCSMultiple(names, root_folder=root_folder, split="train")
                self.val_dataset = VLCSMultiple(names, root_folder=root_folder, split="test")
                self.test_loaders = []
                for index, name in enumerate(dataset_names):
                    if index not in seen_index:
                        dataset = VLCS(name, split="test")
                        loader = DataLoader(
                            dataset,
                            batch_size=flags.batch_size,
                            shuffle=False,
                            num_workers=flags.num_workers,
                            pin_memory=False,

                        )
                        self.test_loaders.append((name, loader))

            else:
                self.train_name = dataset_names[seen_index]
                self.train_dataset = VLCS(
                    dataset_names[seen_index], root_folder=root_folder, split="train"
                )
                self.val_dataset = VLCS(
                    dataset_names[seen_index], root_folder=root_folder, split="test"
                )

                self.test_loaders = []
                for index, name in enumerate(dataset_names):
                    if index != seen_index:
                        dataset = VLCS(name, split="test")
                        loader = DataLoader(
                            dataset,
                            batch_size=flags.batch_size,
                            num_workers=flags.num_workers,
                            shuffle=False,
                        )
                        self.test_loaders.append((name, loader))

        if flags.dataset == "TerraIncognita":
            seen_index = flags.seen_index
            dataset_names = [
                "location_38",
                "location_43",
                "location_46",
                "location_100",
            ]
            root_folder = TerraIncognita_DATA_FOLDER

            if not os.path.exists(flags.logs):
                os.makedirs(flags.logs)

            if type(seen_index) == list:
                names = [dataset_names[i] for i in seen_index]
                self.train_name = "+".join(names)
                self.train_dataset = TerraIncognitaMultiple(names, root_folder=root_folder, split="train")
                self.val_dataset = TerraIncognitaMultiple(names, root_folder=root_folder, split="test")
                self.test_loaders = []
                for index, name in enumerate(dataset_names):
                    if index not in seen_index:
                        ood_dataset = TerraIncognita(name, split="test")
                        ood_loader = DataLoader(
                            ood_dataset,
                            batch_size=flags.batch_size,
                            shuffle=False,
                            num_workers=flags.num_workers,
                            pin_memory=False,

                        )
                        self.test_loaders.append((name, ood_loader))

            else:

                self.train_name = dataset_names[seen_index]

                self.train_dataset = TerraIncognita(dataset_names[seen_index], root_folder=root_folder,
                                                        split="train", ratio=flags.ratio)
                self.val_dataset = TerraIncognita(dataset_names[seen_index], root_folder=root_folder, split="test")
                self.test_loaders = []
                for index, name in enumerate(dataset_names):
                    if index != seen_index:
                        ood_dataset = TerraIncognita(name, split="test")
                        ood_loader = DataLoader(
                            ood_dataset,
                            batch_size=flags.batch_size,
                            num_workers=flags.num_workers,
                            shuffle=False,
                        )
                        self.test_loaders.append((name, ood_loader))

        if flags.dataset == "DomainNet":
            seen_index = flags.seen_index
            dataset_names = [
                "Real",
                "Infograph",
                "Clipart",
                "Painting",
                "Quickdraw",
                "Sketch",
            ]
            root_folder = DomainNet_DATA_FOLDER
            if not os.path.exists(flags.logs):
                os.makedirs(flags.logs)

            if type(seen_index) == list:
                names = [dataset_names[i] for i in seen_index]
                self.train_name = "+".join(names)
                self.train_dataset = DomainNetMultiple(names, root_folder=root_folder, split="train")
                self.val_dataset = DomainNetMultiple(names, root_folder=root_folder, split="test")
                self.test_loaders = []
                for index, name in enumerate(dataset_names):
                    if index not in seen_index:
                        dataset = DomainNet(name, split="test")
                        loader = DataLoader(
                            dataset,
                            batch_size=flags.batch_size,
                            shuffle=False,
                            num_workers=flags.num_workers,
                            pin_memory=False,

                        )
                        self.test_loaders.append((name, loader))

            else:
                self.train_name = dataset_names[seen_index]
                self.train_dataset = DomainNet(
                    dataset_names[seen_index], root_folder=root_folder, split="train"
                )
                self.val_dataset = DomainNet(
                    dataset_names[seen_index], root_folder=root_folder, split="test"
                )

                self.test_loaders = []
                for index, name in enumerate(dataset_names):
                    if index != seen_index:
                        ood_dataset = DomainNet(name, split="test")
                        if flags.loops_min == -1:
                            ood_loader = DataLoader(
                                ood_dataset,
                                batch_size=flags.batch_size,
                                num_workers=flags.num_workers,
                                shuffle=False,
                            )
                        else:
                            ood_loader = DataLoader(
                                ood_dataset,
                                batch_size=flags.batch_size,
                                num_workers=flags.num_workers,
                                shuffle=False,
                                sampler=RandomSampler(ood_dataset, True, 2000),
                            )
                        self.test_loaders.append((name, ood_loader))

        if flags.dataset == "Fundus":
            seen_index = flags.seen_index
            dataset_names = [
                "aptos",
                "deepdr",
                "fgadr",
                "idrid",
                "messidor",
                "rldr",
                "ddr",
                "eyepacs",
            ]
            root_folder = Fundus_DATA_FOLDER
            if not os.path.exists(flags.logs):
                os.makedirs(flags.logs)

            if type(seen_index) == list:
                dataset_names = [
                    "aptos",
                    "deepdr",
                    "fgadr",
                    "idrid",
                    "messidor",
                    "rldr",
                ]
                names = [dataset_names[i] for i in seen_index]
                self.train_name = "+".join(names)
                self.train_dataset = FundusMultiple(names, root_folder=root_folder, split="train")
                self.val_dataset = FundusMultiple(names, root_folder=root_folder, split="test")
                self.test_loaders = []
                for index, name in enumerate(dataset_names):
                    if index not in seen_index:
                        ood_dataset = Fundus(name, split="test")
                        ood_loader = DataLoader(
                            ood_dataset,
                            batch_size=flags.batch_size,
                            shuffle=False,
                            num_workers=flags.num_workers,
                            pin_memory=False,

                        )
                        self.test_loaders.append((name, ood_loader))

            else:

                self.train_name = dataset_names[seen_index]

                self.train_dataset = Fundus(dataset_names[seen_index], root_folder=root_folder, split="train",
                                                ratio=flags.ratio)
                self.val_dataset = Fundus(dataset_names[seen_index], root_folder=root_folder, split="test")
                self.test_loaders = []
                for index, name in enumerate(dataset_names):
                    if index != seen_index:
                        ood_dataset = Fundus(name, split="test")
                        ood_loader = DataLoader(
                            ood_dataset,
                            batch_size=flags.batch_size,
                            num_workers=flags.num_workers,
                            shuffle=False,
                        )
                        self.test_loaders.append((name, ood_loader))




        if flags.dataset == "DomainNet":
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=flags.batch_size,
                shuffle=False,
                num_workers=flags.num_workers,
                pin_memory=False,
                sampler=RandomSampler(
                    self.train_dataset, True, flags.loops_min * flags.batch_size * flags.k
                ),
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=flags.batch_size,
                shuffle=True,
                num_workers=flags.num_workers,
                pin_memory=False,
            )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=flags.batch_size,
            shuffle=False,
            num_workers=flags.num_workers,
            pin_memory=False,
        )


    def configure(self, flags):
        for name, param in self.network.named_parameters():
            print(name, param.size())


        self.optimizerSGD = torch.optim.SGD(
            self.network.parameters(),
            lr=flags.lr,
            weight_decay=flags.weight_decay,
            momentum=flags.momentum,
            nesterov=True,
        )

        # Adam
        # self.optimizer = torch.optim.Adam(
        #     self.network.parameters(),
        #     lr=flags.lr,
        #     weight_decay=flags.weight_decay,
        # )

        # AdamW
        # self.optimizer = torch.optim.AdamW(
        #     self.network.parameters(),
        #     lr=flags.lr,
        #     weight_decay=flags.weight_decay,
        # )

        # RMSprop
        # self.optimizer = torch.optim.RMSprop(
        #     self.network.parameters(),
        #     lr=flags.lr,
        #     weight_decay=flags.weight_decay,
        # )

        # LEAwareSGD
        self.optimizer = LEAwareSGD(
            self.network.parameters(),
            lr=flags.lr,
            momentum=flags.momentum,
            le_weight=flags.le_weight,
            weight_decay=flags.weight_decay
        )


        # self.scheduler = lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, flags.train_epochs * len(self.train_loader)
        # )
        self.scheduler = lr_scheduler.MultiStepLR(
            optimizer=self.optimizer, milestones=[30], gamma=0.1
        )
        # self.scheduler = lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode="max", factor=0.1, patience=5, verbose=True
        # )

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_per_ele = torch.nn.CrossEntropyLoss(reduction="none")


    def save_model(self, file_name, flags):
        if not os.path.exists(flags.model_path):
            os.makedirs(flags.model_path)
        outfile = os.path.join(flags.model_path, file_name)
        torch.save({"state": self.network.state_dict(), "args": flags}, outfile)


    def train(self, flags):
        best_val_acc = 0
        best_test_acc = 0
        best_val_auc = 0
        best_test_auc = 0

        flags_log = os.path.join(flags.logs, "loss_log.txt")
        if not os.path.exists(flags.model_path):
            os.makedirs(flags.model_path)

        for epoch in range(flags.train_epochs):
            loss_avger = Averager()
            self.network.train()
            bn_eval(self.network)

            for ite, (images_train, labels_train, _) in tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    leave=False,
                    desc="train-epoch:{}".format(epoch + 1),
            ):
                inputs, labels = images_train.cuda(), labels_train.cuda()
                outputs, _ = self.network(x=inputs)
                loss = self.loss_fn(outputs, labels)
                self.optimizerSGD.zero_grad()
                loss.backward()
                self.optimizerSGD.step()
                loss_avger.add(loss.item())
            self.scheduler.step()

            val_auc, val_acc, val_f1 = self.batch_test(self.val_loader)
            auc_arr, acc_arr, f1_arr = self.batch_test_workflow()
            mean_auc = np.mean(auc_arr)
            mean_acc = np.mean(acc_arr)
            mean_f1 = np.mean(f1_arr)
            names = [n for n, _ in self.test_loaders]
            if flags.dataset == "Fundus":
                res = (
                        "\n{} ".format(self.train_name)
                        + " ".join(["{}:{:.2f}".format(n, a * 100) for n, a in zip(names, auc_arr)])
                        + "   |  AUC:{:.2f} ACC:{:.2f} F1:{:.2f}".format(mean_auc * 100, mean_acc * 100, mean_f1 * 100)
                )
            else:
                res = (
                        "\n{} ".format(self.train_name)
                        + " ".join(["{}:{:.2f}".format(n, a * 100) for n, a in zip(names, acc_arr)])
                        + "   |  ACC:{:.2f} AUC:{:.2f} F1:{:.2f}".format(mean_acc * 100, mean_auc * 100, mean_f1 * 100)
                )
            msg = "[{}] train_loss:{:.2f} lr:{:.6f} val_auc:{:.2f} val_acc:{:.2f} val_f1:{:.2f}".format(
                epoch,
                loss_avger.item(),
                self.scheduler.get_last_lr()[0],
                val_auc * 100,
                val_acc * 100,
                val_f1 * 100
            )
            if flags.dataset == "Fundus":
                if best_test_auc < mean_auc:
                    best_test_auc = mean_auc
                    self.save_model("best_test_model.tar", flags)
                if best_val_auc < val_auc:
                    best_val_auc = val_auc
                    msg += " (best_AUC)"
                    self.save_model("best_model.tar", flags)
            else:
                if best_test_acc < mean_acc:
                    best_test_acc = mean_acc
                    self.save_model("best_test_model.tar", flags)
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    msg += " (best_ACC)"
                    self.save_model("best_model.tar", flags)

            msg += res
            print(msg)
            write_log(msg, flags_log)


    def batch_test(self, ood_loader):
        self.network.eval()

        all_probs = []
        all_labels = []
        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for images_test, labels_test, _ in tqdm(ood_loader, leave=False, desc="testing"):
                images_test, labels_test = images_test.cuda(), labels_test.cuda()

                out, end_points = self.network(images_test)
                probs = softmax(out).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels_test.cpu().numpy())
                assert np.allclose(probs, end_points['Predictions'].cpu().numpy(), atol=1e-6)
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        try:
            auc = roc_auc_score(labels, probs, average='macro', multi_class='ovo')
        except ValueError:
            auc = 0.0
        return auc, acc, f1

    def batch_test_workflow(self):
        aucs = []
        accs = []
        f1s = []
        with torch.no_grad():
            for name, test_loader in self.test_loaders:
                auc, acc, f1 = self.batch_test(test_loader)
                aucs.append(auc)
                accs.append(acc)
                f1s.append(f1)
        return aucs, accs, f1s


class ModelADA(ModelBaseline):
    def __init__(self, flags):
        super(ModelADA, self).__init__(flags)

    def configure(self, flags):
        super(ModelADA, self).configure(flags)
        self.dist_fn = torch.nn.MSELoss()

    def setup_path(self, flags):
        super(ModelADA, self).setup_path(flags)
        self.image_denormalise = Denormalise(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        self.image_transform = transforms.ToPILImage()

    def maximize(self, flags):
        self.network.eval()
        images, labels = [], []
        self.train_dataset.transform = self.train_dataset.preprocess
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=flags.batch_size,
            shuffle=False,
            num_workers=flags.num_workers,
        )

        for i, (images_train, labels_train, _) in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                leave=False,
                desc="Maximum",
        ):
            inputs, targets = images_train.cuda(), labels_train.cuda()
            out, tuples = self.network(x=inputs)
            inputs_embedding = tuples["Embedding"].detach().clone()
            inputs_embedding.requires_grad_(False)

            inputs_max = inputs.detach().clone()
            inputs_max.requires_grad_(True)
            optimizer = sgd(parameters=[inputs_max], lr=flags.lr_max)

            for ite_max in range(flags.loops_adv):
                out, tuples = self.network(x=inputs_max)
                loss = self.loss_fn(out, targets)
                semantic_dist = self.dist_fn(tuples["Embedding"], inputs_embedding)
                loss = (
                        loss - flags.gamma * semantic_dist + flags.eta * entropy_loss(out)
                )

                self.network.zero_grad()
                optimizer.zero_grad()
                (-loss).backward()
                optimizer.step()

            inputs_max = inputs_max.detach().clone().cpu()
            for j in range(len(inputs_max)):
                input_max = self.image_denormalise(inputs_max[j])
                input_max = self.image_transform(input_max.clamp(min=0.0, max=1.0))
                images.append(input_max)
                labels.append(labels_train[j].item())
        images = np.stack(images, 0)
        labels = np.array(labels)
        images = torch.tensor(images).permute(0, 3, 1, 2)
        labels = torch.tensor(labels)
        return images, labels

    def train(self, flags):
        counter_k = 0
        counter_ite = 0
        best_val_acc = 0
        best_test_acc = 0
        best_val_auc = 0
        best_test_auc = 0
        flags_log = os.path.join(flags.logs, "loss_log.txt")

        for epoch in range(0, flags.train_epochs):
            loss_avger = Averager()

            self.network.train()
            bn_eval(self.network)
            self.train_dataset.transform = self.train_transform
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=flags.batch_size,
                shuffle=True,
                num_workers=flags.num_workers,
                pin_memory=True,
            )

            self.scheduler.T_max = counter_ite + len(self.train_loader) * (
                    flags.train_epochs - epoch
            )
            for ite, (images_train, labels_train, _) in tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    leave=False,
                    desc="train-epoch:{}".format(epoch + 1),
            ):
                counter_ite += 1
                inputs, labels = images_train.cuda(), labels_train.cuda()
                outputs, _ = self.network(x=inputs)
                cls_loss = self.loss_fn(outputs, labels)
                loss = cls_loss - flags.eta_min * entropy_loss(outputs)
                self.optimizerSGD.zero_grad()
                loss.backward()
                self.optimizerSGD.step()
                loss_avger.add(loss.item())
            self.scheduler.step()

            val_auc, val_acc, val_f1 = self.batch_test(self.val_loader)
            auc_arr, acc_arr, f1_arr = self.batch_test_workflow()
            mean_auc = np.mean(auc_arr)
            mean_acc = np.mean(acc_arr)
            mean_f1 = np.mean(f1_arr)
            names = [n for n, _ in self.test_loaders]
            if flags.dataset == "Fundus":
                res = (
                        "\n{} ".format(self.train_name)
                        + " ".join(["{}:{:.2f}".format(n, a * 100) for n, a in zip(names, auc_arr)])
                        + "   |  AUC:{:.2f} ACC:{:.2f} F1:{:.2f}".format(mean_auc * 100, mean_acc * 100, mean_f1 * 100)
                )
            else:
                res = (
                        "\n{} ".format(self.train_name)
                        + " ".join(["{}:{:.2f}".format(n, a * 100) for n, a in zip(names, acc_arr)])
                        + "   |  ACC:{:.2f} AUC:{:.2f} F1:{:.2f}".format(mean_acc * 100, mean_auc * 100, mean_f1 * 100)
                )
            msg = "[{}] train_loss:{:.2f} lr:{:.6f} val_auc:{:.2f} val_acc:{:.2f} val_f1:{:.2f}".format(
                epoch,
                loss_avger.item(),
                self.scheduler.get_last_lr()[0],
                val_auc * 100,
                val_acc * 100,
                val_f1 * 100
            )
            if flags.dataset == "Fundus":
                if best_test_auc < mean_auc:
                    best_test_auc = mean_auc
                    self.save_model("best_test_model.tar", flags)
                if best_val_auc < val_auc:
                    best_val_auc = val_auc
                    msg += " (best_AUC)"
                    self.save_model("best_model.tar", flags)
            else:
                if best_test_acc < mean_acc:
                    best_test_acc = mean_acc
                    self.save_model("best_test_model.tar", flags)
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    msg += " (best_ACC)"
                    self.save_model("best_model.tar", flags)

            msg += res
            print(msg)
            write_log(msg, flags_log)


            if ((epoch) % flags.gen_freq == 0) and (
                    counter_k < flags.k
            ):  # if T_min iterations are passed
                print("Generating adversarial images [iter {}]".format(counter_k))
                images, labels = self.maximize(flags)
                self.train_dataset.x = torch.cat([self.train_dataset.x, images], dim=0)
                self.train_dataset.y = torch.cat([self.train_dataset.y, labels], dim=0)
                self.train_dataset.op_labels = torch.cat(
                    [self.train_dataset.op_labels, torch.ones_like(labels) * (-1)],
                    dim=0,
                )
                counter_k += 1


class ModelLEAware(ModelBaseline):
    def __init__(self, flags):
        super(ModelLEAware, self).__init__(flags)

    def configure(self, flags):
        super(ModelLEAware, self).configure(flags)
        self.dist_fn = torch.nn.MSELoss()
        self.conloss = SupConLoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.image_transform = transforms.ToPILImage()
        self.image_denormalise = Denormalise(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

        if getattr(flags, "loo", None) is None:
            sel_index = None
        else:
            num_ops = len(SemanticPerturbation.semantics_list)
            sel_index = np.array([i for i in range(num_ops) if i != flags.loo])
        self.semantic_config = SemanticPerturbation(sel_index=sel_index)
        self.scheduler = lr_scheduler.MultiStepLR(
            optimizer=self.optimizer, milestones=[30], gamma=0.1
        )


    def save_model(self, file_name, flags):
        outfile = os.path.join(flags.model_path, file_name)

        aug_probs = [
            (
                "-".join(
                    [
                        self.semantic_config.semantic_aug_list[o][0].__name__
                        for o in self.semantic_config.ops[i]
                    ]
                ),
                self.semantic_config.probs[i],
            )
            for i in range(len(self.semantic_config.ops))
        ]
        torch.save(
            {"state": self.network.state_dict(), "args": flags, "aug_probs": aug_probs},
            outfile,
        )

    def load_model(self, flags):
        print("Load model from ", flags.chkpt_path)
        model_dict = torch.load(flags.chkpt_path)
        prob_tuples = model_dict["aug_probs"]
        self.semantic_config.probs = np.array([t[1] for t in prob_tuples])
        self.network.load_state_dict(model_dict["state"])

    def maximize(self, flags):
        self.network.eval()
        images, labels, op_labels = [], [], []
        self.train_dataset.transform = self.train_dataset.preprocess
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=flags.batch_size,
            num_workers=flags.num_workers,
            shuffle=True,
        )
        mean = self.mean.cuda()
        std = self.std.cuda()
        with tqdm(train_loader, leave=False, total=len(train_loader)) as pbar:
            for ite, (images_train, labels_train, _) in enumerate(pbar):
                inputs, targets = images_train.cuda(), labels_train.cuda()
                out, tuples = self.network(x=inputs)
                inputs_embedding = tuples["Embedding"].data.clone()
                inputs_embedding.requires_grad_(False)

                batch_size = len(inputs)
                semantic_perturb = self.semantic_config.sample(batch_size).to(
                    inputs.device
                )
                op_labels.append(np.array([semantic_perturb.op_label] * batch_size))

                diff_loss = 1.0
                prev_loss = 1000
                iter_count = 0

                optimizer = torch.optim.RMSprop(
                    semantic_perturb.parameters(), flags.lr_max
                )
                ori_inputs = (
                        inputs * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
                ).data

                while diff_loss > 0.1 and iter_count < flags.loops_adv:
                    inputs_max = semantic_perturb(ori_inputs.data)
                    inputs_max = (inputs_max - mean.view(1, 3, 1, 1)) / std.view(
                        1, 3, 1, 1
                    )

                    out, tuples = self.network(x=inputs_max)
                    cls_loss = self.loss_fn(out, targets)
                    semantic_loss = self.dist_fn(tuples["Embedding"], inputs_embedding)

                    loss = cls_loss - flags.gamma * semantic_loss + flags.eta * entropy_loss(out)

                    optimizer.zero_grad()
                    (-loss).backward()

                    optimizer.step()

                    diff_loss = abs((loss - prev_loss).item())
                    prev_loss = loss.item()
                    iter_count += 1

                    pbar.set_postfix(
                        {
                            "loss": "{:.4f}".format(loss.item()),
                            "dist": "{:.6f}".format(semantic_loss.item()),
                        }
                    )

                inputs_max = semantic_perturb(ori_inputs.data)
                inputs_max = (inputs_max - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

                images.append(inputs_max.detach().clone().cpu())
                labels.append(targets.cpu())

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)

        op_labels = torch.tensor(np.concatenate(op_labels))
        return images, labels, op_labels

    def train(self, flags):
        counter_k = 0
        best_val_acc = 0
        best_test_acc = 0
        best_val_auc = 0
        best_test_auc = 0
        current_le = 0

        flags_log = os.path.join(flags.logs, "loss_log.txt")
        if not os.path.exists(flags.model_path):
            os.makedirs(flags.model_path)

        train_dataset = deepcopy(self.train_dataset)
        data_pool = DataPool(flags.k + 1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=flags.batch_size,
            num_workers=flags.num_workers,
            shuffle=True
        )

        epsilon = 1e-8
        for epoch in range(1, flags.train_epochs + 1):
            loss_avger = Averager()
            cls_loss_avger = Averager()
            con_loss_avger = Averager()
            entloss_avger = Averager()

            initial_weights = np.concatenate([p.data.cpu().numpy().flatten() for p in self.network.parameters()])
            trajectory = [initial_weights]
            lyap_sum = 0
            pre_le = 0
            ite=0

            for ite, (images_train, labels_train, op_labels) in tqdm(
                    enumerate(train_loader, start=1),
                    total=len(train_loader),
                    leave=False,
                    desc="train-epoch{}".format(epoch),
            ):
                self.network.train()
                inputs, labels = images_train.cuda(), labels_train.cuda()
                img_shape = inputs.shape[-3:]
                outputs, tuples = self.network(x=inputs.reshape(-1, *img_shape))
                cls_loss_ele = self.loss_per_ele(outputs, labels.reshape(-1))
                cls_loss = cls_loss_ele.mean()
                cls_loss_avger.add(cls_loss.item())
                ent_loss = entropy_loss(outputs)
                entloss_avger.add(ent_loss.item())

                if flags.train_mode == "contrastive":
                    projs = tuples["Projection"]
                    projs = projs.reshape(inputs.shape[0], -1, projs.shape[-1])

                    con_loss = self.conloss(projs, labels)
                    loss = (
                            cls_loss
                            + flags.beta * con_loss
                            - flags.eta_min * ent_loss
                    )
                    con_loss_avger.add(con_loss.item())
                else:
                    loss = cls_loss - flags.eta_min * ent_loss

                self.optimizer.zero_grad()
                loss.backward()

                if ite > 1:
                    current_le = lyap_sum / ite
                    self.optimizer.step(current_le=current_le, pre_le=pre_le)
                else:
                    self.optimizer.step()
                loss_avger.add(loss.item())
                pre_le = current_le
                current_weights = np.concatenate([p.data.cpu().numpy().flatten() for p in self.network.parameters()])

                # Calculate weight changes
                trajectory.append(current_weights)
                if ite > 1:
                    delta = np.linalg.norm(current_weights - trajectory[-2]) / np.linalg.norm(
                        trajectory[1] - trajectory[0])
                    # Updated Lyapunov index estimates
                    lyap_sum += np.log(delta + epsilon)
            lyap_exp = lyap_sum / ite
            self.scheduler.step()

            val_auc, val_acc, val_f1 = self.batch_test(self.val_loader)
            auc_arr, acc_arr, f1_arr = self.batch_test_workflow()
            mean_auc = np.mean(auc_arr)
            mean_acc = np.mean(acc_arr)
            mean_f1 = np.mean(f1_arr)
            names = [n for n, _ in self.test_loaders]
            if flags.dataset == "Fundus":
                res = (
                        "\n{} ".format(self.train_name)
                        + " ".join(["{}:{:.2f}".format(n, a * 100) for n, a in zip(names, auc_arr)])
                        + "   |  AUC:{:.2f} ACC:{:.2f} F1:{:.2f}".format(mean_auc * 100, mean_acc * 100, mean_f1 * 100)
                )
            else:
                res = (
                        "\n{} ".format(self.train_name)
                        + " ".join(["{}:{:.2f}".format(n, a * 100) for n, a in zip(names, acc_arr)])
                        + "   |  ACC:{:.2f} AUC:{:.2f} F1:{:.2f}".format(mean_acc * 100, mean_auc * 100, mean_f1 * 100)
                )

            msg = "[{}] train_loss:{:.2f} lr:{:.6f} lyap_exp:{:.6f} val_auc:{:.2f} val_acc:{:.2f} val_f1:{:.2f}".format(
                epoch,
                loss_avger.item(),
                self.scheduler.get_last_lr()[0],
                lyap_exp,
                val_auc * 100,
                val_acc * 100,
                val_f1 * 100
            )
            if flags.dataset == "Fundus":
                if best_test_auc < mean_auc:
                    best_test_auc = mean_auc
                    self.save_model("best_test_model.tar", flags)
                if best_val_auc < val_auc:
                    best_val_auc = val_auc
                    msg += " (best_AUC)"
                    self.save_model("best_model.tar", flags)
            else:
                if best_test_acc < mean_acc:
                    best_test_acc = mean_acc
                    self.save_model("best_test_model.tar", flags)
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    msg += " (best_ACC)"
                    self.save_model("best_model.tar", flags)

            msg += res
            print(msg)
            write_log(msg, flags_log)

            if (
                    epoch % flags.gen_freq == 0
                    and epoch < flags.train_epochs
                    and counter_k < flags.domain_number
            ):  # if T_min iterations are passed
                print("Semantic image generation [iter {}]".format(counter_k))

                images, labels, op_labels = self.maximize(flags)
                data_pool.add((images, labels, op_labels))

                counter_k += 1
            data_batch = data_pool.get()

            gen_x = torch.cat([p[0] for p in data_batch], 0)
            gen_y = torch.cat([p[1] for p in data_batch], 0)
            gen_op_labels = torch.cat([p[2] for p in data_batch], 0)

            train_dataset.x = gen_x
            train_dataset.y = gen_y
            train_dataset.op_labels = gen_op_labels
            train_loader = DataLoader(
                train_dataset,
                batch_size=flags.batch_size,
                num_workers=flags.num_workers,
                shuffle=True,
            )


