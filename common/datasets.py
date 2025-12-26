from __future__ import print_function, absolute_import, division
import h5py
import os
import numpy as np
from PIL import Image
from common.tools import log, resize_image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from config import (OfficeHome_DATA_FOLDER,VLCS_DATA_FOLDER,
                    TerraIncognita_DATA_FOLDER,DomainNet_DATA_FOLDER,
                    Fundus_DATA_FOLDER)

OfficeHome_DATA_DIR = OfficeHome_DATA_FOLDER
VLCS_DATA_DIR = VLCS_DATA_FOLDER
TerraIncognita_DATA_DIR = TerraIncognita_DATA_FOLDER
DomainNet_DATA_DIR = DomainNet_DATA_FOLDER
Fundus_DATA_DIR = Fundus_DATA_FOLDER

class Denormalise(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-12)
        mean_inv = -mean * std_inv
        super(Denormalise, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(Denormalise, self).__call__(tensor.clone())

def preprocess_dataset(x, train, img_mean_mode):
    # Compute image mean if applicable
    if img_mean_mode is not None:
        if train:

            if img_mean_mode == "per_channel":
                x_ = np.copy(x)
                x_ = x_.astype('float32') / 255.0
                img_mean = np.array([np.mean(x_[:, :, :, 0]), np.mean(x_[:, :, :, 1]), np.mean(x_[:, :, :, 2])])

            elif img_mean_mode == "imagenet":
                img_mean = np.array([0.485, 0.456, 0.406])

            else:
                raise Exception("Invalid img_mean_mode..!")
            np.save("img_mean.npy", img_mean)

    return x
def load_OfficeHome(subset, train=True, data_dir="../../datasets"):
    data_path = os.path.join(os.path.dirname(__file__), data_dir)
    subset = subset.lower()
    if subset == "product":
        labelfile = os.path.join(data_path, "product_train.txt") if train else os.path.join(data_path, "product_test.txt")
    elif subset == "clipart":
        labelfile = os.path.join(data_path, "clipart_train.txt") if train else os.path.join(data_path, "clipart_test.txt")
    elif subset == "art":
        labelfile = os.path.join(data_path, "art_train.txt") if train else os.path.join(data_path, "art_test.txt")
    elif subset == "realworld":
        labelfile = os.path.join(data_path, "realworld_train.txt") if train else os.path.join(data_path, "realworld_test.txt")
    else:
        raise ValueError(f"Unknown subset: {subset}")

    imagepath = []
    labels = []
    with open(labelfile, "r") as f_label:
        for line in f_label:
            temp = line[:-1].split(" ")
            imagepath.append(os.path.join(data_path, temp[0]))
            labels.append(int(temp[1]))
    labels = np.array(labels)
    imagepath = np.array(imagepath)

    return imagepath, labels
def load_VLCS(subset, train=True, data_dir="../../dataset"):
    data_path = os.path.join(os.path.dirname(__file__), data_dir)
    subset = subset.lower()
    if subset == "labelme":
        labelfile = os.path.join(data_path, "labelme_train.txt") if train else os.path.join(data_path, "labelme_test.txt")
    else:
        labelfile = os.path.join(data_path, "{}_train.txt".format(subset)) if train else os.path.join(data_path,"{}_test.txt".format(subset))

    imagepath = []
    labels = []
    with open(labelfile, "r") as f_label:
        for line in f_label:
            temp = line[:-1].split(" ")
            imagepath.append(os.path.join(data_path, temp[0]))
            labels.append(int(temp[1]))
    labels = np.array(labels)
    imagepath = np.array(imagepath)

    return (imagepath, labels)
def load_TerraIncognita(subset, train=True, data_dir="../../datasets"):
    data_path = os.path.join(os.path.dirname(__file__), data_dir)
    subset = subset.lower()
    if subset == "location_38":
        labelfile = os.path.join(data_path, "location_38_train.txt") if train else os.path.join(data_path, "location_38_test.txt")
    elif subset == "location_43":
        labelfile = os.path.join(data_path, "location_43_train.txt") if train else os.path.join(data_path, "location_43_test.txt")
    elif subset == "location_46":
        labelfile = os.path.join(data_path, "location_46_train.txt") if train else os.path.join(data_path, "location_46_test.txt")
    elif subset == "location_100":
        labelfile = os.path.join(data_path, "location_100_train.txt") if train else os.path.join(data_path, "location_100_test.txt")
    else:
        raise ValueError(f"Unknown subset: {subset}")
    imagepath = []
    labels = []
    with open(labelfile, "r") as f_label:
        for line in f_label:
            temp = line[:-1].split(" ")
            imagepath.append(os.path.join(data_path, temp[0]))
            labels.append(int(temp[1]))
    labels = np.array(labels)
    imagepath = np.array(imagepath)

    return imagepath, labels
def load_DomainNet(subset, train=True, data_dir="../../datasets"):
    data_path = os.path.join(os.path.dirname(__file__), data_dir)
    subset = subset.lower()
    if subset == "real":
        labelfile = os.path.join(data_path, "real_train.txt") if train else os.path.join(data_path, "real_test.txt")
    else:
        labelfile = os.path.join(data_path, "{}_train.txt".format(subset)) if train else os.path.join(data_path, "{}_test.txt".format(subset))

    imagepath = []
    labels = []
    with open(labelfile, "r") as f_label:
        for line in f_label:
            temp = line[:-1].split(" ")
            imagepath.append(os.path.join(data_path,temp[0]))
            labels.append(int(temp[1]))
    labels = np.array(labels)
    imagepath = np.array(imagepath)

    return (imagepath, labels)
def load_Fundus(subset, train=True, data_dir="../../datasets"):
    data_path = os.path.join(os.path.dirname(__file__), data_dir)
    subset = subset.lower()

    if subset == "aptos":
        labelfile = os.path.join(data_path, "aptos_train.txt") if train else os.path.join(data_path, "aptos_test.txt")
    elif subset == "deepdr":
        labelfile = os.path.join(data_path, "deepdr_train.txt") if train else os.path.join(data_path, "deepdr_test.txt")
    elif subset == "fgadr":
        labelfile = os.path.join(data_path, "fgadr_train.txt") if train else os.path.join(data_path, "fgadr_test.txt")
    elif subset == "idrid":
        labelfile = os.path.join(data_path, "idrid_train.txt") if train else os.path.join(data_path, "idrid_test.txt")
    elif subset == "messidor":
        labelfile = os.path.join(data_path, "messidor_train.txt") if train else os.path.join(data_path, "messidor_test.txt")
    elif subset == "rldr":
        labelfile = os.path.join(data_path, "rldr_train.txt") if train else os.path.join(data_path, "rldr_test.txt")
    elif subset == "ddr":
        labelfile = os.path.join(data_path, "ddr_train.txt") if train else os.path.join(data_path, "ddr_test.txt")
    elif subset == "eyepacs":
        labelfile = os.path.join(data_path, "eyepacs_train.txt") if train else os.path.join(data_path, "eyepacs_test.txt")
    else:
        raise ValueError(f"Unknown subset: {subset}")

    imagepath = []
    labels = []
    with open(labelfile, "r") as f_label:
        for line in f_label:
            temp = line[:-1].split(" ")
            imagepath.append(os.path.join(data_path, temp[0]))
            labels.append(int(temp[1]))
    labels = np.array(labels)
    imagepath = np.array(imagepath)

    return imagepath, labels

class PACS(Dataset):
    def __init__(self, root_folder, name, split='train', transform=None, ratio=None):
        path = os.path.join(root_folder, '{}_{}.hdf5'.format(name, split))
        if split == 'train':
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transform
        else:
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transform

        f = h5py.File(path, "r")
        self.x = np.array(f['images'])
        self.y = np.array(f['labels'])
        self.op_labels = torch.tensor(np.ones(len(self.y), dtype=int) * (-1))
        if ratio is not None:
            num = len(self.x)
            indexes = np.random.permutation(num)
            sel_num = int(ratio * num)
            self.x = self.x[indexes[0:sel_num]]
            self.y = self.y[indexes[0:sel_num]]
            self.op_labels = self.op_labels[indexes[0:sel_num]]
        f.close()

        def resize(x):
            x = x[:, :,
                [2, 1, 0]]  # we use the pre-read hdf5 data file from the download page and need to change BGR to RGB
            x = x.astype(np.uint8)
            return np.array(Image.fromarray(obj=x, mode='RGB').resize(size=(224, 224)))

        self.x = np.array(list(map(resize, self.x)))
        self.x = torch.tensor(self.x).permute(0, 3, 1, 2)
        self.y -= np.min(self.y)
        self.y = torch.tensor(self.y.astype(np.int64))
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.image_denormalise = Denormalise([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]
        if op < 0:
            x = transforms.ToPILImage()(x)
            x = self.transform(x)
        return x, y, op
class OfficeHome(Dataset):
    def __init__(self, name, root_folder=OfficeHome_DATA_DIR, split='train', transform=None, ratio=None):

        if split == 'train':
            train_mode = True
        else:
            train_mode = False

        results = load_OfficeHome(name, train=train_mode, data_dir=root_folder)

        self.x = results[0]
        self.y = results[1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.op_labels = torch.tensor(np.ones(len(self.y), dtype=int) * (-1))

        if ratio is not None:
            num = len(self.x)
            indexes = np.random.permutation(num)
            sel_num = int(ratio * num)
            self.x = self.x[indexes[0:sel_num]]
            self.y = self.y[indexes[0:sel_num]]
            self.op_labels = self.op_labels[indexes[0:sel_num]]

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]

        if op == -1:
            with Image.open(x) as image:
                image = image.convert('RGB')
                x = self.transform(image)
        else:
            if type(x) == str:
                x, y, op = torch.load(x)

        return x, y, op
class VLCS(Dataset):
    def __init__(self, name, root_folder=VLCS_DATA_DIR, split='train', transform=None, ratio=None):
        if split == 'train':
            train_mode = True
        else:
            train_mode = False

        results = load_VLCS(name, train=train_mode, data_dir=root_folder)

        self.x = results[0]
        self.y = results[1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.op_labels = torch.tensor(np.ones(len(self.y), dtype=int) * (-1))

        if ratio is not None:
            num = len(self.x)
            indexes = np.random.permutation(num)
            sel_num = int(ratio * num)
            self.x = self.x[indexes[0:sel_num]]
            self.y = self.y[indexes[0:sel_num]]
            self.op_labels = self.op_labels[indexes[0:sel_num]]

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]

        if op == -1:
            with Image.open(x) as image:
                image = image.convert('RGB')
                x = self.transform(image)
        else:
            if type(x) == str:
                x, y, op = torch.load(x)
        return x, y, op
class TerraIncognita(Dataset):
    def __init__(self, name, root_folder=TerraIncognita_DATA_DIR, split='train', transform=None, ratio=None):

        if split == 'train':
            train_mode = True
        else:
            train_mode = False


        results = load_TerraIncognita(name, train=train_mode, data_dir=root_folder)

        self.x = results[0]
        self.y = results[1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.op_labels = torch.tensor(np.ones(len(self.y), dtype=int) * (-1))

        if ratio is not None:
            num = len(self.x)
            indexes = np.random.permutation(num)
            sel_num = int(ratio * num)
            self.x = self.x[indexes[0:sel_num]]
            self.y = self.y[indexes[0:sel_num]]
            self.op_labels = self.op_labels[indexes[0:sel_num]]

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]

        if op == -1:
            with Image.open(x) as image:
                image = image.convert('RGB')
                x = self.transform(image)
        else:
            if type(x) == str:
                x, y, op = torch.load(x)

        return x, y, op
class DomainNet(Dataset):
    def __init__(self, name, root_folder=DomainNet_DATA_DIR, split='train', transform=None, ratio=None):
        if split == 'train':
            train_mode = True
        else:
            train_mode = False

        results = load_DomainNet(name, train=train_mode, data_dir=root_folder)

        self.x = results[0]
        self.y = results[1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.op_labels = torch.tensor(np.ones(len(self.y), dtype=int) * (-1))
        if ratio is not None:
            num = len(self.x)
            indexes = np.random.permutation(num)
            sel_num = int(ratio * num)
            self.x = self.x[indexes[0:sel_num]]
            self.y = self.y[indexes[0:sel_num]]
            self.op_labels = self.op_labels[indexes[0:sel_num]]

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]

        if op == -1:
            with Image.open(x) as image:
                image = image.convert('RGB')
                x = self.transform(image)
        else:
            if type(x) == str:
                x, y, op = torch.load(x)
        return x, y, op
class Fundus(Dataset):
    def __init__(self, name, root_folder=Fundus_DATA_DIR, split='train', transform=None, ratio=None):

        if split == 'train':
            train_mode = True
        else:
            train_mode = False


        results = load_Fundus(name, train=train_mode, data_dir=root_folder)

        self.x = results[0]
        self.y = results[1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.op_labels = torch.tensor(np.ones(len(self.y), dtype=int) * (-1))

        if ratio is not None:
            num = len(self.x)
            indexes = np.random.permutation(num)
            sel_num = int(ratio * num)
            self.x = self.x[indexes[0:sel_num]]
            self.y = self.y[indexes[0:sel_num]]
            self.op_labels = self.op_labels[indexes[0:sel_num]]

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]

        if op == -1:
            with Image.open(x) as image:
                image = image.convert('RGB')
                x = self.transform(image)
        else:
            if type(x) == str:
                x, y, op = torch.load(x)

        return x, y, op

class PACSMultiple(Dataset):
    def __init__(self, root_folder, names, split='train', transform=None, ratio=1.0):

        if split == 'train':
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.4, .4, .4, .4),
                    # transforms.RandomCrop(224, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transform
        else:
            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transform

        def resize(x):
            x = x[:, :,
                [2, 1, 0]]  # we use the pre-read hdf5 data file from the download page and need to change BGR to RGB
            x = x.astype(np.uint8)
            return np.array(Image.fromarray(obj=x, mode='RGB').resize(size=(224, 224)))

        self.x = []
        self.y = []
        self.op_labels = []  # cn

        for name in names:
            path = os.path.join(root_folder, '{}_{}.hdf5'.format(name, split))
            f = h5py.File(path, "r")
            x = np.array(f['images'])
            y = np.array(f['labels'])
            f.close()
            x = np.array(list(map(resize, x)))
            y -= np.min(y)
            y = y.astype(np.int64)

            # **应用 `ratio` 采样**
            if split == 'train' and ratio < 1.0:
                num_samples = len(y)
                sample_size = int(num_samples * ratio)  # 计算保留的样本数
                indices = np.random.choice(num_samples, sample_size, replace=False)  # 随机采样索引
                x = x[indices]
                y = y[indices]

            self.x.append(x)
            self.y.append(y)
            self.op_labels.append(torch.tensor(np.ones(len(y), dtype=int) * (-1)))  # cn

        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)
        self.op_labels = torch.cat(self.op_labels)  # cn

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]  # cn
        x = transforms.ToPILImage()(x)
        x = self.transform(x)
        return x, y, op
class OfficeHomeMultiple(Dataset):
    def __init__(self, names,root_folder=OfficeHome_DATA_DIR, split='train', transform=None, ratio=None):
        self.x = []
        self.y = []
        self.op_labels = []
        train_mode = (split == 'train')
        if transform is None:
            if train_mode:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        for name in names:
            results = load_OfficeHome(name, train=train_mode, data_dir=root_folder)
            x, y = results
            op_label = torch.tensor(np.ones(len(y), dtype=int) * (-1))

            if ratio is not None:
                num = len(x)
                indexes = np.random.permutation(num)
                sel_num = int(ratio * num)
                x = x[indexes[:sel_num]]
                y = y[indexes[:sel_num]]
                op_label = op_label[indexes[:sel_num]]

            self.x.append(x)
            self.y.append(y)
            self.op_labels.append(op_label)

        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)
        self.op_labels = torch.cat(self.op_labels)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]
        if op == -1:
            with Image.open(x) as image:
                image = image.convert('RGB')
                x = self.transform(image)
        else:
            if type(x) == str:
                x, y, op = torch.load(x)
        return x, y, op
class VLCSMultiple(Dataset):
    def __init__(self, names,root_folder=VLCS_DATA_DIR, split='train', transform=None, ratio=None):
        self.x = []
        self.y = []
        self.op_labels = []

        train_mode = (split == 'train')

        if transform is None:
            if train_mode:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        for name in names:
            results = load_VLCS(name, train=train_mode, data_dir=root_folder)
            x, y = results
            op_label = torch.tensor(np.ones(len(y), dtype=int) * (-1))
            if ratio is not None:
                num = len(x)
                indexes = np.random.permutation(num)
                sel_num = int(ratio * num)
                x = x[indexes[:sel_num]]
                y = y[indexes[:sel_num]]
                op_label = op_label[indexes[:sel_num]]
            self.x.append(x)
            self.y.append(y)
            self.op_labels.append(op_label)

        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)
        self.op_labels = torch.cat(self.op_labels)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]
        if op == -1:
            with Image.open(x) as image:
                image = image.convert('RGB')
                x = self.transform(image)
        else:
            if type(x) == str:
                x, y, op = torch.load(x)
        return x, y, op
class TerraIncognitaMultiple(Dataset):
    def __init__(self, names,root_folder=TerraIncognita_DATA_DIR, split='train', transform=None, ratio=None):
        self.x = []
        self.y = []
        self.op_labels = []

        train_mode = (split == 'train')

        if transform is None:
            if train_mode:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        for name in names:
            results = load_TerraIncognita(name, train=train_mode, data_dir=root_folder)
            x, y = results
            op_label = torch.tensor(np.ones(len(y), dtype=int) * (-1))
            if ratio is not None:
                num = len(x)
                indexes = np.random.permutation(num)
                sel_num = int(ratio * num)
                x = x[indexes[:sel_num]]
                y = y[indexes[:sel_num]]
                op_label = op_label[indexes[:sel_num]]
            self.x.append(x)
            self.y.append(y)
            self.op_labels.append(op_label)

        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)
        self.op_labels = torch.cat(self.op_labels)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]
        if op == -1:
            with Image.open(x) as image:
                image = image.convert('RGB')
                x = self.transform(image)
        else:
            if type(x) == str:
                x, y, op = torch.load(x)
        return x, y, op
class DomainNetMultiple(Dataset):
    def __init__(self, names,root_folder=DomainNet_DATA_DIR, split='train', transform=None, ratio=None):
        self.x = []
        self.y = []
        self.op_labels = []

        train_mode = (split == 'train')

        if transform is None:
            if train_mode:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        for name in names:
            results = load_DomainNet(name, train=train_mode, data_dir=root_folder)
            x, y = results
            op_label = torch.tensor(np.ones(len(y), dtype=int) * (-1))
            if ratio is not None:
                num = len(x)
                indexes = np.random.permutation(num)
                sel_num = int(ratio * num)
                x = x[indexes[:sel_num]]
                y = y[indexes[:sel_num]]
                op_label = op_label[indexes[:sel_num]]
            self.x.append(x)
            self.y.append(y)
            self.op_labels.append(op_label)

        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)
        self.op_labels = torch.cat(self.op_labels)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]
        if op == -1:
            with Image.open(x) as image:
                image = image.convert('RGB')
                x = self.transform(image)
        else:
            if type(x) == str:
                x, y, op = torch.load(x)
        return x, y, op
class FundusMultiple(Dataset):
    def __init__(self, names,root_folder=Fundus_DATA_DIR, split='train', transform=None, ratio=None):
        self.x = []
        self.y = []
        self.op_labels = []

        train_mode = (split == 'train')

        if transform is None:
            if train_mode:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        for name in names:
            results = load_Fundus(name, train=train_mode, data_dir=root_folder)
            x, y = results
            op_label = torch.tensor(np.ones(len(y), dtype=int) * (-1))

            if ratio is not None:
                num = len(x)
                indexes = np.random.permutation(num)
                sel_num = int(ratio * num)
                x = x[indexes[:sel_num]]
                y = y[indexes[:sel_num]]
                op_label = op_label[indexes[:sel_num]]

            self.x.append(x)
            self.y.append(y)
            self.op_labels.append(op_label)
        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)
        self.op_labels = torch.cat(self.op_labels)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.x)


    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        op = self.op_labels[index]

        if op == -1:
            with Image.open(x) as image:
                image = image.convert('RGB')
                x = self.transform(image)
        else:
            if type(x) == str:
                x, y, op = torch.load(x)

        return x, y, op


