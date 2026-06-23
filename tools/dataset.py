import os
import random
import pandas
import pandas as pd
import torch

from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as F, transforms
from PIL import Image

from tools.augment import get_train_augmentation


def readpath(root,txt_path):
    img_list = []
    txt_path=os.path.join(root,txt_path)
    with open(txt_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            tmp=[]
            tmp.append(os.path.join(root,item[0]))
            tmp.append(os.path.join(root,item[1]))
            img_list.append(tmp)
    file_to_read.close()
    return img_list


def generate_pathlist(images_dir, labels_dir):
    img_list = []
    label_list = []

    for filename in os.listdir(images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, filename)
            if os.path.exists(label_path):
                img_list.append(img_path)
                label_list.append(label_path)

    return list(zip(img_list, label_list))

def readpath_Crack3238(root, txt_path):
    img_list = []
    txt_path = os.path.join(root, txt_path)
    with open(txt_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            if len(item) != 1:
                print(f"Warning: Invalid line format in {txt_path}: {lines}")
                continue
            img_name = item[0]
            img_path = os.path.join(root, 'img', img_name)
            label_path = os.path.join(root, 'labelcol', img_name)
            if not os.path.exists(img_path) or not os.path.exists(label_path):
                print(f"Warning: Missing file in {txt_path}: {img_name}")
                continue
            img_list.append([img_path, label_path])
    return img_list
class Dataset(data.Dataset):
    def __init__(self, root, image_size=512, mode='train.txt', augmentation_prob=0.7,savepath='./'):

        self.root = root
        self.savepath=savepath

        self.pathlist = readpath(self.root,mode)
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob

    def __getitem__(self, index):

        image_path = self.pathlist[index][0]
        GT_path = self.pathlist[index][1]
        image = Image.open(image_path)
        GT = Image.open(GT_path)

        p_transform = random.random()
        print(p_transform)

        p = [p_transform]
        prob = pandas.DataFrame([p])
        prob.to_csv(self.savepath + '/seed.csv', mode='a', header=False, index=False)

        if (self.mode == 'train.txt') and p_transform <= self.augmentation_prob:
            print("trans")
            Transform = []

            RotationDegree = random.randint(0, 3)
            RotationDegree = self.RotationDegree[RotationDegree]
            Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))

            RotationRange = random.randint(-10, 10)
            Transform.append(T.RandomRotation((RotationRange, RotationRange)))
            Transform.append(T.Resize((self.image_size, self.image_size)))
            Transform.append(T.ToTensor())
            Transform = T.Compose(Transform)
            image = Transform(image)
            GT = Transform(GT)


            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)
            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)
        else:
            print("no trans")
            Transform = []
            Transform.append(T.Resize((self.image_size, self.image_size)))
            Transform.append(T.ToTensor())
            Transform = T.Compose(Transform)

            image = Transform(image)
            GT = Transform(GT)

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)

        return (image, GT)

    def __len__(self):
        return len(self.pathlist)

class Datasesloader(data.Dataset):
    def __init__(self, root,txt,savepath,imgsize=512):
        super().__init__()
        self.root=root
        self.txt=txt
        self.imgsize=imgsize
        self.savepath=savepath
        self.pathlist = readpath(self.root,txt)
        self.train_transforms=get_train_augmentation((512,512))
        self.normal_trans=T.Compose([
            T.Resize((self.imgsize, self.imgsize)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pathlist)

    def num_of_samples(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        imagepath=self.pathlist[idx][0]
        maskpath=self.pathlist[idx][1]
        image = Image.open(imagepath)
        mask = Image.open(maskpath)
        image = image.convert('RGB')
        image=self.normal_trans(image)
        mask=self.normal_trans(mask)

        p = [random.random()]
        prob = pandas.DataFrame([p])
        prob.to_csv(self.savepath + '/seed.csv', mode='a', header=False, index=False)

        if self.txt=="train.txt":
            image, mask = self.train_transforms(image, mask)
        mask[mask > 0] = 1
        return (image, mask)
class crackDataset_no_augmentation(data.Dataset):
    def __init__(self, root, txt,savepath, imgsize=512):
        super().__init__()
        self.root = root
        self.txt = txt
        self.imgsize = imgsize
        self.savepath=savepath
        self.pathlist = readpath(self.root, txt)

        self.normal_trans = T.Compose([
            T.Resize((self.imgsize, self.imgsize)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        imagepath = self.pathlist[idx][0]
        maskpath = self.pathlist[idx][1]

        image = Image.open(imagepath).convert('RGB')
        mask = Image.open(maskpath)

        image = self.normal_trans(image)
        mask = self.normal_trans(mask)
        p = [random.random()]
        prob = pandas.DataFrame([p])
        prob.to_csv(self.savepath + '/seed.csv', mode='a', header=False, index=False)
        mask[mask > 0] = 1
        return image, mask
class Datasesloader_withname(data.Dataset):
    def __init__(self, root,txt,savepath,imgsize=512):
        super().__init__()
        self.root=root
        self.txt=txt
        self.imgsize=imgsize
        self.savepath=savepath
        self.pathlist = readpath(self.root,txt)
        self.train_transforms=get_train_augmentation((512,512))
        self.normal_trans=T.Compose([
            T.Resize((self.imgsize, self.imgsize)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pathlist)

    def num_of_samples(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        imagepath=self.pathlist[idx][0]
        maskpath=self.pathlist[idx][1]
        image = Image.open(imagepath)
        mask = Image.open(maskpath)
        image = image.convert('RGB')
        image=self.normal_trans(image)
        mask=self.normal_trans(mask)

        p = [random.random()]
        prob = pandas.DataFrame([p])
        prob.to_csv(self.savepath + '/seed.csv', mode='a', header=False, index=False)

        if self.txt=="train.txt":
            image, mask = self.train_transforms(image, mask)
        mask[mask > 0] = 1
        return (image, mask,imagepath)


class DatasetLoader_newDatasets(data.Dataset):

    def __init__(self, root, txt, savepath, imgsize=512):
        super().__init__()
        self.root = root
        self.txt = txt
        self.imgsize = imgsize
        self.savepath = savepath
        self.pathlist = readpath_Crack3238(self.root,txt)
        self.train_transforms = get_train_augmentation((512,512))
        self.normal_trans = T.Compose([
            T.Resize((self.imgsize, self.imgsize)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pathlist)

    def num_of_samples(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        imagepath = self.pathlist[idx][0]
        maskpath = self.pathlist[idx][1]
        image = Image.open(imagepath)
        mask = Image.open(maskpath)
        image = image.convert('RGB')

        # print(f"Loading image: {os.path.basename(imagepath)}")
        image = self.normal_trans(image)
        mask = self.normal_trans(mask)

        p = [random.random()]
        prob = pd.DataFrame([p])
        prob.to_csv(self.savepath + '/seed.csv', mode='a', header=False, index=False)

        if self.txt == "train.txt":
            image, mask = self.train_transforms(image, mask)
        mask[mask > 0] = 1

        return (image, mask, imagepath)

class crackDataset_withname(data.Dataset):
    def __init__(self, root,txt,imgsize=512):
        super().__init__()
        self.root=root
        self.txt=txt
        self.imgsize=imgsize
        self.pathlist = readpath(self.root,txt)
        self.train_transforms=get_train_augmentation((512,512))
        self.normal_trans=T.Compose([
            T.Resize((self.imgsize, self.imgsize)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pathlist)

    def num_of_samples(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        imagepath=self.pathlist[idx][0]
        maskpath=self.pathlist[idx][1]
        image = Image.open(imagepath)
        mask = Image.open(maskpath)
        image = image.convert('RGB')
        image=self.normal_trans(image)
        mask=self.normal_trans(mask)

        if self.txt=="train.txt":
            image, mask = self.train_transforms(image, mask)
        mask[mask > 0] = 1
        return (image, mask,imagepath)

