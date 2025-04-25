import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        label = self.label_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


if __name__ == "__main__":
    df = pd.read_csv('./dataset/train/train.csv')
    image_path_list = []
    label_list = df["label"].to_list()
    for e in df["name"].to_list():
        #print(os.path.join(f'./dataset/train',e))
        image_path_list.append(os.path.join(f'./dataset/train',e))

    """for i,_ in enumerate(image_path_list):
        print(image_path_list[i], label_list[i])"""

    train_data = Dataset(image_path_list, label_list, transform=train_transform)
    # test_data = dataset(test_list, test_label)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=8, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = 8, shuffle= True)

    for imgs, batchsize in train_loader:
        print("Size of train_loader_image:", imgs.size())
        print("Size of train_loader_batchsize:", batchsize.size())
        break