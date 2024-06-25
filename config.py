import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path


class Anime_Data(Dataset):
    def __init__(self, transform):
        super().__init__()
        self.file = Path("./images/faces")
        self.imgs = os.listdir(self.file)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file / self.imgs[index])
        return self.transform(img)

    def __len__(self):
        return len(self.imgs)


class Config:
    resume = False
    img_size = 96
    noise_dim = 100
    batch_size = 64
    # Epoch: 迭代次数
    epochs = 50
    # 网络隐藏特征图尺寸 (判别器的复杂度要略低于生成器,避免过度指导图片生成)
    gen_feature_map = 128
    disc_feature_map = 64
    # 学习率
    gen_lr = 3e-3
    disc_lr = 2e-4
    # 存储路径
    modelSave = False
    data_path = "./images/faces"
    fake_path = "./images/fake/exp"
    if not os.path.exists("./images/fake"):
        os.mkdir("./images/fake")
    real_path = "./images/real/exp"
    if not os.path.exists("./images/real"):
        os.mkdir("./images/real")
    fake_0 = "./images/fake_0/exp"
    if not os.path.exists("./images/fake_0"):
        os.mkdir("./images/fake_0")
    create_model_path = "./checkpoints/exp"
    log_path = "./logging/exp"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
