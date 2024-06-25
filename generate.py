import torch
from datetime import datetime
from model import Generator
from config import Config
from torchvision.utils import save_image


def create(config):
    device = config.device
    time = datetime.now().strftime("%Y-%m%d-%H%M")
    gen = Generator(config).to(device)
    with torch.no_grad():
        gen.load_state_dict(torch.load("./checkpoints/exp/"))
        for i in range(20):
            noise = torch.randn(config.batch_size, config.noise_dim, 1, 1).to(device)
            img = gen.forward(noise)
            save_image(img, f"./images/generate{time}.png")
            print(f"图片已保存至./images/generate/{time}.png")


if __name__ == "__main__":
    cfg = Config()
    create(config=cfg)
