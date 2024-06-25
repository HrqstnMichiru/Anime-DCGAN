from pytorch_fid import fid_score
import lpips
import IS
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import os
from pathlib import Path

real_image_folder = "./images/real/exp4/15"
fake_image_folder = "./images/fake/exp4/15"


def FID(real, fake):
    fid_value = fid_score.calculate_fid_given_paths(
        [real, fake], batch_size=8, device="cuda:0", dims=2048
    )
    return fid_value  # ?越低越好


def ISScore():
    MAX0, IS0 = IS.inception_score(splits=4)
    print("MAX IS is %.4f" % MAX0)
    print("The IS is %.4f" % IS0)


def lpipsScore(real, fake):
    real_img = os.listdir(real)
    fake_img = os.listdir(fake)
    loss_fn = lpips.LPIPS(net="vgg")
    lpips_score = 0
    for real_name, fake_name in zip(real_img, fake_img):
        real_path = real / real_name
        fake_path = fake / fake_name
        img0 = lpips.im2tensor(lpips.load_image(str(real_path)))
        img1 = lpips.im2tensor(lpips.load_image(str(fake_path)))
        lpips_score += loss_fn.forward(img0, img1).item()
    return lpips_score / len(real_img)  # ?越高越相似


def ssimScore(real, fake):
    real_img = os.listdir(real)
    fake_img = os.listdir(fake)
    ssim_score = 0
    for real_name, fake_name in zip(real_img, fake_img):
        real_path = real / real_name
        fake_path = fake / fake_name
        imgr = Image.open(real_path).convert("L")
        imgf = Image.open(fake_path).convert("L")
        imgr, imgf = np.array(imgr), np.array(imgf)
        ssim_score += ssim(imgr, imgf, data_range=1)
    return ssim_score / len(real_img)  # ? 越高越相似


if __name__ == "__main__":
    # fid = fid_value = fid_score.calculate_fid_given_paths(
    #     [real_image_folder, fake_image_folder], batch_size=8, device="cpu", dims=2048
    # )
    # print(fid)
    # ssim_score=ssimScore(Path(real_image_folder),Path(fake_image_folder))
    # print(ssim_score)
    # lpips_score=lpipsScore(Path(real_image_folder),Path(fake_image_folder))
    # print(lpips_score)
    ISScore()
