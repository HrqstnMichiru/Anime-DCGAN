import torch
from model import Generator, Discriminator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from torch.nn import BCELoss
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import increment_path
import warnings
from config import Config, Anime_Data
import sys
import time
import datetime
from pytorch_fid import fid_score

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = "MicroSoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


def train(config):
    device = config.device
    transform = config.transform
    animeData = Anime_Data(transform=transform)

    train_loader = DataLoader(
        dataset=animeData,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    disModel = Discriminator(config)
    genModel = Generator(config)
    if config.resume:
        disModel.load_state_dict(torch.load("./checkpoints/exp2/disc_params.pt"))
        genModel.load_state_dict(torch.load("./checkpoints/exp2/gen_params.pt"))
    criterion = BCELoss()
    optimD = optim.Adam(
        disModel.parameters(), config.disc_lr, betas=(0.5, 0.999), weight_decay=5e-3
    )
    optimG = optim.Adam(genModel.parameters(), config.gen_lr, betas=(0.5, 0.999))
    gen_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimG, T_max=config.epochs * len(train_loader), eta_min=3 * config.disc_lr
    )

    label = torch.FloatTensor(config.batch_size)
    real_label = 1
    fake_label = 0
    disModel = disModel.to(device)
    genModel = genModel.to(device)
    criterion = criterion.to(device)
    label = label.to(device)

    fixed_noise = torch.randn(config.batch_size, config.noise_dim, 1, 1).to(device)
    log_path, _ = increment_path(config.log_path)

    writer = SummaryWriter(
        log_dir=log_path, comment="这是一个有趣的邂逅", flush_secs=60
    )
    step = 0
    model_path, _ = increment_path(config.create_model_path)
    fake_path, _ = increment_path(config.fake_path)
    real_path, _ = increment_path(config.real_path)
    fake_0_path, _ = increment_path(config.fake_0)

    fid_value0 = 1000

    time1 = datetime.datetime.now()
    for epoch in range(config.epochs):
        epoch += 1
        disModel.train()
        genModel.train()
        lossD_all = 0
        lossG_all = 0
        batch_num = 0
        pbar = tqdm(train_loader, leave=True, colour="red")
        start = time.time()
        for batch_idx, real_img in enumerate(pbar):
            batch_num += 1
            pbar.set_description(f"Training,Epoch [{epoch}/{config.epochs}]")

            """train disModel"""
            optimD.zero_grad()
            noise = torch.randn(config.batch_size, config.noise_dim, 1, 1).to(device)

            real_img = real_img.to(device)
            label.resize_(config.batch_size).fill_(real_label)
            output_real = disModel(real_img)
            lossD_real = criterion(output_real, label)
            lossD_real.backward()

            fake_img = genModel(noise)
            label.fill_(fake_label)
            output_fake = disModel(fake_img.clone().detach())
            lossD_fake = criterion(output_fake, label)
            lossD_fake.backward()

            lossD = (lossD_real + lossD_fake) / 2
            lossD_all += lossD.item()
            if lossD > 0.3:
                optimD.step()

            if epoch == 1 and batch_idx == 0:
                writer.add_graph(genModel, noise)
                writer.add_graph(disModel, real_img)

            """train genModel"""
            optimG.zero_grad()
            label.fill_(real_label)
            output_G = disModel(fake_img)
            lossG = criterion(output_G, label)
            lossG_all += lossG.item()
            lossG.backward()
            optimG.step()
            gen_lr_scheduler.step()

            fake = fake_path / f"{epoch}"
            real = real_path / f"{epoch}"
            fake_0_path1 = fake_0_path / f"{epoch}"
            if not fake.exists():
                fake.mkdir()
            if not real.exists():
                real.mkdir()
            if not fake_0_path1.exists():
                fake_0_path1.mkdir()

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    step += 1
                    fake_0 = genModel(fixed_noise).reshape(-1, 3, 96, 96)
                    data = real_img.reshape(-1, 3, 96, 96)
                    fake_img = fake_img.reshape(-1, 3, 96, 96)
                    img_grid_fake = make_grid(fake_0, normalize=True)
                    img_grid_real = make_grid(data, normalize=True)
                    fake_0_grid = make_grid(fake_img, normalize=True)
                    writer.add_image("fake_images", img_grid_fake, global_step=step)
                    writer.add_image("real_image", img_grid_real, global_step=step)
                    writer.add_image("fake_0", fake_0_grid, global_step=step)
                    fake_img = fake / f"otaku{epoch}_{batch_idx}.jpg"
                    real_img = real / f"shojo{epoch}_{batch_idx}.jpg"
                    fake_0_img = fake_0_path1 / f"{epoch}_{batch_idx}.jpg"
                    save_image(img_grid_fake.cpu(), fake_img)
                    save_image(img_grid_real.cpu(), real_img)
                    save_image(fake_0_grid, fake_0_img)
        end = time.time()
        spend = end - start
        lossD_all = lossD_all / batch_num
        lossG_all = lossG_all / batch_num
        pbar.write(
            f"Epoch:{epoch},time:{spend}s, LossD:{lossD_all:.4f}, LossG:{lossG_all:.4f}"
        )
        fid_value1 = fid_score.calculate_fid_given_paths(
            [str(real), str(fake)], batch_size=8, device="cuda:0", dims=2048
        )
        fid_value2 = fid_score.calculate_fid_given_paths(
            [str(real), str(fake_0_path1)], batch_size=8, device="cuda:0", dims=2048
        )
        pbar.write(f"FID value1:{fid_value1},FID value2:{fid_value2}")
        writer.add_scalars(
            "FID", {"fid1": fid_value1, "fid2": fid_value2}, global_step=epoch
        )
        writer.add_scalars(
            "Loss",
            {"Generator": lossG_all, "Discriminator": lossD_all},
            global_step=epoch,
        )
        if config.modelSave:
            torch.save(genModel, model_path / "gen_model.pt")
            torch.save(disModel, model_path / "ids_model.pt")
        if fid_value1 < fid_value0:
            fid_value0 = fid_value1
            torch.save(disModel.state_dict(), model_path / f"disc_params.pt")
            torch.save(genModel.state_dict(), model_path / f"gen_params.pt")
        torch.save(disModel.state_dict(), model_path / "last_dis.pt")
        torch.save(genModel.state_dict(), model_path / "last_gen.pt")
    time2 = datetime.datetime.now()
    time0 = time2 - time1
    writer.close()
    print(f"Training has completed for {time0}")


if __name__ == "__main__":
    cfg = Config()
    sys.exit(train(config=cfg))
