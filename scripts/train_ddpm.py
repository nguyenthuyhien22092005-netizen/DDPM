import os
import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from dataset_ddpm import ImageFolderDDPM
from tqdm import tqdm


# Hàm tạo model diffusion (U-Net 2D)
def create_ddpm_model(img_size=128):
    return UNet2DModel(
        sample_size=img_size,      # Kích thước ảnh (H,W)
        in_channels=3,             # RGB
        out_channels=3,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

# Hàm train DDPM
def train_ddpm(
    images_dir="data/Images",    # ảnh gốc
    ckpt_dir="checkpoints/ddpm", # nơi lưu checkpoint
    img_size = 128,
    batch_size=4,
    epochs=5,
    lr=1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Tạo dataset & dataloader
    dataset = ImageFolderDDPM(images_dir, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Khởi tạo model & noise scheduler
    model = create_ddpm_model(img_size).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # tạo thư mục checkpoint
    os.makedirs(ckpt_dir, exist_ok=True)

    # train loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        pbar = tqdm(loader)

        for batch in pbar:
            batch = batch.to(device)  # ảnh [-1,1]

            # random timestep
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps,
                (batch.shape[0],), device=device
            ).long()

            # thêm noise vào ảnh
            noise = torch.randn_like(batch)
            noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)

            # dự đoán noise
            pred = model(noisy_images, timesteps).sample

            # loss = MSE(noise_predicted, noise_true)
            loss = torch.nn.functional.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss))

        # Lưu checkpoint cuối mỗi epoch
        ckpt_path = os.path.join(ckpt_dir, f"ddpm_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved:", ckpt_path)

# main
if __name__ == "__main__":
    train_ddpm()
