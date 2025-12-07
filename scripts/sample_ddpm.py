import torch
from diffusers import DDPMScheduler, UNet2DModel
from torchvision.utils import save_image
import os

# CẤU HÌNH ĐƯỜNG DẪN
model_path = "checkpoints/ddpm/ddpm_epoch5.pt" #checkpoint đã train

# Thư mục lưu ảnh sinh
os.makedirs("generated/images", exist_ok=True)

# THIẾT LẬP THAM SỐ
device = "cpu"  # dùng CPU
img_size = 128  # kích thước ảnh sinh ra
num_images = 37 # số lượng cần sinh

# KHỞI TẠO MODEL CÙNG CẤU TRÚC
model = UNet2DModel(
    sample_size=img_size,  # kích thước ảnh đầu ra
    in_channels=3,         # RGB
    out_channels=3,        # RGB
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device)

# Load trọng số đã train
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

# TẠO SCHEDULER DDPM
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Sinh ảnh
for i in range(num_images):
    x = torch.randn(1, 3, img_size, img_size).to(device) #khởi tạo ảnh noise trắng

    #chạy qua toàn bộ bước reverse diffusion
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(x, t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    # Lưu ảnh synthetic
    # chuẩn hóa về [0,1] để lưu ảnh
    save_image((x + 1) / 2, f"generated/images/synth_{i}.png")
    print(f"Saved synth_{i}.png")

print("DONE!")