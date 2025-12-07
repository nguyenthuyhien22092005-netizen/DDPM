import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segmentation_dataset import SegmentationDataset
from unet_model import UNet
import os

# Cấu hình
IMG_DIR = "combined/images"   # ảnh train
MASK_DIR = "combined/masks"   # mask train
BATCH = 2                     # batch size khi train
EPOCHS = 20                   # số epoch
LR = 1e-4                     # learning rate
DEVICE = "cpu"                # dùng CPU

# Load Dataset
dataset = SegmentationDataset(IMG_DIR, MASK_DIR, size=256) # resize ảnh về 256x256
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

# Khởi tạo mô hình UNet + Loss + Optimize
model = UNet().to(DEVICE)
criterion = nn.BCELoss()  # dùng Binary Cross Entropy cho segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# thư mục lưu checkpoint
os.makedirs("checkpoints/unet", exist_ok=True)

# Train loop
for epoch in range(EPOCHS):
    total_loss = 0

    for img, mask in loader:
        img = img.to(DEVICE)             # đưa ảnh vào device
        mask = mask.to(DEVICE)           # đưa mask vào device

        pred = model(img)                # UNet dự đoán mask
        loss = criterion(pred, mask)     # tính loss

        optimizer.zero_grad()            # reset gradient
        loss.backward()                  # tính gradient
        optimizer.step()                 # cập nhật trọng số

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss = {total_loss/len(loader):.4f}")
    # lưu checkpoint mỗi epoch
    torch.save(model.state_dict(), f"checkpoints/unet/unet_epoch{epoch+1}.pt")

print("Train UNet hoàn thành!")
