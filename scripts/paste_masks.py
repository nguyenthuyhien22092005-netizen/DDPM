import os
from PIL import Image
import random

# CẤU HÌNH ĐƯỜNG DẪN

real_mask_dir = "data/Masks"           # mask thật
synth_img_dir = "generated/images"     # ảnh synthetic
out_mask_dir  = "generated/masks"      # mask synthetic output

os.makedirs(out_mask_dir, exist_ok = True)

# ĐỌC DANH SÁCH MASK THẬT
real_masks = [
    os.path.join(real_mask_dir, f)
    for f in os.listdir(real_mask_dir)
    if f.lower().endswith(('.png', '.jpg', 'tif', 'tiff'))
]

if len(real_masks) == 0:
    print("Không tìm thấy mask thật trong data/Masks/")
    exit()

print(f"Tìm thấy {len(real_masks)} mask thật.")

# TẠO MASK CHO ẢNH SYNTHETIC
for img_name in os.listdir(synth_img_dir):
    # Bỏ qua file không phải ảnh
    if not img_name.lower().endswith((".png", ".jpg")):
        continue

    synth_path = os.path.join(synth_img_dir, img_name)

    # Chọn ngẫu nhiên 1 mask thật
    real_mask_path = random.choice(real_masks)

    # Load mask thật
    mask = Image.open(real_mask_path).convert("L")

    # Resize mask = kích thước ảnh synthetic
    synth_img = Image.open(synth_path)
    mask = mask.resize(synth_img.size)

    # Lưu mask synthetic
    out_path = os.path.join(out_mask_dir, img_name.replace(".png", "_mask.png"))
    mask.save(out_path)

    print("Saved:", out_path)

    # HOÀN TẤT
print("\nHOÀN TẤT — Đã tạo mask cho ảnh synthetic!")
