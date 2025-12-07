import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from unet_model import UNet

# CẤU HÌNH
MODEL_PATH = "checkpoints/unet/unet_epoch20.pt" # đường dẫn model UNet đã train
INPUT_DIR = "predict/input"                     # input
OUTPUT_DIR = "predict/output"                   # output

# tạo folder output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# lấy danh sách ảnh trong input
input_images = glob.glob(f"{INPUT_DIR}/*.*")
if len(input_images) == 0:
    print("Không tìm thấy ảnh trong predict/input/")
    exit()

# HÀM TÍNH CHỈ SỐ ĐÁNH GIÁ
# hàm tính Dice
def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return 2 * intersection / (pred.sum() + gt.sum() + 1e-8)

# hàm tính IoU
def iou_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)

# DỰ ĐOÁN
# lặp qua từng ảnh trong thư mục input
for IMAGE_PATH in input_images:

    print("Predicting:", IMAGE_PATH)
    DEVICE = "cpu"
    IMG_SIZE = 256

    # load model
    print("Loading UNet model")
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # transform ảnh input
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor()
    ])

    # load ảnh gốc
    print("Loading image")
    img = Image.open(IMAGE_PATH)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # chạy mô hình tạo mask
    with torch.no_grad():
        pred = model(img_tensor)[0][0].cpu().numpy()

    # ngưỡng thành mask nhị phân
    mask = (pred > 0.5).astype(np.uint8)

    # load mask ground-truth tương ứng
    mask_name = os.path.basename(IMAGE_PATH)               # ví dụ: test_01.jpg
    mask_name_no_ext = os.path.splitext(mask_name)[0]      # ví dụ: test_01
    gt_path = f"data/Masks/{mask_name_no_ext}.tif"

    if not os.path.exists(gt_path):
        print("Không tìm thấy mask thật:", gt_path)
        # nếu không có mask thật, bỏ qua ảnh này và không dừng toàn bộ script
        continue

    gt_mask = Image.open(gt_path)
    gt_mask = gt_mask.resize((IMG_SIZE, IMG_SIZE))
    gt_mask = np.array(gt_mask)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    # tính Dice và IoU
    dice = dice_score(mask, gt_mask)
    iou = iou_score(mask, gt_mask)
    print(f"Dice Score: {dice:.4f}")
    print(f"IoU Score : {iou:.4f}")

    # tạo overlay để lưu (màu đỏ cho vùng mask)
    overlay = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    # nếu ảnh grayscale, overlay sẽ có shape (H, W), cần stack thành (H, W, 3)
    if overlay.ndim == 2:
        overlay = np.stack([overlay] * 3, axis=-1)
    overlay[mask == 1] = [255, 0, 0] ## vùng mask tô đỏ

    # lưu kết quả
    base = mask_name_no_ext
    out_mask_path = f"{OUTPUT_DIR}/{base}_mask.png"
    out_overlay_path = f"{OUTPUT_DIR}/{base}_overlay.png"

    plt.imsave(out_mask_path, mask, cmap="gray")
    plt.imsave(out_overlay_path, overlay)

    print("Saved mask   ->", out_mask_path)
    print("Saved overlay->", out_overlay_path)

    # KẾT THÚC
print("DONE. Xem kết quả trong predict/output/")
