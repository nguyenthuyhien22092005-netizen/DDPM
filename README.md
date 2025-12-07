# Diffusion-based Data Augmentation for Nuclei Image Segmentation

## 1. Giới thiệu
Dự án này sử dụng **Diffusion Model (DDPM)** để sinh ảnh tế bào nhằm tăng cường dữ liệu và cải thiện phân đoạn bằng **U-Net**.

## 2. Mục tiêu
- Tạo ảnh synthetic bằng Diffusion.
- Kết hợp dữ liệu thật + synthetic.
- Huấn luyện UNet segmentation.
- Đánh giá bằng Dice & IoU.

## 3. Yêu cầu hệ thống
### Phần mềm:
```
Python 3.10
PyTorch
Diffusers
Torchvision
Matplotlib
Pillow
Tqdm
```
# Cài đặt nhanh
```
pip install -r requirements.txt
```
### Phần cứng:
- CPU chạy được (chậm hơn)
- GPU khuyến khích cho train diffusion & UNet

## 4. Cấu trúc thư mục
```
PythonDiffusion/
│
├── data/
│   ├── Images/           # ảnh thật
│   └── Masks/            # mask thật
│
├── generated/
│   ├── images/           # ảnh synthetic sinh từ diffusion
│   └── masks/            # mask synthetic
│
├── combined/
│   ├── images/           # ảnh kết hợp (real + synth)
│   └── masks/
│
├── checkpoints/
│   ├── ddpm/             # model diffusion
│   └── unet/             # model segmentation
│
├── predict/
│   ├── input/            # ảnh muốn dự đoán
│   └── output/           # ảnh mask + overlay
│
└── scripts/              # toàn bộ mã nguồn

```

## 5. Cách chạy
### Bước 1: Train diffusion Model (DDPM)
```
python scripts/train_ddpm.py
```
#### Model sẽ được lưu vào:
```
checkpoints/ddpm/
```
### Bước 2: Sinh ảnh Synthetic
```
python scripts/sample_ddpm.py
```
#### Kết quả:
```
generated/images/
```
### Bước 3: Tạo mask synthetic
```
python scripts/paste_masks.py
```

### Bước 4: Kết hợp dataset + synthetic
```
python scripts/combine_dataset.py
```
#### Dữ liệu được tạo tại:
```
combined/images/
combined/masks/
```
### Bước 5: Train UNet Segmentation
```
python scripts/train_unet.py
```
#### Checkpoint sẽ nằm tại:
```
checkpoints/unet/
```
### Bước 6: Dự đoán segmentation
#### Cho ảnh vào:
```
predict/input/
```
#### Sau đó chạy:
```
python scripts/predict_unet.py
```
#### Kết quả sẽ nằm tại:
```
predict/output/
```
## 6. Đánh giá Segmentation:
- Dice Scorw
- IoU Score
- Kết quả hiển thị trực tiếp khi chạy ``` predict_unet.py. ```

### Bảng kết quả:
| Image          | Dice Score | IoU Score | Output Mask         | Overlay                |
| -------------- | ---------- | --------- | ------------------- | ---------------------- |
| `image_01.png` | **0.4650** | 0.3030    | `image_01_mask.png` | `image_01_overlay.png` |
| `image_02.png` | **0.6370** | 0.4674    | `image_02_mask.png` | `image_02_overlay.png` |
| `image_03.png` | **0.3502** | 0.2123    | `image_03_mask.png` | `image_03_overlay.png` |

# <a name="results"></a> Kết quả minh họa

## **Ảnh gốc — Mask dự đoán — Overlay**

#### **Ảnh gốc**

### Original

![Original](https://github.com/nguyenthuyhien22092005-netizen/DDPM/blob/master/predict/input/image_02.tif)

### Mask

![Mask](https://github.com/nguyenthuyhien22092005-netizen/DDPM/blob/master/predict/output/image_02_mask.png)

### Overlay

![Overlay](https://github.com/nguyenthuyhien22092005-netizen/DDPM/blob/master/predict/output/image_02_overlay.png)

------------------------------------------------------------------------

## 3. Pipeline Diagram

    [Dataset] --> [DDPM Training] --> [Synthetic Images]
           \                                   /
            \                                 /
             ------> [UNet Training] <--------

------------------------------------------------------------------------

## 7. Tác giả
Nguyễn Thúy Hiền
