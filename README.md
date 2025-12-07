# Diffusion-based Data Augmentation for Nuclei Image Segmentation
*Tác giả: Nguyễn Thúy Hiền - Lê Thị Dịu*

# Table of Contents

1. [Giới thiệu](#-giới-thiệu)
2. [Mục tiêu](#-mục-tiêu)
3. [Pipeline Diagram](#-Pipeline-Diagram)
4. [Yêu cầu hệ thống](#-Yêu-cầu-hệ-thống)
5. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
6. [Huấn luyện mô hình](#️-huấn-luyện-mô-hình)

   * [Huấn luyện Diffusion Model (DDPM)](#-huấn-luyện-diffusion-model-ddpm)
   * [Sinh dữ liệu Synthetic](#-sinh-dữ-liệu-synthetic)
   * [Tạo Synthetic Mask](#-tạo-synthetic-mask)
   * [Kết hợp dataset thực + synthetic](#-kết-hợp-dataset-thực--synthetic)
   * [Huấn luyện UNet Segmentation](#-huấn-luyện-unet-segmentation)
   * [Dự đoán segmentation](#-Dự-đoán-segmentation)
7. [Dự đoán & đánh giá](#-dự-đoán--đánh-giá)
8. [Kết quả minh họa](#-kết-quả-minh-họa)
9. [Cải thiện & hướng phát triển](#-cải-thiện--hướng-phát-triển)
10. [Thông tin liên hệ](#-thông-tin-liên-hệ)

## 1. Giới thiệu
Nghiên cứu này sử dụng **Diffusion Model (DDPM)** để tạo ảnh nhân tạo **(synthetic nuclei images)** nhằm tăng cường dữ liệu cho bài toán phân đoạn nhân tế bào.

Pipeline gồm 2 phần:
1. Tạo dữ liệu synthetic bằng DDPM
2. Huấn luyện UNet segmentation trên dataset:
   Real Data + Synthetic Data

Mục tiêu:
- Giảm phụ thuộc vào dữ liệu thật vốn ít & khó chú thích
- Cải thiện chất lượng phân đoạn khi dữ liệu giới hạn

## 2. Mục tiêu
- Huấn luyện Diffusion Model sinh ảnh tế bào
- Tạo synthetic masks tương 
- Tạo ảnh synthetic bằng Diffusion.
- Kết hợp dữ liệu thật + synthetic.
- Huấn luyện UNet segmentation.
- Đánh giá bằng Dice & IoU.

------------------------------------------------------------------------

## 3. Pipeline Diagram

    [Dataset] --> [DDPM Training] --> [Synthetic Images]
           \                                   /
            \                                 /
             ------> [UNet Training] <--------

------------------------------------------------------------------------

## 4. Yêu cầu hệ thống
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

## 5. Cấu trúc thư mục
```
PythonDiffusion/
│
├── data/                  # Dataset thực
│   ├── Images/
│   └── Masks/
│
├── generated/             # Ảnh synthetic và masks sinh ra
│   ├── images/
│   └── masks/
│
├── combined/              # Dataset kết hợp
│   ├── images/
│   └── masks/
│
├── checkpoints/           # Lưu mô hình
│   ├── ddpm/
│   └── unet/
│
├── predict/               # Dự đoán
│   ├── input/
│   └── output/
│
├── scripts/               # File mã nguồn
│   ├── train_ddpm.py
│   ├── sample_ddpm.py
│   ├── paste_masks.py
│   ├── combine_dataset.py
│   ├── train_unet.py
│   ├── predict_unet.py
│   ├── segmentation_dataset.py
│   └── unet_model.py
│
└── README.md              # File hướng dẫn (file bạn đang đọc)

```

### Dataset dùng trong Project
- (https://drive.google.com/drive/folders/1eeF3NNeLyrtrF3UASUm12Tu7Ey1AEPJ1?usp=drive_link)

Sử dụng bộ dữ liệu (Dataset) có sẵn (online) MoNuSeg (TCGA):
  - 37 ảnh thật (256x256, dạng nuclei microscopy)
  - 37 masks tương ứng
    
## 6. Cách chạy
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
#### Sinh mask tại:
```
generated/masks/
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
#### Model segmentation lưu tại:
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
Kết quả:
- Mask dự đoán
- Overlay
- Dice Score
- IoU Score
  
#### Kết quả sẽ nằm tại:
```
predict/output/
```
## 7. Dự đoán & đánh giá
- Dice Scorw
- IoU Score
- Kết quả hiển thị trực tiếp khi chạy ``` predict_unet.py. ```

## 8. Kết quả minh họa

### **Ảnh gốc — Mask dự đoán — Overlay**

#### **Ảnh gốc**

### Original

![Original](https://github.com/nguyenthuyhien22092005-netizen/DDPM/blob/master/predict/input/image_02.png)

### Mask

![Mask](https://github.com/nguyenthuyhien22092005-netizen/DDPM/blob/master/predict/output/image_02_mask.png)

### Overlay

![Overlay](https://github.com/nguyenthuyhien22092005-netizen/DDPM/blob/master/predict/output/image_02_overlay.png)

### Bảng kết quả: (predict)
| Image          | Dice Score | IoU Score | Output Mask         | Overlay                |
| -------------- | ---------- | --------- | ------------------- | ---------------------- |
| `image_01.png` | **0.4650** | 0.3030    | `image_01_mask.png` | `image_01_overlay.png` |
| `image_02.png` | **0.6370** | 0.4674    | `image_02_mask.png` | `image_02_overlay.png` |
| `image_03.png` | **0.3502** | 0.2123    | `image_03_mask.png` | `image_03_overlay.png` |

-> Kết quả cải thiện đáng kể so với huấn luyện chỉ bằng real data.

## 9. Cải thiện và hướng phát triển
- Dùng UNet++/ Attention UNet
- Dùng Latent Diffusion Model (LDM) để sinh ảnh 512×512
- Tăng số lượng ảnh synthetic
- Augmentation chuyên sâu (elastic transform, stain normalization…)
- Chuyển sang PyTorch Lightning để train nhanh hơn

## 10. Thông tin liên hệ:
Nếu bạn muốn dùng mã nguồn, đóng góp hoặc hỏi thêm:
- Email: nguyenthuyhien22092005@gmail.com
- GitHub: https://github.com/nguyenthuyhien22092005-netizen
  
