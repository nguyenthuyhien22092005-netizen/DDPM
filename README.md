# Diffusion-based Data Augmentation for Nuclei Image Segmentation
*Tác giả: Nguyễn Thúy Hiền - Lê Thị Dịu*

## Table of Contents

- [Giới thiệu](#-giới-thiệu)
- [Tính năng chính](#-tính-năng)
- [Pipeline Diagram](#-Pipeline-Diagram)
- [Yêu cầu hệ thống](#-Yêu-cầu-hệ-thống)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Huấn luyện mô hình](#️-huấn-luyện-mô-hình)
- [Dự đoán & đánh giá](#-dự-đoán--đánh-giá)
- [Kết quả minh họa](#kết-quả-minh-họa)
- [Cải thiện & hướng phát triển](#cải-thiện--hướng-phát-triển)
- [Thông tin liên hệ](#thông-tin-liên-hệ)

#kết-quả-minh-họa
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

#cải-thiện--hướng-phát-triển

- Dùng UNet++/ Attention UNet
- Dùng Latent Diffusion Model (LDM) để sinh ảnh 512×512
- Tăng số lượng ảnh synthetic
- Augmentation chuyên sâu (elastic transform, stain normalization…)
- Chuyển sang PyTorch Lightning để train nhanh hơn

#thông-tin-liên-hệ

Nếu bạn muốn dùng mã nguồn, đóng góp hoặc hỏi thêm:
- Email: nguyenthuyhien22092005@gmail.com
- GitHub: https://github.com/nguyenthuyhien22092005-netizen
  
