import os
import shutil

# CONFIG - THIẾT LẬP ĐƯỜNG DẪN
real_img_dir  = "data/Images" #Thư mục chứa ảnh gốc
real_mask_dir = "data/Masks" #Thư mục chứa mask gốc

synth_img_dir  = "generated/images" #Thư mục chứa ảnh synthetic (do DDPM sinh ra)
synth_mask_dir = "generated/masks"  #Thư mục chứa mask synthetic tương ứng

#Thư mục kết hợp ảnh gốc và ảnh synthetic
combined_img_dir  = "combined/images"
combined_mask_dir = "combined/masks"

#Tạo thư mục gốc
os.makedirs(combined_img_dir, exist_ok=True)
os.makedirs(combined_mask_dir, exist_ok=True)

# COPY REAL DATA
print("Copy ảnh gốc")
for f in os.listdir(real_img_dir): #Lặp qua tất cả file ảnh trong data
    if f.lower().endswith((".png", ".jpg", ".tif")): #Chỉ nhận file đúng định dạng
        shutil.copy(os.path.join(real_img_dir, f), #Copy file vào combine
                    os.path.join(combined_img_dir, f))

for f in os.listdir(real_mask_dir): #Lặp qua tất cả file mask trong data
    if f.lower().endswith((".png", ".jpg", ".tif")):
        shutil.copy(os.path.join(real_mask_dir, f),
                    os.path.join(combined_mask_dir, f))

print("Ảnh gốc đã được copy!")

# COPY SYNTHETIC DATA (DDPM GENERATE)
print("Copy ảnh synthetic")
for f in os.listdir(synth_img_dir): #Ảnh synth nằm generated/images dạng PNG
    if f.lower().endswith(".png"):
        shutil.copy(os.path.join(synth_img_dir, f),
                    os.path.join(combined_img_dir, f))

for f in os.listdir(synth_mask_dir): #Ảnh synth nằm generated/masks dạng PNG
    if f.lower().endswith(".png"):
        shutil.copy(os.path.join(synth_mask_dir, f),
                    os.path.join(combined_mask_dir, f))

print("Ảnh synthetic và mask synthetic đã được copy!")

# HOÀN TẤT
print("HOÀN TẤT — Dataset kết hợp nằm tại:")
print("combined/images/")
print("combined/masks/")
