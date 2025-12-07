import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# Dataset dùng cho UNet segmentation
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=256):
        self.img_dir = img_dir   #ảnh
        self.mask_dir = mask_dir #mask

        # Lọc ảnh theo đúng đuôi png, ipg, tif
        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".tif"))
        ])

        # transform cho ảnh đầu vào
        self.transform_img = T.Compose([
            T.Grayscale(num_output_channels=1), # chuyển sang ảnh xám 1 channel
            T.Resize((size, size)),             # resize về kích thước cố định
            T.ToTensor(),                       # chuyển về sang tensor [0,1]
        ])

        # transform cho mask
        self.transform_mask = T.Compose([
            T.Grayscale(num_output_channels=1), # mask cũng chuyển 1 channel
            T.Resize((size, size)),             # resize mask
            T.ToTensor(),                       # to tensor
        ])

    # số lượng phần tử trong dataset
    def __len__(self):
        return len(self.images)

    # lấy 1 mẫu ảnh + mask
    def __getitem__(self, idx):
        name = self.images[idx]

        # load ảnh
        img = Image.open(os.path.join(self.img_dir, name))
        mask = Image.open(os.path.join(self.mask_dir, name))

        # transform
        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        return img, mask
