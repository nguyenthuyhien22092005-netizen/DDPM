import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageFolderDDPM(Dataset):
    def __init__(self, images_dir, img_size=128):
        # images_dir: thư mục chứa ảnh để train DDPM
        # img_size  : kích thước mà toàn bộ ảnh sẽ được resize về

        # Lấy danh sách tất cả ảnh hợp lệ trong thư mục
        # Chấp nhận các đuôi: png, jpg, jpeg, tif
        # sorted() để đảm bảo thứ tự file ổn định
        self.images = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ])

        # Định nghĩa các phép biến đổi ảnh
        # 1. Resize về img_size x img_size
        # 2. Chuyển sang tensor (0–1)
        # 3. Chuẩn hóa về [-1, 1] vì DDPM yêu cầu input dạng này
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5],   ## mean
                        [0.5, 0.5, 0.5])   ## std
        ])

    def __len__(self):
        ## Trả về tổng số lượng ảnh trong dataset
        return len(self.images)

    def __getitem__(self, idx):
        # Hàm lấy ra 1 ảnh theo index
        # img_path: đường dẫn ảnh
        img_path = self.images[idx]

        # Mở file ảnh và convert về RGB
        # Nếu ảnh là grayscale hoặc 4 kênh, convert sẽ chuẩn hóa về 3 kênh
        img = Image.open(img_path).convert("RGB")

        # Trả về tensor đã transform
        return self.transform(img)
