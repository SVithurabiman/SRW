from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image_paths, H, W, low_limit=0, up_limit=10000):
        self.image_paths = [
            os.path.join(image_paths, img) for img in os.listdir(image_paths)
        ][low_limit:up_limit]
        print(f"Number of images in dataset: {len(self.image_paths)}", flush=True)
        self.H = H
        self.W = W

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.H, self.W), antialias=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image
