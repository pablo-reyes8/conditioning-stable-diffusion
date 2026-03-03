import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image

from src.data.constants import DEFAULT_ATTRIBUTES


ATTRS = DEFAULT_ATTRIBUTES


class ZipFaceDataset(Dataset):
    def __init__(self, zip_path, labels_path, attrs=ATTRS, img_size=512, zip_prefix="train/"):
        self.zip_path = zip_path
        self.zip_prefix = zip_prefix
        self.attrs = attrs

        if labels_path.endswith(".parquet"):
            df = pd.read_parquet(labels_path)
        else:
            df = pd.read_csv(labels_path)

        assert "Filename" in df.columns, "No encuentro columna Filename"
        self.filenames = df["Filename"].astype(str).tolist()
        self.conds = df[attrs].astype(np.float32).values

        # v2 transforma más rápido y evita el cuello de PIL
        self.tf = v2.Compose([
            v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.ToDtype(torch.float32, scale=True), # Pasa a [0, 1]
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Pasa a [-1, 1]
        ])

        self._zip = None

    def _get_zip(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, "r")
        return self._zip

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        z = self._get_zip()
        rel = self.filenames[idx].replace("\\", "/")
        zname = f"{self.zip_prefix}{rel}"

        try:
            raw = z.read(zname)
        except KeyError:
            raw = z.read(rel)

        # decodificación nativa a tensor (B, C, H, W) mucho más rápida que PIL
        tensor_img = decode_image(torch.frombuffer(bytearray(raw), dtype=torch.uint8), mode=ImageReadMode.RGB)

        x = self.tf(tensor_img)
        c = torch.from_numpy(self.conds[idx])
        return x, c
    

def denorm(x):
    return (x + 1) / 2

def show_samples(ds, attrs, n=8, seed=0):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=n, replace=False)

    plt.figure(figsize=(14, 6))
    for i, idx in enumerate(idxs, start=1):
        x, c = ds[idx]
        x_show = denorm(x).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

        # atributos activos
        c_np = c.cpu().numpy()
        active = [attrs[j] for j, v in enumerate(c_np) if v > 0.5]
        title = f"idx={idx}\n" + (", ".join(active) if active else "(no positive attrs)")

        plt.subplot(2, (n + 1)//2, i)
        plt.imshow(x_show)
        plt.axis("off")
        plt.title(title, fontsize=9)

    plt.tight_layout()
    plt.show()
