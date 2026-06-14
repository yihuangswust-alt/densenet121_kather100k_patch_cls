import os
import h5py
import torch
import timm
import openslide
import numpy as np
from PIL import Image
from tqdm import tqdm
from safetensors.torch import load_file
from torchvision import transforms

# =========================
# paths
# =========================
PATCH_H5_DIR = "/yzyStorage/public/Feature_hy/Feature_ALL_univ1/output/overlap_overlap0_batchsize224_univ2/20.0x_224px_0px_overlap/patches"
WSI_ROOT = "/yzyStorage/public/Orion_CRC"
MODEL_PATH = "/yzyStorage/public/Feature_hy/Feature_ALL_univ1/output/overlap_overlap0_batchsize224_univ2/densenet121_kather100k/model.safetensors"
OUT_DIR = "/yzyStorage/public/Feature_hy/Feature_ALL_univ1/output/overlap_overlap0_batchsize224_univ2/20.0x_224px_0px_overlap/patch_cls_h5"

PATCH_SIZE = 224
BATCH_SIZE = 256
NUM_CLASSES = 9

LABEL_NAMES = [
    "ADI", "BACK", "DEB", "LYM", "MUC",
    "MUS", "NORM", "STR", "TUM"
]

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# model
# =========================
model = timm.create_model(
    "densenet121",
    pretrained=False,
    num_classes=NUM_CLASSES
)
weights = load_file(MODEL_PATH)
model.load_state_dict(weights, strict=True)
model = model.to(device).eval()

# ImageNet normalization, consistent with most timm DenseNet inference
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

def h5_to_wsi_path(h5_name):
    # CRC01_18459_xxx-registered_patches.h5
    # -> /yzyStorage/public/Orion_CRC/CRC01/18459_xxx-registered.ome.tif
    case_id = h5_name.split("_")[0]
    base = h5_name[len(case_id) + 1:]
    base = base.replace("_patches.h5", ".ome.tif")
    return os.path.join(WSI_ROOT, case_id, base)

def read_patch(slide, x, y):
    img = slide.read_region(
        (int(x), int(y)),
        0,
        (PATCH_SIZE, PATCH_SIZE)
    ).convert("RGB")
    return img

def classify_one_h5(h5_path):
    h5_name = os.path.basename(h5_path)
    out_name = h5_name.replace("_patches.h5", "_patch_cls.h5")
    out_path = os.path.join(OUT_DIR, out_name)

    if os.path.exists(out_path):
        print("[SKIP]", out_path)
        return

    wsi_path = h5_to_wsi_path(h5_name)

    if not os.path.exists(wsi_path):
        print("[MISS WSI]", h5_name)
        print("expected:", wsi_path)
        return

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]

    slide = openslide.OpenSlide(wsi_path)

    probs_all = []
    preds_all = []

    batch_imgs = []

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(coords, desc=h5_name)):
            img = read_patch(slide, x, y)
            batch_imgs.append(tfm(img))

            if len(batch_imgs) == BATCH_SIZE or i == len(coords) - 1:
                x_tensor = torch.stack(batch_imgs, dim=0).to(device)

                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                probs_all.append(probs.cpu().numpy().astype(np.float32))
                preds_all.append(preds.cpu().numpy().astype(np.int64))

                batch_imgs = []

    probs_all = np.concatenate(probs_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("coords", data=coords, compression="gzip")
        f.create_dataset("prob", data=probs_all, compression="gzip")
        f.create_dataset("pred", data=preds_all, compression="gzip")
        f.create_dataset("label_names", data=np.array(LABEL_NAMES, dtype="S"))

    print("[DONE]", out_path)
    print("coords:", coords.shape)
    print("prob:", probs_all.shape)
    print("pred:", preds_all.shape)

def main():
    h5_files = sorted([
        os.path.join(PATCH_H5_DIR, f)
        for f in os.listdir(PATCH_H5_DIR)
        if f.endswith("_patches.h5")
    ])

    print("total h5:", len(h5_files))

    for h5_path in h5_files:
        classify_one_h5(h5_path)

if __name__ == "__main__":
    main()