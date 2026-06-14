import os
import h5py
import argparse
import numpy as np


DEFAULT_LABEL_NAMES = [
    "ADI",   # adipose
    "BACK",  # background
    "DEB",   # debris
    "LYM",   # lymphocytes
    "MUC",   # mucus
    "MUS",   # smooth muscle
    "NORM",  # normal colon mucosa
    "STR",   # cancer-associated stroma
    "TUM",   # colorectal adenocarcinoma epithelium
]


def decode_label_names(x):
    names = []
    for v in x:
        if isinstance(v, bytes):
            names.append(v.decode("utf-8"))
        else:
            names.append(str(v))
    return names


def check_one_h5(h5_path):
    print("=" * 80)
    print("file:", h5_path)

    with h5py.File(h5_path, "r") as f:
        print("keys:", list(f.keys()))

        coords = f["coords"][:] if "coords" in f else None
        prob = f["prob"][:] if "prob" in f else None
        pred = f["pred"][:] if "pred" in f else None

        if "label_names" in f:
            label_names = decode_label_names(f["label_names"][:])
        else:
            label_names = DEFAULT_LABEL_NAMES

    if coords is not None:
        print("coords shape:", coords.shape)

    if prob is not None:
        print("prob shape:", prob.shape)
        print("prob min/max:", float(prob.min()), float(prob.max()))

        if pred is None:
            pred = np.argmax(prob, axis=1)

    if pred is None:
        print("No pred/prob found.")
        return

    pred = pred.astype(int)
    num_classes = int(pred.max()) + 1

    if len(label_names) < num_classes:
        label_names = label_names + [f"class_{i}" for i in range(len(label_names), num_classes)]

    total = len(pred)
    print("total patches:", total)
    print("-" * 80)
    print(f"{'idx':<6}{'label':<12}{'count':<12}{'ratio':<12}")

    for i in range(max(num_classes, len(label_names))):
        count = int(np.sum(pred == i))
        ratio = count / total if total > 0 else 0
        name = label_names[i] if i < len(label_names) else f"class_{i}"
        print(f"{i:<6}{name:<12}{count:<12}{ratio:.4f}")

    print("-" * 80)

    if prob is not None:
        mean_prob = prob.mean(axis=0)
        print("mean probability per class:")
        for i, p in enumerate(mean_prob):
            name = label_names[i] if i < len(label_names) else f"class_{i}"
            print(f"{i:<6}{name:<12}{p:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='/yzyStorage/public/Feature_hy/Feature_ALL_univ1/output/overlap_overlap0_batchsize224_univ2/20.0x_224px_0px_overlap/patch_cls_h5', help="patch_cls.h5 file or directory")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        h5_files = sorted([
            os.path.join(args.path, f)
            for f in os.listdir(args.path)
            if f.endswith(".h5")
        ])
    else:
        h5_files = [args.path]

    for h5_path in h5_files:
        check_one_h5(h5_path)


if __name__ == "__main__":
    main()