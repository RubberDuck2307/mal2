import os
import os.path as osp
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


def augment_dataset(
        num_augs_per_image: int = 2,
        valid_extensions=(".jpg", ".jpeg", ".png", ".bmp"),
):
    os.makedirs("banana-detection/augmented/images", exist_ok=True)

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.4),
            A.RandomRotate90(p=0.4),
            A.Affine
            (
                scale=(0.8, 1.2),
                rotate=(-30, 30),
                p=0.5
            ),

        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,

        ),
    )

    base_path = Path("banana-detection/bananas_train/images")
    image_paths = sorted(
        [str(p) for p in base_path.iterdir() if
         p.suffix.lower() in valid_extensions]
    )

    for img_path in tqdm(image_paths, desc=f"Augmenting'"):
        img_base, img_ext = osp.splitext(osp.basename(img_path))
        lbl_path = osp.join("banana-detection/bananas_train/images/labels", img_base + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            tqdm.write(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        bboxes, class_labels = read_yolo_labels(lbl_path)

        cv2.imwrite(
            osp.join("banana-detection/augmented/images", img_base + img_ext),
            img
        )

        write_yolo_labels(
            osp.join("banana-detection/augmented/images/labels", img_base + ".txt"),
            bboxes,
            class_labels,
        )

        for i in range(num_augs_per_image):
            transformed = transform(image=img, bboxes=bboxes,
                                    class_labels=class_labels)
            aug_img = transformed["image"]
            aug_bboxes = transformed["bboxes"]
            aug_labels = transformed["class_labels"]

            if len(aug_bboxes) == 0 and len(bboxes) > 0:
                continue

            out_img_name = f"{img_base}_aug{i}{img_ext}"
            out_lbl_name = f"{img_base}_aug{i}.txt"

            cv2.imwrite(
                osp.join("banana-detection/augmented/images", out_img_name),
                aug_img)

            write_yolo_labels(
                osp.join("banana-detection/augmented/images/labels", out_lbl_name),
                aug_bboxes,
                aug_labels,
            )

    print(f"\nAugmentation complete!")


def read_yolo_labels(label_path):
    bboxes = []
    class_labels = []

    if not osp.exists(label_path):
        return bboxes, class_labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:5]))
                class_labels.append(class_id)
                bboxes.append(bbox)

    return bboxes, class_labels


def write_yolo_labels(label_path, bboxes, class_labels):
    os.makedirs(osp.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        for class_id, bbox in zip(class_labels, bboxes):
            bbox_str = " ".join(map(str, bbox))
            f.write(f"{class_id} {bbox_str}\n")


if __name__ == "__main__":
    augment_dataset(
        valid_extensions=(".jpg", ".jpeg", ".png", ".bmp"),
        num_augs_per_image=3
    )
