import tensorflow as tf
import random
from pathlib import Path
from typing import Optional, Dict

import torch
from PIL import Image, ImageDraw





def generate_anchor_boxes(h, w, aspect_ratio, scale):
    """
    :param aspect_ratio: list of height of the box divided by w of the box
    :param scale: list of normalized scales os the boxes
    :return:
    """
    number_of_boxes = len(aspect_ratio) + len(scale) - 1
    t = []
    for j, s in enumerate(scale):
        if j == 0:
            for ar in aspect_ratio:
                t.append(create_anchor_box(ar, s))
            continue
        t.append(create_anchor_box(aspect_ratio[0], s))

    t = tf.stack(t, axis=0)

    t_expanded = tf.expand_dims(tf.expand_dims(t, axis=0), axis=0)
    t_out = tf.tile(t_expanded, multiples=[h, w, 1, 1])
    return t_out


def create_anchor_box(ar, s):
    h, w = s ** 0.5, s ** 0.5
    h = h * ar ** 0.5
    w = w / ar ** 0.5
    return tf.constant([0.5, 0.5, h, w])



def annotate_image(
        img: Image.Image,
        targets,
        labels,
        scores,
        normalized: bool = True,
        save_path: Optional[str] = None,
        file_name: Optional[str] = None,
        class_names: Optional[Dict[int, str]] = None,
) -> Image.Image:
    img_copy = img.copy()
    width, height = img_copy.size
    draw = ImageDraw.Draw(img_copy)

    box_width = max(1, int(min(width,
                               height) * 0.005))  # 0.5% of smaller dimension
    font_size = max(10,
                    int(min(width, height) * 0.03))  # 3% of smaller dimension

    palette = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "yellow", "lime", "pink",
    ]
    class_to_color = {}

    for box, label, conf in zip(targets, labels, scores):
        label_int = int(label.numpy().item())
        if label_int not in class_to_color:
            class_to_color[label_int] = (
                palette[label_int % len(palette)]
                if label_int < len(palette)
                else "#{:06x}".format(random.randint(0, 0xFFFFFF))
            )

        x1, y1, x2, y2, *_ = box.numpy()
        if normalized:
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height

        color = class_to_color[label_int]
        label_text = (
            class_names[label_int]
            if class_names and label_int in class_names
            else str(label_int)
        )

        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        draw.text((x1 - font_size * 2, max(0, y1 - 10)), f"{conf.numpy().item():.2f}", font_size=font_size)

    # draw.text((x1, max(0, y1 - 10)), label_text, fill=color, font_size=font_size)

    if save_path and file_name:
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        full_path = path / file_name
        print("Saving annotated image to:", full_path)
        img_copy.save(full_path)

    return img_copy
