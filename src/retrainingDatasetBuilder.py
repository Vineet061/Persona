import os,shutil,time,yaml,random
from ultralytics import YOLO

from src.classification import *
from src.security import validate_url

import os
import shutil
import random
import json
import yaml
from PIL import Image
from datetime import datetime
import requests


def save_img(url):
    if os.path.exists(url):
        return url
    if not validate_url(url):
        raise ValueError(f"Refused to fetch unsafe/invalid URL: {url}")
    imgName = datetime.now()
    dt_string = imgName.strftime("%Y_%m_%d_%H_%M_%S")

    with open(dt_string + '.jpg', 'wb') as handle:
        response = requests.get(url, stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)
    return dt_string + '.jpg'


def _normalize_annotations(annotations):
    if annotations is None:
        return []

    if isinstance(annotations, str):
        try:
            annotations = json.loads(annotations)
        except (TypeError, json.JSONDecodeError):
            return []

    if isinstance(annotations, dict):
        return [annotations]

    if isinstance(annotations, (list, tuple)):
        return [ann for ann in annotations if isinstance(ann, dict)]

    return []



def create_retraining_dataset(image_path, annotations, main_dataset_path, output_dataset="retrain_dataset"):
    image_path = save_img(image_path)
    annotations = _normalize_annotations(annotations)

    for split in ["train", "valid", "test"]:
        os.makedirs(
            os.path.join(output_dataset, split, "images"),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(output_dataset, split, "labels"),
            exist_ok=True
        )


    yaml_path = os.path.join(main_dataset_path, "data.yaml")

    with open(yaml_path) as f:
        data_yaml = yaml.safe_load(f)

    class_names = data_yaml["names"]

    class_map = {
        name: idx
        for idx, name in enumerate(class_names)
    }

    image_name = os.path.basename(image_path)

    shutil.copy(
        image_path,
        os.path.join(
            output_dataset,
            "train/images",
            image_name
        )
    )

    img = Image.open(image_path)

    w, h = img.size

    label_file = os.path.join(
        output_dataset,
        "train/labels",
        image_name.rsplit(".", 1)[0] + ".txt"
    )

    with open(label_file, "w") as f:
        for ann in annotations:
            label = ann.get("label")
            if label is None:
                continue

            if label not in class_map:
                raise ValueError(f"Unknown label '{label}'. Available labels: {class_names}")

            cls = class_map[label]

            x1 = ann.get("x1")
            y1 = ann.get("y1")
            x2 = ann.get("x2")
            y2 = ann.get("y2")

            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h

            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            f.write(
                f"{cls} "
                f"{x_center} "
                f"{y_center} "
                f"{bw} "
                f"{bh}\n"
            )

    for split in ["train", "valid", "test"]:

        img_dir = os.path.join(
            main_dataset_path,
            split,
            "images")

        lbl_dir = os.path.join(
            main_dataset_path,
            split,
            "labels")

        imgs = os.listdir(img_dir)

        sample_count = min(
            20,
            len(imgs))

        selected = random.sample(
            imgs,
            sample_count)

        for img_file in selected:

            label_file = (
                os.path.splitext(img_file)[0]
                + ".txt")

            shutil.copy(
                os.path.join(img_dir, img_file),
                os.path.join(
                    output_dataset,
                    split,
                    "images"
                )
            )

            shutil.copy(
                os.path.join(lbl_dir, label_file),
                os.path.join(
                    output_dataset,
                    split,
                    "labels"
                )
            )

   
    new_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(class_names),
        "names": class_names}

    with open(
        os.path.join(output_dataset, "data.yaml"),
        "w"
    ) as f:
        yaml.dump(
            new_yaml,
            f,
            sort_keys=False)
    

        model_loader = ModelLoader()
        model = model_loader.modelCurrentDetection()
        print(model,"This is the model")

        model.train(data="retrain_dataset"+'/data.yaml',
                    epochs=4,
                    batch=16,                  # number of images in a batch
                    imgsz=640,                 # image size
                    # device=0,                 # GPU number
                    patience = 6,             # for early stopping, initializing the patience here with 10
                    save = True,               # used for saving the training checkpoints
                    # pretrained = True,         # used to resume training from previously trained model
                    optimizer = 'auto',        # according to the model conf, it detects the optimizer
                    cache=True)

    # model.save(updatedWeights, save_format='h5')
    # upload_file_to_s3("models/updated.weights.h5", "bxscore", "bx-models/doc_verification")
    return("Retraining dataset created.")




