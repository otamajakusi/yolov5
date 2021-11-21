import os
import glob
import random


def get_ext(basename, exts=["jpg", "png"]):
    for ext in exts:
        if os.path.exists(f"{basename}.{ext.upper()}"):
            return ext.upper()
        if os.path.exists(f"{basename}.{ext.lower()}"):
            return ext.lower()
    return None


def split_data(dataset_path, percentage_test=20):
    # Populate the folders
    p = percentage_test / 100

    os.makedirs("tiles/images/valid", exist_ok=True)
    os.makedirs("tiles/labels/valid", exist_ok=True)
    os.makedirs("tiles/images/train", exist_ok=True)
    os.makedirs("tiles/labels/train", exist_ok=True)
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.txt")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        img_ext = get_ext(f"{dataset_path}/{title}")
        if img_ext is None:
            print(f"WARN: image not found for {pathAndFilename}")
            continue

        if random.random() <= p:
            os.system(f"cp {dataset_path}/{title}.{img_ext} tiles/images/valid")
            os.system(f"cp {dataset_path}/{title}.txt tiles/labels/valid")
        else:
            os.system(f"cp {dataset_path}/{title}.{img_ext} tiles/images/train")
            os.system(f"cp {dataset_path}/{title}.txt tiles/labels/train")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", required=True, help="dataset path")
    args = parser.parse_args()
    split_data(args.datapath)
