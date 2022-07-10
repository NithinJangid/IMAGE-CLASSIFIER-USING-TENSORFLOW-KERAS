import argparse
import os
import shutil
import cv2
from tqdm import tqdm

arg_parse = argparse.ArgumentParser()

arg_parse.add_argument("--data_dir", required=True,
                       help="Specify dataset path containing class folders", default=None, type=str)

arg_parse.add_argument("--out_train_dir", required=False,
                       help="Specify output train dir", default='data/training_data/train', type=str)

arg_parse.add_argument("--out_val_dir", required=False,
                       help="Specify output val dir", default='data/training_data/val', type=str)

args = arg_parse.parse_args()

data_dir = args.data_dir
out_train_dir = args.out_train_dir
out_val_dir = args.out_val_dir

for folder in os.listdir(data_dir):
    print(f"Processing on folder -> {folder}")

    to_train_dir = os.path.join(out_train_dir, folder)
    to_val_dir = os.path.join(out_val_dir, folder)
    os.makedirs(to_train_dir, exist_ok=True)
    os.makedirs(to_val_dir, exist_ok=True)

    split_val = len(os.listdir(os.path.join(data_dir, folder))) * 0.2

    for ind, imagename in enumerate(tqdm(os.listdir(os.path.join(data_dir, folder)))):

        cv2_image = cv2.imread(os.path.join(data_dir, folder, imagename))
        cv2_image = cv2.resize(cv2_image, (256, 256))

        if ind < split_val:
            # shutil.copy(os.path.join(data_dir, folder, imagename), to_val_dir)
            cv2.imwrite(os.path.join(to_val_dir, imagename), cv2_image)
        else:
            # shutil.copy(os.path.join(data_dir, folder, imagename), to_train_dir)
            cv2.imwrite(os.path.join(to_train_dir, imagename), cv2_image)