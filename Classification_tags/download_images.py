import pandas as pd
import os, shutil, requests, sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split

PATH = "/datasets/train_classifier/daria/tag_classifier/dataset/"
TRAIN_DIR = "train_images"
VALID_DIR = "valid_images"

TRAIN_PATH = os.path.join(PATH, TRAIN_DIR)
VALID_PATH = os.path.join(PATH, VALID_DIR)
RANDOM = 17

def download_images_from_csv(path=None):
    CSV_PATH = path or os.path.join(PATH, "PEPSICO_PO1_PLACE-05-02-2020.csv")

    if os.path.exists(TRAIN_PATH):
        shutil.rmtree(TRAIN_PATH, ignore_errors=False, onerror=None)

    if os.path.exists(VALID_PATH):
        shutil.rmtree(VALID_PATH, ignore_errors=False, onerror=None)

    os.mkdir(TRAIN_PATH)
    os.mkdir(VALID_PATH)

    dataset_csv = pd.read_csv(CSV_PATH)

    X_train, X_valid = train_test_split(dataset_csv, test_size=0.1, stratify=dataset_csv['tag'], random_state=RANDOM)

    for dataset, dataset_path in [(X_train, TRAIN_PATH), (X_valid, VALID_PATH)]:
        dataset.to_csv(os.path.join(PATH, dataset_path.split("/")[-1] + ".csv"), index=False)

        for i in tqdm(range(len(dataset)), total=len(dataset)):
            img = dataset.values[i]
            img_url = img[0]
            label = img[1]

            r = requests.get(img_url)
            file_name = img_url.split("/")[-1]
            if r.status_code == requests.codes.ok:
                label_path = os.path.join(dataset_path, str(label))
                if not os.path.exists(label_path):
                    os.mkdir(label_path)
                out = open("{}/{}".format(label_path, file_name), "wb")
                out.write(r.content)
                out.close()

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    download_images_from_csv(path)

# USE:
# cd /code/ir-classifier-pytorch/ds_0001
# python3 ./download_images.py