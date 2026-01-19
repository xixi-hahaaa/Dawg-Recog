import os
import shutil
import random

# reproducibility
random.seed(42)

# paths
RAW_DATA_DIR = "./raw_dogs"
OUTPUT_DIR = "./dogs"

# split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# create output directories
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# loop over each breed
for breed in os.listdir(RAW_DATA_DIR):
    breed_path = os.path.join(RAW_DATA_DIR, breed)
    if not os.path.isdir(breed_path):
        continue

    images = os.listdir(breed_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    # create breed folders
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, breed), exist_ok=True)

    # copy files
    for img in train_imgs:
        shutil.copy(
            os.path.join(breed_path, img),
            os.path.join(OUTPUT_DIR, "train", breed, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(breed_path, img),
            os.path.join(OUTPUT_DIR, "val", breed, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(breed_path, img),
            os.path.join(OUTPUT_DIR, "test", breed, img)
        )

    print(f"{breed}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

print("Dataset split complete.")
