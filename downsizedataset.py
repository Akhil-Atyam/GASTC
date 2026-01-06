import os
import random
import shutil
from PIL import Image
import imagehash
from tqdm import tqdm

# -------- CONFIG --------
BASE_DIR = "datasets"
OUT_IMAGES = "dataset_small/images"
OUT_LABELS = "dataset_small/labels"

TOTAL_IMAGES = 500          # total final dataset size
HASH_THRESHOLD = 3          # lower = stricter duplicate removal
# ------------------------

os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)

# Step 1: Discover datasets
datasets = []
for d in os.listdir(BASE_DIR):
    img_dir = os.path.join(BASE_DIR, d, "images")
    lbl_dir = os.path.join(BASE_DIR, d, "labels")
    if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
        datasets.append((d, img_dir, lbl_dir))

if not datasets:
    raise RuntimeError("No valid datasets found.")

# Step 2: Count images per dataset
dataset_info = []
total_images_available = 0

for name, img_dir, lbl_dir in datasets:
    images = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    count = len(images)
    dataset_info.append({
        "name": name,
        "img_dir": img_dir,
        "lbl_dir": lbl_dir,
        "images": images,
        "count": count
    })
    total_images_available += count

print(f"Found {total_images_available} images across {len(dataset_info)} datasets")

# Step 3: Compute proportional targets
for d in dataset_info:
    ratio = d["count"] / total_images_available
    d["target"] = max(1, round(ratio * TOTAL_IMAGES))

# Fix rounding drift
current_total = sum(d["target"] for d in dataset_info)
while current_total != TOTAL_IMAGES:
    diff = TOTAL_IMAGES - current_total
    step = 1 if diff > 0 else -1
    dataset_info[0]["target"] += step
    current_total += step

print("\nSampling plan:")
for d in dataset_info:
    print(f"  {d['name']}: {d['target']} images")

# Step 4: Hash-based deduplication + sampling
global_hashes = []
final_pairs = []

for d in dataset_info:
    kept = []
    random.shuffle(d["images"])

    for img_name in tqdm(d["images"], desc=f"Hashing {d['name']}"):
        if len(kept) >= d["target"]:
            break

        img_path = os.path.join(d["img_dir"], img_name)
        try:
            img = Image.open(img_path)
        except Exception:
            continue

        h = imagehash.phash(img)

        if any(h - prev < HASH_THRESHOLD for prev in global_hashes):
            continue

        kept.append(img_name)
        global_hashes.append(h)

    for img_name in kept:
        base, _ = os.path.splitext(img_name)
        final_pairs.append((
            os.path.join(d["img_dir"], img_name),
            os.path.join(d["lbl_dir"], base + ".txt")
        ))

# Step 5: Copy output
for img_path, lbl_path in tqdm(final_pairs, desc="Copying files"):
    shutil.copy(img_path, OUT_IMAGES)
    if os.path.exists(lbl_path):
        shutil.copy(lbl_path, OUT_LABELS)

print("\n🎉 Done!")
print(f"Final dataset size: {len(final_pairs)} images")
print("Saved to dataset_small/")
