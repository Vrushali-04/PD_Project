import os
import shutil
import random

def copy_images(source_dir, target_dir, max_images=None):
    os.makedirs(target_dir, exist_ok=True)

    images = [
        img for img in os.listdir(source_dir)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not images:
        print(f"❌ No images found in {source_dir}")
        return

    if max_images:
        images = random.sample(images, min(max_images, len(images)))

    for img in images:
        shutil.copy(
            os.path.join(source_dir, img),
            os.path.join(target_dir, img)
        )

    print(f"✅ Copied {len(images)} images → {target_dir}")


# ================================
# YOUR EXISTING DATA (NO RENAME)
# ================================
train_source = "datasets/images/train/healthy"
test_source  = "datasets/images/test/healthy"

# ================================
# NEW scan_type STRUCTURE
# ================================
train_target = "datasets/scan_type/train/non_brain"
test_target  = "datasets/scan_type/test/non_brain"

copy_images(train_source, train_target)
copy_images(test_source, test_target)
