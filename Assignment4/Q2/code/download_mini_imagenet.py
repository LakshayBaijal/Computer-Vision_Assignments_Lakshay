import os
import shutil
from bing_image_downloader import downloader

from imagenet_labels import load_imagenet_class_index
from comparison_eval import DEFAULT_CLASS_HINTS, choose_target_class_indices

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
MINI_IMAGENET_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "mini_imagenet", "val"))

def main():
    os.makedirs(MINI_IMAGENET_ROOT, exist_ok=True)
    class_index = load_imagenet_class_index(cache_dir=RESULTS_DIR)
    target_classes = choose_target_class_indices(class_index, DEFAULT_CLASS_HINTS)
    
    for class_idx in target_classes:
        synset = class_index[class_idx]["synset"]
        label = class_index[class_idx]["label"]
        print(f"\n--- Downloading images for {label} ({synset}) ---")
        
        target_dir = os.path.join(MINI_IMAGENET_ROOT, synset)
        if os.path.exists(target_dir):
            print(f"Directory {target_dir} already exists, skipping download.")
            continue
            
        downloader.download(
            f"a photo of a {label}",
            limit=4,
            output_dir=MINI_IMAGENET_ROOT,
            adult_filter_off=False,
            force_replace=False,
            timeout=60,
            verbose=False
        )
        
        # bing_image_downloader creates a folder named after the query,
        # we need to rename it to the synset ID so it matches ImageNet format
        downloaded_folder = os.path.join(MINI_IMAGENET_ROOT, f"a photo of a {label}")
        if os.path.exists(downloaded_folder):
            os.rename(downloaded_folder, target_dir)
            print(f"Renamed {downloaded_folder} to {target_dir}")

if __name__ == "__main__":
    main()
