import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from imagenet_labels import load_imagenet_class_index

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
VIS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "visualizations"))

def main():
    os.makedirs(VIS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "comparison_cases.json")
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}. Run comparison_eval.py first.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    class_index = load_imagenet_class_index(cache_dir=RESULTS_DIR)

    for i, case in enumerate(cases):
        target_label = case["target_label"]
        image_path = case["image_path"]
        
        # Determine who won
        if case["clip_top5_hit"] and not case["rn50_top5_hit"]:
            winner = "CLIP"
        elif case["rn50_top5_hit"] and not case["clip_top5_hit"]:
            winner = "RN50"
        else:
            winner = "TIE (or both missed)"

        fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            img = Image.open(image_path).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        ax.axis("off")
        
        # Build text string for top 5
        clip_labels = [class_index[idx]["label"] for idx in case["clip_top5"]]
        rn50_labels = [class_index[idx]["label"] for idx in case["rn50_top5"]]

        text_str = f"Target: {target_label} | Winner: {winner}\n\n"
        text_str += "CLIP Top 5:\n"
        for j, l in enumerate(clip_labels):
            text_str += f"  {j+1}. {l}\n"
            
        text_str += "\nRN50 Top 5:\n"
        for j, l in enumerate(rn50_labels):
            text_str += f"  {j+1}. {l}\n"

        plt.figtext(0.95, 0.5, text_str, fontsize=12, va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        # Adjust layout to make room for text
        plt.subplots_adjust(right=0.85)

        out_name = f"case_{i:02d}_{target_label.replace(' ', '_')}_{winner}.png"
        out_path = os.path.join(VIS_DIR, out_name)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        
        print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
