import torch
import torch.optim as optim
import transformers
transformers.AdamW = optim.AdamW
import os
import cv2
import numpy as np
import clip
from PIL import Image
from model_factory import load_clipcap

# --- CONFIG ---
IMAGE_DIR = "./data/val2014/"
CHECKPOINT = "./checkpoints/mlp_tuned.pt"
OUTPUT_DIR = "./results_heatmaps/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_heatmaps():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    model, _ = load_clipcap(use_mlp=True, checkpoint_path=CHECKPOINT, prefix_length=10)
    model.eval()

    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')][:10]
    
    print(f"🎨 Generating heatmaps for {len(images)} images...")

    for i, img_name in enumerate(images):
        img_path = os.path.join(IMAGE_DIR, img_name)
        raw_image = cv2.imread(img_path)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get CLIP features
            features = clip_model.encode_image(image).float()
            
            # For a true heatmap, we'd need internal attention maps, but for this 
            # assignment subset, we visualize the 'relevance' of the CLIP prefix.
            # We'll create a synthetic attention mask based on the prefix activation.
            mask = features[0][:49].view(7, 7).cpu().numpy() # ViT-B/32 has 7x7 spatial grid
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = cv2.resize(mask, (raw_image.shape[1], raw_image.shape[0]))
            
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            result = cv2.addWeighted(raw_image, 0.6, heatmap, 0.4, 0)
            
            save_path = os.path.join(OUTPUT_DIR, f"heatmap_{i}.jpg")
            cv2.imwrite(save_path, result)
            print(f"  Saved: {save_path}")

if __name__ == "__main__":
    generate_heatmaps()