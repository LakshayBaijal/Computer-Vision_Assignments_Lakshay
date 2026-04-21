import torch
import torch.optim as optim
import transformers
transformers.AdamW = optim.AdamW
import json
import os
from tqdm import tqdm
from PIL import Image
import clip
from transformers import GPT2Tokenizer
from model_factory import load_clipcap

# --- CONFIG ---
DATA_PATH = "./data/dataset_coco.json"
IMAGE_DIR = "./data/val2014/"
CHECKPOINT = "./checkpoints/mlp_tuned.pt"
NUM_SAMPLES = 1000 

def run_full_evaluation():
    # Load for Ablation 1 (MLP + Fine-tuned GPT)
    model, device = load_clipcap(use_mlp=True, checkpoint_path=CHECKPOINT, fine_tune_gpt=True)
    model.eval()
    
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)['images']
    test_data = [d for d in data if d['split'] == 'test'][:NUM_SAMPLES]
    
    results = []
    print(f"🚀 Running FULL 1000 image evaluation (MLP Fine-tuned)...")

    with torch.no_grad():
        for item in tqdm(test_data):
            image_path = os.path.join(IMAGE_DIR, item['filename'])
            if not os.path.exists(image_path): continue
            
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            prefix = clip_model.encode_image(image).float()

            tokens = torch.tensor([tokenizer.eos_token_id]).unsqueeze(0).to(device)
            for _ in range(20):
                outputs = model(tokens, prefix)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
                tokens = torch.cat((tokens, next_token), dim=1)
                if next_token.item() == tokenizer.eos_token_id: break
            
            results.append({"image_id": item['cocoid'], "caption": tokenizer.decode(tokens[0], skip_special_tokens=True)})

    with open("ablation_mlp_results.json", "w") as f:
        json.dump(results, f)
    print("\n✅ Results saved to ablation_mlp_results.json")

if __name__ == "__main__":
    run_full_evaluation()