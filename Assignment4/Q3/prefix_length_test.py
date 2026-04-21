import torch
import torch.optim as optim
import transformers
transformers.AdamW = optim.AdamW
import os
import clip
from PIL import Image
from transformers import GPT2Tokenizer
from model_factory import load_clipcap

IMAGE_DIR = "./data/val2014/"
K_VALUES = [1, 5, 10, 20, 40]

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Get one single test image to keep it fast
    test_img = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')][0]
    img_path = os.path.join(IMAGE_DIR, test_img)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prefix_raw = clip_model.encode_image(image).float()

        for k in K_VALUES:
            print(f"\n--- Testing k={k} ---")
            # PASSING K HERE IS CRITICAL
            model, _ = load_clipcap(use_mlp=True, checkpoint_path=None, prefix_length=k)
            model.eval()

            tokens = torch.tensor([tokenizer.eos_token_id]).unsqueeze(0).to(device)
            for _ in range(5): # Just 5 tokens to prove it works
                outputs = model(tokens, prefix_raw)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
                tokens = torch.cat((tokens, next_token), dim=1)
            
            print(f"Result for k={k}: {tokenizer.decode(tokens[0])}")

if __name__ == "__main__":
    test()