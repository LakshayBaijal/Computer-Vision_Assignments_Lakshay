import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate():
    # Load your generated results
    with open("ablation_mlp_results.json", "r") as f:
        results = json.load(f)
    
    # Load the ground truth from dataset_coco.json
    with open("./data/dataset_coco.json", "r") as f:
        gt_data = json.load(f)['images']
    
    # Map cocoid to its ground truth captions
    gt_dict = {img['cocoid']: [cap['raw'] for cap in img['sentences']] for img in gt_data}
    
    bleu_scores = []
    smooth = SmoothingFunction().method1
    
    for res in results:
        img_id = res['image_id']
        if img_id in gt_dict:
            reference = [cap.split() for cap in gt_dict[img_id]]
            candidate = res['caption'].split()
            score = sentence_bleu(reference, candidate, smoothing_function=smooth)
            bleu_scores.append(score)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    print(f"\n--- FINAL METRICS ---")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Total Samples Evaluated: {len(bleu_scores)}")

if __name__ == "__main__":
    calculate()