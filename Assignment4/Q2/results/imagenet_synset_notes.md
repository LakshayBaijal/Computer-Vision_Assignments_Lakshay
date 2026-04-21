# ImageNet Synset Notes

## Label hierarchy used by ImageNet
ImageNet-1K categories are linked to WordNet noun synsets. WordNet forms a lexical-semantic hierarchy
with hypernym-hyponym relations (is-a relations), so each class belongs to a concept graph rather than
a flat independent list.

## What a synset means
A synset (synonym set) is a set of words/lemmas that represent a single concept sense in WordNet.
In ImageNet, each class index maps to one WordNet synset ID (wnid such as n02123045).

## Why synset-based grouping can be problematic for visual recognition
- Semantic similarity does not always equal visual similarity.
- Some synsets are visually broad and include high intra-class variation.
- Fine-grained neighboring synsets can be visually near-indistinguishable in unconstrained photos.

## Three visual differences inside one synset
1. Viewpoint and pose variation.
2. Scale, occlusion, and background-context variation.
3. Lighting, color appearance, and capture-domain variation (studio vs in-the-wild, motion blur, etc).
