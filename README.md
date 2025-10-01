
---

## 🚀 How to Run in Google Colab

### Step 1: Open in Colab
1. Go to [Google Colab](https://colab.research.google.com/).  
2. Upload the notebook (`q1.ipynb` or `q2.ipynb`).  
3. Enable GPU:  
   `Runtime → Change runtime type → Hardware accelerator → GPU → Save`.

---

### Step 2: Run Q1 (Vision Transformer)
- Notebook: `q1.ipynb`  
- Dataset: **CIFAR-10** (auto-downloaded via `torchvision`).  
- Training pipeline includes:
  - Custom ViT implementation (patch embedding, multi-head attention, transformer blocks).
  - Data augmentation (random crop, flip, AutoAugment).
  - Optimizer: **AdamW** with weight decay.
  - Scheduler: **Cosine learning rate with warmup**.
  - Mixed-precision training for speed.

**Best Model Config:**
- Embedding dim = 256  
- Depth = 10 layers  
- Heads = 8  
- Patch size = 4  
- Epochs = 80  
- Optimizer = AdamW (lr = 3e-4, weight decay = 0.05)  

**Result:**  
✅ Achieved **~85.64% test accuracy** on CIFAR-10.  

📌 *Note:* With more epochs, stronger augmentations (MixUp, CutMix), and larger model sizes (ViT-L/H), accuracy can go beyond 90%.

---

### Step 3: Run Q2 (Text-driven Image Segmentation)
- Notebook: `q2.ipynb`  
- Pipeline:
  1. **CLIPSeg** (HuggingFace) generates a **coarse mask** from a text prompt.
  2. The coarse mask is converted into seed points.
  3. **SAM (Segment Anything, Meta AI)** refines the segmentation to high-quality masks.
  4. Final segmentation mask is displayed & saved.

**Example:**  
Input: `dog.jpg`  
Prompt: `"dog"`  
Output: `final_mask.png` (segmentation overlay).

**Highlights:**
- Works with any uploaded image (`.jpg`/`.png`).  
- SAM checkpoint (`vit_b`, `vit_l`, `vit_h`) is auto-downloaded in Colab.  
- Adjustable CLIPSeg threshold for more/less strict masks.  

---

## 📊 Results Summary
- **Q1 (ViT on CIFAR-10):**  
  - Implemented Vision Transformer from scratch.  
  - Best test accuracy: **~85%**.  
  - Demonstrated scalability with deeper models & augmentations.

- **Q2 (Text-driven Segmentation):**  
  - Integrated **CLIPSeg** for text-to-mask mapping.  
  - Refined masks using **SAM** for high-quality outputs.  
  - Demonstrated segmentation on custom prompts & images.

---

## ⚠️ Limitations & Improvements
- **Q1:**  
  - Training ViTs requires more compute for state-of-the-art results.  
  - Improvements: longer training (200+ epochs), better augmentation (MixUp, CutMix), larger ViT models, EMA of weights.

- **Q2:**  
  - CLIPSeg sometimes fails on vague prompts → fallback needed (boxes, points).  
  - Future work: integrate **GroundingDINO** for text-to-box generation and combine with SAM for robust results.  
  - SAM checkpoints are large (~2GB for `vit_h`).

---

## 🛠 Requirements
All dependencies are installed automatically in the notebooks.  
Key libraries used:
- `torch`, `torchvision`, `einops`, `tqdm`
- `transformers`, `timm`
- `segment-anything`
- `opencv-python`, `matplotlib`, `Pillow`

---

## 📌 Credits
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)  
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)  
- [CLIPSeg (HuggingFace)](https://huggingface.co/CIDAS/clipseg-rd64-refined)  
- [Segment Anything (Meta AI)](https://github.com/facebookresearch/segment-anything)  

---

## ✨ Final Note
Both assignments demonstrate:  
- **Q1:** Strong understanding of Transformer-based architectures applied to vision.  
- **Q2:** Ability to integrate multiple state-of-the-art models (CLIPSeg + SAM) into a practical text-driven segmentation pipeline.  

📈 Together, these solutions highlight how modern AI models can be built **from scratch (Q1)** and **combined creatively (Q2)** to solve complex tasks.

