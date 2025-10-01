
---

## 🧑‍💻 Q1 — Vision Transformer on CIFAR-10

### 🔬 Approach
- Images are split into patches → embedded → passed through transformer blocks.  
- Self-attention captures **global relationships** across patches.  
- Classification is performed using the `[CLS]` token representation.  

### ⚙️ Best Model Configuration
| Parameter      | Value |
|----------------|-------|
| Embedding dim  | 256   |
| Depth (layers) | 10    |
| Heads          | 8     |
| Patch size     | 4     |
| Optimizer      | AdamW (lr=3e-4, wd=0.05) |
| Scheduler      | Cosine LR with warmup |
| Epochs         | 80    |

### 📊 Results
- ✅ **~85% test accuracy** on CIFAR-10.  
- 📈 With larger models (ViT-L/H) and longer training, accuracy can exceed **90%**.  

---

## 🎨 Q2 — Text-driven Image Segmentation (CLIPSeg + SAM)

### 🔬 Pipeline
1. **CLIPSeg (HuggingFace):** Generates a coarse segmentation mask from a text prompt.  
2. **Seed Extraction:** High-confidence regions are converted to seed points.  
3. **SAM (Segment Anything, Meta):** Refines segmentation to produce high-quality masks.  
4. **Visualization:** Final mask is overlaid on the image and saved.  

### 🖼️ Example
- Input: `dog.jpg`  
- Prompt: `"dog"`  
- Output: `final_mask.png`  

✨ The result is a **refined segmentation mask** corresponding to the text prompt.

---

## ⚠️ Limitations & Future Work
- **Q1 (ViT):**  
  - Limited compute → trained medium-scale ViT.  
  - 🚀 Future: MixUp, CutMix, longer training, and pretrained ViT models for higher accuracy.  

- **Q2 (Segmentation):**  
  - CLIPSeg sometimes fails on vague prompts.  
  - 🚀 Future: integrate **GroundingDINO** (text-to-box) + SAM for robust results.  
  - SAM checkpoints are large (~2GB for ViT-H).  

---

## 💻 How to Run in Colab

### 1️⃣ Open in Colab
- Go to [Google Colab](https://colab.research.google.com/).  
- Upload `q1.ipynb` or `q2.ipynb`.  
- Enable GPU:  
  `Runtime → Change runtime type → Hardware accelerator → GPU → Save`.

### 2️⃣ Run Q1 (Vision Transformer)
- Run cells top-to-bottom.  
- CIFAR-10 dataset auto-downloads.  
- Best checkpoint saved as: **`best_vit_cifar10.pth`**.

### 3️⃣ Run Q2 (Segmentation)
- Run cells top-to-bottom.  
- Upload an input image (e.g., `dog.jpg`).  
- Enter a text prompt (e.g., `"dog"`).  
- Output mask saved as **`final_mask.png`**.

---

## 📊 Results Summary

| Task | Model / Method | Dataset | Result |
|------|----------------|---------|--------|
| Q1   | Vision Transformer (from scratch) | CIFAR-10 | ~85% accuracy |
| Q2   | CLIPSeg + SAM | Custom images | High-quality text-driven segmentation |

---

## 🛠 Requirements
All dependencies are installed automatically in the notebooks.  

**Key Libraries:**  
- PyTorch, Torchvision, Timm  
- Transformers (HuggingFace)  
- Segment Anything (Meta AI)  
- OpenCV, Matplotlib, Pillow, Einops  

---

## 📌 Credits
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)  
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)  
- [CLIPSeg (HuggingFace)](https://huggingface.co/CIDAS/clipseg-rd64-refined)  
- [Segment Anything (Meta AI)](https://github.com/facebookresearch/segment-anything)  

---

## ✨ Final Note
This assignment demonstrates two complementary AI skills:  
- Building **models from scratch** (Q1, Vision Transformer).  
- Combining **pretrained state-of-the-art models** into a creative pipeline (Q2, CLIPSeg + SAM).  

📈 Together, they highlight the power of both **theory + practice** in modern AI.  
