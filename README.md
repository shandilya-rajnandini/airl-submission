
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
- ✅ **~85.64% test accuracy** on CIFAR-10.  
- 📈 With larger models (ViT-L/H) and longer training, accuracy can exceed **90%**.  

---

# Q2: Text-Driven Image Segmentation

<p align="center">
  <em>A robust pipeline that segments any object in an image based on a simple text prompt, using state-of-the-art models CLIPSeg and SAM.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.x-yellow.svg" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/Made%20with-Colab-lightgrey.svg" alt="Google Colab">
</p>

---

## 🚀 Project Showcase

This project demonstrates a powerful, two-stage approach to zero-shot image segmentation. First, the **CLIPSeg** model generates a coarse mask from a user's text prompt. Then, the state-of-the-art **Segment Anything Model (SAM)** refines this mask to produce a high-fidelity final output.

<p align="center">
  <img src="link_to_your_final_dog_image.jpg" alt="Segmentation Demo" width="600"/>
</p>

---

## ✨ Key Features

- **Zero-Shot Pipeline:** Segment any object in an image using only a text prompt, without needing any model retraining.
- **Two-Stage Refinement:** Uses **CLIPSeg** to generate a coarse mask from text, which is then refined into a high-fidelity mask by the **Segment Anything Model (SAM)**.
- **Interactive & Flexible:** Works on any user-uploaded image and allows for easy adjustment of model parameters.

---

## 🛠️ Tech Stack

- **Core:** Python, PyTorch
- **Models & Libraries:** Hugging Face Transformers, `segment-anything` (Meta AI)
- **Tools:** OpenCV, Supervision, Matplotlib, NumPy
- **Environment:** Google Colab (GPU required)


## ⚠️ Limitations & Future Work
- **Q1 (ViT):**  
  - Limited compute → trained medium-scale ViT.  
  - 🚀 Future: MixUp, CutMix, longer training, and pretrained ViT models for higher accuracy.  

- **Q2 (Segmentation):**  
 - The pipeline's success depends on the initial coarse mask from CLIPSeg, which can struggle with ambiguous prompts.
- Future improvements could integrate **GroundingDINO** to generate bounding box prompts for SAM, creating an even more robust system. 

---

## 💻 How to Run in Colab

### 1️⃣ Open in Colab
- Go to [Google Colab](https://colab.research.google.com/).  
- Upload `q1.ipynb` or `q2.ipynb`.  
- Enable GPU:  
  `Runtime → Change runtime type → Hardware accelerator → GPU → Save`.

### 2️⃣ Run Q1 (Vision Transformer)
1. Run cells top-to-bottom.  
2. CIFAR-10 dataset auto-downloads.  
3. Best checkpoint saved as: **`best_vit_cifar10.pth`**.

### 3️⃣ Run Q2 (Segmentation)
1.  **Open in Colab:** Upload the `q2.ipynb` notebook to [Google Colab](https://colab.research.google.com/).
2.  **Enable GPU:** In the menu, go to `Runtime` → `Change runtime type` and select `GPU`.
3.  **Run All:** Execute all cells in the notebook. All dependencies are installed automatically, and you will be prompted to upload an image.

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
