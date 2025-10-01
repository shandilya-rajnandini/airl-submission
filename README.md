# Vision Transformer & Text-Driven Segmentation

<p align="center">
  <em>Solutions for the AIRL Internship Coding Assignment, showcasing a from-scratch Vision Transformer and a state-of-the-art text-to-image segmentation pipeline.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.x-yellow.svg" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/Made%20with-Colab-lightgrey.svg" alt="Google Colab">
</p>

---

## 🚀 Project Showcase

This project tackles two core deep learning challenges. **Part 1** involves building a **Vision Transformer (ViT)** from the ground up to achieve high accuracy on CIFAR-10. **Part 2** demonstrates a robust pipeline for **text-driven image segmentation** by integrating the CLIPSeg and Segment Anything (SAM) models.

<p align="center">
  <img src="link_to_your_final_dog_image.jpg" alt="Segmentation Demo" width="600"/>
</p>

---

## ✨ Key Features

### Q1: Vision Transformer (ViT)
- **Scratch Implementation:** ViT architecture built using PyTorch, including patch embeddings, multi-head self-attention, and transformer encoder blocks.
- **Modern Training:** Trained using AdamW optimizer, a cosine annealing learning rate scheduler, and mixed-precision for efficiency.
- **High Performance:** Achieved **~85% test accuracy** on the CIFAR-10 dataset.

### Q2: Text-Driven Segmentation
- **Zero-Shot Pipeline:** Segment any object in an image using only a text prompt, without needing any model retraining.
- **Two-Stage Refinement:** Uses **CLIPSeg** to generate a coarse mask from text, which is then refined into a high-fidelity mask by the **Segment Anything Model (SAM)**.
- **Interactive & Flexible:** Works on any user-uploaded image and allows for easy adjustment of model parameters.

---

## 📊 Results

| Model | Task | Dataset | Best Accuracy |
| :--- | :--- | :---: | :---: |
| **Vision Transformer** | Image Classification | CIFAR-10 | **~85%** |

---

## 🛠️ Tech Stack

- **Core:** Python, PyTorch
- **Models & Libraries:** Hugging Face Transformers, TIMM, `segment-anything` (Meta AI)
- **Tools:** OpenCV, Supervision, Matplotlib, NumPy
- **Environment:** Google Colab (GPU required)

---

## ⚙️ Setup and Usage

1.  **Clone or Download:** Get the `q1.ipynb` and `q2.ipynb` files.
2.  **Open in Colab:** Upload the notebooks to [Google Colab](https://colab.research.google.com/).
3.  **Enable GPU:** In the menu, go to `Runtime` → `Change runtime type` and select `GPU`.
4.  **Run All:** Execute all cells in the notebooks. All dependencies are installed automatically.

---
