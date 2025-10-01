# AIRL Internship Coding Submission

This repository contains my solutions for the AIRL coding assignment. It features a Vision Transformer (ViT) built from scratch for image classification on CIFAR-10, and a text-driven image segmentation pipeline that integrates CLIPSeg and the Segment Anything Model (SAM).

---

## 📊 Results Summary

### Q1: Vision Transformer on CIFAR-10
A custom Vision Transformer was implemented and trained on the CIFAR-10 dataset.

| Model Architecture | Best Test Accuracy |
| :--- | :---: |
| ViT (Depth: 10, Heads: 8) | **~85%** |

### Q2: Text-Driven Image Segmentation
A pipeline was created to segment objects in an image based on a text prompt, using CLIPSeg for initial mask generation and SAM for high-quality refinement. The pipeline works effectively on custom images and prompts.

---

## 🚀 How to Run in Google Colab

1.  **Open Notebooks:** Upload `q1.ipynb` and `q2.ipynb` to [Google Colab](https://colab.research.google.com/).
2.  **Enable GPU:** In the Colab menu, navigate to `Runtime` → `Change runtime type` and select `GPU` as the hardware accelerator.
3.  **Run All Cells:** Execute all cells from top to bottom in each notebook. All dependencies are installed automatically.

---

## 🛠️ Architectures and Pipelines

### Q1: Vision Transformer
The ViT implementation includes:
-   **Patch & Positional Embeddings:** Images are converted into a sequence of flattened patches with learnable positional embeddings.
-   **Transformer Encoder:** A stack of 10 Transformer blocks (Multi-Head Self-Attention + MLP) processes the sequence.
-   **Training:** The model was trained for 80 epochs using the AdamW optimizer, a cosine learning rate schedule, and mixed-precision training for efficiency.

### Q2: Text-to-Image Segmentation
The pipeline operates in two stages:
1.  **Prompt Seeding:** The **CLIPSeg** model takes an image and a text prompt (e.g., "a dog") and generates a coarse probability map of the object's location.
2.  **Mask Refinement:** Points are sampled from the coarse map and fed as prompts to the **Segment Anything Model (SAM)**, which produces a precise, high-quality final segmentation mask.

---

## ⚠️ Limitations and Future Work

* **Q1 (ViT):** Training state-of-the-art Vision Transformers is computationally intensive. Accuracy could be further improved with longer training schedules (200+ epochs), stronger augmentations like MixUp/CutMix, and larger model variants.
* **Q2 (Segmentation):** The pipeline's success depends on the initial coarse mask from CLIPSeg, which can struggle with ambiguous prompts. Future work could integrate **GroundingDINO** to generate bounding box prompts for SAM, creating a more robust system.

---

## 📚 Credits and References
- [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929)
- [Segment Anything (Meta AI)](https://github.com/facebookresearch/segment-anything)
- [CLIPSeg (HuggingFace)](https://huggingface.co/CIDAS/clipseg-rd64-refined)

