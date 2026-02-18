# 🎨 Studio AI: End-to-End Stable Diffusion App

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YOUR_USERNAME]/[YOUR_REPO_NAME]/blob/main/notebooks/research_nb.ipynb)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)

**Studio AI** is a full-stack Generative AI web application built with **Streamlit** that leverages **Stable Diffusion v1.5** for high-fidelity image generation and manipulation.

This project demonstrates end-to-end AI engineering capabilities, focusing on inference pipeline optimization, GPU memory management, and interactive UI/UX for advanced image editing tasks like Inpainting and Outpainting.

---

## ✨ Key Features

Beyond standard text-to-image generation, this application includes advanced computer vision workflows:

* **🎨 Text-to-Image Generation**: Generate high-quality images from natural language prompts using the Stable Diffusion v1.5 model.
* **🛠️ Inpainting (Object Editing)**: Interactively modify specific areas of an image using a brush tool and text prompts (e.g., changing a dog to a cat while keeping the background intact).
* **🔍 Outpainting (Canvas Expansion)**: Extends the boundaries of an image (Zoom Out effect) while maintaining context consistency.
* **⚡ Batch Generation**: Supports batch processing to generate multiple image variations in a single run.
* **⚙️ Advanced Inference Controls**: Full control over generation parameters including *Inference Steps*, *Guidance Scale (CFG)*, *Seed*, and *Schedulers* (Euler A, DPM++, DDIM).
* **🧹 VRAM Management**: Includes a manual "Flush RAM" utility to optimize GPU memory usage and prevent Out-of-Memory (OOM) errors on limited hardware (e.g., Google Colab T4).

---

## 🚀 Quick Start (Google Colab)

The easiest way to run this application is via Google Colab (Free T4 GPU supported).

1.  Click the **Open In Colab** badge at the top of this README.
2.  Ensure the Runtime type is set to **T4 GPU** (*Runtime > Change runtime type > T4 GPU*).
3.  Run all cells in the notebook.
4.  Click the **public URL** generated at the bottom of the notebook (usually ending in `.ngrok-free.app` or similar).

---

## 🛠️ Tech Stack
* **Python:** Core programming language.
* **Streamlit:** Framework for building the interactive web interface.
* **Diffusers (Hugging Face):** State-of-the-art library for diffusion models.
* **PyTorch:** Deep learning framework for tensor computation.
* **Streamlit Drawable Canvas:** DComponent for interactive masking in Inpainting mode.

## 📥 How to Run
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ariellaacahyani/stable-diffusion-streamlit
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Notebook:**
    streamlit run app.py

---
**Created by [Ariella Cahyani]**
