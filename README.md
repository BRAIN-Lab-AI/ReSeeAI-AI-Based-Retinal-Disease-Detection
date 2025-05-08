# ReSeeAI: Vision Reimagined — AI-Powered Retinal Disease Detection

## 🌟 Introduction: A New Dawn in Retinal Diagnostics
In a world where millions silently suffer from retinal diseases, early detection remains a key to preserving sight. **ReSeeAI** aims to change that. Inspired by the groundbreaking advancements in foundation models like [RETFound](https://www.nature.com/articles/s41586-023-06555-x), we set out to build an AI system that can detect retinal diseases from fundus images with unprecedented accuracy, transparency, and reliability. Harnessing the power of deep transfer learning, smart data strategies, and robust validation, our project reimagines what is possible in retinal disease AI detection.

---

# 📚 Project Metadata

**Team:** Shabaaz Hussain, Sheharyar Khan  
**Supervisor:** Muzammil Behzad  
**Affiliations:** KFUPM

**Reference Paper:**  
- [RETFound - Foundation Model for Retinal Imaging](https://www.nature.com/articles/s41586-023-06555-x)

**Datasets Used:**
- Fundus Dataset: [Peacein/color-fundus-eye](https://huggingface.co/datasets/Peacein/color-fundus-eye)
- OCT Dataset: [MaybeRichard/OCT-retina-classification-2017](https://huggingface.co/datasets/MaybeRichard/OCT-retina-classification-2017)

**Deliverables**
- ReeSee: [Presentation](https://reesee.my.canva.site/)
- ReeSee: [Report](https://github.com/BRAIN-Lab-AI/ReSeeAI-AI-Based-Retinal-Disease-Detection/blob/main/Final_report.pdf)

---

# 🔥 Motivation: Problem Statements

- **Incomplete Detection:** Existing CNNs struggle with complex and subtle retinal disease patterns.
- **Poor Generalization:** Models trained on small, localized datasets fail on global populations.
- **Opaque Decisions:** Lack of explainability in predictions hinders clinical trust.

# 🚪 Loopholes in Existing Systems

- **Tiny and Biased Datasets** → Risk of Overfitting
- **Shallow Models** → Insufficient feature learning
- **Limited Evaluation Metrics** → Misleading performance claims

# 🧠 Problem vs Ideation: Our Solution Journey

| Problem | Ideation | 
|:---|:---|
| Overfitting | Use balanced stratified sampling + data augmentation |
| Shallow feature capture | Fine-tune deep foundation models (ViT/RETFound) |
| Lack of robustness | Introduce full fine-tuning and adapters with careful validation |
| Data Validation    | Sourced test data from different sources uploaded to hugging face |

---

# 🛠️ Technical Walkthrough

## 🚀 Model Architecture
- Based on **ViT-Large-Patch16** backbone with global pooling.
- Customized **2-layer MLP head** replacing original classifier.
- Pretrained weights initialized from **RETFound-MAE**.
- Fine-tuning modes supported: Linear Probe | Partial FT | Full FT | Adapters.

## 📦 Data Processing Pipeline
- **Fundus Data:** Downloaded via HuggingFace clone; manually curated testing set.
- **OCT Data:** Organized clean 4-class dataset.
- **Train/Val/Test Split:**
  - Balanced per-class sampling
  - 80/20 stratified split for training/validation
  - Separate held-out test set

## ⚙️ How We Trained
```bash
# Download data
# Clone RETFound checkpoint
# Train models using train_loader and val_loader
python train.py --method full_ft --epochs 10 --batch_size 16
```

**Training Details:**
- **Loss:** CrossEntropyLoss with class weights
- **Optimizer:** Adam
- **Epochs:** 6–10
- **Batch Size:** 16
- **Learning Rate:** 3e-5
- **Weight Decay:** 1e-4

## 🎯 How to Run Inference
```python
from model import load_model, predict_image

model = load_model("best_model.pth")
prediction = predict_image(model, "path_to_local_image.jpg")
print(prediction)
```

## 🛠️ How to Setup
```bash
git clone https://github.com/yourusername/reseeai.git
cd reseeai
pip install -r requirements.txt
```

---

# 🧪 Experimental Results

| Method          | Validation Accuracy (%) | Final Train Loss |
|:----------------|:------------------------|:-----------------|
| Linear Probe    | 56.6                     | 2.27              |
| Partial Fine-Tuning | 65.4                 | 1.63              |
| **Full Fine-Tuning** | **92.0**             | 0.67              |
| Adapters        | 59.2                     | 2.18              |

## ✨ Key Improvements
- Dropping low-sample classes improved baseline from ~62% to ~87%.
- **Full fine-tuning + hyperparameter** tuning finally achieved 92% accuracy.
- Transferred best techniques to OCT dataset, achieving 94%!
- **Grad-CAM Visualizations:** Understanding "where" model looks.

---

# 🌈 Future Vision
- **Multi-Modal Learning:** Combining fundus + OCT + clinical records.
- **External Validation:** Testing across global datasets.
- **Deployments:** Web app via Gradio for public demos.

---

# 🙏 Acknowledgments
- Huge thanks to open-source contributors from HuggingFace and PyTorch!
- Special appreciation to the RETFound team for releasing such a transformational foundation model.
- Heartfelt gratitude to Dr. Muzammil for mentorship.

---

# With ❤️ to Deep Learning!

