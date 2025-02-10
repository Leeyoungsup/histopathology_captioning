


# AI Caption Generation Model for Digital Pathology of Adenocarcinoma in Endoscopic Histopathology using Multi-Instance Attention Mechanisms

## Overview
This repository contains the implementation of the **Multi-Instance Self-Attention Captioning (MIAC) model**, developed for **digital pathology caption generation** in **gastric adenocarcinoma** histopathology images. The model leverages **multi-instance learning (MIL)** and **self-attention mechanisms** to generate descriptive diagnostic captions from endoscopic biopsy whole-slide images (WSIs).

## Key Features
- **Multi-Instance Learning (MIL):** Aggregates patch-level features to represent whole-slide images.
- **Self-Attention Mechanism:** Enhances feature extraction and ensures robust caption generation.
- **EfficientNetV2-S as Feature Extractor:** Provides efficient image representation.
- **Supports External Validation:** Evaluated on both **PatchGastricADC22** dataset and **external dataset from Gachon University Gil Medical Center**.
- **Stain Normalization (Macenko Method):** Improves model robustness across different datasets.

## Dataset
### Internal Dataset
- **PatchGastricADC22**: Publicly available dataset containing **histopathologic patch images** from **H&E-stained WSIs**.
- **Source**: [PatchGastricADC22 Dataset](https://www.kaggle.com/datasets/sanikapadegaonkar/patchgastricadc22)
- **Composition:**
  - 991 WSIs (262,777 patches)
  - Training: 800 WSIs
  - Validation: 91 WSIs
  - Test: 100 WSIs
![Figure1](https://github.com/user-attachments/assets/a9310a3e-fbd0-419a-9204-afa073fb8733)

### External Dataset
- Collected from **Gil Hospital of Gachon University**.
- Contains **105 WSIs**, each with corresponding diagnostic captions written by pathologists.
- **Ethical Approval:** GBIRB2024-121
- Preprocessed using **Macenko normalization** to align staining variations across datasets.
![Figure2](https://github.com/user-attachments/assets/cc1738d1-e656-4c21-91f7-bd5640e2eb18)
## Model Architecture
The **MIAC model** consists of two key components:
1. **Encoder:**
   - Extracts patch-level features using **EfficientNetV2-S**.
   - Uses **self-attention** for feature aggregation.
2. **Decoder:**
   - Generates diagnostic captions using **transformer-based sequence modeling**.
   - Incorporates **Punkt Tokenization** for structured sentence generation.
![Figure3](https://github.com/user-attachments/assets/0474ea39-2a1f-468e-bf61-b9e020dfbc2f)
## Performance Metrics
The model is evaluated using standard **natural language processing (NLP) metrics**:
- **BLEU@4** (Higher values indicate better n-gram overlap)
- **ROUGE-L** (Measures longest common subsequence similarity)
- **METEOR** (Considers synonyms and stemming)
- **CIDEr** (Measures term frequency–inverse document frequency similarity)

### Internal Dataset Performance (PatchGastricADC22)
| Model (Training Patches) | BLEU@4 | ROUGE-L | METEOR | CIDEr |
|--------------------------|--------|---------|--------|-------|
| **MIAC ×50 (ours)** | **0.617** | **0.731** | **0.506** | **5.588** |
| SGMT ×32 Inference ×64 | 0.551 | 0.697 | 0.432 | 4.836 |
| SGMT ×16 | 0.346 | 0.600 | 0.353 | 2.634 |

### External Dataset Performance
| Model (Training Patches) | BLEU@4 | ROUGE-L | METEOR | CIDEr |
|--------------------------|--------|---------|--------|-------|
| **MIAC ×50 (normalized)** | **0.375** | **0.577** | **0.336** | **4.382** |
| MIAC ×50 (raw) | 0.357 | 0.550 | 0.318 | 4.150 |


![Figure4](https://github.com/user-attachments/assets/e59800a7-f41c-4196-90e5-799047cce69a)
## Installation
```bash
git clone https://github.com/Leeyoungsup/histopathology_captioning.git
cd histopathology_captioning
pip install -r requirements.txt
```
## Ethical Statement
This study was approved by the **Institutional Review Board (IRB) of Gil Hospital of Gachon University (GBIRB2024-121)**. The requirement for patient consent was waived due to the retrospective nature of the study.

## Citation
If you use this code or dataset, please cite:
```bibtex
@article{lee2025miac,
  author = {Youngseop Lee, Kyungah Bai, Youngjae Kim, Jisup Kim, Kwanggi Kim},
  title = {AI Caption Generation Model for Digital Pathology of Adenocarcinoma in Endoscopic Histopathology using Multi-Instance Attention Mechanisms},
  journal = {Diagnostics},
  year = {2025}
}
```

## License
This project is licensed under the MIT License.

## Contact
For further inquiries, please contact:
- **Youngseop Lee** (Lead Developer): leeyoungsup@gachon.ac.kr
- **Kwanggi Kim** (Principal Investigator): kimkg@gachon.ac.kr
