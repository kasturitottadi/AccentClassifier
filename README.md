# 🎙️ Native Language Identification of Indian English Speakers Using HuBERT

A comprehensive AI system that identifies the native language (L1) of Indian speakers from their English speech by analyzing accent patterns.

## 📋 Project Overview

This project develops a Native Language Identification (NLI) system that classifies Indian speakers' native language (Telugu, Tamil, Malayalam, Kannada, Hindi, Gujarati) from their English speech using:
- **MFCC Features** (Traditional signal processing)
- **HuBERT Embeddings** (Self-supervised deep learning)

**Dataset:** IndicAccentDb - 8,116 audio samples from Hugging Face
**Link:** https://huggingface.co/datasets/DarshanaS/IndicAccentDb

---

## 📦 Requirements

### System Requirements
- **Platform:** Google Colab (Recommended) or Local machine with GPU
- **GPU:** T4 or better (16GB+ VRAM recommended)
- **RAM:** 12GB+ system RAM
- **Storage:** 10GB free space (for dataset and models)
- **Internet:** Required for downloading dataset and models

### Software Requirements
- **Python:** 3.8 or higher
- **CUDA:** 11.8 or higher (for GPU support)

---

## 🔧 Installation & Setup

### Option 1: Google Colab (Recommended - No Installation Needed)

**Why Colab?**
- ✅ Free GPU access (T4)
- ✅ Pre-installed libraries
- ✅ No local setup required
- ✅ Easy to share and run

**Steps:**
1. Go to https://colab.research.google.com/
2. Sign in with Google account
3. Upload notebook files (see execution steps below)

### Option 2: Local Installation

**Step 1: Install Python**
```bash
# Check Python version (must be 3.8+)
python --version

# If not installed, download from:
# https://www.python.org/downloads/
```

**Step 2: Create Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv accent_env

# Activate (Windows)
accent_env\Scripts\activate

# Activate (Linux/Mac)
source accent_env/bin/activate
```

**Step 3: Install PyTorch with CUDA**
```bash
# For CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CPU only (slower)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

**Step 4: Install Required Packages**
```bash
pip install -r requirements.txt
```

---

## 📋 Package Dependencies

### Core Dependencies (with versions)

```txt
# Deep Learning Framework
torch==2.1.0
torchaudio==2.1.0
torchcodec==0.1.0

# Transformers & NLP
transformers==4.35.0
datasets==3.0.1

# Audio Processing
librosa==0.10.1
soundfile==0.12.1

# Machine Learning
scikit-learn==1.3.2
numpy==1.24.3

# Visualization
matplotlib==3.8.2
seaborn==0.13.0

# Web Interface
gradio==4.8.0

# Utilities
tqdm==4.66.1
joblib==1.3.2
pandas==2.1.3
```

### Installation Commands

**All at once:**
```bash
pip install torch==2.1.0 torchaudio==2.1.0 torchcodec==0.1.0 transformers==4.35.0 datasets==3.0.1 librosa==0.10.1 soundfile==0.12.1 scikit-learn==1.3.2 numpy==1.24.3 matplotlib==3.8.2 seaborn==0.13.0 gradio==4.8.0 tqdm==4.66.1 joblib==1.3.2 pandas==2.1.3
```

**Or use requirements.txt:**
```bash
pip install -r requirements.txt
```

---

## 📁 Project Files

### Training Notebooks
| File | Purpose | Runtime | Output |
|------|---------|---------|--------|
| `Untitled3.ipynb` | Train MFCC model | ~50 min | `mfcc_best_model.pt` |
| `HuBERT_Training.ipynb` | Train HuBERT model | ~70 min | `hubert_best_model.pt` |
| `MFCC_Retrain_Fixed.ipynb` | Backup MFCC training | ~50 min | `mfcc_best_model_fixed.pt` |

### Analysis Notebooks
| File | Purpose | Runtime | Output |
|------|---------|---------|--------|
| `Layer_Analysis.ipynb` | HuBERT layer analysis | ~12-15 hrs | Layer comparison plots |
| `Age_Generalization.ipynb` | Age study (theoretical) | ~3-4 hrs | Age comparison plots |
| `Word_vs_Sentence.ipynb` | Context comparison | ~4-5 hrs | Context comparison plots |

### Application
| File | Purpose | Runtime | Output |
|------|---------|---------|--------|
| `Gradio_Demo_With_Cuisine.ipynb` | Interactive demo | ~2 min | Web interface URL |

### Documentation
- `README.md` - This file
- `PROJECT_REPORT.md` - Comprehensive project report
- `requirements.txt` - Package dependencies
- `QUICK_START.md` - Quick start guide
- `EXECUTION_GUIDE.md` - Detailed instructions
- `SYSTEM_EXPLANATION.md` - Technical details
- `TROUBLESHOOTING.md` - Common issues

---

## 🚀 Step-by-Step Execution Guide

### STEP 1: Train MFCC Model (Required)

**Time:** ~50 minutes | **GPU:** Required

**1.1 Open Google Colab**
- Go to: https://colab.research.google.com/
- Sign in with your Google account

**1.2 Upload Notebook**
- Click "File" → "Upload notebook"
- Select `Untitled3.ipynb` from your computer
- Wait for upload to complete

**1.3 Enable GPU**
- Click "Runtime" → "Change runtime type"
- Select "T4 GPU" from Hardware accelerator dropdown
- Click "Save"

**1.4 Mount Google Drive**
- Run the first cell (click play button or Shift+Enter)
- Click "Connect to Google Drive" when prompted
- Allow permissions
- Your models will be saved to `/content/drive/MyDrive/IndicAccent_Project/`

**1.5 Run All Cells**
- Click "Runtime" → "Run all"
- Or run cells one by one with Shift+Enter
- Monitor progress in output

**1.6 Expected Output**
```
✅ Drive mounted successfully!
✅ Working directory: /content/drive/MyDrive/IndicAccent_Project
✅ Dependencies installed!
✅ Dataset loaded successfully!
...
Processing samples 0 → 300
100%|██████████| 300/300 [01:15<00:00, 3.98it/s]
...
Epoch 20/20 | Train Loss: 0.0170 Acc: 0.9941 | Val Loss: 0.0098 Acc: 0.9969
✅ Training complete!
   Best validation accuracy: 0.9975 (99.75%)
✅ Model saved successfully!
```

**1.7 Verify Output Files**
Check your Google Drive:
- `/MyDrive/IndicAccent_Project/mfcc_best_model.pt` (trained model)
- `/MyDrive/IndicAccent_Project/mfcc_chunks/` (feature cache)

---

### STEP 2: Train HuBERT Model (Required)

**Time:** ~70 minutes | **GPU:** Required

**2.1 Upload Notebook**
- In Google Colab, click "File" → "Upload notebook"
- Select `HuBERT_Training.ipynb`

**2.2 Enable GPU**
- Click "Runtime" → "Change runtime type"
- Select "T4 GPU"
- Click "Save"

**2.3 Run All Cells**
- Click "Runtime" → "Run all"
- The notebook will:
  - Mount Google Drive
  - Install dependencies
  - Load HuBERT model (facebook/hubert-base-ls960)
  - Extract embeddings (this takes longest)
  - Train classifier
  - Save model

**2.4 Expected Output**
```
✅ HuBERT loaded and frozen!
   Model: facebook/hubert-base-ls960
   Output dim: 768
...
Processing samples 0 → 200
100%|██████████| 200/200 [02:30<00:00, 1.33it/s]
...
Epoch 15/15 | Train Loss: 0.0213 Acc: 0.9234 | Val Loss: 0.0156 Acc: 0.8892
✅ Training complete!
   Best validation accuracy: 0.8892 (88.92%)
✅ Model saved!
```

**2.5 Verify Output Files**
- `/MyDrive/IndicAccent_Project/hubert_best_model.pt`
- `/MyDrive/IndicAccent_Project/hubert_chunks/`

---

### STEP 3: Launch Interactive Demo (Required)

**Time:** ~2 minutes | **GPU:** Optional (CPU works)

**3.1 Upload Notebook**
- Upload `Gradio_Demo_With_Cuisine.ipynb` to Colab

**3.2 Run All Cells**
- Click "Runtime" → "Run all"
- Wait for Gradio interface to launch

**3.3 Expected Output**
```
✅ MFCC model loaded
✅ HuBERT model loaded
✅ Interface created!

Running on public URL: https://xxxxx.gradio.live

✅ Demo launched with cuisine info!
```

**3.4 Use the Demo**
- Click the public URL (https://xxxxx.gradio.live)
- Opens in new tab
- Three tabs available:
  - **MFCC Model:** Fast predictions
  - **HuBERT Model:** Better accuracy
  - **Compare Both:** Side-by-side comparison

**3.5 Test the System**
1. Click "Upload Audio" or microphone icon
2. Upload an audio file (5-10 seconds of English speech)
3. Click "Predict" button
4. See results:
   - Accent prediction with confidence
   - Regional cuisine recommendations
   - Famous dishes from that region

**Example:**
```
Predicted Accent: Telugu (87.3%)

🌶️ Telugu Cuisine (Andhra Pradesh & Telangana)

Specialty: Known for spicy and tangy flavors

Famous Dishes:
  • 🍛 Hyderabadi Biryani
  • 🌶️ Gongura Pachadi
  • 🥘 Pulihora
```

---

### STEP 4: Run Analysis Notebooks (Optional)

#### 4A: Layer-wise Analysis

**Time:** ~12-15 hours | **GPU:** Required

**Purpose:** Identify which HuBERT layer best captures accent information

**Steps:**
1. Upload `Layer_Analysis.ipynb` to Colab
2. Enable GPU
3. Run all cells
4. **Warning:** This extracts features from all 13 layers - very time consuming!

**Output:**
- Layer-wise accuracy comparison
- Visualization showing best layer (typically Layer 9-11)
- Saved plot: `layer_analysis.png`

#### 4B: Age Generalization Study

**Time:** ~3-4 hours | **GPU:** Required

**Purpose:** Analyze model performance across age groups (theoretical)

**Steps:**
1. Upload `Age_Generalization.ipynb` to Colab
2. Enable GPU
3. Run all cells

**Note:** Dataset contains only adults, so this provides theoretical analysis

**Output:**
- Age comparison plots
- Performance drop analysis
- Saved plot: `age_generalization.png`

#### 4C: Word vs Sentence Comparison

**Time:** ~4-5 hours | **GPU:** Required

**Purpose:** Compare accent detection at word-level vs sentence-level

**Steps:**
1. Upload `Word_vs_Sentence.ipynb` to Colab
2. Enable GPU
3. Run all cells

**Output:**
- Accuracy comparison
- Context analysis
- Saved plot: `word_vs_sentence.png`

---

## 🎯 Quick Test (5 Minutes)

Want to test quickly without training? Use pre-trained models:

**Option 1: Use Demo Only**
1. Upload `Gradio_Demo_With_Cuisine.ipynb`
2. Modify model loading to use pre-trained models (if available)
3. Run demo

**Option 2: Test with Sample Audio**
1. Record 5-10 seconds of English speech
2. Upload to demo
3. See prediction

---

## 📊 Expected Results

### Training Results

| Model | Validation Accuracy | Training Time | Model Size |
|-------|-------------------|---------------|------------|
| MFCC | 99.75% | ~50 min | ~500 KB |
| HuBERT | 85-92% | ~70 min | ~2 MB |

### Per-Class Accuracy (MFCC)

```
Telugu:    99.4%
Tamil:     100%
Malayalam: 100%
Kannada:   100%
Hindi:     100%
Gujarati:  99.2%
```

---

## 🐛 Troubleshooting

### Common Issues

**1. GPU Not Available**
```
Error: CUDA not available
```
**Solution:**
- In Colab: Runtime → Change runtime type → Select GPU
- Restart runtime after changing

**2. Out of Memory**
```
Error: CUDA out of memory
```
**Solution:**
- Reduce batch size in training cells
- Restart runtime: Runtime → Restart runtime
- Use smaller chunk size

**3. Dataset Download Fails**
```
Error: Connection timeout
```
**Solution:**
- Check internet connection
- Retry the cell
- Dataset is large (3.2GB), may take time

**4. Module Not Found**
```
ImportError: No module named 'torchcodec'
```
**Solution:**
```python
!pip install torchcodec
```

**5. Audio Decoding Error**
```
ImportError: To support decoding audio data, please install 'torchcodec'
```
**Solution:**
```python
!pip install -q torchcodec
# Restart runtime
```

### Getting Help

- Check `TROUBLESHOOTING.md` for detailed solutions
- Check notebook outputs for error messages
- Verify all dependencies installed correctly
- Ensure GPU is enabled in Colab

---

## 📚 Additional Resources

### Documentation
- **[PROJECT_REPORT.md](PROJECT_REPORT.md)** - Complete project report with methodology and results
- **[QUICK_START.md](QUICK_START.md)** - Condensed quick start guide
- **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** - Detailed execution instructions
- **[SYSTEM_EXPLANATION.md](SYSTEM_EXPLANATION.md)** - Technical architecture details
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

### External Links
- **Dataset:** https://huggingface.co/datasets/DarshanaS/IndicAccentDb
- **HuBERT Model:** https://huggingface.co/facebook/hubert-base-ls960
- **Google Colab:** https://colab.research.google.com/
- **PyTorch:** https://pytorch.org/
- **Gradio:** https://gradio.app/

### Tutorials
- **Colab Tutorial:** https://colab.research.google.com/notebooks/intro.ipynb
- **PyTorch Tutorial:** https://pytorch.org/tutorials/
- **Hugging Face:** https://huggingface.co/docs

---

## 📞 Support

**Issues?**
- Check `TROUBLESHOOTING.md` first
- Review error messages in notebook outputs
- Verify all installation steps completed

**Contact:**
- Email: [Your Email]
- GitHub Issues: [Repository URL]/issues

---

## ✅ Checklist

Before running, ensure:
- [ ] Google Colab account created
- [ ] GPU enabled in runtime settings
- [ ] Google Drive mounted successfully
- [ ] All dependencies installed (run install cells)
- [ ] Sufficient storage in Google Drive (10GB+)
- [ ] Stable internet connection

---

## 🎓 Citation

If you use this project, please cite:

```bibtex
@project{indian-accent-detection,
  title={Native Language Identification of Indian English Speakers Using HuBERT},
  author={[Your Name]},
  year={2024},
  institution={[Your Institution]},
  url={[Repository URL]}
}
```

---

**Last Updated:** [Current Date]
**Version:** 1.0
**Status:** Ready for Submission ✅

## 🚀 Quick Start

### Prerequisites
- Google Colab account (free)
- Google Drive (for saving models)
- GPU runtime (T4 recommended)

### Step 1: Train MFCC Model (~50 min)
```bash
1. Upload Untitled3.ipynb to Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells
```
**Output:** MFCC model with 99.75% validation accuracy

### Step 2: Train HuBERT Model (~70 min)
```bash
1. Upload HuBERT_Training.ipynb to Google Colab
2. Enable GPU
3. Run all cells
```
**Output:** HuBERT model with 85-92% validation accuracy (more generalizable)

### Step 3: Launch Interactive Demo (~2 min)
```bash
1. Upload Gradio_Demo_With_Cuisine.ipynb to Google Colab
2. Run all cells
3. Get public URL to share
```
**Features:**
- Upload audio or record voice
- Real-time accent prediction
- Regional cuisine recommendations
- Compare MFCC vs HuBERT

### Step 4: Run Analyses (Optional)
```bash
# Layer-wise analysis (~12-15 hours)
Layer_Analysis.ipynb

# Age generalization study (~3-4 hours)
Age_Generalization.ipynb

# Word vs Sentence comparison (~4-5 hours)
Word_vs_Sentence.ipynb
```

## 📁 Project Structure

```
indian-accent-detection/
├── README.md                          # This file
├── PROJECT_REPORT.md                  # Comprehensive project report
├── requirements.txt                   # Python dependencies
│
├── Training Notebooks/
│   ├── Untitled3.ipynb               # MFCC model training ✅
│   ├── HuBERT_Training.ipynb         # HuBERT model training ✅
│   └── MFCC_Retrain_Fixed.ipynb      # Backup training script
│
├── Analysis Notebooks/
│   ├── Layer_Analysis.ipynb          # HuBERT layer-wise analysis
│   ├── Age_Generalization.ipynb      # Age generalization study
│   └── Word_vs_Sentence.ipynb        # Context comparison
│
├── Application/
│   └── Gradio_Demo_With_Cuisine.ipynb # Interactive demo ✅
│
└── Documentation/
    ├── QUICK_START.md                # Quick start guide
    ├── EXECUTION_GUIDE.md            # Detailed instructions
    ├── SYSTEM_EXPLANATION.md         # Technical details
    └── TROUBLESHOOTING.md            # Common issues
```

## 🎯 Results Summary

| Model | Feature | Accuracy | Generalization | Speed |
|-------|---------|----------|----------------|-------|
| **MFCC** | 80-dim MFCC | 99.75% | May overfit | Fast ⚡ |
| **HuBERT** | 768-dim embeddings | 85-92% | Better ✅ | Slower 🐢 |

### Key Findings
- ✅ **MFCC:** Excellent validation accuracy but may overfit
- ✅ **HuBERT:** More realistic and generalizable performance
- ✅ **Sentence-level:** 15-20% better than word-level
- ✅ **Layer 9-11:** Optimal HuBERT layers for accent detection
- ⚠️ **Age limitation:** Dataset contains only adults

## 🛠️ Installation

### For Google Colab (Recommended)
```python
# Run in notebook cell
!pip install -q datasets==3.0.1 transformers torch torchaudio
!pip install -q librosa soundfile scikit-learn gradio matplotlib tqdm
!pip install -q torchcodec
```

### For Local Setup
```bash
# Clone repository
git clone https://github.com/yourusername/indian-accent-detection.git
cd indian-accent-detection

# Install dependencies
pip install -r requirements.txt
```

## 📦 Dependencies

```
python>=3.8
torch>=2.0.0
torchaudio>=2.0.0
torchcodec>=0.1.0
transformers>=4.30.0
datasets>=3.0.0
librosa>=0.10.0
soundfile>=0.12.0
scikit-learn>=1.3.0
gradio>=4.0.0
matplotlib>=3.7.0
numpy>=1.24.0
tqdm>=4.65.0
```

## 🍛 Application: Accent-Aware Cuisine Recommendation

**Concept:** When a customer speaks English, the system:
1. Analyzes their accent
2. Detects probable native language
3. Recommends authentic regional cuisine

**Example:**
```
User speaks: "I would like to order some food"
↓
System detects: Malayalam accent (87%)
↓
Recommends: Kerala Fish Curry, Appam with Stew, Sadya
```

### Cuisine Mapping

| Accent | Region | Famous Dishes |
|--------|--------|---------------|
| Telugu | Andhra Pradesh | Hyderabadi Biryani, Gongura Pachadi, Pulihora |
| Tamil | Tamil Nadu | Chettinad Chicken, Dosa & Idli, Sambar |
| Malayalam | Kerala | Kerala Fish Curry, Appam with Stew, Sadya |
| Kannada | Karnataka | Bisi Bele Bath, Mysore Masala Dosa, Ragi Mudde |
| Hindi | North India | Butter Chicken, Dal Makhani, Chole Bhature |
| Gujarati | Gujarat | Dhokla, Undhiyu, Thepla |

## 📊 Model Architectures

### MFCC Classifier
```
Input (80 MFCC features)
    ↓
Dense(256) + ReLU + Dropout(0.3)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Output(6 classes)
```

### HuBERT Classifier
```
Input (768 HuBERT embeddings)
    ↓
Dense(256) + ReLU + Dropout(0.4)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Output(6 classes)
```

## 💡 Usage Tips

### For Best Results:
- ✅ Speak in **English** (the model detects accent in English speech)
- ✅ Record **5-10 seconds** of clear speech
- ✅ Use **quiet environment**
- ✅ Speak **naturally** (don't exaggerate accent)
- ✅ Use **good microphone** quality

### Common Issues:
- ❌ Speaking native language instead of English
- ❌ Too short audio (< 3 seconds)
- ❌ Background noise
- ❌ Poor microphone quality

## 🔬 Research Contributions

1. **Comparative Analysis:** MFCC vs HuBERT for Indian accent detection
2. **Layer Analysis:** Identified optimal HuBERT layers (9-11) for accent
3. **Context Study:** Demonstrated sentence-level superiority over word-level
4. **Practical Application:** Accent-aware personalization system
5. **Methodology:** Comprehensive framework for accent classification

## 📈 Performance Metrics

### Per-Class Accuracy (MFCC Model)
```
Telugu:    99.4%
Tamil:     100%
Malayalam: 100%
Kannada:   100%
Hindi:     100%
Gujarati:  99.2%
```

### Confusion Patterns
- Telugu ↔ Kannada (geographically close)
- Tamil ↔ Malayalam (language family)
- Hindi ↔ Gujarati (North Indian)

## 🚧 Limitations

1. **Age:** Dataset contains only adult speakers
2. **Recording Quality:** Trained on clean audio
3. **English Proficiency:** Assumes reasonable English fluency
4. **Regional Variations:** Limited to 6 major accents
5. **Code-Mixing:** May struggle with heavy code-mixing

## 🔮 Future Work

1. **Expand Dataset:**
   - Include child speakers
   - Add more regional accents (Bengali, Marathi, Punjabi)
   - Diverse recording conditions

2. **Model Improvements:**
   - Fine-tune HuBERT on Indian English
   - Ensemble methods (MFCC + HuBERT)
   - Attention mechanisms

3. **Applications:**
   - Real-time streaming
   - Mobile deployment
   - Multi-modal integration (audio + text)

## 📚 Documentation

- **[PROJECT_REPORT.md](PROJECT_REPORT.md)** - Comprehensive project report
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide
- **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** - Detailed execution steps
- **[SYSTEM_EXPLANATION.md](SYSTEM_EXPLANATION.md)** - Technical deep dive
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational purposes. Dataset license follows IndicAccentDb terms.

## 👥 Authors

- **[Your Name]** - [Your Email]
- **[Institution]** - [Course/Project]

## 🙏 Acknowledgments

- **Dataset:** DarshanaS/IndicAccentDb (Hugging Face)
- **HuBERT Model:** Facebook AI Research
- **Libraries:** PyTorch, Transformers, Librosa, Gradio

## 📞 Contact

- **Email:** [Your Email]
- **GitHub:** [Your GitHub]
- **Project Link:** [Repository URL]

## 📊 Citation

If you use this work, please cite:
```bibtex
@project{indian-accent-detection,
  title={Native Language Identification of Indian English Speakers Using HuBERT},
  author={[Your Name]},
  year={2024},
  institution={[Your Institution]}
}
```

---

**⭐ Star this repository if you find it helpful!**

**🐛 Report issues:** [GitHub Issues](https://github.com/yourusername/indian-accent-detection/issues)

**📧 Questions?** Contact: [Your Email]
