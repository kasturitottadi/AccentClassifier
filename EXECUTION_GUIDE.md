# 🚀 EchoAccent Execution Guide

## Prerequisites
```bash
pip install -r requirements.txt
```

## Step-by-Step Execution

### Step 1: Extract MFCC Features
```bash
python 1_data_extraction_mfcc.py
```
**What it does:**
- Downloads IndicAccentDb dataset
- Extracts 40 MFCCs from each audio
- Computes mean + std → 80-dim features
- Processes in chunks of 250 samples
- Saves to `mfcc_features_all.pkl`

**Expected output:**
- `mfcc_chunks/` folder with chunk files
- `mfcc_features_all.pkl` (merged dataset)
- Shape: (N, 80) where N ≈ 8000

**Time:** ~30-60 minutes depending on hardware

---

### Step 2: Extract HuBERT Embeddings
```bash
python 2_data_extraction_hubert.py
```
**What it does:**
- Loads facebook/hubert-base-ls960 model
- Extracts 768-dim embeddings from audio
- Processes in chunks of 200 samples
- Saves to `hubert_features_all.pkl`

**Expected output:**
- `hubert_chunks/` folder with chunk files
- `hubert_features_all.pkl` (merged dataset)
- Shape: (N, 768)

**Time:** ~1-2 hours (GPU recommended)

---

### Step 3: Train MFCC Model
```bash
python 3_train_mfcc_model.py
```
**What it does:**
- Loads MFCC features
- Splits 80/20 train/val
- Trains 80→256→128→6 network
- Saves best model

**Expected output:**
- `mfcc_full_6class.pt` (trained model)
- Training logs showing accuracy per epoch
- Target: >90% validation accuracy

**Time:** ~5-15 minutes

---

### Step 4: Train HuBERT Model
```bash
python 4_train_hubert_model.py
```
**What it does:**
- Loads HuBERT embeddings
- Splits 80/20 train/val
- Trains 768→256→128→6 network
- Saves best model

**Expected output:**
- `hubert_full_6class.pt` (trained model)
- Training logs
- Target: >90% validation accuracy

**Time:** ~5-15 minutes

---

### Step 5: Launch Gradio App
```bash
python 5_gradio_app.py
```
**What it does:**
- Loads both trained models
- Creates web interface
- Allows audio upload/recording
- Performs real-time inference

**Expected output:**
- Local URL: http://127.0.0.1:7860
- Public URL (if share=True)

**Usage:**
1. Upload audio or record from mic
2. Select model (MFCC or HuBERT)
3. Click "Predict Accent"
4. View results + probability chart

---

## 🔄 Reloading Models in New Session

If you start a fresh Colab/Jupyter session:

```python
# Run this code
exec(open('6_reload_model_guide.py').read())
```

Or manually:
```python
import torch
import torch.nn as nn

# Define architectures (copy from training scripts)
class MFCCClassifier(nn.Module):
    # ... (same as training)

class HuBERTClassifier(nn.Module):
    # ... (same as training)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mfcc_model = MFCCClassifier().to(device)
mfcc_model.load_state_dict(torch.load("mfcc_full_6class.pt", map_location=device))
mfcc_model.eval()

hubert_model = HuBERTClassifier().to(device)
hubert_model.load_state_dict(torch.load("hubert_full_6class.pt", map_location=device))
hubert_model.eval()
```

---

## 📊 Expected Results

### MFCC Model
- Training accuracy: ~95-98%
- Validation accuracy: ~90-93%
- Inference time: <100ms per sample

### HuBERT Model
- Training accuracy: ~98-99%
- Validation accuracy: ~92-95%
- Inference time: ~200-500ms per sample

---

## 🐛 Troubleshooting

### Out of Memory Error
- Reduce CHUNK_SIZE in extraction scripts
- Use CPU instead of GPU for extraction
- Process fewer samples at a time

### Model Not Loading
- Ensure model architecture matches training
- Check file paths
- Verify .pt files exist

### Low Accuracy
- Train for more epochs
- Adjust learning rate
- Try data augmentation
- Check class balance

### Gradio Not Launching
- Check port 7860 is available
- Try different port: `demo.launch(server_port=7861)`
- Disable share: `demo.launch(share=False)`

---

## 💡 Tips

1. **Save to Google Drive** (if using Colab):
```python
from google.colab import drive
drive.mount('/content/drive')
# Save models to /content/drive/MyDrive/echoaccent/
```

2. **Monitor GPU usage**:
```python
!nvidia-smi
```

3. **Resume interrupted extraction**:
- Check which chunks already exist
- Modify start_idx in extraction scripts

4. **Test single sample**:
```python
import librosa
audio, sr = librosa.load("test.wav")
features = extract_mfcc_features(audio, sr)
# ... predict
```

---

## ✅ Success Checklist

- [ ] All dependencies installed
- [ ] Dataset downloaded successfully
- [ ] MFCC features extracted
- [ ] HuBERT embeddings extracted
- [ ] MFCC model trained (>90% val acc)
- [ ] HuBERT model trained (>90% val acc)
- [ ] Both .pt files saved
- [ ] Gradio app launches
- [ ] Predictions work correctly
- [ ] Models saved to persistent storage

---

## 🎉 You're Done!

Your EchoAccent system is now ready to classify Indian accents!
