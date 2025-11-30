# 🚀 Quick Start Guide

## One-Command Setup (Recommended for Colab)

```python
# Install dependencies
!pip install datasets librosa numpy torch transformers scikit-learn gradio matplotlib tqdm soundfile

# Clone or upload project files
# Then run each step sequentially
```

## Step-by-Step Commands

### 1️⃣ Install Dependencies
```bash
python 0_install_dependencies.py
```
OR
```bash
pip install -r requirements.txt
```

### 2️⃣ Extract MFCC Features (~30-60 min)
```bash
python 1_data_extraction_mfcc.py
```

### 3️⃣ Extract HuBERT Embeddings (~1-2 hours, GPU recommended)
```bash
python 2_data_extraction_hubert.py
```

### 4️⃣ Train MFCC Model (~5-15 min)
```bash
python 3_train_mfcc_model.py
```

### 5️⃣ Train HuBERT Model (~5-15 min)
```bash
python 4_train_hubert_model.py
```

### 6️⃣ Launch Gradio App
```bash
python 5_gradio_app.py
```

## 🎯 Expected Outputs

After completion, you should have:
- ✅ `mfcc_features_all.pkl` - MFCC features
- ✅ `hubert_features_all.pkl` - HuBERT embeddings
- ✅ `mfcc_full_6class.pt` - Trained MFCC model
- ✅ `hubert_full_6class.pt` - Trained HuBERT model
- ✅ Gradio web interface running

## 💡 Pro Tips

1. **For Google Colab**: Mount Drive first
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/echoaccent
```

2. **Check GPU availability**
```python
import torch
print(torch.cuda.is_available())
```

3. **Monitor progress**: All scripts show progress bars

4. **Resume interrupted runs**: Chunks are saved incrementally

## 🆘 Need Help?
- See `EXECUTION_GUIDE.md` for detailed instructions
- See `SYSTEM_EXPLANATION.md` for how it works
- Check `TROUBLESHOOTING.md` for common issues
