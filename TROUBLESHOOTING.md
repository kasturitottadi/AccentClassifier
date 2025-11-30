# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### 1. Out of Memory (OOM) Error

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `CHUNK_SIZE` in extraction scripts (try 100-150)
- Use CPU instead of GPU for extraction
- Close other programs using GPU
- Restart kernel and clear cache:
```python
import torch
torch.cuda.empty_cache()
```

### 2. Dataset Download Fails

**Symptoms:**
```
ConnectionError: Couldn't reach the Hugging Face Hub
```

**Solutions:**
- Check internet connection
- Try again (temporary server issue)
- Use VPN if blocked in your region
- Download manually and load locally

### 3. Model Not Loading

**Symptoms:**
```
RuntimeError: Error(s) in loading state_dict
```

**Solutions:**
- Ensure model architecture matches training
- Check if .pt file exists and is not corrupted
- Verify input dimensions match
- Re-train if necessary

### 4. Low Accuracy (<90%)

**Possible Causes:**
- Insufficient training epochs
- Learning rate too high/low
- Class imbalance
- Poor feature extraction

**Solutions:**
- Increase epochs to 100
- Try different learning rates (0.0001, 0.0005)
- Check class distribution:
```python
import numpy as np
unique, counts = np.unique(y_all, return_counts=True)
print(dict(zip(unique, counts)))
```
- Add data augmentation

### 5. Gradio Won't Launch

**Symptoms:**
```
OSError: [Errno 48] Address already in use
```

**Solutions:**
- Change port:
```python
demo.launch(server_port=7861)
```
- Kill existing process:
```bash
# Linux/Mac
lsof -ti:7860 | xargs kill -9

# Windows
netstat -ano | findstr :7860
taskkill /PID <PID> /F
```
- Disable sharing:
```python
demo.launch(share=False)
```

### 6. Audio File Not Recognized

**Symptoms:**
```
LibsndfileError: Error opening file
```

**Solutions:**
- Install ffmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# Mac
brew install ffmpeg

# Windows
# Download from ffmpeg.org
```
- Convert audio to WAV format
- Check file is not corrupted

### 7. HuBERT Model Download Fails

**Symptoms:**
```
OSError: Can't load tokenizer for 'facebook/hubert-base-ls960'
```

**Solutions:**
- Check internet connection
- Clear Hugging Face cache:
```python
import shutil
shutil.rmtree("~/.cache/huggingface", ignore_errors=True)
```
- Download manually:
```python
from transformers import HubertModel
model = HubertModel.from_pretrained("facebook/hubert-base-ls960", force_download=True)
```

### 8. Slow Training

**Symptoms:**
- Training takes hours instead of minutes

**Solutions:**
- Verify GPU is being used:
```python
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```
- Reduce batch size if GPU memory limited
- Use mixed precision training:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### 9. Chunk Files Corrupted

**Symptoms:**
```
EOFError: Ran out of input
```

**Solutions:**
- Delete corrupted chunk
- Re-run extraction for that chunk
- Check disk space
- Use try-except when loading:
```python
try:
    with open(chunk_path, 'rb') as f:
        data = pickle.load(f)
except:
    print(f"Skipping corrupted chunk: {chunk_path}")
```

### 10. Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'librosa'
```

**Solutions:**
- Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```
- Check Python version (3.8+ recommended)
- Use virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 🆘 Still Having Issues?

1. Check Python version: `python --version` (3.8+ required)
2. Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`
3. Verify CUDA (if using GPU): `nvidia-smi`
4. Clear all caches and restart
5. Try on Google Colab (free GPU)

## 📝 Reporting Bugs

If you encounter a new issue:
1. Note the exact error message
2. Check which step failed
3. Verify all previous steps completed
4. Check system resources (RAM, disk space)
5. Try with smaller dataset first
