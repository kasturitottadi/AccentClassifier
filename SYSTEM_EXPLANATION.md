# EchoAccent System Explanation

## 🎯 Project Overview
EchoAccent is an Indian accent classification system that predicts which of 6 Indian accents a speaker has based on their audio.

## 📊 Dataset
- **Source**: DarshanaS/IndicAccentDb (Hugging Face)
- **Size**: 8116 audio samples
- **Classes**: Telugu, Tamil, Malayalam, Kannada, Hindi, Gujarati
- **Preprocessing**: Shuffled with seed=42, resampled to 16kHz

## 🔧 Feature Extraction

### 1. MFCC Features (80-dimensional)
- Extract 40 MFCCs from audio
- Compute mean across time → 40 values
- Compute std across time → 40 values
- Concatenate → 80-dim feature vector
- **Why MFCC?** Captures spectral characteristics of speech

### 2. HuBERT Embeddings (768-dimensional)
- Use facebook/hubert-base-ls960 pretrained model
- Convert raw waveform → hidden states
- Mean pooling over time → 768-dim embedding
- **Why HuBERT?** Deep learned representations capture accent nuances

## 🧠 Model Architectures

### MFCC Classifier
```
Input: 80 features
  ↓
Dense(256) + ReLU + Dropout(0.3)
  ↓
Dense(128) + ReLU + Dropout(0.3)
  ↓
Dense(6) → Output logits
```

### HuBERT Classifier
```
Input: 768 embeddings
  ↓
Dense(256) + ReLU + Dropout(0.3)
  ↓
Dense(128) + ReLU + Dropout(0.3)
  ↓
Dense(6) → Output logits
```

## 📈 Training Strategy
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 64
- **Epochs**: 50
- **Split**: 80% train, 20% validation
- **Target**: >90% validation accuracy

## 💾 Memory Management
- **Problem**: 8116 samples can cause OOM errors
- **Solution**: Chunk-based processing
  - MFCC: 250 samples per chunk
  - HuBERT: 200 samples per chunk (more memory intensive)
- Save each chunk to disk
- Merge all chunks after extraction

## 🚀 Deployment (Gradio)
- Upload audio or record from microphone
- Select model (MFCC or HuBERT)
- Extract features on-the-fly
- Run inference
- Display:
  - Predicted accent
  - Confidence percentage
  - Probability bar chart for all 6 accents

## 🔄 Workflow

1. **Data Extraction** (Steps 1-2)
   - Download dataset from Hugging Face
   - Process in chunks to avoid memory issues
   - Extract MFCC features → save to mfcc_features_all.pkl
   - Extract HuBERT embeddings → save to hubert_features_all.pkl

2. **Model Training** (Steps 3-4)
   - Load extracted features
   - Split 80/20 train/val
   - Train neural network
   - Save best model based on validation accuracy

3. **Deployment** (Step 5)
   - Load trained models
   - Create Gradio interface
   - Real-time inference on uploaded audio

## 📁 Output Files
- `mfcc_features_all.pkl` - Extracted MFCC features
- `hubert_features_all.pkl` - Extracted HuBERT embeddings
- `mfcc_full_6class.pt` - Trained MFCC model weights
- `hubert_full_6class.pt` - Trained HuBERT model weights

## 🎓 Key Concepts

### Why Two Models?
- **MFCC**: Traditional signal processing, fast, interpretable
- **HuBERT**: Deep learning, captures complex patterns, potentially higher accuracy

### Why Chunk Processing?
- Prevents memory overflow
- Allows processing large datasets on limited hardware
- Can resume if interrupted

### Why 16kHz Sampling Rate?
- Standard for speech recognition
- Balances quality and computational efficiency
- Required by HuBERT model

## 🔍 How Prediction Works
1. User uploads audio
2. Audio resampled to 16kHz
3. Features extracted (MFCC or HuBERT)
4. Features fed to trained model
5. Model outputs 6 logits
6. Softmax converts to probabilities
7. Highest probability = predicted accent
