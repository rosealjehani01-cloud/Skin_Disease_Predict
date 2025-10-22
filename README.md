# Skin Disease Prediction App

An intelligent skin disease classification system powered by Vision Transformer (ViT) model.

## Supported Diseases

1. **Acne**
2. **Eczema**
3. **Basal Cell Carcinoma (BCC)**
4. **Melanoma**
5. **Normal Skin**
6. **Psoriasis**
7. **Seborrheic Keratosis (SK)**

## Features

✅ Clean and simple user interface  
✅ Automatic prediction on image upload  
✅ High accuracy disease classification  
✅ Professional color scheme  
✅ Instant results with confidence scores  

## Installation

### 1. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 2. File Structure

Make sure your files are organized as follows:

```
your-project/
├── skin_disease_app.py
├── requirements.txt
├── model_torchscript.pt
├── metadata.json
└── labels.txt
```

### 3. Update File Paths

Open `skin_disease_app.py` and update the paths at the top of the file:

```python
# ===== FILE PATHS - Edit these paths to match your file locations =====
MODEL_PATH = '/path/to/your/model_torchscript.pt'
METADATA_PATH = '/path/to/your/metadata.json'
# =====================================================================
```

### 4. Run the Application

```bash
streamlit run skin_disease_app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Open the app** in your browser
2. **Upload a skin image** (JPG, JPEG, or PNG)
3. **View results** automatically - disease name and confidence score

## Deployment

### Deploy on Streamlit Cloud (Free)

1. Create a GitHub account and repository
2. Upload all files to your repository
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub account
5. Select your repository and main file
6. Click "Deploy"

### Deploy on Hugging Face Spaces (Free)

1. Create account at [huggingface.co](https://huggingface.co)
2. Create a new Space with Streamlit
3. Upload all project files
4. App will deploy automatically

## Technical Requirements

- Python 3.11 or higher
- PyTorch 2.0+
- Streamlit
- 2GB RAM minimum

## Model Information

- **Architecture:** Vision Transformer (ViT)
- **Model:** vit_tiny_patch16_224
- **Input Size:** 224x224
- **Number of Classes:** 7

## Warning

This system is for guidance only and is not a substitute for professional medical diagnosis. Always consult a healthcare professional for medical advice.

## Technologies Used

- **PyTorch** - Deep Learning Framework
- **Streamlit** - Web Application Framework
- **Vision Transformer** - Image Classification Model
- **timm** - PyTorch Image Models

---

Made with ❤️ using PyTorch & Streamlit

