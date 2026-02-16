# 🧠 AI Diabetic Retinopathy Detection — Backend

Deep learning based backend API for detecting Diabetic Retinopathy stages from retinal fundus images.

---

## 🚀 Live API

HuggingFace Space Deployment:

👉 https://ashutoh12-retinopathy-backend.hf.space

---

## 📌 Model Details

- Architecture: **DenseNet121**
- Framework: TensorFlow / Keras
- Input Size: 224x224 RGB
- Output Classes:
  - No_DR
  - Mild
  - Moderate
  - Severe
  - Proliferative_DR

---

## 🏗 Architecture

```
Client (Frontend - React)
        ↓
Axios POST (Image)
        ↓
Flask API
        ↓
Preprocessing
        ↓
DenseNet Model
        ↓
Prediction + Confidence
        ↓
JSON Response
```

---

## 📂 Project Structure

```
backend/
│
├── app.py              # Flask API
├── train.py            # Model training script
├── check_model.py      # Model validation
├── split_dataset.py    # Dataset splitting
├── requirements.txt    # Dependencies
└── model/
    └── model.h5        # Trained DenseNet model
```

---

## 🧪 API Endpoint

### POST `/predict`

### Request:
FormData:
```
file: image.jpg
```

### Response:
```json
{
  "prediction": "Moderate",
  "confidence": 87.45
}
```

---

## 🛠 Local Setup

```bash
git clone https://github.com/Ashu777767/retinopathy-backend.git
cd retinopathy-backend

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Server runs at:

```
http://localhost:5000
```

---



---

## 👨‍💻 Author

Ashutosh Kumar Jha  
B.E Computer Science
