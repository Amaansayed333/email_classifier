# Email Classifier API 

This project classifies emails into predefined categories using machine learning. The model is trained with TF-IDF features and a Naive Bayes classifier.

##  Features

- Text preprocessing with `scikit-learn` and `re`
- Email classification using `MultinomialNB`
- Deployment-ready using FastAPI on Hugging Face Spaces
- `/classify` API endpoint that accepts JSON input

## Folder Structure
.
├── main.py # FastAPI app with /classify endpoint
├── models.py # Model loading and prediction
├── utils.py # Text cleaning and utility functions
├── naive_bayes_model.pkl # Trained classifier
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── label_encoder.pkl # Label encoder
├── requirements.txt
├── .gitignore

##  Model

- **Algorithm:** Naive Bayes
- **Vectorizer:** TF-IDF with max 5000 features
- **Target Labels:** Encoded using `LabelEncoder`

## 🔧 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/email-classifier-api.git
cd email-classifier-api
```

python -m venv myenv
#on windows
 myenv\Scripts\activate  
 # or on mac 
 source env/bin/activate

 pip install -r requirements.txt

 Deploy on Hugging Face

    Upload all .py files, .pkl files, and requirements.txt to your Hugging Face Space.

    Set the SDK to FastAPI

    App will be hosted at:

https://<username>-<space-name>.hf.space/classify

#example post request
import requests

url = "https://<username>-<space-name>.hf.space/classify"
data = {"email": "Please review the financial report by EOD."}
res = requests.post(url, json=data)
print(res.json())
