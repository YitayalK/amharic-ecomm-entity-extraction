# Amharic Named Entity Recognition (NER) for EthioMart

## Project Overview
EthioMart is building a centralized e-commerce platform that consolidates product listings from multiple Telegram channels in Ethiopia. This project focuses on fine-tuning a Named Entity Recognition (NER) model to extract key business entities such as product names, prices, and locations from Amharic-language messages shared on these Telegram channels.

## Features
- **Real-time Data Extraction:** Fetch messages from Ethiopian e-commerce Telegram channels.
- **NER Model Fine-Tuning:** Train models to identify and extract entities.
- **Model Comparison:** Evaluate multiple models (XLM-Roberta, BERT-Tiny-Amharic, AfroXMLR).
- **Model Interpretability:** Use SHAP and LIME to explain model predictions.
- **Business Intelligence:** Provide structured e-commerce data for EthioMart.

## Project Structure
```
├── data/               # Raw and preprocessed datasets
├── models/             # Trained model checkpoints
├── notebooks/          # Jupyter notebooks for analysis and training
├── scripts/            # Python scripts for data ingestion and model training
├── results/            # Evaluation results and reports
├── README.md           # Project documentation
└── requirements.txt    # Required dependencies
```

## Installation
### Prerequisites
- Python 3.8+
- Jupyter Notebook / Google Colab
- Hugging Face Transformers
- Telegram API access

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Amharic-NER-EthioMart.git
   cd Amharic-NER-EthioMart
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Collection & Preprocessing
1. Use a Telegram scraper to fetch messages from selected e-commerce channels.
2. Preprocess text (tokenization, normalization, stopword removal, etc.).
3. Label a subset of messages in CoNLL format for training.

## Model Training & Fine-Tuning
1. Load pre-trained models (XLM-Roberta, BERT-Tiny-Amharic, AfroXMLR).
2. Fine-tune using the labeled Amharic dataset.
3. Evaluate models using Precision, Recall, and F1-score.
4. Select the best-performing model for deployment.

## Model Evaluation & Interpretability
- Compare model performance on validation sets.
- Use SHAP and LIME for interpretability.
- Analyze model errors and improvement areas.

## Usage
### Running the NER Model
```python
from transformers import pipeline
ner_pipeline = pipeline("ner", model="path_to_trained_model")
result = ner_pipeline("ዋጋ 1000 ብር ባለው እቃ በቦሌ ይገኛል")
print(result)
```

### Running the Telegram Scraper
```bash
python scripts/telegram_scraper.py
```

## Results
- **Best Model:** XLM-Roberta with **F1-score: 89%**
- **Entities Extracted:** Product names, Prices, Locations
- **Business Impact:** Improved e-commerce searchability on EthioMart
