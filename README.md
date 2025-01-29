# Fine-Tuned DistilBERT for Sentiment Analysis on IMDB Dataset

This repository contains a fine-tuned DistilBERT model for sentiment analysis on the IMDB movie review dataset. The model is trained to classify movie reviews as either **positive** or **negative** and includes scripts for training, inference, and evaluation.

---

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project fine-tunes the **DistilBERT** model using the Hugging Face `transformers` library on the IMDB dataset for binary sentiment classification. The lightweight and efficient architecture of DistilBERT makes it suitable for production deployment.

Key Features:
- Fine-tuned for binary sentiment classification (positive and negative)
- Hugging Face `Trainer` API for streamlined training
- High performance with efficient inference

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-imdb.git
   cd sentiment-analysis-imdb
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training
To fine-tune the DistilBERT model on the IMDB dataset, run the training script:
```bash
python train.py
```

- The training script utilizes the Hugging Face `Trainer` API.
- After training, the model and tokenizer will be saved in the `model/` and `tokenizer/` directories, respectively.

### Inference
To use the fine-tuned model for sentiment analysis, run the inference script:
```bash
python predict.py --text "This movie was fantastic!"
```
- The script will output the predicted sentiment (positive or negative).

---

## Model Performance

The fine-tuned model achieves the following performance on the IMDB test set:
- **Accuracy:** ~84.65%
- **F1 Score:** ~84.83%
- **Precision:** ~84.20%
- **Recall:** ~85.47%

These metrics indicate the effectiveness of the fine-tuned DistilBERT for sentiment analysis.

---

## Dataset

The IMDB dataset consists of 50,000 movie reviews labeled as positive or negative. You can download the dataset from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) or use the Hugging Face `datasets` library to load it:

```python
from datasets import load_dataset

dataset = load_dataset("jahjinx/IMDb_movie_reviews")
```

The dataset is divided into training, validation, and test subsets for better evaluation and fine-tuning.

---

## Requirements

The following dependencies are required for this project:

```plaintext
transformers==4.30.0
torch==2.0.1
datasets==2.13.0
scikit-learn==1.2.2
numpy==1.23.5
streamlit==1.10.0
pandas==1.5.2
```

To install all the dependencies:
```bash
pip install -r requirements.txt
```

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **Hugging Face:** for the `transformers` library.
- **Kaggle:** for hosting the IMDB dataset.
- **Scikit-learn:** for evaluation metrics.

Thank you for using this repository! Feel free to contribute or report any issues.

