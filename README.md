# 📰 Fake News Detection Using Deep Learning Methods

### NEUROMATCH 2024 - NATURAL LANGUAGE PROCESSING
#### A Novel Approach for Detecting Fake News using Deep Learning Methods

---

## Project Overview

In this notebook, we propose several embedding and model pairs to classify news articles as either real or fake. The following approaches were explored:

1. **GloVe Embeddings + LSTM (both uni/bidirectional models)**
2. **TFIDF + Logistic Regression**
3. **CountVectorizer + Logistic Regression**
4. **Pretrained Tokenizer + Transformer Model from BERT**

---

## Installation and Setup

To run this project, you'll need to install the required libraries. Use the following commands to set up your environment:

```bash
pip install numpy pandas matplotlib seaborn torch torchtext scikit-learn tqdm
```
## Usage

### Import the Notebook 

###  Import the Dataset

You can either import the preprocessed data (`news_df_processed.csv`) or the raw dataframe that is of the form `{'label': (0 or 1), 'content': (article string)}`.

```python
news_df = pd.read_csv('path/to/your/news_df_processed.csv')
```


## Acknowledgements

We would like to thank the Neuromatch Academy for providing the platform and resources for this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or suggestions, please contact:

Boran Aybak Kilic
boranaybak34@gmail.com
