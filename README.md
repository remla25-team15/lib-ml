# lib-ml

This repository contains the **preprocessing logic** used across both `model-training` and `model-service`. It provides reusable text transformation utilities for training and inference pipelines.

---

## Preprocessing Features

The preprocessing pipeline includes:

- Lowercasing and tokenization  
- Removal of punctuation and non-letter characters  
- Snowball stemming (English)  
- Stopword removal (excluding "not")  
- TF-IDF vectorization  
- Message length as an additional numeric feature  

These transformations are compatible with `sklearn` pipelines and `CountVectorizer`.

---

## Environment Configuration

The library has minimal configuration and works out of the box. Ensure the following:

- Python 3.10
- Installed dependencies from `requirements.txt`

To set up the environment locally:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

---

## Development Setup

### Local Development

```bash
# Clone the repo
git clone https://github.com/remla25-team15/lib-ml.git
cd lib-ml

# Install the library
pip install .

This installs the library as `libml`, exposing `libml.preprocessing`.


## Versioning

The library is versioned using **semantic versioning** and released through GitHub using **repository tags**.

* Do **not** publish to PyPi or public registries.
* Install via Git tag:

```text
git+https://github.com/remla25-team15/lib-ml.git@v0.1.2
```

---

## Docker & CI/CD

This repo is not meant to be containerized but integrates with containerized environments consuming the library (e.g., model-service). GitHub Actions workflows are used for release automation on new tags.

---

## File Structure

```
lib-ml/
├── libml/
│   ├── __init__.py
│   └── preprocessing.py
|   |__output
|
|_ output
|
|_retaurantreviewsdata
    |_a1_RestaurantReviews_HistoricDump.tsv
    |_a2_RestaurantReviews_FreshDump.tsv
|  
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## Dataset

This repository assumes access to a dataset stored in the `restaurantreviewsdata/` directory, containing `.tsv` files with restaurant reviews.

```
