# Machine Learning for Named Entity Recognition and Classification

> A comparative study of Logistic Regression, Naive Bayes, SVM (with and without word embeddings), and fine-tuned BERT for Named Entity Recognition on the CoNLL-2003 dataset — including hyperparameter tuning, a feature ablation study, and a detailed error analysis.

**Author:** Ino van de Wouw

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML%20models-orange)
![BERT](https://img.shields.io/badge/transformers-BERT-yellow)
![Task](https://img.shields.io/badge/task-NER-lightgrey)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the full pipeline](#running-the-full-pipeline)
  - [Script-by-script documentation](#script-by-script-documentation)
  - [BERT fine-tuning notebook](#bert-fine-tuning-notebook)
- [Features](#features)
- [Models](#models)
- [Feature Ablation Study](#feature-ablation-study)
- [Error Analysis](#error-analysis)
- [Confusion Matrices](#confusion-matrices)
- [Discussion & Limitations](#discussion--limitations)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Overview

Named Entity Recognition (NER) is the task of automatically detecting and classifying named mentions of entities — such as persons, organizations, locations, and miscellaneous named entities — within unstructured text. This project evaluates and compares multiple machine learning approaches for NER on the **CoNLL-2003** dataset:

🔹 **Logistic Regression** <br>
🔹 **Support Vector Machine (SVM)** — with and without Google News word embeddings <br>
🔹 **Naive Bayes** <br>
🔹 **Fine-tuned BERT** (3 random seeds) <br>

Alongside model comparison, the project includes:
- **Hyperparameter tuning** via `RandomizedSearchCV`
- A **feature ablation study** isolating the contribution of token, lemma, POS tag, token shape, and previous-token features
- A **token-level error analysis** examining how capitalization and internal character composition (periods, hyphens, digits) relate to classification errors

📄 The full write-up, methodology, and discussion are available in the accompanying paper: [`Machine_Learning_for_Named_Entity_Recognition_and_Classification_Ino_van_de_Wouw.pdf`](./Machine_Learning_for_Named_Entity_Recognition_and_Classification_Ino_van_de_Wouw.pdf).

---

## Key Results

| Model | Recall | Precision | F1-score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 75.7% | 78.2% | 76.6% | 94.9% |
| Naive Bayes | 63.0% | 78.5% | 67.4% | 92.6% |
| SVM (with Embeddings) | 64.4% | 67.0% | 65.5% | 93.1% |
| **SVM (standard)** | 77.0% | 76.9% | 76.7% | 95.1% |
| BERT (seed 1) | 95.2% | 94.1% | 94.6% | 98.7% |
| BERT (seed 2) | 95.1% | 94.1% | 94.6% | 98.7% |
| **BERT (seed 3)** | 95.1% | 94.4% | **94.7%** | **98.7%** |

> 🏆 **Fine-tuned BERT clearly outperforms all traditional approaches**, achieving F1-scores of ~94.6–94.7% and accuracy of 98.7% across all three random seeds, compared to the best traditional model (standard SVM) at 76.7% F1.
>
> Interestingly, adding word embeddings to the SVM **degraded** performance rather than improving it (65.5% vs. 76.7% F1) — see [Discussion & Limitations](#discussion--limitations) for possible explanations.

---

## Repository Structure

```
.
├── README.md
├── Machine_Learning_for_Named_Entity_Recognition_and_Classification_Ino_van_de_Wouw.pdf
├── code/
│   ├── calls_all_scripts.py          # Runs the full traditional-ML pipeline end-to-end
│   ├── tuning.py                     # Hyperparameter tuning (RandomizedSearchCV)
│   ├── feature_ablation_study.py     # Feature ablation experiments
│   ├── classification_script.py      # Training, classification & evaluation
│   ├── results_analysis.py           # Token-level error analysis
│   └── bert_finetunen.ipynb          # BERT fine-tuning (3 seeds)
├── data/
│   ├── conll2003/                    # Train / dev / test CoNLL-2003 splits
│   └── classified_data/
│       ├── model_predictions/        # Model output .conll files
│       ├── confusion_matrixes/       # Generated confusion matrix plots
│       ├── feature_ablation_study/   # Predictions per ablated feature
│       └── mistakes/                 # Per-shape misclassification dumps
└── models/
    ├── model_parameters/             # Best hyperparameters per model (from tuning.py)
    ├── models_as_pkl/                # Trained models, pickled
    ├── feature_ablation_study/       # Ablation models, pickled
    └── GoogleNews-vectors-negative300.bin.gz   # Pretrained word embeddings (not included — see below)
```

> ℹ️ The scripts use **relative paths** (`../data/...`, `../models/...`), so they are designed to be run from inside the `code/` directory. Make sure the `data/` and `models/` folders exist as siblings of `code/` before running anything.

---

## Dataset

This project uses the **CoNLL-2003** English NER dataset (Sang & De Meulder, 2003), containing 203,621 tokens. Each line represents a token annotated with:

1. The **token** itself
2. A **Part-of-Speech (PoS) tag**
3. A **chunk tag** (not used as a model feature in this project)
4. A **Named Entity (NE) tag**, in BIO format:
   - `O` — outside any named entity
   - `B-<TYPE>` — beginning of a named entity
   - `I-<TYPE>` — continuation (inside) of a named entity

Entity types: `PER` (Person), `LOC` (Location), `ORG` (Organization), `MISC` (Miscellaneous).

> ⚠️ The CoNLL-2003 dataset itself is **not redistributed** in this repository — see the [official CoNLL-2003 shared task page](https://www.clips.uantwerpen.be/conll2003/ner/) for access instructions, then place the train/dev/test files under `data/conll2003/`.

---

## Installation

### Requirements

- Python 3.9+
- The following Python packages:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy nltk gensim
```

### NLTK data

The lemmatizer requires the WordNet corpus:

```python
import nltk
nltk.download('wordnet')
```

### Word embeddings (SVM with embeddings only)

The SVM-with-embeddings experiment requires Google's pretrained **Word2Vec News vectors**:

1. Download `GoogleNews-vectors-negative300.bin.gz` (~1.5GB) from the [official source](https://code.google.com/archive/p/word2vec/).
2. Place it at `models/GoogleNews-vectors-negative300.bin.gz`.

### BERT fine-tuning notebook

`bert_finetunen.ipynb` additionally requires:

```bash
pip install transformers datasets torch
```

A **GPU is strongly recommended** for fine-tuning BERT — training on CPU will be extremely slow. If using Google Colab or a similar hosted notebook environment, select a GPU runtime before running the notebook.

---

## Usage

### Running the full pipeline

`calls_all_scripts.py` executes the traditional-ML pipeline end-to-end, in order:

```bash
cd code/
python calls_all_scripts.py
```

This runs, in sequence:

1. **`tuning.py`** — hyperparameter search
2. **`feature_ablation_study.py`** — feature ablation experiments
3. **`classification_script.py`** — final training, classification & evaluation
4. **`results_analysis.py`** — error analysis on the SVM's predictions

> 💡 Note: `results_analysis.py` and parts of `feature_ablation_study.py` / `classification_script.py` currently hard-code file paths (e.g. `../data/classified_data/model_predictions/output_SVM_no.conll`) inside their `main()` functions rather than reading them from `sys.argv`. Review these paths before running, or adapt them to point to your own data/output locations.

### Script-by-script documentation

#### `tuning.py`

Performs hyperparameter tuning for Logistic Regression, Naive Bayes, and SVM using `RandomizedSearchCV`.

- **Input:** CoNLL-2003 training file
- **Features extracted per token:** token, PoS tag, lemma, token shape, previous token (see [Features](#features))
- **Search spaces:**
  - Logistic Regression: `C ~ Uniform(0, 4)`, `penalty = l2`
  - Naive Bayes: `alpha ~ Uniform(0, 4)`
  - SVM: `C ~ Uniform(0, 4)`, `penalty ∈ {l1, l2}`, `loss ∈ {hinge, squared_hinge}`
- **Output:** Best parameters per model, written to `../models/model_parameters/best_parameters<MODEL>.txt`

```bash
python tuning.py
```

#### `feature_ablation_study.py`

Systematically removes one feature (or feature combination) at a time from the Logistic Regression model and re-evaluates performance, to isolate each feature's contribution.

- **Ablation conditions tested:** `token`, `pos`, `lemma`, `shape`, `no token and no lemma`, `pos and shape`, `previous token`, and the full `baseline`
- **Output:**
  - Predictions per condition → `../data/classified_data/feature_ablation_study/output_without_<feature>.conll`
  - Trained models per condition → `../models/feature_ablation_study/`
  - Printed recall / precision / F1 / accuracy / confusion matrix per condition

```bash
python feature_ablation_study.py
```

#### `classification_script.py`

The core training, classification, and evaluation script. Trains Logistic Regression, Naive Bayes, and SVM (both with and without Google News word embeddings) on the tuned hyperparameters, classifies the test set, and evaluates performance.

- **Input:** CoNLL-2003 train/test files, tuned hyperparameters from `tuning.py`
- **Output:**
  - Trained models (pickled) → `../models/models_as_pkl/<model>.pkl`
  - Predictions → `../data/classified_data/model_predictions/output_<model>_<embeddings?>.conll`
  - Confusion matrix plots → `../data/classified_data/confusion_matrixes/<model>ConfusionMatrix.png`
  - Printed evaluation metrics (macro recall / precision / F1 / accuracy)

```bash
python classification_script.py
```

#### `results_analysis.py`

Performs a token-level error analysis on a model's predictions (by default, the standard SVM's `.conll` output), categorizing errors by:

- **Capitalization:** uppercase, lowercase, title case, digit, other
- **Internal character composition:** presence of periods, hyphens

- **Input:** `../data/classified_data/model_predictions/output_SVM_no.conll`
- **Output:**
  - Printed error/no-error counts per shape category
  - Per-category mistake dumps (token, true label, predicted label) → `../data/classified_data/mistakes/<category>_mistakes_SVM.txt`

```bash
python results_analysis.py
```

#### `calls_all_scripts.py`

Convenience script that runs all four scripts above in sequence via `exec()`. See [Running the full pipeline](#running-the-full-pipeline).

### BERT fine-tuning notebook

`bert_finetunen.ipynb` fine-tunes a pretrained BERT model on the CoNLL-2003 NER task across **3 random seeds**, to assess result stability. Run it in Jupyter, JupyterLab, or a hosted notebook environment (e.g. Google Colab) with GPU support:

```bash
jupyter notebook code/bert_finetunen.ipynb
```

Each seed's fine-tuned model achieves F1-scores of 94.6–94.7% and accuracy of 98.7% — see [Key Results](#key-results).

---

## Features

Each token is represented using the following features:

| Feature | Description |
|---|---|
| **Token** | The raw token string |
| **PoS tag** | Part-of-speech tag from the dataset |
| **Lemma** | Lemmatized form of the token (via NLTK's `WordNetLemmatizer`) |
| **Shape** | Encodes capitalization pattern (`Upper`, `Lower`, `First_letter_capital`, `Digit`, `Other`) combined with internal characters (`Has_period`, `Has_hyphen`) |
| **Previous token** | The token immediately preceding the current one, capturing local sequential context |
| **Word embedding** *(SVM only)* | 300-dimensional Google News Word2Vec embedding of the token |

The **chunk tag** provided in the raw CoNLL-2003 data was **excluded** due to its similarity to the PoS tag.

---

## Models

<details>
<summary><strong>Logistic Regression</strong></summary>

Models the likelihood of a token/feature vector belonging to each entity type, optimizing a loss function via iterative gradient-based updates.
</details>

<details>
<summary><strong>Support Vector Machine (SVM)</strong></summary>

Projects tokens and their features into a high-dimensional space, finding hyperplanes that maximize the margin between entity classes, using hinge loss. Evaluated both **with** and **without** Google News word embeddings as an additional feature.
</details>

<details>
<summary><strong>Naive Bayes</strong></summary>

A probabilistic classifier applying Bayes' theorem under a conditional-independence assumption between features, selecting the entity class with the highest posterior probability.
</details>

<details>
<summary><strong>Fine-tuned BERT</strong></summary>

A transformer-based model using bidirectional contextual embeddings (as opposed to the static features used by the other models). Fine-tuned directly for the NER task across 3 random seeds, using WordPiece tokenization to robustly handle rare/out-of-vocabulary words.
</details>

---

## Feature Ablation Study

Conducted on the Logistic Regression model, systematically excluding one feature (or feature pair) at a time:

| Feature left out | Recall | Precision | F1-score | Accuracy |
|---|---|---|---|---|
| **Baseline (all features)** | 80.7% | 86.5% | 83.2% | 96.8% |
| Token | 80.4% | 86.2% | 82.9% | 96.7% |
| Lemma | 80.4% | 86.1% | 82.9% | 96.7% |
| PoS | 80.2% | 86.2% | 82.8% | 96.6% |
| Shape | 80.1% | 85.5% | 82.3% | 96.3% |
| PoS + Shape | 73.3% | 90.0% | 80.4% | 95.1% |
| Previous token | 69.9% | 78.3% | 72.7% | 94.6% |
| Token + Lemma | 65.0% | 70.6% | 66.7% | 92.6% |

**Key takeaways:**
- Removing **token** or **lemma** individually barely affects performance — the two features appear to substitute for each other.
- Removing **both** token and lemma causes the sharpest performance drop, confirming they jointly encode essential lexical information.
- Removing **token shape** hurts more than PoS, highlighting the importance of surface-level capitalization/punctuation cues for recognizing proper nouns.
- Removing the **previous token** causes the second-largest drop, underscoring the value of local sequential context for detecting multi-word entities.

---

## Error Analysis

Error analysis was performed on the best-performing traditional model, the **standard SVM**, examining errors by token shape:

| Category | No error | Error | Total | Error rate |
|---|---|---|---|---|
| **Capitalization** — Upper | 1,639 | 568 | 2,207 | 25.7% |
| **Capitalization** — Lower | 23,779 | 56 | 23,835 | 0.2% |
| **Capitalization** — Title | 7,109 | 1,589 | 8,698 | 18.3% |
| **Internal shape** — Digit | 3,962 | 7 | 3,969 | 0.2% |
| **Internal shape** — Period | 2,759 | 47 | 2,806 | 1.7% |
| **Internal shape** — Hyphen | 1,622 | 91 | 1,713 | 5.3% |
| **Internal shape** — Other | 7,666 | 60 | 7,726 | 0.8% |
| **Total** | 48,536 | 2,418 | 50,954 | 4.7% |

**Key takeaways:**
- **Title-case tokens** show the highest error frequency, mostly confusions between countries classified as locations vs. locations classified as countries.
- **Upper-case tokens** are the second most error-prone, mainly organizations misclassified as locations or as non-entities.
- **Lower-case tokens** are highly reliable, with errors largely limited to sports-related terms and conjunctions (`and`, `of`) inside miscellaneous entities.
- **Digit-containing tokens** are classified almost perfectly; the rare errors involve years not recognized as the start of a MISC entity, or non-entity numbers being wrongly tagged.
- **Hyphenated tokens** show the highest internal-shape error rate, often confusing locations/organizations with persons.
- The **"Other"** category (special characters, e.g. `McIntosh`, `trans-Atlantic`) mostly shows entities being missed entirely (classified as `O`).

---

## Confusion Matrices

Confusion matrices for each model are generated automatically by `classification_script.py` (traditional models) and saved to [`data/classified_data/confusion_matrixes/`](./data/classified_data/confusion_matrixes/), using the naming convention `<model>ConfusionMatrix.png`.

### Logistic Regression

![Logistic Regression Confusion Matrix](./data/classified_data/confusion_matrixes/logregConfusionMatrix.png)

> 📌 Add the remaining confusion matrices to [`data/classified_data/confusion_matrixes/`](./data/classified_data/confusion_matrixes/) and reference them here, e.g.:
> ```markdown
> ![Naive Bayes Confusion Matrix](./data/classified_data/confusion_matrixes/NBConfusionMatrix.png)
> ![Standard SVM Confusion Matrix](./data/classified_data/confusion_matrixes/SVMConfusionMatrix.png)
> ![SVM with Embeddings Confusion Matrix](./data/classified_data/confusion_matrixes/SVMwithEmbeddingsConfusionMatrix.png)
> ```

---

## Discussion & Limitations

- **BERT's dominance:** BERT's bidirectional contextual embeddings and lack of dependence on hand-engineered features give it a substantial edge (94.6–94.7% F1) over all traditional approaches (65.5–76.7% F1), consistent with its ability to capture long-range dependencies that static, per-token feature representations cannot.
- **Word embeddings hurt SVM performance:** Somewhat counter-intuitively, adding Google News word embeddings to the SVM *degraded* performance relative to the standard feature-based SVM. Two possible explanations are discussed in the paper: (1) the chosen embedding strategy may not have captured semantic relationships useful for entity recognition, or (2) computational constraints required limiting the added feature set (only shape + embedding, due to computational limitations), potentially losing valuable information present in the full feature set used by the other models.
- **Contextual features matter:** The feature ablation study shows that removing the previous-token feature causes one of the largest performance drops among traditional features, aligning with BERT's strength in modeling context and suggesting sequential/contextual information is a key lever for further improving traditional models.
- **Capitalization sensitivity:** Traditional models struggle disproportionately with Title-case and Upper-case tokens (entity type confusions), suggesting potential future work in richer capitalization-aware or gazetteer-based features.

---

## Citation

If you use this work, please cite:

```
van de Wouw, I. (2024). Machine Learning for Named Entity Recognition and Classification.
```

This project builds on the CoNLL-2003 shared task dataset:

```
Erik Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the CoNLL-2003 shared task:
Language-independent named entity recognition. In Proceedings of the Seventh Conference on
Natural Language Learning at HLT-NAACL 2003, pages 142–147.
```

---

## Acknowledgements

- The `extract_embeddings_as_features_and_gold` function in `classification_script.py` was partially inspired by code from the [HLT course (cltl/ma-hlt-labs)](https://github.com/cltl/ma-hlt-labs/), accessed May 2020.
- Fine-tuned BERT relies on pretrained weights and the WordPiece tokenizer as described in Devlin et al. (2018), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
