# SHL Grammar Scoring Engine 2026

An end-to-end **Machine Learning pipeline for automated spoken-English grammar scoring**, developed for the **SHL Hiring Assessment 2026 Kaggle competition**.

The system takes spoken-English audio recordings as input, extracts linguistic, grammatical, semantic, acoustic, and prosodic features, and predicts a **grammar score between 1 and 5**.

---

## 📌 Project Overview

The goal of this project is to predict the grammatical quality of spoken English from approximately **45–60 second audio recordings**.

The pipeline combines:

- Speech-to-text transcription
- Grammar correction
- NLP features
- Sentence embeddings
- Speech embeddings
- Grammar analysis
- Language-model perplexity
- Acoustic features
- Prosody features
- Feature selection
- Multiple regression models
- Out-of-fold ensemble optimization
- Calibration
- Prediction distribution optimization

The final output is:

```text
submission.csv
```

containing predicted grammar scores for the competition test set.

---

## 🎯 Problem Statement

Given an audio recording of a candidate speaking English, predict the candidate's grammatical proficiency on a scale from:

```text
1 → Lowest grammar proficiency
5 → Highest grammar proficiency
```

The system aims to capture both explicit grammatical errors and characteristics associated with fluent and grammatically complex speech.

---

# 🏗️ System Architecture

```text
                    Audio Recording
                           │
                           ▼
                 Speech Transcription
                     (Whisper)
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
      Grammar Correction          Audio Processing
           (T5)                         │
              │                         │
              ▼                         ▼
       Grammar Features          Acoustic Features
              │                         │
              ▼                         ▼
        LanguageTool                Prosody
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                    NLP Features
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
          MiniLM        Wav2Vec2       GPT-2
        Embeddings      Embeddings   Perplexity
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                  Feature Combination
                           │
                           ▼
                    Feature Selection
                           │
                           ▼
               ┌───────────────────────┐
               │     ML Models         │
               │                       │
               │ Ridge Regression      │
               │ SVR                   │
               │ ExtraTrees             │
               │ XGBoost               │
               └───────────┬───────────┘
                           │
                           ▼
                  OOF Ensemble Learning
                           │
                           ▼
                      Calibration
                           │
                           ▼
              Distribution Optimization
                           │
                           ▼
                 Grammar Score 1–5
                           │
                           ▼
                    submission.csv
```

---

# ✨ Features

## 1. Speech Transcription

**Whisper** is used to convert the spoken audio into text.

This provides the linguistic representation required for downstream NLP processing.

---

## 2. Grammar Correction

A T5-based grammar correction model:

```text
vennify/t5-base-grammar-correction
```

is used to generate corrected versions of the transcripts.

The original and corrected transcripts are compared to derive features such as:

- Original word count
- Corrected word count
- Word-count difference
- Correction ratio
- Grammar error characteristics

---

## 3. Sentence Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to generate semantic embeddings from transcripts.

Embedding size:

```text
384 dimensions
```

These embeddings capture semantic and linguistic information that handcrafted features may miss.

---

## 4. Speech Embeddings

The project uses:

```text
facebook/wav2vec2-base-960h
```

to extract speech representations directly from the audio.

Mean and standard-deviation pooling are used to obtain fixed-size speech embeddings.

These features capture information directly from the spoken signal.

---

# 📝 Grammar Features

The pipeline extracts grammatical characteristics such as:

- Word count
- Character count
- Sentence count
- Average sentence length
- Maximum sentence length
- Vocabulary diversity
- Repetition ratio
- Noun frequency
- Verb frequency
- Adjective frequency
- Adverb frequency
- Pronoun frequency
- Determiner frequency
- Preposition/adposition frequency
- Conjunction frequency
- Auxiliary verb frequency

---

# 🔎 LanguageTool Features

LanguageTool is used to detect potential grammatical and writing errors.

Features include:

- Grammar error count
- Grammar error rate
- Errors per 100 words
- Spelling errors
- Grammar-only errors
- Punctuation errors
- Sentence-length statistics
- Long-sentence ratio
- Short-sentence ratio
- Conjunction density
- Type-token ratio
- Repetition rate

---

# 🧠 GPT-2 Perplexity

GPT-2 is used to estimate the linguistic predictability of the transcript.

The resulting:

```text
Perplexity
```

is used as an additional language-quality feature.

Additional complexity features include:

- Average word length
- Word-length standard deviation
- Long-word ratio
- Unique-word ratio
- Sentence-length statistics
- Punctuation density
- Question ratio
- Conjunction density

---

# 🎙️ Acoustic & Prosody Features

Audio-level characteristics are extracted using `librosa`.

Features include:

### Acoustic

- RMS energy
- Zero-crossing rate
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- MFCC statistics

### Prosody

- Speech duration
- Silence duration
- Silence ratio
- Pause count
- Average pause duration
- Maximum pause duration
- Pause density
- RMS statistics
- Fundamental frequency (F0)
- F0 range
- Voiced ratio
- Words per minute
- Words per second

These features help capture characteristics of spoken delivery that cannot be obtained from text alone.

---

# 🤖 Machine Learning Models

Multiple regression models were evaluated.

### Ridge Regression

Used as a strong regularized linear baseline, particularly useful with high-dimensional embeddings.

### Support Vector Regression

SVR is used to model nonlinear relationships between extracted features and grammar scores.

### ExtraTrees Regressor

An ensemble of randomized decision trees is used to capture nonlinear feature interactions.

### XGBoost

Gradient-boosted decision trees are used to capture complex nonlinear relationships.

---

# 🔬 Feature Selection

Because the combined feature space can become very large, feature selection is applied before the final model.

The project evaluates different feature groups and uses:

```text
SelectKBest
```

to identify useful features.

This helps reduce noise and improve generalization.

---

# 🔀 Ensemble Learning

Instead of relying on a single model, predictions from multiple models are combined.

The ensemble uses **out-of-fold (OOF) predictions** to prevent information leakage during model selection.

Model weights are optimized using OOF predictions.

The ensemble includes predictions from:

```text
Ridge
SVR
ExtraTrees
XGBoost
Selected Ridge
Prosody model
```

---

# 📊 Calibration

Regression models often compress predictions toward the mean.

To address this, a calibration layer is trained using OOF predictions.

The calibration step helps correct:

- Systematic bias
- Prediction compression
- Mean-shift
- Score distribution mismatch

Predictions are finally clipped to:

```text
1 ≤ score ≤ 5
```

---

# 📈 Evaluation

The project evaluates predictions using:

### RMSE

Root Mean Squared Error measures the magnitude of prediction errors.

```text
RMSE = sqrt(mean((y - ŷ)²))
```

Lower is better.

### Pearson Correlation

Measures the linear relationship between predicted and actual grammar scores.

Higher is better.

---

# 📁 Project Structure

Recommended repository structure:

```text
shl-grammar-scoring-engine/
│
├── grammar_scoring_engine.ipynb
│
├── submission.csv
│
├── README.md
│
├── requirements.txt
│
└── .gitignore
```

The competition audio and input dataset are **not included** in this repository.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/shl-grammar-scoring-engine.git
cd shl-grammar-scoring-engine
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Main Libraries

The project uses technologies including:

```text
Python
NumPy
Pandas
Scikit-learn
PyTorch
Transformers
Sentence-Transformers
Librosa
LanguageTool
XGBoost
SciPy
Jupyter
```

---

# 🚀 Running the Project

1. Download the competition dataset from Kaggle.
2. Place the dataset in the expected directory structure.
3. Open:

```text
grammar_scoring_engine.ipynb
```

4. Install all dependencies.
5. Run the notebook sequentially.
6. The final predictions are generated as:

```text
submission.csv
```

---

# 📤 Final Submission Format

The final file contains exactly two columns:

```text
filename,label
```

Example:

```csv
filename,label
audio_001.wav,3.42
audio_002.wav,4.17
audio_003.wav,2.81
```

The final submission contains **216 test samples**.

---

# ⚠️ Dataset

The competition dataset is intentionally **not included** in this repository because it contains the competition's audio/input data.

Users should obtain the dataset directly through the official Kaggle competition and follow its applicable terms.

---

# 🔮 Future Improvements

Potential improvements include:

- Whisper-large transcription
- Better word-level alignment
- More robust pronunciation features
- Speaker-independent acoustic normalization
- Advanced transformer fine-tuning
- Cross-validation with repeated folds
- Deep learning regression models
- Better temporal speech representations
- More sophisticated ensemble optimization
- Human-level grammar scoring analysis

---

# 🏆 Project Highlights

- Multimodal **audio + text** approach
- Transformer-based speech representations
- Semantic sentence embeddings
- Automated grammar-error analysis
- Language-model perplexity
- Acoustic and prosodic analysis
- Leakage-aware OOF ensemble learning
- Feature selection
- Prediction calibration
- Distribution optimization
- Automated Kaggle submission generation

---

# 👨‍💻 Author

**Shashi Bhushan Raj**

B.Tech — Electronics & Communication Engineering  
IIIT Una

---

## 📜 Disclaimer

This project is developed for educational, research, and competition purposes. The predicted grammar score should not be considered a definitive assessment of a person's overall English proficiency.

---

**Final output: `submission.csv`**
