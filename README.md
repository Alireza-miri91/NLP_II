# Question-Aware Paragraph Retrieval: A Comparison of Semantic and Lexical Approaches

This repository contains the implementation and evaluation code for a comprehensive comparison of semantic and lexical retrieval methods for paragraph-level question answering on scientific text using the QASPER dataset.

## Project Overview

This project addresses the challenge of retrieving relevant paragraphs for question answering in scientific domains. We compare five different retrieval methods across two granularities (paragraph and sentence-level) to understand when semantic embeddings outperform traditional lexical methods like BM25.

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- Internet connection for downloading QASPER dataset and pre-trained models

### Dependencies

The project requires the following Python packages:

```bash
# Core ML/NLP libraries
pip install torch torchvision torchaudio
pip install transformers>=4.20.0
pip install sentence-transformers>=2.2.0

# Haystack framework for retrieval
pip install haystack-ai>=2.0.0

# Data processing and ML
pip install scikit-learn>=1.1.0
pip install datasets>=2.0.0
pip install numpy>=1.21.0
pip install pandas>=1.4.0

# Optional: Visualization and utilities
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install tqdm>=4.64.0
```

### Installation Steps

1. Clone or download the repository:
```bash
git clone <repository-url>
cd NLP_II
```
Or extract from zip file and navigate to the NLP_II directory.

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation by running a simple test:
```bash
python -c "import haystack, transformers, datasets; print('Installation successful!')"
```

### Required Tools and Libraries

The implementation uses the following key frameworks and libraries:

- **Haystack AI**: Framework for building retrieval pipelines and document stores
- **Transformers**: Hugging Face library for transformer models and tokenization
- **Sentence Transformers**: Library for generating sentence embeddings
- **Datasets**: Hugging Face library for loading QASPER dataset
- **Scikit-learn**: For TF-IDF, LSI, and similarity computations
- **PyTorch**: Deep learning framework for transformer models
- **NumPy/Pandas**: Data manipulation and analysis

### System Requirements

- **Memory**: At least 8GB RAM recommended (16GB+ for full evaluation)
- **Storage**: ~5GB for models and dataset (downloaded automatically)
- **GPU**: CUDA-compatible GPU recommended for faster processing
- **Python**: Version 3.8 or higher

## Usage

### Quick Start

1. **Test basic retrieval** with a sample question:
```bash
python retrieve_qasper.py --question "What methods are used in this paper?" --top_k 3
```

2. **Run a quick evaluation** on a small sample:
```bash
python eval_qasper.py --mode embeddings --limit 50
```

3. **Compare all methods** on a small sample:
```bash
python compare_all_qasper.py --limit 50 --modes "embeddings,bm25" --granularities "paragraph"
```

### Basic Retrieval

To run a simple retrieval example with a custom question:

```bash
python retrieve_qasper.py --question "What are the main limitations of the proposed approach?" --top_k 5
```

Available options:
- `--question`: Your question (required)
- `--top_k`: Number of paragraphs to retrieve (default: 5)
- `--model`: SentenceTransformer model name (default: sentence-transformers/all-MiniLM-L6-v2)
- `--split`: Dataset split to use (default: train)

### Document Indexing

To index documents for retrieval (usually done automatically):

```bash
python index_qasper.py --split train --granularity paragraph
```

Available options:
- `--model`: SentenceTransformer model name (default: sentence-transformers/all-MiniLM-L6-v2)
- `--split`: Dataset split (train/validation/test, default: train)
- `--granularity`: Indexing granularity (paragraph/sentence, default: paragraph)

### Individual Method Evaluation

Evaluate specific retrieval methods:

```bash
# Sentence embeddings
python eval_qasper.py --mode embeddings --limit 200

# BM25
python eval_qasper.py --mode bm25 --limit 200

# TF-IDF
python eval_qasper.py --mode tfidf --limit 200

# DPR
python eval_qasper.py --mode dpr --limit 200

# LSI
python eval_qasper.py --mode lsi --limit 200

# SPLADE
python eval_qasper.py --mode splade --limit 200
```

Available options for eval_qasper.py:
- `--mode`: Retrieval method (embeddings, bm25, tfidf, lsi, dpr, splade, answerability)
- `--model`: Model name for embeddings/DPR (default: sentence-transformers/all-MiniLM-L6-v2)
- `--split`: Dataset split (default: train)
- `--k`: Number of top documents to retrieve (default: 5)
- `--limit`: Number of questions to evaluate (default: 200, 0 = all)
- `--granularity`: Retrieval granularity (paragraph/sentence, default: paragraph)

### Answerability Evaluation

Evaluate answerability prediction performance:

```bash
python eval_qasper.py --mode answerability --limit 200
```

### Comparison

To run evaluation across all methods:

```bash
python compare_all_qasper.py --limit 200 --include-answerability
```

Available options for compare_all_qasper.py:
- `--split`: Dataset split (default: train)
- `--k`: Number of top documents to retrieve (default: 5)
- `--limit`: Number of questions to evaluate (-1 = auto-detect, 0 = all, default: -1)
- `--model`: Model name (default: sentence-transformers/all-MiniLM-L6-v2)
- `--modes`: Comma-separated retrieval modes (default: embeddings,bm25,tfidf,lsi,dpr,splade)
- `--granularities`: Comma-separated granularities (default: paragraph,sentence)
- `--include-answerability`: Include answerability evaluation
- `--out`: Output JSON file name (default: compare_results_TIMESTAMP.json)

## Project Structure

```
NLP_II/
├── qasper_loader.py          # QASPER dataset loading and preprocessing
├── index_qasper.py           # Document indexing with sentence embeddings
├── retrieve_qasper.py        # Retrieval pipeline for single questions
├── eval_qasper.py            # evaluation framework
├── compare_all_qasper.py     # Comparison across all methods
├── answerability_eval.py     # Answerability prediction evaluation
├── baselines.py              # Implementation of baseline methods (TF-IDF, LSI, DPR, SPLADE)
├── compare_results_*.json    # Evaluation results (JSON format)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

### Core Files Description

- **`qasper_loader.py`**: Loads and preprocesses the QASPER dataset, handles paragraph/sentence extraction
- **`index_qasper.py`**: Creates embeddings and indexes documents using Haystack framework
- **`retrieve_qasper.py`**: Implements retrieval pipeline for answering single questions
- **`eval_qasper.py`**: Main evaluation script supporting multiple retrieval methods and metrics
- **`compare_all_qasper.py`**: Runs comprehensive comparison across all methods and generates results
- **`answerability_eval.py`**: Evaluates whether questions can be answered given retrieved context
- **`baselines.py`**: Implements baseline retrieval methods (TF-IDF, LSI, DPR, SPLADE)

## Methods Implemented

### Lexical Methods
- **BM25**: Traditional probabilistic ranking function
- **TF-IDF**: Term frequency-inverse document frequency with cosine similarity

### Semantic Methods
- **Sentence Embeddings**: Using sentence-transformers/all-MiniLM-L6-v2
- **Dense Passage Retrieval (DPR)**: Using Facebook's DPR models
- **Latent Semantic Indexing (LSI)**: Truncated SVD on TF-IDF matrices

## Dataset

We use the QASPER dataset (Dasigi et al., 2021) which contains:
- 1,585 scientific papers
- 5,049 question-answer pairs
- Paragraph-level evidence annotations
- Multiple scientific domains (ML, NLP, etc.)

## Evaluation Metrics

- **Precision@k**: Fraction of retrieved documents that are relevant
- **Recall@k**: Fraction of relevant documents that are retrieved
- **F1@k**: Harmonic mean of precision and recall
- **Hit@k**: Whether the correct paper is retrieved within top-k
- **Answerability Accuracy**: Performance on answerability prediction

## Reproducibility

All results can be reproduced using the provided scripts. The evaluation uses a fixed random seed for consistency. Results are saved in JSON format for further analysis.

## Technical Details

- **Batch Processing**: Optimized for RTX A6000 with batch sizes 32-64 for embedding generation
- **GPU Utilization**: Automatic GPU detection and utilization for transformer models
- **Memory Management**: Efficient in-memory document stores using Haystack framework
- **Dataset**: Automatically downloads QASPER dataset from Hugging Face on first run
- **Models**: Downloads pre-trained models automatically (sentence-transformers, DPR, etc.)
- **Output Format**: Results saved in JSON format for further analysis

## Limitations

- Evaluation limited to 200 questions due to computational constraints (configurable via --limit)
- Single embedding model evaluated (sentence-transformers/all-MiniLM-L6-v2)
- Domain-specific to scientific text (QASPER dataset)
- Results may not generalize to other domains
- Requires significant computational resources for full evaluation

## Future Work

- Evaluate domain-specific embedding models
- Implement hybrid lexical-semantic approaches
- Extend evaluation to full QASPER dataset
- Explore different chunking strategies


## Contact

Seyed Alireza Miri (Seyed-Alireza.Miri@stud.uni-regensburg.de)
Matrikelnummer: 2384735
Computational Sciences 
University of Regensburg
