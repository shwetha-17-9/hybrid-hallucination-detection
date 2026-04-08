# Hybrid Hallucination Detection and Reduction in LLMs

## Overview

This project implements a hybrid framework for detecting and reducing hallucinations in Large Language Models (LLMs). The approach combines semantic consistency (SelfCheckGPT), retrieval-based verification (RAG using FAISS), and knowledge graph validation to improve reliability of generated responses.

The system computes a hybrid hallucination score and applies threshold-based classification to detect and correct unreliable outputs.

---

## Key Features

* Multi-sample response generation
* Semantic consistency checking (SelfCheckGPT-based)
* Retrieval-based verification using FAISS
* Knowledge graph-based validation
* Hybrid scoring mechanism
* Threshold-based hallucination detection
* Automatic correction using retrieved knowledge
* Evaluation using PR curves, ROC curves, and statistical graphs

---

## Project Structure

```
├── selfcheckgpt/                 # SelfCheckGPT (used as baseline)
├── experiment.py                 # Main pipeline execution
├── evaluation.py                 # Metrics calculation
├── comparison_graphs.py          # PR curve comparisons
├── graphs.py                     # Threshold & distribution graphs
├── retrieval_module.py           # Retrieval using FAISS
├── knowledge_graph.py            # Knowledge graph validation
├── llm_module.py                 # LLM interaction
├── pipeline.py                   # Full pipeline integration
├── scoring.py                    # Hybrid scoring logic
├── results.csv                   # Final results
├── README.md                     # Documentation
```

---

## Methodology

### 1. Multi-response generation

Multiple responses are generated for each query.

### 2. Semantic consistency (SelfCheckGPT)

Responses are compared to detect internal inconsistencies.

### 3. Retrieval-based verification

Relevant documents are retrieved using FAISS and compared with generated responses.

### 4. Knowledge graph validation

Facts are verified using structured knowledge representation.

### 5. Hybrid scoring

The final hallucination score is computed as:

S_hybrid = α(1 − S_sem) + β(1 − S_ret) + γ(1 − S_kg)

Where:

* α = 0.2 (semantic)
* β = 0.45 (retrieval)
* γ = 0.35 (knowledge graph)

### 6. Threshold-based classification

* T = 0.60 → High recall (lenient)
* T = 0.65 → Balanced
* T = 0.70 → High precision (strict)

### 7. Correction

Hallucinated responses are replaced using retrieved factual content.

---

## Graphs and Evaluation

### From comparison_graphs.py

* PR curves (SelfCheckGPT vs Hybrid)
* SelfCheckGPT variants
* Hybrid variants
* ROC curve comparison
* Correlation analysis (SelfCheck vs Hybrid)

### From graphs.py

* Detection rate vs threshold
* Score distribution (histogram)
* Boxplot (spread and outliers)

---

## Results Summary

* Hybrid model outperforms SelfCheckGPT in PR curves
* Higher AUC in ROC analysis
* Improved recall and F1-score
* More stable score distribution
* Weak correlation indicates hybrid captures additional signals beyond internal consistency

---

## Installation

Install required libraries:

pip install numpy pandas matplotlib scikit-learn scipy faiss-cpu

---

## How to Run

Step 1: Run experiment
python experiment.py

Step 2: Evaluate results
python evaluation.py

Step 3: Generate graphs
python comparison_graphs.py
python graphs.py

---

## Dataset

* ~200 manually curated queries
* Covers multiple domains (science, history, general knowledge)
* Pseudo ground truth generated using:

  * Retrieval agreement
  * Knowledge graph validation

---

## Notes

* SelfCheckGPT is used only as a baseline component
* Hybrid model combines internal and external verification
* Graphs follow standard evaluation methods used in hallucination detection research