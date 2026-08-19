# The Magic Behind the Text: A Deep Learning Comparison for Named Entity Recognition and Relation Extraction in the Harry Potter Universe

## Overview
This repository contains a Deep Learning project implementing Named Entity Recognition (NER) and Relation Extraction (RE) for the Harry Potter universe.

## Research Questions
- **RQ1**: How effectively can Deep Learning architectures identify named entities in Harry Potter text?
- **RQ2**: How does sequential modelling affect NER performance compared with a feed-forward token-classification baseline?
- **RQ3**: Does bidirectional recurrent modelling improve entity recognition compared with a unidirectional recurrent architecture?
- **RQ4**: Does incorporating attention improve contextual token classification and entity-level recognition?
- **RQ5** (Optional): Can the annotated entity and relation information be used to construct a meaningful Harry Potter knowledge graph?

## Objectives
- To develop a Deep Learning-based Named Entity Recognition system for identifying characters, houses, magical items, spells, and locations in Harry Potter text.
- To compare feed-forward, recurrent, bidirectional LSTM, and attention-enhanced recurrent architectures using a common experimental protocol.
- To evaluate whether sequential context and attention improve token-level and entity-level NER performance.
- To investigate the feasibility of relation extraction and knowledge-graph construction from the annotated fictional text.

## Dataset
Kaggle dataset: Harry Potter (NER + RE)
URL: [https://www.kaggle.com/datasets/mkdsps/harry-potter-ner-re](https://www.kaggle.com/datasets/mkdsps/harry-potter-ner-re)

## Environment Setup
### Virtual Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name hp-ner-dl --display-name "Harry Potter NER DL"
```

## Running the Notebook
Open the `notebooks/harry_potter_ner_relation_extraction.ipynb` file in Jupyter Notebook or JupyterLab, select the "Harry Potter NER DL" kernel, and run all cells sequentially.

## Models
1. **MLP Token Classifier**: Feed-forward baseline.
2. **Simple RNN**: Baseline sequential model.
3. **Bidirectional LSTM**: Advanced sequential context model.
4. **Bidirectional LSTM + Attention**: Contextual token model with self-attention.

## Academic Integrity / AI Assistance Disclosure
*Note: External AI assistance may have been used during the development of this codebase, subject to the university/course policy.*
