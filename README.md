# ML benchmark for Cancer-Associated ncORF Prediction

## Introduction
Non-canonical open reading frames (ncORFs) within non-coding RNAs (ncRNAs) have arisen as a hidden layer of gene regulation, encoding peptides and microproteins that represent a new class of cancer regulators with diagnostic and therapeutic potential. However, linking ncORFs to specific cancer types remains challenging and requires computational approaches for accurate prediction. We present a systematic evaluation of machine learning models and
feature extraction approaches to predict cancer-associated ncORFs across 15 cancer types. We benchmarked seven traditional algorithms (Random Forest, Support Vector Machine, Multilayer Perceptron, Logistic Regression, XGBoost, Naive Bayes, and k-nearest neighbors) combined with three feature extraction methods: k-mer frequency, Word2Vec embeddings, and genomic language model (gLM)-based embeddings. 

## Content

- [Download](#download)
- [Usage](#usage)
  - [Training and Evaluate](#training-and-evaluate)
  - [Inference](#inference)
- [Data](#data)
- [Citation](#citation)
  
## Download
You can download ONCOsORF in two different ways:

### 1. Direct Download

You can download the repository as a ZIP file and unpack it via the command line.
```
wget https://github.com/fabianagoes/ncORF-MLBench/archive/refs/heads/main.zip
unzip ncORF-MLBench-main.zip
```

### 2. Cloning the repository

Alternatively, you can clone the repository:
```
git clone https://github.com/fabianagoes/ncORF-MLBench.git
```

## Usage

### Training and Evaluate

To run the full experimental analysis using all 15 datasets, including **cross-validation** and testing on the **held-out datasets**, execute the following commands depending on the feature type:

- **For k-mer features:**
```
python ./kmer.py
```
- **For Word2Vec features:**
```
python ./w2v.py
```
- **For DNABERT2 or Nucleotide Transformer features:**
```
python ./llm.py --llm dnabert2
```

#### Input and Output Directories

Users can optionally specify the paths to the dataset and results directories using the `--dataset_path` and `--output_path` arguments, respectively.

If these arguments are not provided, the framework uses the following default configuration:

- The input datasets are expected to be located in the `datasets` directory at the root of the project (`./datasets/`).
- Evaluation results are saved to the `results` directory at the root of the project (`./results/`). This directory is created automatically if it does not already exist.
- Trained models are saved to the `models` directory (`./models/`) and organized by cancer type.

For example, custom paths can be specified as follows:

```
python ./kmer.py --dataset_path /path/to/datasets --output_path /path/to/results
```

#### Generated Models and Results

Running the training and evaluation scripts automatically generates two main types of outputs:

- `models/`: contains the trained machine learning models saved as `.pkl` files. The models are organized by dataset and feature representation and can subsequently be used for inference on new, unseen sequences.
- `results/`: contains the results from the model evaluation procedure, including cross-validation results and performance metrics obtained on the held-out test datasets.

The generated result files provide information about the evaluated classifiers, the hyperparameters explored during cross-validation, the selected best hyperparameters, and the corresponding model performance metrics.

The GitHub repository includes example `models/` and `results/` outputs generated using **k-mer features**. These examples illustrate the expected directory and file structure without requiring users to rerun the complete training and evaluation workflow.

### Inference

To apply a trained model to **new, unseen data** that were not included in the training or test datasets, users can run the inference procedure using:

```
python <inference_script.py> [arguments]
```

The input sequences must be provided in **FASTA format**. The inference script loads the selected trained model, processes the input sequences according to the feature representation used during training, predicts their class, and saves the results to a CSV file.

The following arguments are available:

- `--fasta`: Path to the input FASTA file containing the sequences to be classified.
- `--model`: Path to the trained model (`.pkl`) used for prediction.
- `--feature`: Feature representation used during model training (e.g., `w2v`, `dnabert2`, `nt`, or `rnafm`).
- `--pooling`: Pooling strategy used to generate embeddings from genomic language models (`cls`, `mean`, or `max`). This argument is required when using LLM-based features.
- `--output`: Path to the output CSV file containing the predictions.

For conventional feature representations, such as Word2Vec-based features, the corresponding trained pipeline is directly applied to the input sequences.

For **genomic language model (LLM)-based features**, including DNABERT2, Nucleotide Transformer (NT), and RNA-FM, an additional feature extraction step is performed before classification. The corresponding pretrained language model is loaded and used to generate sequence embeddings with the same pooling strategy employed during model training. These embeddings are then provided to the trained machine-learning classifier.

For example, inference using Nucleotide Transformer embeddings with CLS pooling can be performed as follows:

```
python inference.py --fasta example.fasta --model models/Breast/nt_cls_knn.pkl --feature nt --pooling cls --output predictions.csv
```

The resulting CSV file contains the following columns:

- `header`: Sequence identifier obtained from the FASTA header.
- `label`: Predicted class.
- `probability`: Probability associated with the predicted class.
- `result`: Textual interpretation of the prediction (`associated` or `non-associated`).

## Data
The `datasets` directory contains subfolders, each corresponding to a different cancer type (e.g., `Anal_canal`, `Bile_duct`, `Bladder`, etc.).  
All files inside these folders are **randomly generated** and are provided **only for illustration and testing purposes**.
The real datasets used in the experimental analysis can be obtained from the following repository:

https://github.com/Johnsunnn/CoraL/tree/main/Datasets

Users can also run the framework on **their own datasets**, including datasets from different problems or application domains. To do so, simply ensure that the input data follow the **same format as the example datasets** provided in the `datasets` directory.

## Citation
```bibtex
@unpublished{oncosorf,
  author       = {Fabiana Goes, Makanaka Mazheke, Aparajita Karmakar, Amanda Piveta Schnepper, Nayane de Souza, Robson Francisco Carvalho, Mark Basham, and Alexandre Rossi Paschoal},
  title        = {Systematic Evaluation of Feature Representations Improves Cancer-Associated sORF Prediction in Non-coding RNA},
  note         = {Manuscript in preparation},
  year         = {2025}
}
```
