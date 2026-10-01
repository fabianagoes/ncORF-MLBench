import argparse
import pandas as pd
import numpy as np
import torch
import dill
from Bio import SeqIO
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from multimolecule import AutoModelForSequencePrediction

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def get_embeddings(sequences, tokenizer, model, pooling_strategy, batch_size=8):

    model.eval()
    model.to(device)
    embeddings = []

    for i in range(0, len(sequences), batch_size):

        batch = sequences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs[0]

            if pooling_strategy == "cls":
                pooled_emb = last_hidden_state[:, 0, :].cpu().numpy()

            elif pooling_strategy == "max":
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                last_hidden_state_masked = last_hidden_state.masked_fill(mask_expanded == 0, -1e9)
                pooled_emb = torch.max(last_hidden_state_masked, dim=1)[0].cpu().numpy()

            elif pooling_strategy == "mean":
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled_emb = (sum_embeddings / sum_mask).cpu().numpy()
            else:
                raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")

            embeddings.extend(pooled_emb)

    return np.asarray(embeddings)


def load_llm(feature):

    feature = feature.lower()

    if feature == "dnabert2":
        print("Loading DNABERT2...")
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        llm_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

    elif feature == "nt":
        print("Loading Nucleotide Transformer...")
        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", trust_remote_code=True)
        llm_model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", trust_remote_code=True)
    else:
        raise ValueError(f"Unknown LLM feature: {feature}")

    llm_model.eval()
    llm_model.to(device)

    return tokenizer, llm_model


def predict_fasta(fasta_path, model_path, feature, output_csv, pooling=None):
    print("aqui")
    feature = feature.lower()
    print(f"Feature: {feature}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")

    with open(model_path, "rb") as f:
        pipeline = dill.load(f)

    print("Classifier loaded successfully.")
    headers = []
    sequences = []

    for record in SeqIO.parse(fasta_path, "fasta"):

        headers.append(record.id)
        seq = str(record.seq).upper()
        seq = seq.replace("U", "T") # All the features representations were trained with DNA
        sequences.append(seq)

    print(f"Number of sequences: {len(sequences)}")

    if len(sequences) == 0:
        raise ValueError("No sequences were found in the FASTA file.")

    # -----------------------------
    # Feature extraction
    # -----------------------------

    llm_features = ["dnabert2", "nt"]

    if feature in llm_features:
        if pooling is None:
            raise ValueError("For LLM features, --pooling must be specified: cls, mean or max.")

        tokenizer, llm_model = load_llm(feature)
        print("Generating embeddings...")
        X = get_embeddings(sequences, tokenizer, llm_model, pooling)
        print(f"Embedding matrix shape: {X.shape}")
        
        del llm_model # Free LLM memory before classification

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Classifier receives LLM embeddings
        preds = pipeline.predict(X)
        probs = pipeline.predict_proba(X)

    else:
        print(f"Using saved pipeline for feature extraction: {feature}")
        preds = pipeline.predict(sequences)
        probs = pipeline.predict_proba(sequences)

    # Prediction probabilities
    classes = pipeline.classes_

    results = []

    for header, pred, prob_vector in zip(headers, preds, probs):

        class_idx = np.where(classes == pred)[0][0]
        prob_pred_class = prob_vector[class_idx]

        result_text = "associated" if pred == 1 else "non-associated"

        results.append([header, pred, prob_pred_class, result_text])

    df = pd.DataFrame(results, columns=["header", "label", "probability", "result"])
    df.to_csv(output_csv, index=False)
    print(f"CSV file generated: {output_csv}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predict DNA/RNA sequences with trained model")

    parser.add_argument("--fasta", type=str, required=True, help="Path to input FASTA file")
    parser.add_argument("--feature", type=str, required=True, help="Feature type: w2v, dnabert2, nt, rnafm, etc.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--pooling", type=str, choices=["cls", "mean", "max"], default=None, help="Pooling strategy for LLM features")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file path")

    args = parser.parse_args()

    predict_fasta(fasta_path=args.fasta, model_path=args.model, feature=args.feature, pooling=args.pooling, output_csv=args.output)

    # predict_fasta("/ceph/groups/aibds/fabiana/workspace1/sorf_cancer/example/example.fasta", 
    #               "/ceph/groups/aibds/fabiana/workspace1/sorf_cancer/models/Anal_canal/nt_cls_knn.pkl",
    #               "nt",
    #               "/ceph/groups/aibds/fabiana/workspace1/ncORF-MLBench/pred.csv", "cls")
    