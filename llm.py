from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, confusion_matrix, classification_report, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoTokenizer, AutoModel
from multimolecule import AutoModelForSequencePrediction
from transformers.models.bert.configuration_bert import BertConfig
from IPython.display import display
import torch
import pandas as pd
import numpy as np
import os
import time
import json
import argparse
import dill
import glob
import re
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Function to extract gLM embeddings
def get_embeddings(sequences, tokenizer, model, pooling_strategy, batch_size=8):
    model.eval()
    model.to(device)
    embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        # Tokenize batch with attention mask
        inputs = tokenizer(batch,return_tensors="pt",padding=True,truncation=True,max_length=512).to(device)
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
                pooled_emb = torch.max(last_hidden_state_masked, dim=1)[0]
                pooled_emb = pooled_emb.cpu().numpy()

            elif pooling_strategy == "mean":
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled_emb = sum_embeddings / sum_mask
                pooled_emb = pooled_emb.cpu().numpy()

            embeddings.extend(pooled_emb)

    return embeddings

def load_and_preprocess(train_path, test_path, convert=True):
    train = pd.read_csv(train_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')

    train.rename(columns={'sequence': 'seq_nt'}, inplace=True)
    test.rename(columns={'sequence': 'seq_nt'}, inplace=True)

    if convert:
        print("Converting from RNA to DNA...")
        train['seq_nt'] = train['seq_nt'].str.replace('U', 'T')
        test['seq_nt'] = test['seq_nt'].str.replace('U', 'T')

    train_seqs = train['seq_nt']
    test_seqs = test['seq_nt']
    train_labels = train['label']
    test_labels = test['label']

    return train_seqs, test_seqs, train_labels, test_labels

# Take the grid as an argument, train the model and fit only the best params on the test data
def test(grid, X_train, y_train, X_test, y_test, cancer, llm, pooling_strategy):

    # Fiting the data on the training data set
    grid.fit(X_train, y_train)

    # Returning the best k and parameters
    best_estimator = grid.best_estimator_
    
    # Evaluating on test set using best params only
    y_pred = best_estimator.predict(X_test) # test the model

    # Computing metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_mcc = matthews_corrcoef(y_test, y_pred)   
    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    try:
        if hasattr(best_estimator, "predict_proba"):
            y_probs = best_estimator.predict_proba(X_test)
            if y_probs.ndim == 2:
                auc = roc_auc_score(y_test, y_probs[:, 1])
            else:
                auc = roc_auc_score(y_test, y_probs)
        elif hasattr(best_estimator, "decision_function"):
            scores = best_estimator.decision_function(X_test)
            auc = roc_auc_score(y_test, scores)
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    cm = confusion_matrix(y_test, y_pred)
    sensitivity = specificity = None
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    classifier = best_estimator.named_steps['model']
    classifier_name = type(classifier).__name__
    os.makedirs(f"./models/{cancer.replace(' ', '_').lower().capitalize()}", exist_ok=True)

    if classifier_name == "RandomForestClassifier":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/{llm}_{pooling_strategy}_rf.pkl"
    elif classifier_name == "SVC":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/{llm}_{pooling_strategy}_svm.pkl"
    elif classifier_name == "MLPClassifier":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/{llm}_{pooling_strategy}_mlp.pkl"
    elif classifier_name == "LogisticRegression":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/{llm}_{pooling_strategy}_logreg.pkl"
    elif classifier_name == "GaussianNB":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/{llm}_{pooling_strategy}_nb.pkl"
    elif classifier_name == "XGBClassifier":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/{llm}_{pooling_strategy}_xgb.pkl"    
    elif classifier_name == "KNeighborsClassifier":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/{llm}_{pooling_strategy}_knn.pkl"
    else:
        output_file = f"{classifier_name}.pkl"

    with open(output_file, "wb") as f:
        dill.dump(best_estimator, f)
    print(f"Pipeline saved: {output_file}")

    return test_accuracy, test_mcc, test_macro_f1, report_df, auc, sensitivity, specificity, best_estimator

def evaluate_llm_models(train_path, test_path, cancer, llm, rna_model, pooling_strategy, output):
    os.makedirs(output, exist_ok=True)
    start_time = time.time()
    
    # Loadinf the gLM
    if llm == "dnabert2":
        print("DNABERT2...")
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        llm_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
    elif llm == "nt": 
        print("Nucleotide Transformer...")  
        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", trust_remote_code=True)
        llm_model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref", trust_remote_code=True)
        # llm_model = AutoModel.from_pretrained("./models_llms/Colon/nt/final_model", trust_remote_code=True)
    else:
        print("RNA-FM...")
        tokenizer = AutoTokenizer.from_pretrained("multimolecule/rnafm")
        llm_model = AutoModelForSequencePrediction.from_pretrained("multimolecule/rnafm")

    llm_model.eval()
    llm_model.to(device)
    if llm in ["dnabert2","nt"]:
        print("Converting to DNA...")
        train_seqs, test_seqs, train_labels, test_labels = load_and_preprocess(train_path, test_path)
    else:
        print("Not converting to DNA...")
        train_seqs, test_seqs, train_labels, test_labels = load_and_preprocess(train_path, test_path, convert=False)

    # Generate embeddings for train and test sets
    train_embeddings = get_embeddings(train_seqs.tolist(), tokenizer, llm_model, pooling_strategy)
    test_embeddings = get_embeddings(test_seqs.tolist(), tokenizer, llm_model, pooling_strategy)
    print("Embeddings generated successfully!")
    torch.cuda.empty_cache()
    
    X_train, X_test = train_embeddings, test_embeddings
    y_train, y_test = train_labels.values, test_labels.values
    
    df_train = pd.DataFrame(train_embeddings)
    df_train["label"] = train_labels
    df_test = pd.DataFrame(test_embeddings)
    df_test["label"] = test_labels
    
    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label']
    X_test = df_test.drop(columns=['label'])
    y_test = df_test['label']

    # Dictionaries to store metrics for the best k only
    metrics_dict = {'Accuracy': {}, 'MCC': {}, 'Macro F1': {}, 'AUC': {}, 'Sensitivity': {}, 'Specificity': {}}
    report_dfs = {'RF': pd.DataFrame(), 'SVM': pd.DataFrame(), 'MLP': pd.DataFrame(), 'LogReg': pd.DataFrame(),
                  'NB': pd.DataFrame(), 'XGB': pd.DataFrame(), 'KMeans': pd.DataFrame()}
    
    # Defining grids for hyparameters
    rf_grid = {
        'model__n_estimators': [200, 300, 400]
    }
    svm_grid = {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    }
    mlp_grid = {
        'model__hidden_layer_sizes': [(32,16), (100,), (128, 64)],
        'model__alpha': [0.0001, 0.001]
    }
    logreg_grid = {
        'model__C': [0.01, 0.1, 1, 10],
        'model__penalty': ['l1', 'l2']
    }
    xgb_grid = {
        'model__n_estimators': [200, 300, 400],
        'model__learning_rate': [0.01, 0.1]
    }
    knn_grid = {
        'model__n_neighbors': [3, 5, 7, 10]
    } 
    nb_grid = {
        'model__var_smoothing': [1e-9, 1e-8, 1e-7]
    }  

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=34)

    for model_name, base_model, param_grid in [
        ('RF', RandomForestClassifier(random_state=34), rf_grid),
        ('SVM', svm.SVC(probability=True, random_state=34), svm_grid),
        ('MLP', MLPClassifier(activation='relu',solver='adam',max_iter=3000, random_state=34,verbose=False, learning_rate = 'adaptive',learning_rate_init = 0.001), mlp_grid),
        ('LogReg', LogisticRegression(max_iter=1000, solver='liblinear', random_state=34, n_jobs=None), logreg_grid),
        ('NB', GaussianNB(), nb_grid),
        ('XGB', XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_jobs=-1, random_state=34, tree_method='hist'), xgb_grid),
        ('KNN', KNeighborsClassifier(), knn_grid)
    ]:
        pipeline = Pipeline([("model", base_model)])

        # Acoring for gridsearch
        scoring = {'accuracy': make_scorer(accuracy_score),'f1_macro': make_scorer(f1_score, average='macro'),'mcc': make_scorer(matthews_corrcoef)}
        # Aridsearch, refitting on mcc
        grid = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1, refit='mcc')
        # Obtaining results from the test data set using best params
        test_accuracy, test_mcc, test_macro_f1, report_df, auc, sensitivity, specificity, best_estimator = test(grid, X_train, y_train, X_test, y_test, cancer, llm, pooling_strategy)

        # Store metrics per model
        metrics_dict['Accuracy'][model_name] = test_accuracy
        metrics_dict['MCC'][model_name] = test_mcc
        metrics_dict['Macro F1'][model_name] = test_macro_f1
        metrics_dict['AUC'][model_name] = auc
        metrics_dict['Sensitivity'][model_name] = sensitivity
        metrics_dict['Specificity'][model_name] = specificity

        # Store classification report
        report_dfs[model_name] = report_df

        # Saving all results to CSV
        row = {
            'cancer': cancer,
            'model': model_name,
            'feature': llm,
            'best_params': json.dumps(grid.best_params_),
            'accuracy': test_accuracy,
            'f1': test_macro_f1,
            'mcc': test_mcc,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
        
        csv_path = f'{output}/{llm}_{pooling_strategy}_test_results_all.csv'
        write_header = not os.path.exists(csv_path)
        pd.DataFrame([row]).to_csv(csv_path, mode='a', header=write_header, index=False)
        # Saving cross val results
        cv_df = pd.DataFrame(grid.cv_results_)
        cv_flat = pd.concat([cv_df.drop(columns=['params']), pd.json_normalize(cv_df['params'])], axis=1)
        cv_flat['cancer'] = cancer
        # Appending to CSV
        cv_out_path = f'{output}/{model_name}_{llm}_{pooling_strategy}_cv_results_all.csv'
        write_header = not os.path.exists(cv_out_path)
        cv_flat.to_csv(cv_out_path, mode='a', header=write_header, index=False)

        # Save best estimator summary
        best_row = {
            'cancer': cancer,
            'model': model_name,
            'best_params': json.dumps(grid.best_params_),
            'best_score': grid.best_score_,
            'best_index': grid.best_index_,
            'feature': 'kmer'
        }
        best_out_path = f'{output}/{llm}_{pooling_strategy}_cv_best_summary.csv'
        write_header_best = not os.path.exists(best_out_path)
        pd.DataFrame([best_row]).to_csv(best_out_path, mode='a', header=write_header_best, index=False)

    if os.path.exists(f'{output}/{llm}_{pooling_strategy}_test_results_all.csv'):
        best_models_perfomance = pd.read_csv(f'{output}/{llm}_{pooling_strategy}_test_results_all.csv')
        try:
            display(best_models_perfomance[best_models_perfomance['cancer'] == cancer])
        except Exception:
            display(best_models_perfomance)
    print("--- %s seconds --- run time" % (time.time() - start_time))


def treat_files(llm, output):
    templates = [
        f"{llm}_*_cv_best_summary",
        f"{llm}_*_test_results_all",
        f"MLP_{llm}_*_cv_results_all",
        f"RF_{llm}_*_cv_results_all",
        f"SVM_{llm}_*_cv_results_all"
    ]

    for template in templates:
        files = glob.glob(os.path.join(output, f"{template}.csv"))
        dfs = [
            pd.read_csv(f).assign(pooling_type=re.search(r"_(cls|max|mean)_", f).group(1))
            for f in files
        ]
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(os.path.join(output, template.replace("*_", "") + ".csv"), index=False)
        print("Salvando os arquivos finais de resultados do LLM...")
        print(os.path.join(output, template.replace("*_", "") + ".csv"))


parser = argparse.ArgumentParser()
parser.add_argument("--llm", type=str, required=True, help="LLM")
parser.add_argument("--rna_model", action="store_true", help="Whether an RNA model is used")
parser.add_argument( "--dataset_path", type=str, default="./datasets/", help="Path to the datasets directory" )
parser.add_argument( "--output_path", type=str, default="./results/", help="Path to the output directory" )
args = parser.parse_args()
print("LLM:", args.llm)
print("RNA model:", args.rna_model)
print("Dataset path:", args.dataset_path)
print("Output path:", args.output_path)

directory = args.dataset_path
output_folder = args.output_path

folders = [name for name in os.listdir(directory)
          if os.path.isdir(os.path.join(directory, name))]
folders = sorted(folders)

for folder in folders:
    data_dir = f"{directory}/{folder}"
    cancer_name = folder.replace('_', ' ').title()
    print(f"Running the dataset: {cancer_name}...")

    for pooling_strategy in ['mean','max','cls']:
        print(f"Running the pooling: {pooling_strategy}...")
        evaluate_llm_models(
            f"{data_dir}/{folder}_train.tsv",
            f"{data_dir}/{folder}_test.tsv",
            cancer=cancer_name, llm=args.llm, rna_model=args.rna_model,
            pooling_strategy=pooling_strategy,
            output=f"./{output_folder}/{args.llm}"
        )

    treat_files(args.llm,f"./{output_folder}/{args.llm}")        
