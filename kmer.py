from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, confusion_matrix, classification_report, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from IPython.display import display
import pandas as pd
import numpy as np
import argparse
import os
import time
import json
import dill

# Custom transformer for k-mer extraction and normalization
class KmerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k=3, dense_output=False):
        self.k = k
        self.dense_output = dense_output
        self.vectorizer = None

    def kmer_analyzer(self, seq):
        return [seq[i:i+self.k] for i in range(len(seq)-self.k+1)] if len(seq) >= self.k else []

    def fit(self, X, y=None):
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(self.k, self.k), lowercase=False)
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        X_counts = self.vectorizer.transform(X)  
        X_freq = normalize(X_counts, norm='l1', axis=1, copy=True)

        if self.dense_output == 'NB':
            return X_freq.toarray()
    
        return X_freq

def load_and_preprocess_kmer(train_path, test_path):
    train = pd.read_csv(train_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')

    train.rename(columns={'sequence': 'seq_nt'}, inplace=True)
    test.rename(columns={'sequence': 'seq_nt'}, inplace=True)

    train['seq_nt'] = train['seq_nt'].str.replace('U', 'T', regex=False)
    test['seq_nt'] = test['seq_nt'].str.replace('U', 'T', regex=False)

    train_seqs = train['seq_nt']
    test_seqs = test['seq_nt']
    train_labels = train['label']
    test_labels = test['label']

    return train_seqs, test_seqs, train_labels, test_labels

# Take the grid as an argument, train the model and fit only the best params on the test data
def test(grid, X_train, y_train, X_test, y_test, cancer):

    # Fiting the data on the training data set
    grid.fit(X_train, y_train)

    # Returning the best k and parameters
    best_k = grid.best_params_.get('kmer__k', None)
    best_estimator = grid.best_estimator_
    
    # Evaluating on test set using best params only
    y_pred = best_estimator.predict(X_test)

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
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/kmer_rf.pkl"
    elif classifier_name == "SVC":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/kmer_svm.pkl"
    elif classifier_name == "MLPClassifier":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/kmer_mlp.pkl"
    elif classifier_name == "LogisticRegression":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/kmer_logreg.pkl"
    elif classifier_name == "GaussianNB":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/kmer_nb.pkl"
    elif classifier_name == "XGBClassifier":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/kmer_xgb.pkl"    
    elif classifier_name == "KNeighborsClassifier":
        output_file = f"./models/{cancer.replace(' ', '_').lower().capitalize()}/kmer_knn.pkl"  
    else:
        output_file = f"{classifier_name}.pkl"

    with open(output_file, "wb") as f:
        dill.dump(best_estimator, f)
    print(f"Pipeline saved: {output_file}")

    return test_accuracy, test_mcc, test_macro_f1, report_df, auc, sensitivity, specificity, best_k, best_estimator


def evaluate_kmer_models(train_path, test_path, cancer, output):
    os.makedirs(output, exist_ok=True)
    start_time = time.time()

    train_seqs, test_seqs, train_labels, test_labels = load_and_preprocess_kmer(train_path, test_path) # load and process data
    X_train = train_seqs.tolist()
    X_test = test_seqs.tolist()
    y_train = train_labels.values
    y_test = test_labels.values

    # Dictionaries to store metrics for the best k only
    metrics_dict = {'Accuracy': {}, 'MCC': {}, 'Macro F1': {}, 'AUC': {}, 'Sensitivity': {}, 'Specificity': {}}
    report_dfs = {'RF': pd.DataFrame(), 'SVM': pd.DataFrame(), 'MLP': pd.DataFrame(), 'LogReg': pd.DataFrame(),
                  'NB': pd.DataFrame(), 'XGB': pd.DataFrame(), 'KMeans': pd.DataFrame()}
    
    # Defining grids for hyparameters
    rf_grid = {
        'model__n_estimators': [200, 300, 400],
    }
    svm_grid = {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    }
    mlp_grid = {
        'model__hidden_layer_sizes': [(32,16), (100,), (128, 64)],
        'model__alpha': [0.0001, 0.001],
        
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
        pipeline = Pipeline([
            ('kmer', KmerTransformer(dense_output=model_name)),
            ('model', base_model)
        ])

        # Include k as hyperparameter
        param_grid = dict(param_grid)  
        param_grid['kmer__k'] = [2, 3, 4, 5, 6, 7]
        # Scoring for gridsearch
        scoring = {'accuracy': make_scorer(accuracy_score),'f1_macro': make_scorer(f1_score, average='macro'),'mcc': make_scorer(matthews_corrcoef)}
        # Gridsearch, refitting on mcc
        grid = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1, refit='mcc')
        
        # Obtaining results from the test data set using best params
        test_accuracy, test_mcc, test_macro_f1, report_df, auc, sensitivity, specificity, best_k, best_estimator = test(grid,
                                                                                                                        X_train, y_train, X_test, y_test, cancer)

        # Store metrics per model
        metrics_dict['Accuracy'][model_name] = test_accuracy
        metrics_dict['MCC'][model_name] = test_mcc
        metrics_dict['Macro F1'][model_name] = test_macro_f1
        metrics_dict['AUC'][model_name] = auc
        metrics_dict['Sensitivity'][model_name] = sensitivity
        metrics_dict['Specificity'][model_name] = specificity

        # Store classification report
        report_df['k'] = best_k
        report_dfs[model_name] = report_df

        # Saving all results to CSV
        row = {
            'cancer': cancer,
            'model': model_name,
            'feature': 'kmer',
            'k': best_k,
            'best_params': json.dumps(grid.best_params_),
            'accuracy': test_accuracy,
            'f1': test_macro_f1,
            'mcc': test_mcc,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

        csv_path = f'{output}/kmer_test_results_all.csv'
        write_header = not os.path.exists(csv_path)
        pd.DataFrame([row]).to_csv(csv_path, mode='a', header=write_header, index=False)
        # Saving cross val results
        cv_df = pd.DataFrame(grid.cv_results_)
        cv_flat = pd.concat([cv_df.drop(columns=['params']), pd.json_normalize(cv_df['params'])], axis=1)
        cv_flat['cancer'] = cancer
        # Append to CSV
        cv_out_path = f'{output}/{model_name}_kmer_cv_results_all.csv'
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
        best_out_path = f'{output}/kmer_cv_best_summary.csv'
        write_header_best = not os.path.exists(best_out_path)
        pd.DataFrame([best_row]).to_csv(best_out_path, mode='a', header=write_header_best, index=False)

    if os.path.exists(f'{output}/kmer_test_results_all.csv'):
        best_models_perfomance = pd.read_csv(f'{output}/kmer_test_results_all.csv')
        # display rows for this cancer if present
        try:
            display(best_models_perfomance[best_models_perfomance['cancer'] == cancer])
        except Exception:
            display(best_models_perfomance)
    print("--- %s seconds --- run time" % (time.time() - start_time))


parser = argparse.ArgumentParser()
parser.add_argument( "--dataset_path", type=str, default="./datasets/", help="Path to the datasets directory" )
parser.add_argument( "--output_path", type=str, default="./results/", help="Path to the output directory" )
args = parser.parse_args()
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
    
    evaluate_kmer_models(
        f"{data_dir}/{folder}_train.tsv",
        f"{data_dir}/{folder}_test.tsv",
        cancer=cancer_name, output=f"./{output_folder}/kmer"
    )
