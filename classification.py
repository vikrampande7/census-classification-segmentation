import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib
import shap

column_names = []
with open('data/census-bureau.columns', 'r') as f:
    for line in f:
        column_names.append(line.strip().replace(" ","_"))

df = pd.read_csv("data/census-bureau.data", delimiter=",")
df.columns = column_names

df = df.copy()
df['label'] = df['label'].map({'- 50000.':0, '50000+.':1})
df['label'].value_counts()

X = df.drop(columns=['label'])
y = df['label']

# Get the Numeric and Categorical Features Separately
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Transform the Features to Impute Missing values, with methods like (median, most_frequent, mean, etc)
# Scale the numeric values
# Encode the categorical values
numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
)

# Preprocess with the above Transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Divide the dataset into train test and validation
# Keep a part of data completete UNSEEN as test set for inference
# Out of the remaining data, divide the data furthermore into training and validation to get offline metrics
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=7)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1765, stratify=y_train_full, random_state=7)

# SMOTENC for mixed types, SMOTE for numeric
smote = SMOTENC(
    categorical_features = [X.columns.get_loc(c) for c in categorical_features], random_state = 7
)

# Define Model Pipelines

# Logistic Regression
lrc = ImbPipeline(
    steps=[
        ('preprocessor', preprocessor),
        #('smote', SMOTE(sampling_strategy='auto', random_state=7)),
        #('smote', smote),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=1))
    ]
)

# Random Forest Classifier
rfc = ImbPipeline(
    steps=[
        ('preprocessor', preprocessor),
        #('smote', smote),
        ('clf', RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=7))
    ]
)

# Histogram Gradient Boosting
hgb = ImbPipeline(
    steps=[
        ('preprocessor', preprocessor),
        #('smote', smote),
        ('clf', HistGradientBoostingClassifier(random_state=7))
    ]
)

logistic_regression_params = {
    'clf__C': np.logspace(-2,2,20),
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs', 'liblinear']
}

random_forest_params = {
    'clf__n_estimators': [200],
    'clf__max_depth': [10],
    'clf__min_samples_split': [2],
    'clf__min_samples_leaf': [1],
    'clf__max_features': ['sqrt']
}

hist_gb_params = {
    'clf__learning_rate': [0.1],
    'clf__max_depth': [5],
    'clf__max_leaf_nodes': [31],
    'clf__min_samples_leaf': [10],
    'clf__l2_regularization': [0.0]
}

# Stratified Cross Validation Step
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

# Randomized search CV per model
def run_search(pipeline, param_config, X_train, y_train, cv, n_iter=25):
  search = RandomizedSearchCV(
      pipeline,
      param_distributions=param_config,
      n_iter=n_iter,
      scoring='roc_auc',
      cv=cv,
      n_jobs=1,
      verbose=3,
      random_state=7
  )
  search.fit(X_train, y_train)
  return search

# Train the models
hist_grad_boost_search = run_search(pipeline=hgb, param_config=hist_gb_params, X_train=X_train, y_train=y_train, cv=cv)
random_forests_search = run_search(pipeline=rfc, param_config=random_forest_params, X_train=X_train, y_train=y_train, cv=cv)
logistic_regression_search = run_search(pipeline=lrc, param_config=logistic_regression_params, X_train=X_train, y_train=y_train, cv=cv)

results = [
    ('LogisticRegression', logistic_regression_search.best_score_, logistic_regression_search.best_estimator_),
    ('RandomForests', random_forests_search.best_score_, random_forests_search.best_estimator_),
    ('HistogramGradientBoost', hist_grad_boost_search.best_score_, hist_grad_boost_search.best_estimator_)
]

# Get top results
top_results = sorted(results, key=lambda x:x[1], reverse=True)

# Get the Best Model
best_model_name, best_cv_auc, best_model = top_results[0]
print(f"Best Model: {best_model_name} with CV ROC-AUC={best_cv_auc: .4f}")
print("Best params: ", best_model.get_params())

best_model.fit(X_train, y_train)

# Predict the probabilities on validation dataset
y_val_prob = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, 'predict_proba') else best_model.decision_function(X_val)

# Threshold to maximize the F1 metrics on validatino dataset
thresholds = np.linspace(0.1, 0.9, 81)
f1s = []
for t in thresholds:
  y_val_pred = (y_val_prob >= t).astype(int)
  f1s.append(f1_score(y_val, y_val_pred))
best_t = thresholds[int(np.argmax(f1s))]
print(f"Chosen Threshold={best_t} (F1 on validation={max(f1s)})")

# Final Evaluation on Test Dataset
y_test_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else best_model.decision_function(X_test)
y_test_pred = (y_test_proba >= best_t).astype(int)

test_metrics = {
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1': f1_score(y_test, y_test_pred),
    'ROC_AUC': roc_auc_score(y_test, y_test_pred),
    'PR_AUC': average_precision_score(y_test, y_test_pred)
}

print("Test Metrics ", {key: round(value, 4) for key, value in test_metrics.items()})

fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_proba)

## Save the model and use for deployment inference
joblib.dump({
    'model': best_model,
    'threshold': float(best_t),
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
}, 'classifier.joblib')

## Use the deployed model for inference
artifact = joblib.load('classifier.joblib')
model = artifact['model']
threshold = artifact['threshold']

prob = model.predict_proba(X_test[:10])[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test[:10])
pred = (prob >= threshold).astype(int)