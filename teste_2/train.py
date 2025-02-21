import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from itertools import chain


def symmetrical_uncertainty(X, y):
    """Compute symmetrical uncertainty for each feature in X with respect to target y."""
    H_y = np.log2(len(np.unique(y)))
    mutual_info = mutual_info_classif(X, y)
    H_x = np.log2(X.shape[0])  # Assuming H(X) approximates log2(N) for discrete data
    su = 2 * mutual_info / (H_x + H_y)
    return su


class FCBF(BaseEstimator, TransformerMixin):
    """Fast Correlation-Based Filter for Feature Selection."""

    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.selected_features_ = []

    def fit(self, X, y):
        su_scores = symmetrical_uncertainty(X, y)
        features = [(score, i) for i, score in enumerate(su_scores) if score > self.threshold]
        features.sort(reverse=True)  # Sort by score descending
        self.selected_features_ = self._eliminate_redundant(features, X)
        return self

    def transform(self, X):
        return X[:, self.selected_features_]

    def _eliminate_redundant(self, features, X):
        selected = []
        for score, i in features:
            redundant = False
            for _, j in selected:
                su_ij = symmetrical_uncertainty(X[:, [i]], X[:, j])
                if su_ij >= score:
                    redundant = True
                    break
            if not redundant:
                selected.append((score, i))
        return [i for _, i in selected]


import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib  # For saving and loading models
from sklearn.decomposition import PCA

# Constants
INPUT_CSV = "open3l_results/openl3_embeddings_segmented_500_audio_name.csv"  # The features CSV
MODEL_PATH = "audio_classification_model_svm.pkl"  # Path to save/load the model
CLASS_COLUMN = "Song Name"  # The target column

# Step 1: Load the CSV
print("Loading data...")
data = pd.read_csv(INPUT_CSV,header=0)

# Ensure the target column exists
if CLASS_COLUMN not in data.columns:
    raise ValueError(f"Target column '{CLASS_COLUMN}' not found in CSV.")

# Check for and remove any extra headers accidentally parsed
if data.columns.duplicated().any():
    print("Detected duplicate headers. Attempting to fix...")
    data = pd.read_csv(INPUT_CSV, header=1)

# Step 2: Preprocess the target column
print("Preprocessing target column...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data[CLASS_COLUMN])


# Step 3: Preprocess feature columns
print("Preprocessing feature columns...")
X = data.drop(columns=[CLASS_COLUMN])  # Features

# Ensure all columns are consistently typed
for col in X.columns:
    if X[col].dtype == 'object':
        try:
            print(X[col])
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except ValueError:
            pass

# Handle non-numeric columns
non_numeric_columns = X.select_dtypes(include=['object']).columns
if not non_numeric_columns.empty:
    print(f"Non-numeric columns detected: {non_numeric_columns}")
    # Encode non-numeric columns using one-hot encoding
    column_transformer = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(), non_numeric_columns)],
        remainder="passthrough"  # Leave numeric columns unchanged
    )
    X = column_transformer.fit_transform(X)
else:
    print("All feature columns are numeric.")
normScaler1=StandardScaler()
normScaler1.fit(X)
X_prenorm=normScaler1.transform(X)

pca = PCA(n_components=3)



# Step 4: Split data into training and testing sets
print("Splitting data...")
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y )

# Step 5: Train or load the model
if 1:
# if not os.path.exists(MODEL_PATH):
    print("Training model...")
    # Initialize and train the model
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    model=NuSVC(nu=0.15,kernel='rbf', random_state=42)
    # model =
    model.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
else:
    print("Loading saved model...")
    # Load the trained model
    model = joblib.load(MODEL_PATH)

# Step 6: Perform inference
print("Performing inference...")
y_pred = model.predict(X_test)

# Decode predicted labels
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Step 7: Evaluate the model
print("Evaluating model...")
print("Accuracy:", accuracy_score(y_test, y_pred))
# Get the unique classes in y_test
unique_classes_in_test = sorted(set(y_test))

# Map the unique class indices to their corresponding labels
target_names = [label_encoder.inverse_transform([cls])[0] for cls in unique_classes_in_test]

print("Classification Report:\n", classification_report(
    y_test,
    y_pred,
    labels=unique_classes_in_test,
    target_names=target_names
))

# Use FCBF for feature selection
fcbf = FCBF(threshold=0.01)
pca = PCA(n_components=0.95, random_state=42)
scaler = StandardScaler()
classifier = NuSVC(nu=0.1, kernel='rbf', random_state=42)

pipeline = Pipeline([
    ("feature_selection", fcbf),
    ("dimensionality_reduction", pca),
    ("normalization", scaler),
    ("classification", classifier)
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict and evaluate
# y_pred = pipeline.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Classification pipeline

# # print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
#
# # Optional: Save predictions for further analysis
# predictions_df = pd.DataFrame(X_test.toarray() if hasattr(X_test, 'toarray') else X_test)  # Handle sparse matrices
# predictions_df[CLASS_COLUMN] = label_encoder.inverse_transform(y_test)  # Actual labels
# predictions_df["Predicted"] = y_pred_labels
# predictions_df.to_csv("predictions.csv", index=False)
# print("Predictions saved to 'predictions.csv'")

# Define classifiers
nusvc_classifier = NuSVC(nu=0.1, kernel='rbf', random_state=42)
logreg_classifier = LogisticRegression(random_state=42, max_iter=1000)

# Define pipelines
nusvc_pipeline = Pipeline([
    ("feature_selection", fcbf),
    ("dimensionality_reduction", PCA(n_components=0.95, random_state=42)),
    ("normalization", StandardScaler()),
    ("classification", nusvc_classifier)
])

logreg_pipeline = Pipeline([
    ("feature_selection", fcbf),
    ("dimensionality_reduction", PCA(n_components=0.95, random_state=42)),
    ("normalization", StandardScaler()),
    ("classification", logreg_classifier)
])

# Train and evaluate NuSVC
nusvc_pipeline.fit(X_train, y_train)
y_pred_nusvc = nusvc_pipeline.predict(X_test)
print("NuSVC Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nusvc))
print("Classification Report:\n", classification_report(y_test, y_pred_nusvc, target_names=label_encoder.classes_))

# Train and evaluate Logistic Regression
logreg_pipeline.fit(X_train, y_train)
y_pred_logreg = logreg_pipeline.predict(X_test)
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Classification Report:\n", classification_report(y_test, y_pred_logreg, target_names=label_encoder.classes_))