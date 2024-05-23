# ML-based-variant-prediction-by-incorporating-Dynamics-based-features
Variant effect prediction of  TEM-1 β-lactamases by incorporating Dynamic based features


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Load your Excel file
file_path = "/content/drive/MyDrive/roshan/TEM-1/WT_R_S.xlsx"
data = pd.read_excel(file_path)  # Use pd.read_excel for Excel files

# Extract relevant columns for PCA
features = data.iloc[:, 2:]  # Assuming columns 3 and onwards are the features

# Center and scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)
features=features.drop(columns=data.columns[:2])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)
print(scaled_data)


# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(scaled_data)
# Create a DataFrame with principal components
pc_columns = [f'PC{i}' for i in range(1, pca.n_components_ + 1)]
pc_df = pd.DataFrame(data=principal_components, columns=pc_columns)



###RANDOM FOREST
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Assuming `result_df` contains the DataFrame with principal components and other data
# Make sure to adapt this based on your actual DataFrame structure

# Extract features (principal components)
X = result_df.iloc[:, -pca.n_components_:]

# Extract target variable (e.g., 'GROUP' column)
y = result_df['GROUP']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)




##VARIOUS MODELS MCC, PRECISION, ACCURACY
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder

# Assuming `result_df` contains the DataFrame with principal components and other data
# Make sure to adapt this based on your actual DataFrame structure

# Encode the target variable
le = LabelEncoder()
result_df['GROUP'] = le.fit_transform(result_df['GROUP'])

# Extract features (principal components)
X = result_df.iloc[:, -pca.n_components_:]

# Extract target variable (encoded)
y = result_df['GROUP']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    'Linear Regression': LinearRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Perform 5-fold cross-validation and print results
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name} - Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}")

# Perform 10-fold cross-validation and print results
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=10)
    print(f"{name} - Mean Cross-Validation Accuracy (10-fold): {cv_scores.mean():.2f}")

# Train and evaluate models on the test set
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):  # Checking if the model has predict_proba method
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_prob = None
    if hasattr(model, 'score'):  # Checking if the model has a score method (for regression models)
        accuracy = model.score(X_test, y_test)
    else:
        accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n{name} - Test Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Matthews Correlation Coefficient: {mcc:.2f}")
    print("\nClassification Report:\n", classification_rep)

##SVM CROSS VALIDATION
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder

# Assuming `result_df` contains the DataFrame with principal components and other data
# Make sure to adapt this based on your actual DataFrame structure

# Encode the target variable
le = LabelEncoder()
result_df['GROUP'] = le.fit_transform(result_df['GROUP'])

# Extract features (principal components)
X = result_df.iloc[:, -pca.n_components_:]

# Extract target variable (encoded)
y = result_df['GROUP']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    'SVM': SVC()
}

# Perform 5-fold cross-validation and print results
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name} - Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}")


# Train and evaluate models on the test set
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):  # Checking if the model has predict_proba method
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_prob = None
    if hasattr(model, 'score'):  # Checking if the model has a score method (for regression models)
        accuracy = model.score(X_test, y_test)
    else:
        accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n{name} - Test Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Matthews Correlation Coefficient: {mcc:.2f}")
    print("\nClassification Report:\n", classification_rep)



##FEATURE IMPORTANCE
# Map 'S' and 'R' to numerical values for the target variable
data['GROUP'] = data['GROUP'].map({'S': 0, 'R': 1})

# Split the data into features (X) and target variable (y)
X = data.drop(['GROUP', 'TYPE'], axis=1)
X = X.drop(X.columns[526:532], axis=1)
y = data['GROUP']

# Initialize and train the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)

# Get feature importances from the trained Random Forest model
feature_importances = rf_classifier.feature_importances_

# Get the column names (feature names) from the DataFrame
feature_names = X.columns

# Create a DataFrame for easier sorting and plotting
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the DataFrame by Importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Take the top 15 variables
top_features = feature_importance_df.head(15)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Top 15 Variable Importance in Random Forest Model')
plt.show()

# Evaluate model accuracy using cross-validation
accuracy = cross_val_score(rf_classifier, X, y, cv=5).mean()
print(f"Model Accuracy: {accuracy:.2f}")



##GRAPH GENERATION FOR FREQUENCY DISTRIBUTION OF TOP 15 IMPORTANT FEATURES IDENTIFIED

import pandas as pd
import matplotlib.pyplot as plt

# List of Excel files
excel_files = ['Q39K.xlsx', 'E240K.xlsx', 'N175S.xlsx', 'E104K.xlsx', 'A237T.xlsx', 'G238S.xlsx']

# Initialize lists to store data for plotting
all_phi_columns = []
all_psi_columns = []

# Loop through each file and read the data
for file in excel_files:
    df = pd.read_excel(file)

    # Extract phi and psi columns
    phi_columns = [col for col in df.columns if 'phi' in col]
    psi_columns = [col for col in df.columns if 'psi' in col]

    all_phi_columns.append((file, phi_columns, df[phi_columns]))
    all_psi_columns.append((file, psi_columns, df[psi_columns]))

# Plot histograms for phi angles
fig, axes = plt.subplots(nrows=len(excel_files), ncols=len(all_phi_columns[0][1]), figsize=(20, 5 * len(excel_files)), sharey=True)
fig.suptitle('Frequency Distribution of Phi Angles', fontsize=16, y=1.05)
for i, (file, phi_columns, phi_data) in enumerate(all_phi_columns):
    for j, phi_angle in enumerate(phi_columns):
        axes[i, j].hist(phi_data[phi_angle], bins=10, range=(-180, 180), edgecolor='black')
        axes[i, j].set_title(f'{phi_angle} ({file})')
        axes[i, j].set_xlabel('Values')
        axes[i, j].set_xlim(-180, 180)  # Set x-axis limits from -180 to 180
        axes[i, j].set_ylim(0, 8000)  # Set y-axis limits from 0 to 8000
        axes[i, j].grid(True)
axes[0, 0].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot histograms for psi angles
fig, axes = plt.subplots(nrows=len(excel_files), ncols=len(all_psi_columns[0][1]), figsize=(20, 5 * len(excel_files)), sharey=True)
fig.suptitle('Frequency Distribution of Psi Angles', fontsize=16, y=1.05)
for i, (file, psi_columns, psi_data) in enumerate(all_psi_columns):
    for j, psi_angle in enumerate(psi_columns):
        axes[i, j].hist(psi_data[psi_angle], bins=10, range=(-180, 180), edgecolor='black')
        axes[i, j].set_title(f'{psi_angle} ({file})')
        axes[i, j].set_xlabel('Values')
        axes[i, j].set_xlim(-180, 180)  # Set x-axis limits from -180 to 180
        axes[i, j].set_ylim(0, 8000)  # Set y-axis limits from 0 to 8000
        axes[i, j].grid(True)
axes[0, 0].set_ylabel('Frequency')
plt.tight_layout()
plt.show()
