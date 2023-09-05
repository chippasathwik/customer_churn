import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shap
import matplotlib.pyplot as plt

# Title and description
st.title("Customer Churn Prediction Dashboard")
st.write("This dashboard allows you to load a customer churn dataset, perform machine learning model training, and visualize feature importance.")

# Upload dataset
uploaded_file = st.file_uploader("Upload your customer churn dataset (Excel)", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    # Drop non-numeric columns like 'CustomerID' and 'Name'
    data = data.drop(['CustomerID', 'Name'], axis=1)

    # Encode categorical variables like 'Gender' and 'Location' using one-hot encoding
    data = pd.get_dummies(data, columns=['Gender', 'Location'], drop_first=True)

    # Define your features and target variable
    X = data.drop('Churn', axis=1)  # Assuming 'Churn' is the target variable
    y = data['Churn']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to balance the classes
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Define a parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],           # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],           # Maximum depth of each tree
        'min_samples_split': [2, 5, 10],          # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],            # Minimum number of samples required to be at a leaf node
        'max_features': ['sqrt', 'log2'],         # Number of features to consider at each split
    }

    # Initialize GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the resampled training data to find the best hyperparameters
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Get the best hyperparameters from the grid search
    best_params = grid_search.best_params_

    # Create a Random Forest Classifier with the best hyperparameters
    best_rf_classifier = RandomForestClassifier(random_state=42, **best_params)

    # Train the best model on the resampled training data
    best_rf_classifier.fit(X_train_resampled, y_train_resampled)

    # Make predictions on the test data
    y_pred = best_rf_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate precision
    precision = precision_score(y_test, y_pred)

    # Calculate recall
    recall = recall_score(y_test, y_pred)

    # Calculate F1-score
    f1 = f1_score(y_test, y_pred)

    # Generate and print the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # SHAP values for feature importance
    explainer = shap.TreeExplainer(best_rf_classifier)
    shap_values = explainer.shap_values(X_test)

    # Summary plot of feature importance
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    # Display results
    st.header("Model Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1-Score: {f1:.4f}")
    st.subheader("Confusion Matrix")
    st.write(conf_matrix)

    st.header("Feature Importance")
    st.pyplot(plt)

else:
    st.write("Please upload a dataset to get started.")
