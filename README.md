# Hotel Booking Status Prediction

This project is a machine learning-based implementation for predicting hotel booking statuses (canceled or confirmed) using various classification algorithms. The model is trained on a dataset of hotel reservations and leverages several preprocessing techniques to enhance prediction accuracy.

## Features:
- **Data Cleaning**:
  - Removal of unnecessary columns (`Booking_ID`).
  - Label encoding of categorical variables.
  - Conversion of data types to numeric for efficient processing.
- **Data Visualization**:
  - Correlation heatmap and count plots for target variable (`booking_status`).
  - Box plots to analyze feature distribution.
- **Handling Imbalanced Data**:
  - Use of SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
- **Model Training**:
  - **Random Forest Classifier**
  - **Logistic Regression**
  - **Decision Tree Classifier**
  - **Naive Bayes**
  - **Support Vector Machine (SVM)**
- **Performance Evaluation**:
  - Accuracy scores and classification reports for each model to assess performance.

## How to Use:
1. **Load Data**: Ensure the dataset is correctly loaded from the specified path.
2. **Data Preprocessing**: Execute the preprocessing steps including label encoding and scaling.
3. **Model Training and Evaluation**: Train various classification models and evaluate their performance using accuracy and classification reports.
4. **Visualization**: Generate plots to visualize feature distributions and correlations.

## Example Code:
```python
# Load dataset
data = pd.read_csv('C:/Users/Faris Hassan/Downloads/hotel/Hotel Reservations.csv')

# Drop unnecessary columns
data = data.drop('Booking_ID', axis=1)

# Encode categorical variables
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col].astype(str))

# Train and evaluate models
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
