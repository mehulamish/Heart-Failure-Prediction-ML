# README for Heart Failure Prediction Using Machine Learning

## Project Overview

This project focuses on predicting heart failure events using a variety of machine learning models and neural networks. The dataset used includes clinical records of heart failure patients. The overall goal is to build models that effectively predict whether a patient is at risk of death based on various health indicators.

### Key Features:
- **Dataset**: The dataset contains various features such as age, serum creatinine, ejection fraction, high blood pressure, serum sodium, smoking, and more, all of which are used to predict death events (i.e., whether the patient will face a heart failure event).
- **Target Variable**: `DEATH_EVENT`, which is a binary classification (0 for No Death, 1 for Death).

## New Additions and Enhancements for Better Results

1. **Combined Risk Factor**:
   - Created a new feature `combined_risk_factor` that sums up multiple risk factors into a single score. This provides a simplified representation of a patient’s overall risk level.
   - This feature considers factors such as age, serum creatinine, ejection fraction, high blood pressure, and smoking status.

2. **Time-of-Day Feature**:
   - Added a new `time_of_day` column by binning the `time` variable into four categories representing different quarters of the day. This allows exploration of time-dependent patterns, which may impact heart failure predictions.

3. **Multiple Model Training**:
   - **Models used**:
     - **Logistic Regression**
     - **Decision Tree**
     - **Random Forest**
   - These models are trained and evaluated to compare their performance on the task of predicting heart failure events. Random Forest achieved the highest accuracy in this case.

4. **Neural Network**:
   - Implemented a Multi-Layer Perceptron (MLP) neural network with hidden layers of size (100, 50) to explore a deeper, more complex model. The neural network also provided competitive accuracy compared to traditional machine learning models.

5. **Feature Scaling**:
   - Standardized all features using `StandardScaler` to ensure that models work more effectively, particularly the ones sensitive to feature magnitudes like Logistic Regression and Neural Networks.

## Model Performance Metrics

For each model, the following metrics were computed:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Score**

### Model Results:
- **Logistic Regression**:
  - Accuracy: 0.847
  - Precision: 0.770
  - Recall: 0.733
  - F1-Score: 0.751
  - ROC-AUC: 0.816
- **Decision Tree**:
  - Accuracy: 0.985
  - Precision: 0.980
  - Recall: 0.972
  - F1-Score: 0.976
  - ROC-AUC: 0.981
- **Random Forest**:
  - Accuracy: 0.993
  - Precision: 0.992
  - Recall: 0.986
  - F1-Score: 0.989
  - ROC-AUC: 0.991
- **Neural Network**:
  - Accuracy: 0.936

## Visualizations

1. **Correlation Heatmap**:
   - A heatmap showing the correlation between features, which highlights important relationships, such as the impact of certain health indicators on the target variable (`DEATH_EVENT`).

2. **Box Plot**:
   - A box plot showing the relationship between the combined risk factor and the occurrence of death events, giving insights into how the risk score aligns with patient outcomes.

3. **ROC Curve**:
   - A Receiver Operating Characteristic (ROC) curve for the Random Forest model to visualize its performance in distinguishing between death and no-death events.

4. **Decision Tree Visualization**:
   - A visualization of a specific tree from the Random Forest model to help understand decision-making paths and splits used in predictions.

5. **Count Plots and Stacked Bar Charts**:
   - Plots for analyzing death events by categories like time of day and sex to identify any patterns in the data.

6. **Crosstabs**:
   - Crosstabs showing relationships between features (e.g., time of day, sex, and diabetes) and death events, to provide additional context on how these factors influence outcomes.

## How to Use

1. **Install Dependencies**:
   Install the required libraries using pip:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. **Run the Notebook**:
   - Load the dataset and execute the cells sequentially to preprocess the data, train the models, and visualize the results.

3. **Evaluate Models**:
   - Analyze the model performances using the printed metrics and visualizations to decide the best approach for predicting heart failure in patients.

## Future Improvements
- Experiment with additional advanced machine learning techniques such as boosting algorithms (e.g., XGBoost, AdaBoost).
- Perform hyperparameter tuning to optimize the models.
- Explore other feature engineering methods to further improve the predictive power of the models.

