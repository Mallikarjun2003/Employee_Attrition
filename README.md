## Employee Attrition Prediction using Random Forest

This project investigates employee attrition using a Random Forest classification model. It employs the `scikit-learn` library for data manipulation, model building, and evaluation.

### Project Structure

The project is organized as follows:

- `Employee_Attrition.ippynb`: Main Python script for data analysis, model development, and evaluation.
- `Employee_Attrition_data`: Cntaining the dataset (replace with the actual path to your CSV file).

### Dependencies

This project requires the following Python libraries:

- pandas
- numpy
- matplotlib.pyplot
- seaborn
- sklearn (specifically `train_test_split`, `LabelEncoder`, `StandardScaler`, `RandomForestClassifier`, `confusion_matrix`, `accuracy_score`, `classification_report`, `cross_val_score`, `GridSearchCV`)

### Running the Script

1. Ensure you have the required libraries installed (`pip install <library_name>` for each library).
2. Replace the file path in `data = pd.read_csv("C:\\Users\\91868\\OneDrive\\Desktop\\DATA SCIENCE PROJECTS\\RandomForest\\Employee_Attrition_data.csv")` with the actual location of your CSV file.
3. Run the script from your terminal using `python employee_attrition.py`.

### Project Overview

The script performs the following steps:

1. **Data Loading and Exploration:** Loads the employee data from a CSV file, explores its shape, content, data types, and summary statistics.
2. **Data Cleaning:** Identifies and removes duplicate entries, checks for missing values (which you may need to address in a future iteration).
3. **Data Understanding:** Analyzes the relationship between employee departure and features like salary and department using visualizations.
4. **Feature Engineering:** Converts categorical features (salary, department) into numerical representations suitable for the model.
5. **Model Building and Evaluation:**
   - Splits the data into training and testing sets.
   - Applies standard scaling to numerical features in the training set and transforms the testing set using the fitted scaler.
   - Trains a Random Forest classification model on the scaled training data.
   - Makes predictions on the test set.
   - Evaluates the model's performance using:
      - Confusion matrix to visualize correct and incorrect predictions.
      - Accuracy score to measure overall prediction accuracy.
      - Heatmap to visually represent the confusion matrix.
      - Classification report for detailed metrics like precision, recall, and F1-score.
6. **Feature Importance:** Extracts feature importances from the Random Forest model to identify relatively more important features for prediction.
7. **K-Fold Cross-Validation:** Performs k-fold cross-validation to obtain a more robust estimate of the model's generalization performance.
8. **Hyperparameter Tuning:**
   - Defines a grid of hyperparameter values to search for optimal settings for the Random Forest model (n_estimators and max_features).
   - Performs GridSearchCV to find the best hyperparameter combination based on the chosen scoring metric (accuracy in this case).
9. **New Model Building with Best Hyperparameters:** Creates a new Random Forest model with the identified best hyperparameters and refits it to potentially improve performance.
10. **K-Fold Cross-Validation (Again):** Re-evaluates the model's performance with the tuned hyperparameters using k-fold cross-validation.

### Further Improvements

The project provides a solid foundation for employee attrition prediction using Random Forest. Here are some potential areas for future exploration:

- Address missing values in the data (e.g., imputation or deletion).
- Experiment with different hyperparameter values or try other classification algorithms for comparison.
- Save the trained model for future predictions using `joblib` or pickle.
- Explore feature engineering techniques like dimensionality reduction or creating new features.
- Implement feature selection to identify the most informative features for the model.


This README provides a clear overview of the project's purpose, structure, dependencies, usage instructions, and potential areas for improvement. 
