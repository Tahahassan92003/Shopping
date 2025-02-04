# Shopping
This Python script is designed to load a dataset from a CSV file, preprocess it, and use a K-Nearest Neighbors (KNN) classifier to predict .
This Python script is designed to load a dataset from a CSV file, preprocess it, and use a **K-Nearest Neighbors (KNN)** classifier to predict whether a customer made a purchase (represented by the "Revenue" column) based on various features. The script splits the dataset into training and testing sets, trains the model, makes predictions, and evaluates the model using **sensitivity** and **specificity**. Here's a breakdown of the components:

### **Key Components and Workflow**:

1. **Loading and Preprocessing Data (`load_data`)**:
   - The data is loaded from a CSV file into a pandas DataFrame.
   - **Mapping categorical values to numerical**: 
     - **Month**: The 'Month' column is converted from text (e.g., 'Jan', 'Feb') to numerical values (0 for Jan, 1 for Feb, etc.) using a dictionary.
     - **VisitorType**: This column is converted into binary values (1 for 'Returning_Visitor' and 0 for others).
     - **Weekend**: This column is converted to an integer, where `True` becomes `1` and `False` becomes `0`.
   - **Missing Values**: Any missing values (`NaN`) in the dataset are filled with the **mean** of the respective column.
   - **Splitting Data**: The data is divided into features (evidence) and labels (target variable, i.e., 'Revenue'). The feature set (`evidence`) excludes the 'Revenue' column, while the target set (`labels`) contains the values of the 'Revenue' column.

2. **Training the Model (`train_model`)**:
   - The **KNeighborsClassifier** from `sklearn` is used with `n_neighbors=1`. This means that the classifier will predict the label based on the nearest neighbor.
   - The model is trained using the training data (`evidence` and `labels`), allowing it to learn the patterns for making predictions.

3. **Evaluating the Model (`evaluate`)**:
   - After the model makes predictions on the test set, the script calculates two performance metrics:
     - **Sensitivity (True Positive Rate)**: This is the proportion of actual positives (customers who made a purchase) that are correctly identified as such by the model.
     - **Specificity (True Negative Rate)**: This is the proportion of actual negatives (customers who did not make a purchase) that are correctly identified as such by the model.
   - The script counts the number of **True Positives (TP)** and **True Negatives (TN)** by comparing the actual labels with the predicted labels.

4. **Splitting Data for Training and Testing**:
   - **`train_test_split`** from `sklearn.model_selection` is used to split the data into training and test sets with a specified **test size** (40% of the data in this case).
   - This ensures that the model is trained on a portion of the data and tested on unseen data to evaluate its performance.

5. **Printing Results**:
   - After the model is trained and predictions are made, the script outputs the following information:
     - **Correct Predictions**: The number of predictions where the model's output matches the actual label.
     - **Incorrect Predictions**: The number of predictions where the model's output does not match the actual label.
     - **True Positive Rate (Sensitivity)**: The percentage of correctly identified purchases.
     - **True Negative Rate (Specificity)**: The percentage of correctly identified non-purchases.

### **Usage**:
To run the script, you must provide the path to a CSV file containing the dataset. For example:
```bash
python shopping.py data.csv
```

### **Detailed Breakdown**:

- **Input**: A CSV file containing a dataset with columns like 'Month', 'VisitorType', 'Weekend', 'Revenue', etc.
- **Processing**: 
  - Categorical columns ('Month', 'VisitorType', 'Weekend') are converted to numerical values.
  - Missing data is filled with the mean of the column.
  - The data is split into training and test sets.
- **Model**:
  - The K-Nearest Neighbors classifier is trained using the training data.
  - The classifier makes predictions on the test set.
- **Evaluation**:
  - The model’s performance is evaluated using **sensitivity** and **specificity**.
  - The results (correct, incorrect, sensitivity, specificity) are printed.

### **Key Libraries**:
- **pandas**: For handling the dataset and preprocessing.
- **sklearn**: 
  - For **KNeighborsClassifier** (machine learning model).
  - For **train_test_split** (data splitting).
  
### **Example Output**:
```bash
Unique values in 'Month' column: ['Feb' 'Mar' 'Apr' 'May' 'Jun' 'Jul' 'Aug' 'Sep' 'Oct' 'Nov' 'Dec' 'Jan']
Correct: 700
Incorrect: 300
True Positive Rate: 85.00%
True Negative Rate: 90.00%
```

### **Key Points**:
- The **K-Nearest Neighbors (KNN)** algorithm is a simple, instance-based learning method used for classification.
- **Sensitivity** and **Specificity** are essential metrics for evaluating classification models, especially in imbalanced datasets.
- The script provides a simple, end-to-end solution for training and evaluating a KNN model on customer data, with a focus on predicting whether a customer will make a purchase.



NOTE: USE YOUR OWN CSV OR A THIRD PASRTY CSV FILE.
