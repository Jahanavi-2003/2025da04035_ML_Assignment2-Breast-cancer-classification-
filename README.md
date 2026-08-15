# Breast Cancer Classification Using Machine Learning

## a. Problem Statement

Breast cancer is one of the most common forms of cancer, and early and accurate diagnosis is important for effective treatment. Machine learning classification techniques can be used to analyze medical diagnostic features and classify tumors as either benign or malignant.

The objective of this project is to implement and compare different machine learning classification algorithms on the Breast Cancer Wisconsin (Diagnostic) dataset. The models are evaluated using multiple performance metrics to determine which model performs best for the selected dataset.

The classification models implemented in this project are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

The models are evaluated using the following metrics:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

## b. Dataset Description

### Dataset Name

**Breast Cancer Wisconsin (Diagnostic) Dataset**

### Dataset Source

The dataset was obtained from a public dataset repository on Kaggle.

### Dataset Characteristics

The dataset is a binary classification dataset containing diagnostic measurements of breast cancer tumors.

The target variable represents two classes:

* **Benign (B)**
* **Malignant (M)**

The dataset contains:

* **569 instances**
* **30 numerical features**
* **2 target classes**

The dataset satisfies the minimum requirements specified in the assignment:

| Requirement         |      Minimum Required | Dataset |
| ------------------- | --------------------: | ------: |
| Number of Features  |                    12 |      30 |
| Number of Instances |                   500 |     569 |
| Classification Type | Binary or Multi-class |  Binary |

The ID column was removed because it is an identifier and does not provide meaningful predictive information for classification.

The target variable was encoded as:

* Benign → 0
* Malignant → 1

### Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset from the CSV file.
2. Checked the dataset dimensions.
3. Checked for missing values.
4. Removed the ID column.
5. Separated the independent variables and target variable.
6. Encoded the target variable into numerical values.
7. Divided the dataset into training and testing sets using an 80:20 split.
8. Feature scaling was applied for models that require standardized numerical features, such as Logistic Regression and kNN.

---

## c. GitHub Repository Link

The complete project, including the application, requirements file, README, test data, and saved machine learning models, is available in the GitHub repository:

**GitHub Repository:**
  https://github.com/Jahanavi-2003/2025da04035_ML_Assignment2-Breast-cancer-classification-

---

## d. Models Used

Five classification models were implemented using the same dataset.

### 1. Logistic Regression

Logistic Regression is a supervised classification algorithm that predicts the probability of an observation belonging to a particular class.

### 2. Decision Tree

Decision Tree Classifier uses a tree-based structure to make classification decisions by recursively splitting the dataset based on feature values.

### 3. K-Nearest Neighbors (kNN)

KNN classifies an observation based on the classes of its nearest neighboring observations.

### 4. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm that assumes that continuous features follow a Gaussian distribution within each class.

### 5. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees and aggregates their predictions to improve classification performance and reduce overfitting.

---

# Evaluation Metrics

The models were evaluated using six performance metrics.

### Accuracy

Accuracy measures the proportion of correctly classified observations out of all observations.

### AUC Score

AUC represents the Area Under the Receiver Operating Characteristic Curve. A higher AUC indicates better ability to distinguish between the two classes.

### Precision

Precision measures the proportion of predicted positive observations that are actually positive.

### Recall

Recall measures the proportion of actual positive observations that are correctly identified.

### F1 Score

F1 Score is the harmonic mean of Precision and Recall and provides a balance between the two metrics.

### Matthews Correlation Coefficient (MCC)

MCC measures the quality of binary classifications by considering true positives, true negatives, false positives, and false negatives.

---

# Model Comparison

The following table presents the evaluation results obtained from the implemented classification models.

| ML Model Name            |     Accuracy |      AUC |    Precision |       Recall |           F1 |          MCC |
| ------------------------ | -----------: | -------: | -----------: | -----------: | -----------: | -----------: |
| Logistic Regression      |     0.964912 | 0.996032 |     0.975000 |     0.928571 |     0.951220 |     0.924518 |
| Decision Tree            |     0.929825 | 0.924603 |     0.904762 |     0.904762 |     0.904762 |     0.849206 |
| kNN                      |     0.956140 | 0.982308 |     0.974359 |     0.904762 |     0.938272 |     0.905824 |
| Gaussian Naive Bayes     |     0.938596 | 0.993386 |     1.000000 |     0.833333 |     0.909091 |     0.871489 |
| Random Forest (Ensemble) | **0.973684** | 0.992890 | **1.000000** | **0.928571** | **0.962963** | **0.944155** |

---

# Observations on Model Performance

| ML Model Name                | Observation about Model Performance                                                                                                                                                                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Logistic Regression**      | Logistic Regression performed very well, achieving an accuracy of 96.49% and the highest AUC score of 0.9960 among all models. Its high AUC indicates excellent ability to distinguish between benign and malignant cases.                                                           |
| **Decision Tree**            | Decision Tree achieved an accuracy of 92.98%, which was the lowest accuracy among the implemented models. Its AUC of 0.9246 and MCC of 0.8492 were also lower than those of the other models, indicating comparatively weaker performance on this dataset.                           |
| **kNN**                      | kNN achieved an accuracy of 95.61% and an AUC of 0.9823. It provided good overall performance, although its recall of 90.48% was slightly lower than Logistic Regression and Random Forest.                                                                                          |
| **Gaussian Naive Bayes**     | Gaussian Naive Bayes achieved an accuracy of 93.86% and a very high AUC of 0.9934. It achieved perfect precision of 1.0000, meaning that all observations predicted as malignant were actually malignant. However, its recall of 83.33% was the lowest among the implemented models. |
| **Random Forest (Ensemble)** | Random Forest achieved the highest accuracy of 97.37%, the highest F1 score of 0.9630, and the highest MCC score of 0.9442. It also achieved perfect precision of 1.0000 and a recall of 92.86%, making it the strongest overall model for this dataset.                             |

---

# Best Model According to Each Metric

The best-performing model for each evaluation metric was:

| Metric         | Best Model                           |      Score |
| -------------- | ------------------------------------ | ---------: |
| Best Accuracy  | Random Forest                        | **0.9737** |
| Best AUC Score | Logistic Regression                  | **0.9960** |
| Best Precision | Gaussian Naive Bayes / Random Forest | **1.0000** |
| Best Recall    | Logistic Regression / Random Forest  | **0.9286** |
| Best F1 Score  | Random Forest                        | **0.9630** |
| Best MCC Score | Random Forest                        | **0.9442** |

---

# Overall Winner

## Random Forest

Based on the overall results, **Random Forest is the overall winner for this dataset**.

Random Forest achieved the highest accuracy of **97.37%**, the highest F1 score of **0.9630**, and the highest MCC score of **0.9442**. It also achieved a precision of **1.0000** and recall of **0.9286**.

Although Logistic Regression achieved the highest AUC score of **0.9960**, Random Forest demonstrated stronger overall performance across the majority of the evaluation metrics.

Therefore, **Random Forest is selected as the best overall classification model for the Breast Cancer Wisconsin dataset.**

---

# Conclusion

The five classification models were successfully implemented and evaluated using the same Breast Cancer Wisconsin dataset.

Among the implemented models, Random Forest demonstrated the best overall performance, achieving an accuracy of **97.37%**. Logistic Regression achieved the highest AUC score of **0.9960**, indicating excellent class discrimination. Gaussian Naive Bayes and Random Forest both achieved perfect precision of **1.0000**.

The results demonstrate that ensemble-based Random Forest classification is highly effective for this dataset and provides the strongest overall balance between the different evaluation metrics.

---



