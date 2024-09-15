Decision Tree Classifier for Product Purchase Prediction

Overview:
This task is part of my Data Science internship at Prodigy InfoTech, where I built a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. I used a dataset similar to the Bank Marketing dataset from the UCI Machine Learning Repository.

The main goal was to develop a machine learning model that can classify customers based on demographic and campaign data, indicating whether they will subscribe to a product ("yes") or not ("no").

Objective :
The objective of this task was to:
Load, clean, and preprocess the dataset to make it suitable for training a machine learning model.
Build a decision tree classifier using demographic and behavioral data as features.
Evaluate the classifier's performance using accuracy, confusion matrix, and classification report.
Visualize the decision tree for interpretability and understanding of the decision rules.

Key Features of the Code :
1)Data Loading :
The dataset was loaded using Pandas, with a semicolon (;) delimiter. The file bank-additional-full.csv was used for building the classifier.

2)Data Preprocessing :
Categorical features were encoded using LabelEncoder to convert strings into numerical values.
The target variable (y, whether the customer purchased the product) was encoded into binary labels: 1 for "yes" and 0 for "no."
Irrelevant features like duration were dropped, as they introduce leakage into the model (known only after customer contact).

3)Splitting the Dataset :
The dataset was split into training and testing sets (80% training, 20% testing) using the train_test_split function.

4)Model Training :
A decision tree classifier was initialized and trained using the DecisionTreeClassifier from Scikit-learn. The classifier was trained on the training set to learn decision rules based on the input features.

5)Model Evaluation :
The classifier’s performance was evaluated using:
Accuracy: Measures the proportion of correct predictions.
Confusion Matrix: Provides insight into true positives, true negatives, false positives, and false negatives.
Classification Report: Gives precision, recall, and F1-score for both classes.

6)Decision Tree Visualization :
The decision tree was visualized using tree.plot_tree() from Scikit-learn. To improve interpretability, the depth of the tree was limited, and the visualization was saved as a PNG image for viewing.

Learning Outcomes :
Gained experience in encoding categorical features using LabelEncoder for machine learning models.
Learned how to preprocess datasets for classification tasks by handling unnecessary columns and preparing the target variable.
Understood how to train and evaluate a decision tree classifier using Scikit-learn.
Acquired skills in visualizing decision trees and interpreting their structure to understand the decision-making process.

Tools & Technologies
Python (Version 3.x)
Pandas (For data handling)
Scikit-learn (For building the decision tree model)
Matplotlib (For visualization)
LabelEncoder (For encoding categorical variables)
DecisionTreeClassifier (For model training)


Memory updated
Here’s a detailed overview of the task you are working on with the decision tree classifier:

Task: Decision Tree Classifier for Product Purchase Prediction
Overview
This task is part of my Data Science internship at Prodigy InfoTech, where I built a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. I used a dataset similar to the Bank Marketing dataset from the UCI Machine Learning Repository.

The main goal was to develop a machine learning model that can classify customers based on demographic and campaign data, indicating whether they will subscribe to a product ("yes") or not ("no").

Objective
The objective of this task was to:

Load, clean, and preprocess the dataset to make it suitable for training a machine learning model.
Build a decision tree classifier using demographic and behavioral data as features.
Evaluate the classifier's performance using accuracy, confusion matrix, and classification report.
Visualize the decision tree for interpretability and understanding of the decision rules.
Key Features of the Code
Data Loading
The dataset was loaded using Pandas, with a semicolon (;) delimiter. The file bank-additional-full.csv was used for building the classifier.

Data Preprocessing

Categorical features were encoded using LabelEncoder to convert strings into numerical values.
The target variable (y, whether the customer purchased the product) was encoded into binary labels: 1 for "yes" and 0 for "no."
Irrelevant features like duration were dropped, as they introduce leakage into the model (known only after customer contact).
Splitting the Dataset
The dataset was split into training and testing sets (80% training, 20% testing) using the train_test_split function.

Model Training
A decision tree classifier was initialized and trained using the DecisionTreeClassifier from Scikit-learn. The classifier was trained on the training set to learn decision rules based on the input features.

Model Evaluation
The classifier’s performance was evaluated using:

Accuracy: Measures the proportion of correct predictions.
Confusion Matrix: Provides insight into true positives, true negatives, false positives, and false negatives.
Classification Report: Gives precision, recall, and F1-score for both classes.
Decision Tree Visualization
The decision tree was visualized using tree.plot_tree() from Scikit-learn. To improve interpretability, the depth of the tree was limited, and the visualization was saved as a PNG image for viewing.

Learning Outcomes
Gained experience in encoding categorical features using LabelEncoder for machine learning models.
Learned how to preprocess datasets for classification tasks by handling unnecessary columns and preparing the target variable.
Understood how to train and evaluate a decision tree classifier using Scikit-learn.
Acquired skills in visualizing decision trees and interpreting their structure to understand the decision-making process.
Tools & Technologies
Python (Version 3.x)
Pandas (For data handling)
Scikit-learn (For building the decision tree model)
Matplotlib (For visualization)
LabelEncoder (For encoding categorical variables)
DecisionTreeClassifier (For model training)

Sample Output :
Accuracy: X%
Confusion Matrix: Shows how well the model predicted both classes (purchase/no purchase).
Tree Visualization: Displays the decision-making path based on customer attributes like age, education, and campaign history.

This task provided valuable experience in working with real-world datasets, handling data preprocessing, building classification models, and evaluating machine learning performance.

