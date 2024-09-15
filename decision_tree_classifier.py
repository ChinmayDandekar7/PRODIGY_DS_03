# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Load dataset
file_path = r"C:\\Users\\hp\\OneDrive\\Desktop\\PRODIGY_DS_03\\bank-additional-full.csv"  
data = pd.read_csv(file_path, sep=';')

# Check for any missing values 
print("Missing values:\n", data.isnull().sum())

# Drop irrelevant or redundant features 
data = data.drop(columns=['duration'])

# Encoding categorical features
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'y':  # We don't want to encode the target variable yet
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Encode the target variable ('y') into 0 (no) and 1 (yes)
data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Define features (X) and target (y)
X = data.drop(columns=['y'])
y = data['y']

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=4)
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Visualize the Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['no', 'yes'], filled=True)
plt.show()

# Save the decision tree visualization to a file
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['no', 'yes'], filled=True)
plt.savefig("decision_tree.png", dpi=300)  # Save as PNG file
