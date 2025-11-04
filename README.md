# New-Project.ML--III
The ML PROJECT-III is about Heart Disease dataset 
# Data handling
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Loading Dataset 
df = pd.read_csv("DOC-20251103-WA0019..")

df.head()

# Dataset information
print("Dataset Info:")
df.info()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical summary
print("\nDataset Summary:")
print(df.describe())

# Target Distribution 
sns.countplot(x='target', data=df, palette='coolwarm')
plt.title("Heart Disease Distribution (1 = Disease, 0 = No Disease)")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='Reds')
plt.title("Feature Correlation Heatmap")
plt.show()

# Age vs Max heart
plt.figure(figsize=(8,5))
sns.scatterplot(x='age', y='thalach', hue='target', data=df, palette='coolwarm')
plt.title("Age vs Max Heart Rate (Colored by Target)")
plt.show()

# Split the Data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
# (a) Linear regression 
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Model Evaluation 
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Predict for new patient datset
# Example patient data
sample = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
sample_scaled = scaler.transform(sample)

prediction = rf_model.predict(sample_scaled)

if prediction[0] == 1:
    print(" Patient is likely to have Heart Disease.")
else:
    print(" Patient is unlikely to have Heart Disease.")




