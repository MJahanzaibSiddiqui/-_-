import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


data = pd.read_csv("german_credit_data.csv")

X = data.drop("Risk", axis=1)
y = data["Risk"]

X = pd.get_dummies(X, columns=["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

conf_matrix = confusion_matrix(y_test, y_pred)

class_labels = ['Good Risk', 'Bad Risk']
class_accuracy = [conf_matrix[i, i] / sum(conf_matrix[i, :]) for i in range(len(class_labels))]

# Plot bar chart
plt.figure(figsize=(6, 4))
plt.bar(class_labels, class_accuracy, color='skyblue')
plt.xlabel('Risk')
plt.ylabel('Accuracy')
plt.title('Accuracy by Risk Category')
plt.ylim(0, 1)  # Set y-axis limit to range from 0 to 1
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
