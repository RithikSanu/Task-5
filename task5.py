# Decision Trees and Random Forests - Heart Disease Dataset
# Install needed packages if not already installed
# !pip install scikit-learn pandas matplotlib graphviz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import graphviz

# 1. Load Dataset
df = pd.read_csv('heart.csv')  # Make sure heart.csv is in the same directory
X = df.drop('target', axis=1)
y = df['target']

# 2. Train/Test Split and Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Train a Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")

# 4. Visualize the Decision Tree
dot_data = export_graphviz(dt, out_file=None,
                           feature_names=X.columns,
                           class_names=["No Disease", "Disease"],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)  # saves as decision_tree.png

# 5. Control Tree Depth to Avoid Overfitting
dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_shallow.fit(X_train, y_train)
print(f"Shallow Tree Accuracy (max_depth=3): {dt_shallow.score(X_test, y_test):.4f}")

# 6. Train Random Forest and Compare
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# 7. Feature Importances
importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.barh(range(len(importances)), importances[indices], align='center')
plt.yticks(range(len(importances)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()

# 8. Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"Random Forest Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# 9. Summary Output
print("\nSummary:")
print("Decision Tree overfits on training data. Controlling depth reduces overfitting.")
print("Random Forest performs better due to ensembling (bagging).")
print("Feature importance helps understand key predictors.")
