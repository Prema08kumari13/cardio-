"""
Cardiovascular Disease Prediction Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CARDIOVASCULAR DISEASE PREDICTION")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('cardio_train (1).csv', sep=';')
print(f"\nDataset loaded: {len(df)} records")

# ============================================================
# 2. DATA PRE-PROCESSING
# ============================================================
print("\n--- DATA PRE-PROCESSING ---")

# Drop ID column
df = df.drop('id', axis=1)

# Convert age from days to years
df['age'] = df['age'] / 365.25

# Calculate BMI
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

# Remove outliers
df = df[(df['ap_hi'] > 50) & (df['ap_hi'] < 250)]
df = df[(df['ap_lo'] > 30) & (df['ap_lo'] < 200)]
df = df[df['ap_lo'] < df['ap_hi']]

print(f"After cleaning: {len(df)} records")
print(f"Features: {list(df.columns)}")

# ============================================================
# 3. VISUALIZATIONS
# ============================================================
print("\n--- CREATING VISUALIZATIONS ---")

# Plot 1: Target distribution
plt.figure(figsize=(6, 5))
df['cardio'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'salmon'])
plt.title('Heart Disease Distribution')
plt.ylabel('')
plt.savefig('1_target_distribution.png')
plt.close()

# Plot 2: Age distribution
plt.figure(figsize=(8, 5))
plt.hist(df['age'], bins=30, color='steelblue', edgecolor='black')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.savefig('2_age.png')
plt.close()

# Plot 3: BMI distribution
plt.figure(figsize=(8, 5))
plt.hist(df['bmi'], bins=30, color='green', edgecolor='black')
plt.xlabel('BMI')
plt.ylabel('Count')
plt.title('BMI Distribution')
plt.savefig('3_bmi.png')
plt.close()

# Plot 4: Cholesterol vs Disease
plt.figure(figsize=(8, 5))
pd.crosstab(df['cholesterol'], df['cardio']).plot(kind='bar', color=['lightblue', 'salmon'])
plt.title('Cholesterol vs Heart Disease')
plt.xlabel('Cholesterol Level')
plt.legend(['No Disease', 'Heart Disease'])
plt.savefig('4_cholesterol.png')
plt.close()

# Plot 5: Glucose vs Disease
plt.figure(figsize=(8, 5))
pd.crosstab(df['gluc'], df['cardio']).plot(kind='bar', color=['lightblue', 'salmon'])
plt.title('Glucose vs Heart Disease')
plt.xlabel('Glucose Level')
plt.legend(['No Disease', 'Heart Disease'])
plt.savefig('5_glucose.png')
plt.close()

# Plot 6: Smoking vs Disease
plt.figure(figsize=(8, 5))
pd.crosstab(df['smoke'], df['cardio']).plot(kind='bar', color=['lightblue', 'salmon'])
plt.title('Smoking vs Heart Disease')
plt.legend(['No Disease', 'Heart Disease'])
plt.savefig('6_smoking.png')
plt.close()

# Plot 7: Physical Activity vs Disease
plt.figure(figsize=(8, 5))
pd.crosstab(df['active'], df['cardio']).plot(kind='bar', color=['lightblue', 'salmon'])
plt.title('Physical Activity vs Heart Disease')
plt.legend(['No Disease', 'Heart Disease'])
plt.savefig('7_activity.png')
plt.close()

# Plot 8: Blood Pressure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df['ap_hi'], bins=30, color='orange', edgecolor='black')
axes[0].set_title('Systolic BP')
axes[0].set_xlabel('mmHg')
axes[1].hist(df['ap_lo'], bins=30, color='purple', edgecolor='black')
axes[1].set_title('Diastolic BP')
axes[1].set_xlabel('mmHg')
plt.savefig('8_blood_pressure.png')
plt.close()

# Plot 9: Age by Disease
plt.figure(figsize=(8, 5))
df.boxplot(column='age', by='cardio')
plt.title('Age by Heart Disease')
plt.suptitle('')
plt.savefig('9_age_boxplot.png')
plt.close()

# Plot 10: BP by Disease
plt.figure(figsize=(8, 5))
df.boxplot(column='ap_hi', by='cardio')
plt.title('Systolic BP by Heart Disease')
plt.suptitle('')
plt.savefig('10_bp_boxplot.png')
plt.close()

# Plot 11: BMI by Disease
plt.figure(figsize=(8, 5))
df.boxplot(column='bmi', by='cardio')
plt.title('BMI by Heart Disease')
plt.suptitle('')
plt.savefig('11_bmi_boxplot.png')
plt.close()

# Plot 12: Scatter - Age vs BP
plt.figure(figsize=(10, 6))
for i in [0, 1]:
    subset = df[df['cardio'] == i]
    plt.scatter(subset['age'], subset['ap_hi'], c='blue' if i == 0 else 'red', 
                alpha=0.5, label='No Disease' if i == 0 else 'Heart Disease')
plt.xlabel('Age')
plt.ylabel('Systolic BP')
plt.title('Age vs Blood Pressure')
plt.legend()
plt.savefig('12_age_bp_scatter.png')
plt.close()

# Plot 13: Gender distribution
plt.figure(figsize=(6, 5))
df['gender'].value_counts().plot(kind='bar', color=['salmon', 'lightblue'])
plt.title('Gender Distribution')
plt.xlabel('1=Male, 2=Female')
plt.savefig('13_gender.png')
plt.close()

# Plot 14: Weight by Disease
plt.figure(figsize=(8, 5))
df.boxplot(column='weight', by='cardio')
plt.title('Weight by Heart Disease')
plt.suptitle('')
plt.savefig('14_weight_boxplot.png')
plt.close()

# Plot 15: Height by Disease
plt.figure(figsize=(8, 5))
df.boxplot(column='height', by='cardio')
plt.title('Height by Heart Disease')
plt.suptitle('')
plt.savefig('15_height_boxplot.png')
plt.close()

print("15 visualizations saved!")

# ============================================================
# 4. CORRELATION MATRIX
# ============================================================
print("\n--- CORRELATION MATRIX ---")

plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()
print("Correlation matrix saved!")

# ============================================================
# 5. MACHINE LEARNING MODELS
# ============================================================
print("\n--- MACHINE LEARNING MODELS ---")

# Prepare data
X = df.drop('cardio', axis=1)
y = df['cardio']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# 1. Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
results['Logistic Regression (LR)'] = lr_acc
print(f"Logistic Regression: {lr_acc*100:.2f}%")

# 2. K-Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_pred)
results['K-Nearest Neighbor (KNN)'] = knn_acc
print(f"K-Nearest Neighbor: {knn_acc*100:.2f}%")

# 3. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
dt_pred = dt.predict(X_test_scaled)
dt_acc = accuracy_score(y_test, dt_pred)
results['Decision Tree (DT)'] = dt_acc
print(f"Decision Tree: {dt_acc*100:.2f}%")

# 4. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
results['Random Forest (RF)'] = rf_acc
print(f"Random Forest: {rf_acc*100:.2f}%")

# 5. Support Vector Machine
svm = SVC()
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_pred)
results['Support Vector Machine (SVM)'] = svm_acc
print(f"Support Vector Machine: {svm_acc*100:.2f}%")

# ============================================================
# 6. FINAL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("ACCURACY RESULTS")
print("=" * 60)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for model, acc in sorted_results:
    print(f"{model}: {acc*100:.2f}%")

best = sorted_results[0]
print("\n" + "=" * 60)
print(f"BEST MODEL: {best[0]} - {best[1]*100:.2f}%")
print("=" * 60)

# Save accuracy chart
plt.figure(figsize=(10, 6))
models = [m.split(' (')[0] for m in results.keys()]
accs = list(results.values())
colors = ['blue', 'green', 'red', 'purple', 'orange']
plt.bar(models, accs, color=colors, edgecolor='black')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0.65, 0.80)
for i, v in enumerate(accs):
    plt.text(i, v + 0.005, f'{v*100:.2f}%', ha='center')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('model_accuracy.png')
plt.close()

# Feature importance
plt.figure(figsize=(10, 6))
imp = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
imp = imp.sort_values('importance', ascending=True)
plt.barh(imp['feature'], imp['importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nAll files saved!")
print("Project completed!")
