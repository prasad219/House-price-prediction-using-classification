import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('MagicBricks_Updated_Clear_Locality_draft15122023.csv')

Q1 = df['Price'].quantile(0.10)
Q3 = df['Price'].quantile(0.90)

PQ1 = df['Per_Sqft'].quantile(0.10)
PQ3 = df['Per_Sqft'].quantile(0.90)

AQ1 = df['Area'].quantile(0.10)
AQ3 = df['Area'].quantile(0.90)

df1=df[(df['Price']>Q1) & (df['Price']<Q3) & (df['Per_Sqft']>PQ1) & (df['Per_Sqft']<PQ3) & (df['Area']>AQ1) & (df['Area']<AQ3)]
df1=df1.dropna()

threshold_low = 6000000
threshold_high = 10000000

def compare_with_threshold(value):
    if value < threshold_low:
        return 'Low'
    elif threshold_low <= value <= threshold_high:
        return 'Medium'
    else:
        return 'High'

df1['Price_range'] = df1['Price'].apply(compare_with_threshold)

label_encoder = LabelEncoder()

columns_to_encode = ['Furnishing', 'Status', 'Transaction', 'Type','Locality','Price_range']

for column in columns_to_encode:
    df1[column] = label_encoder.fit_transform(df1[column])

X=df1.drop(["Price_range",'Price'],axis=1)
y=df1['Price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=33)

classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB()
}

for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy - {accuracy:.4f}")
    
model = RandomForestClassifier()
model.fit(X_train, y_train)

pickle.dump(model,open('model_house_pricing','wb'))