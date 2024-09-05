import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
try:
    df = pd.read_csv("data/csv_Mental_Health.csv")  # Ensure the path is correct
    print("CSV file loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    raise
except pd.errors.EmptyDataError:
    print("File is empty. Please provide a valid CSV file.")
    raise
except pd.errors.ParserError:
    print("Error parsing the CSV file. Please check the file format.")
    raise
if df.empty:
    raise ValueError("The dataframe is empty. Please check the CSV file content.")
df = df.dropna(subset=['Result'])
X = df.drop('Result', axis='columns')
y = df['Result']
if len(X) == 0 or len(y) == 0:
    raise ValueError("No samples to split. Please check the CSV file content.")
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.1)
model = XGBClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print('Accuracy level of this model:', accuracy)
patient_data = input("Enter the patient's data (comma-separated values): ")
arr = [int(x) for x in patient_data.split(",")]
probabilities = model.predict_proba([arr])[0]
random_index = np.random.choice(range(len(probabilities)), p=probabilities)
if random_index == 0:
    print('This person has Stress and Anxiety')
elif random_index == 1:
    print('This person has Depression')
elif random_index == 2:
    print('This person has Generalized Anxiety Disorder (GAD) and Obsessive-Compulsive Disorder (OCD)')
elif random_index == 3:
    print('This person has No Mental Issues')
else:
    print('Invalid data')