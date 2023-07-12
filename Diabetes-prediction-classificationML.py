
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv('diabetes.csv')


x = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]


rf = RandomForestClassifier(n_estimators=100)


rf.fit(x, y)


new_instance = []
feature=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','age']
print('Sample:\nPregnancies:6\tGlucose:148\tBloodPressure:72\tSkinThickness:35\nInsulin:0\tBMI:33.6\tDiabetesPedigreeFunction:0.6\tAge:50\n')
for i in range(8):
    
    value = float(input(f"Enter value for {feature[i]}: "))
    new_instance.append(value)

# Classify the new instance
prediction = rf.predict([new_instance])

# Print the predicted class
if prediction[0] == 1 :
    print("\nIs a diabetic")
else:
    print("\nNot diabetic")
