#importing required modules
import pickle
import pandas as pd
import numpy as np

#loading the trained model
with open('logistic_reg.pkl', 'rb') as file:
    lreg=pickle.load(file)

#loading the fitted standard scaler
with open('standard_scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)

#taking input from user
print('---------- INPUT ----------')
pregnancies=int(input('Number of times pregnant: '))
glucose=int(input('Plasma glucose concentration a 2 hours in an oral glucose tolerance test: '))
bloodpressure=int(input('Diastolic blood pressure (mm Hg): '))
skinthickness=int(input('Triceps skin fold thickness (mm): '))
insulin=int(input('2-Hour serum insulin (mu U/ml): '))
bmi=float(input('Body mass index (weight in kg/(height in m)^2): '))
diabetespedigreefunction=float(input('Diabetes pedigree function: '))
age=int(input('Age (years): '))

#converting the input to dataframe
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [bloodpressure],
    'SkinThickness': [skinthickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetespedigreefunction],
    'Age': [age]
})

#feature scaling on the dataframe
input_data=pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

#predicting result
predicted_result=lreg.predict(input_data)

#printing the result
print('\n\n---------- RESULT ----------')
if(predicted_result[0]==1):
    print('Diabetes')
else:
    print('No Diabetes')
