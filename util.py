from flask import Flask, render_template, request
import pickle
import pandas as pd
####################### Load transformers #####################
with open(r"static/transformers/model.pkl", "rb") as f:
	model = pickle.load(f)

with open(r"static/transformers/onehotencoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open(r"static/transformers/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
########################### Helper functions ##################
def age_group(x):
    if x<13: return "Child"
    elif 13<x<20: return "Teenager"
    elif 20<x<=60: return "Adult"
    else: return "Elder"

def bmi_group(x):
    if x<18.5 : return "UnderWeight"
    elif 18.5<x<25: return "Healthy"
    elif 25<x<30: return "OverWeight"
    else: return "Obese"

# features
features = ['gender', 'age', 'hypertension', 'heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']

numerical_features = ['age', 'avg_glucose_level', 'bmi']

categorical_features = ['gender', 'hypertension', 'heart_disease','ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age_group', 'bmi_group']

def predict_class(x):
    X = pd.DataFrame(x, columns=features)
    # converting numerical features as float dtype
    X.loc[:, numerical_features] = X.loc[:, numerical_features].astype('float64')
    # add new features
    X["age_group"] = X.age.apply(age_group)
    X["bmi_group"] = X.bmi.apply(bmi_group)
    
    # converting categorical features as category dtype
    X.loc[:, categorical_features] = X.loc[:, categorical_features].astype('category')
    # Categorical encoding
    cols = encoder.get_feature_names(categorical_features)

    X.loc[:, cols] = encoder.transform(X[categorical_features])

    # Drop categorical features
    X.drop(categorical_features, axis=1, inplace=True)

    # Feature scaling
    X.loc[:, numerical_features] = scaler.transform(X[numerical_features])
    proba = round(model.predict_proba(X)[:, 1][0], 4)*100
    return proba