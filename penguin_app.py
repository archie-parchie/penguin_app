import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)
def prediction(model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex):
  species=model.predict([[island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]])[0]
  return species
st.title("Penguin Prediction App")
bl=st.slider("Bill Length (mm)",30.0,60.0)
bd=st.slider("Bill Depth (mm)",10.0,25.0)
fl=st.slider("Flipper Length (mm)",175,235)
bm=st.slider("Body Mass (mm)",2500,6500)
b1=st.selectbox("Sex",("Male","Female"))
b2=st.selectbox("Island",("Biscoe","Dream","Torgersen"))
if b1=="Male":
  b1=0
else:
  b1=1
if b2=="Biscoe":
  b2=0
elif b2=="Dream":
  b2=1
else:
  b2=2
b3=st.selectbox("Model",("Support Vector Machine Classification","Logistic Regression","Random Forest Classifier"))
lst=["Adelie","Chinstrap","Gentoo"]
if st.button("Predict"):
  if b3=="Support Vector Machine Classification":
    spec=prediction(svc,b2,bl,bd,fl,bm,b1)
    st.write("The species is of your penguin is ",lst[spec])
    st.write("The accuracy of this model is",svc_score)
  elif b3=="Logistic Regression":
    spec=prediction(log_reg,b2,bl,bd,fl,bm,b1)
    st.write("The species is of your penguin is ",lst[spec])
    st.write("The accuracy of this model is",log_reg_score)
  else:
    spec=prediction(rf_clf,b2,bl,bd,fl,bm,b1)
    st.write("The species is of your penguin is ",lst[spec])
    st.write("The accuracy of this model is",rf_clf_score)