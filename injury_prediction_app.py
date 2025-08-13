import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.drop('Position', axis=1, inplace=True)
    
    cols = list(df.columns)
    bmi_index = cols.index('BMI')
    injury_index = cols.index('Injury_Next_Season')
    cols[bmi_index], cols[injury_index] = cols[injury_index], cols[bmi_index]
    df = df[cols]
    
    return df

df = load_data()

X = df.drop('Injury_Next_Season', axis=1)
y = df['Injury_Next_Season']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

st.title("🏥 Injury Prediction App")
st.write("Enter player details to predict if injury is likely next season.")

user_input = {}
for col in X.columns:
    if np.issubdtype(df[col].dtype, np.number):
        user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))
    else:
        user_input[col] = st.selectbox(f"{col}", df[col].unique())

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.error("⚠️ Injury Likely Next Season")
    else:
        st.success("✅ No Injury Likely Next Season")

if st.checkbox("Show Model Evaluation"):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"**Test Accuracy:** {accuracy*100:.2f}%")
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Injury', 'Injury'],
                yticklabels=['No Injury', 'Injury'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')
    st.pyplot(fig)
