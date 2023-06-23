import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model_methods import predict
classes = {0:'Account Services', 1:'Others', 2:'Mortgage/Loan', 3:'Credit card or prepaid card', 4:'Theft/Dispute Reporting'}
class_labels = list(classes.values())
st.title("Classification of IT service support Tickets")
st.markdown('**Objective** : Given text details of support ticket, model predicts the Catergories.')
st.markdown('The model can predict if it belongs to the following three Categories : **Account Services, Others, Mortgage, Credit card, Theft** ')

def predict_class():
    data = [ticket_text]
    result, probs = predict(data)
    st.write("The predicted class is ",result)
    probs = [np.round(x,6) for x in probs]
    ax = plt.barh(class_labels, probs)
#     ax.set_yticklabels(class_labels,rotation=0)
    plt.title("Probabilities of the Ticket belonging to each class")
    for index, value in enumerate(probs):
        plt.text(value, index,str(value))
    st.pyplot()
st.markdown("**Please enter the details of ticket**")
ticket_text = st.text_input('Text', '')

if st.button("Predict"):
    predict_class()