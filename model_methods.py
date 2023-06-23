import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
def predict(arr):
    # Load the model
    with open('finalized_model.sav', 'rb') as f:
        model = pickle.load(f)
    classes = {0:'Account Services', 1:'Others', 2:'Mortgage/Loan', 3:'Credit card or prepaid card', 4:'Theft/Dispute Reporting'}
    # return prediction as well as class probabilities
    preds = model.predict_proba([arr])[0]
    return (classes[np.argmax(preds)], preds)