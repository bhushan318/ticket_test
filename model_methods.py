import re
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from cust_fun import clean_series


def predict(arr):
    # Load the model
    model = pickle.load(open('finalized_model.sav', 'rb'))
    classes = {0:'Account Services', 1:'Others', 2:'Mortgage/Loan', 3:'Credit card or prepaid card', 4:'Theft/Dispute Reporting'}
    preprocessor = pickle.load(open('preprocessor.sav', 'rb'))      
    prepro_text = preprocessor.transform(pd.Series([arr])) 
    preds = model.predict_proba(prepro_text)[0]
    return (classes[np.argmax(preds)], preds)







