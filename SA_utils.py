import pickle
import pandas as pd
def load_m_n_v():
    load_model=pickle.load(open("senti_finalized_model.sav",'rb'))
    vectorizer=pickle.load(open("tfidf_vectorizer.pkl",'rb'))
    return load_model,vectorizer


def output_lable(n):
    if n == 0:
        return "The Text Sentiment is Negative"
    elif n == 4:
        return "The Text Sentiment is Positive"
def wp(text):
    return text.lower()

# Define the manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wp)
    new_x_test = new_def_test["text"]
    return new_x_test