import streamlit as st

st.sidebar.image("logo.png")
st.sidebar.info("This is a Natural Language Processing project maintained by **DecafSunrise**. Feel free to take a peek at the  [**source code**](https://github.com/DecafSunrise/Yeah-Right), or check out the rest of my [**awesome projects on GitHub**](https://github.com/DecafSunrise/)")

from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from joblib import load

vectorizer = load('././models/sarcasm_vectorizer.joblib')
lr_clf = load("././models/logreg_clf.pkl")
rf_clf = load("././models/rf_clf.pkl")

lr_pipe = make_pipeline(vectorizer, lr_clf)
rf_pipe = make_pipeline(vectorizer, rf_clf)

explainer = LimeTextExplainer(class_names=["sarcastic", "not-sarcastic"])

st.title('"Yeah-Right": Sarcasm Classifier')

st.markdown("_Classifiers_ are type of machine learning machine learning model that determines which bin an input should belong to: '__Hot Dog__' or '__Not Hot Dog__'.<br/><br/>"
            "Want to try it yourself? Type some text below, and play with the provided models!", unsafe_allow_html=True)

inst_text = st.text_input(label="What text do you want to classify?", placeholder="Your super sarcastic text here")

if inst_text:
    st.write("**Input:**", inst_text)

    exp = explainer.explain_instance(inst_text, lr_pipe.predict_proba)

    st.write(' LogisticRegression Probability(Sarcastic) =', round(lr_pipe.predict_proba([inst_text])[0, 1],3))
    st.write(' RandomForest Probability(Sarcastic) =', round(lr_pipe.predict_proba([inst_text])[0, 1],3))
    st.write("(Prediction probability below 0.5 would mean 'Not Sarcastic')")

    st.header('Logistic Regression:')
    lr_explanation = explainer.explain_instance(inst_text, lr_pipe.predict_proba)
    st.pyplot(fig=lr_explanation.as_pyplot_figure())

    st.header('Random Forest:')
    rf_explanation = explainer.explain_instance(inst_text, rf_pipe.predict_proba)
    st.pyplot(fig=rf_explanation.as_pyplot_figure())
