import numpy as np
import streamlit as st
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

st.title("20 Newsgroups Text Classification")

# Task 1: introduction text
st.write(
    "This app uses these models for classification: "
    "Naïve Bayes, SVM, Random Forest, Decision Tree."
)

# Task 2: classifier options in the sidebar select box
cs = ["Naive Bayes", "SVM", "Random Forest", "Decision Tree"]
classification_space = st.sidebar.selectbox("Choose a classifier", cs)


@st.cache_data(show_spinner="Loading the 20 newsgroups dataset...")
def load_data():
    """Load the dataset once and reuse it, so it is never downloaded twice at the same time."""
    train = fetch_20newsgroups(subset="train", shuffle=True)
    test = fetch_20newsgroups(subset="test", shuffle=True)
    return train.data, train.target, test.data, test.target


def build_classifier(name):
    """Return the scikit-learn classifier for the selected option."""
    if name == "Naive Bayes":
        return MultinomialNB()
    if name == "SVM":
        return SGDClassifier(loss="hinge", penalty="l1", alpha=0.0005, l1_ratio=0.17)
    if name == "Random Forest":
        return RandomForestClassifier(max_depth=2, random_state=0)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=0)


# Task 3: train, predict and report accuracy for the selected classifier
if st.sidebar.button("Classify"):
    st.write(f"{classification_space} selected")

    train_x, train_y, test_x, test_y = load_data()

    pipeline = Pipeline([
        ("bow", CountVectorizer()),
        ("vector", TfidfTransformer()),
        ("classifier", build_classifier(classification_space)),
    ])

    with st.spinner(f"Training {classification_space}... this can take a minute."):
        pipeline = pipeline.fit(train_x, train_y)
        predictions = pipeline.predict(test_x)

    st.write(f"Accuracy of {classification_space}:")
    st.write(np.mean(predictions == test_y))
