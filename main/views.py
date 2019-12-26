from django.shortcuts import render
from django.views.generic import TemplateView
import pandas as pd
import os

from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np

from .models import Article

from scipy.sparse import csr_matrix
from scipy.sparse import hstack

from collections import Counter

import pickle
import sys
import re


class CleanUpTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        output = list()
        for text in X:
            text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
            text = re.sub(r'\W+', ' ', text, flags=re.M)
            output.append(
                ' '.join([word[:-2] for word in text.lower().split() if len(word) > 3]) # Cut off the words endings
            )
        print('[Preprocessing] Text Cleanup Completed.')
        return np.array(output)

class VectorizeTransformer(TransformerMixin):
    def fit(self, X, y=None, vocab_size=1000):
        vectorizer = CountVectorizer(max_features=vocab_size)
        vectorizer.fit(X)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        print('[Preprocessing] Fitting Vectorizer Completed.')
        return self

    def transform(self, X, y=None):
        try:
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
        except FileNotFoundError:
            print('[Error] Couldn`t locate vectorizer file.')
            return np.empty([0])

        vector = vectorizer.transform(X)
        print('[Preprocessing] Text vectorization completed.')

        return vector


X_transform = Pipeline([
    ("CleanUp", CleanUpTransformer()),
    ("Vectorize", VectorizeTransformer()),
])


# Create your views here.

class IndexView(TemplateView):
    template_name = "index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['current_page'] = 'index'
        return context

class ArticlesView(TemplateView):
    template_name = "articles.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['current_page'] = 'articles'

        context['articles'] = Article.objects.all()
        return context

class ArticleView(TemplateView):
    template_name = "article.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['article'] = Article.objects.get(pk=self.kwargs['article'])
        return context


def output_view(request):
    text = request.POST['TextArea']
    types = ['дача', 'здоровье', 'лайфхаки', 'новости', 'тренды']
    output = []
    for type, proba in zip(types, predict(text)):
        output.append({'type': type, 'proba': proba})
    return render(request, 'output.html', context={'output': output})

def predict(text):
    model = load_model('model.h5')
    text = X_transform.transform([text]).toarray()
    scaler = MinMaxScaler(feature_range=(0, 100))
    print(model.predict_proba(text))
    print(scaler.fit_transform(model.predict_proba(text))[0])
    return scaler.fit_transform(model.predict_proba(text))[0]
