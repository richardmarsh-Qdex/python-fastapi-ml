from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = MultinomialNB()
        self.is_trained = False
    
    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
    
    def predict(self, text):
        if not self.is_trained:
            raise ValueError("Model not trained")
        X = self.vectorizer.transform([text])
        return self.classifier.predict(X)[0]
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
