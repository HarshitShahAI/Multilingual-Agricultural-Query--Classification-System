import pickle


class GujaratiModel:
    def __init__(self):
        with open("models/Gujarati/first_level_model.pkl", "rb") as f:
            self.first_model = pickle.load(f)
        with open("models/Gujarati/second_level_xgb_6_vs_8.pkl", "rb") as f:
            self.second_model = pickle.load(f)
        with open("models/Gujarati/gujarati_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, query):
        X = self.vectorizer.transform([query])
        pred = self.first_model.predict(X)[0]
        if pred in [6, 8]:
            refined = self.second_model.predict(X)[0]
            pred = 6 if refined == 0 else 8
        return pred


class HindiModel:
    def __init__(self):
        with open("models/Hindi/model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("models/Hindi/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, query):
        X = self.vectorizer.transform([query])
        return self.model.predict(X)[0]


class EnglishModel:
    def __init__(self):
        with open("models/english/english_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("models/english/english_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, query):
        X = self.vectorizer.transform([query])
        return self.model.predict(X)[0]


class ModelHandler:
    def __init__(self, lang):
        if lang == "guj":
            self.model = GujaratiModel()
        elif lang == "hin":
            self.model = HindiModel()
        elif lang == "eng":
            self.model = EnglishModel()
        else:
            raise ValueError("Unsupported language: choose 'guj', 'hin', or 'eng'")

    def predict(self, query):
        return self.model.predict(query)
