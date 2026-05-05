import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

document = """
Artificial Intelligence is changing the world.
Machine learning helps computers learn from data.
Natural language processing helps machines understand text.
"""

print("Original Document:\n")
print(document)

tokens = word_tokenize(document)

print("Tokens:\n")
print(tokens)

pos_tags = pos_tag(tokens)

print("\nPOS Tags:\n")
print(pos_tags)

stop_words = set(stopwords.words("english"))

filtered = [
    word for word in tokens
    if word.lower() not in stop_words and word.isalpha()
]

print("\nAfter Stopword Removal:\n")
print(filtered)

stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered]

print("\nStemmed Words:\n")
print(stemmed)

lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered]

print("\nLemmatized Words:\n")
print(lemmatized)

docs = [
    "Artificial intelligence uses data",
    "Machine learning uses algorithms",
    "Text analytics uses language processing"
]

cv = CountVectorizer()
tf = cv.fit_transform(docs)

tf_df = pd.DataFrame(
    tf.toarray(),
    columns=cv.get_feature_names_out()
)

print("\nTerm Frequency Matrix:\n")
print(tf_df)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(docs)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

print("\nTF-IDF Matrix:\n")
print(tfidf_df)