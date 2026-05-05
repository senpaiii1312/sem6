111111111111111111111111111111111111111111111111111111111111111111111

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset

df = pd.read_csv(r"C:\Users\Sujal\Desktop\MingW2bin\bin_73564975682_ds\titanic.csv")

# -----------------------------

# Normalize column names (IMPORTANT FIX)

# -----------------------------

df.columns = df.columns.str.strip().str.lower()

# Basic Info

print("First 5 Rows\n", df.head())
print("\nDataset Shape\n", df.shape)
print("\nColumn Names\n", df.columns)
print("\nData Types\n", df.dtypes)
print("\nMissing Values\n", df.isnull().sum())
print("\nStatistical Summary\n", df.describe(include="all"))

# -----------------------------

# Handling Missing Values (SAFE for both datasets)

# -----------------------------

if "age" in df.columns:
df["age"].fillna(df["age"].mean(), inplace=True)

if "embarked" in df.columns:
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

if "cabin" in df.columns:
df["cabin"].fillna("unknown", inplace=True)

if "deck" in df.columns:
df["deck"].fillna("unknown", inplace=True)

if "embark_town" in df.columns:
df["embark_town"].fillna(df["embark_town"].mode()[0], inplace=True)

print("\nMissing Values After Filling\n", df.isnull().sum())

# -----------------------------

# Data Type Conversion (only if present)

# -----------------------------

type_map = {
"survived": "int",
"pclass": "int",
"fare": "float"
}

for col, dtype in type_map.items():
if col in df.columns:
df[col] = df[col].astype(dtype)

# -----------------------------

# Label Encoding (only existing columns)

# -----------------------------

label = LabelEncoder()

categorical_cols = [
"sex", "embarked", "class", "who",
"deck", "embark_town", "alive",
"alone", "adult_male", "cabin"
]

for col in categorical_cols:
if col in df.columns:
df[col] = label.fit_transform(df[col])

# -----------------------------

# Feature Scaling

# -----------------------------

scaler = MinMaxScaler()

scale_cols = ["age", "fare"]

for col in scale_cols:
if col in df.columns:
df[[col]] = scaler.fit_transform(df[[col]])

# -----------------------------

# Final Output

# -----------------------------

print("\nFormatted Data Types\n", df.dtypes)
print("\nFinal Dataset Sample\n", df.head())

22222222222222222222222222222222222222222222222222222222222222222

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

np.random.seed(10)

data = {
"Roll_No": range(1, 21),
"Name": [
"A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
"K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"
],
"Maths": [78, 85, 90, np.nan, 66, 72, 88, 95, 43, 77, 82, 91, 58, 64, 70, 300, 76, 84, 69, 73],
"Science": [80, 79, 92, 88, 67, np.nan, 85, 94, 40, 76, 81, 89, 60, 62, 71, 96, 75, 83, 68, 72],
"English": [75, 82, 88, 84, 65, 70, 86, 91, 45, 74, 80, 87, 59, 61, 69, 93, 74, 81, 66, 71],
"Attendance": [92, 95, 98, 90, 85, 87, 93, 97, 70, 88, 91, 96, 82, 84, 86, 99, 89, 94, 83, 85]
}

df = pd.DataFrame(data)

print("Original Dataset")
print(df)

print("\nMissing Values")
print(df.isnull().sum())

# ✅ FIX 1: removed inplace (better practice)

df["Maths"] = df["Maths"].fillna(df["Maths"].mean())
df["Science"] = df["Science"].fillna(df["Science"].mean())

# Handling inconsistencies (same logic)

df["Maths"] = df["Maths"].clip(lower=0, upper=100)
df["Science"] = df["Science"].clip(lower=0, upper=100)
df["English"] = df["English"].clip(lower=0, upper=100)
df["Attendance"] = df["Attendance"].clip(lower=0, upper=100)

print("\nAfter Handling Missing Values and Inconsistencies")
print(df)

numeric_cols = ["Maths", "Science", "English", "Attendance"]

# Outlier detection (Z-score)

z_scores = np.abs(zscore(df[numeric_cols]))
outliers = (z_scores > 3)

print("\nOutlier Positions")
print(outliers)

# Outlier treatment (IQR)

for col in numeric*cols:
q1 = df[col].quantile(0.25)
q3 = df[col].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 \_ iqr

    # ✅ FIX 2: used np.clip instead of two np.where (cleaner, same logic)
    df[col] = np.clip(df[col], lower, upper)

print("\nAfter Outlier Treatment")
print(df)

# Data transformation

df["Log_Maths"] = np.log(df["Maths"] + 1)

print("\nAfter Data Transformation")
print(df[["Maths", "Log_Maths"]])

# Visualization

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df["Maths"], bins=10)
plt.title("Original Maths Marks")

plt.subplot(1, 2, 2)
plt.hist(df["Log_Maths"], bins=10)
plt.title("Log Transformed Maths Marks")

plt.tight_layout()
plt.show()

33333333333333333333333333333333333333333333333333333333333333333333

import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Sujal\Desktop\MingW2bin\bin_73564975682_ds\iris.csv")

print("First Five Records")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nSummary Statistics Grouped by Species")
grouped = df.groupby("species")["sepal_length"].agg(
["mean", "median", "min", "max", "std"]
)

print(grouped)

species_numeric = {
"Iris-setosa": 0,
"Iris-versicolor": 1,
"Iris-virginica": 2
}

df["species_value"] = df["species"].map(species_numeric)

print("\nNumeric Value for Each Category")
print(df[["species", "species_value"]].drop_duplicates())

species_list = df["species_value"].tolist()

print("\nList of Numeric Values")
print(species_list)

print("\nStatistical Details of Each Species")

for flower in df["species"].unique():
print("\nSpecies:", flower)

    subset = df[df["species"] == flower]

    print(subset.describe())

    print("Percentiles")
    print(subset.quantile([0.25, 0.50, 0.75], numeric_only=True))

4444444444444444444444444444444444444444444444444444444444444444444444444444444444444444

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset

raw_df = pd.read_csv(r"C:\Users\Sujal\Desktop\MingW2bin\bin_73564975682_ds\housing.csv")

# 🔥 Ensure even rows (VERY IMPORTANT)

if len(raw_df) % 2 != 0:
raw_df = raw_df.iloc[:-1]

# Create X and y

X = np.hstack([
raw_df.values[::2, :],
raw_df.values[1::2, :2]
])

y = raw_df.values[1::2, 2]

# 🔥 CHECK SHAPE

print("Shape of X:", X.shape)

# Create dynamic column names (AUTO FIX)

columns = [f"Feature\_{i+1}" for i in range(X.shape[1])]

X = pd.DataFrame(X, columns=columns)

print("\nFirst Five Rows")
print(X.head())

print("\nShape of Dataset")
print(X.shape)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Model

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Results

print("\nCoefficients")
print(model.coef\_)

print("\nIntercept")
print(model.intercept\_)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

# Plot

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.grid(True)
plt.show()

5555555555555555555555555555555555555555555555555555555555555555555555555555

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv(r"C:\Users\Sujal\Desktop\MingW2bin\bin_73564975682_ds\Social_Network_Ads.csv")

print("First Five Rows")
print(df.head())

X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.25, random_state=0
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = (FP + FN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("\nConfusion Matrix")
print(cm)

print("\nTrue Positive :", TP)
print("False Positive :", FP)
print("True Negative :", TN)
print("False Negative :", FN)

print("\nAccuracy :", accuracy)
print("Error Rate :", error_rate)
print("Precision :", precision)
print("Recall :", recall)

6666666666666666666666666666666666666666666666666666666666666666666666666

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv(r"C:\Users\Sujal\Desktop\MingW2bin\bin_73564975682_ds\iris.csv")

print(df.head())

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.30, random_state=0
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

print("\nAccuracy =", accuracy)
print("Error Rate =", error_rate)

TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
TN = np.sum(cm) - (TP + FP + FN)

for i in range(len(model.classes\_)):
precision = TP[i] / (TP[i] + FP[i])
recall = TP[i] / (TP[i] + FN[i])

    print("\nClass =", model.classes_[i])
    print("TP =", TP[i])
    print("FP =", FP[i])
    print("TN =", TN[i])
    print("FN =", FN[i])
    print("Precision =", precision)
    print("Recall =", recall)

777777777777777777777777777777777777777777777777777777777777777777777777777777777

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

8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

print("First Five Rows")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

plt.figure(figsize=(8, 5))
sns.countplot(x="survived", data=df)
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="sex", hue="survived", data=df)
plt.title("Survival Based on Gender")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="pclass", hue="survived", data=df)
plt.title("Passenger Class vs Survival")
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df["fare"].dropna(), bins=30, edgecolor="black")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()

99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

print("First Five Rows")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))

sns.boxplot(
x="sex",
y="age",
hue="survived",
data=df
)

plt.title("Age Distribution by Gender and Survival")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

print("\nObservations:")
print("1. Female passengers had higher survival compared to males.")
print("2. Many children were among survivors.")
print("3. Male non-survivors are higher in count.")
print("4. Median age differs slightly between groups.")
print("5. Several outliers are visible in higher age ranges.")

1010101010101010101010101010101010101010101010101010101010101010101010101010101010101010

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("iris")

print("First Five Rows")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nFeature Types")
print(df.dtypes)

print("\nInference:")
print("sepal_length -> Numeric")
print("sepal_width -> Numeric")
print("petal_length -> Numeric")
print("petal_width -> Numeric")
print("species -> Nominal")

features = [
"sepal_length",
"sepal_width",
"petal_length",
"petal_width"
]

for col in features:
plt.figure(figsize=(7, 4))
plt.hist(df[col], bins=15, edgecolor="black")
plt.title(f"Histogram of {col}")
plt.xlabel(col)
plt.ylabel("Frequency")
plt.show()

for col in features:
plt.figure(figsize=(7, 4))
sns.boxplot(x=df[col])
plt.title(f"Boxplot of {col}")
plt.xlabel(col)
plt.show()

print("\nComparison and Outlier Detection:")
print("1. Petal length and petal width show clear grouped distributions.")
print("2. Sepal width contains some outliers.")
print("3. Sepal length has moderate spread.")
print("4. Petal features separate species better than sepal features.")
print("5. Boxplots clearly show extreme values in sepal_width.")
