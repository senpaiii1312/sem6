import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset("titanic")

# Boxplot
sns.boxplot(x="sex", y="age", hue="survived", data=df)

# Title
plt.title("Age Distribution by Gender and Survival")

# Show plot
plt.show()

# Observations
print("1. Females had higher survival rate.")
print("2. More male passengers did not survive.")
print("3. Children had better survival chances.")
print("4. Some age outliers are present.")