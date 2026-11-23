import math
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("========== MATH LIBRARY ==========")
number = 25
print("Square root of", number, ":", math.sqrt(number))
print("Power (2^5):", math.pow(2, 5))
print("Factorial of 5:", math.factorial(5))
print("Value of Pi:", math.pi)
print("Ceiling of 4.3:", math.ceil(4.3))
print("Floor of 4.7:", math.floor(4.7))

print("\n========== NUMPY LIBRARY ==========")
marks = np.array([70, 76, 84, 78, 92])
print("Marks:", marks)
print("Sum:", np.sum(marks))
print("Mean:", np.mean(marks))
print("Median:", np.median(marks))
print("Standard Deviation:", np.std(marks))
print("Maximum:", np.max(marks))
print("Minimum:", np.min(marks))

print("\n========== SCIPY LIBRARY ==========")
z_scores = stats.zscore(marks)
print("Z-Scores:", z_scores)
mode_result = stats.mode(marks, keepdims=True)
print("Mode:", mode_result.mode[0])

print("\n========== PANDAS LIBRARY ==========")
data = {
    "Student": ["Rohan", "Ananya", "Karan", "Pooja", "Siddharth"],
    "Maths": [81, 87, 74, 92, 85],
    "Python": [84, 91, 78, 94, 88],
    "AI": [79, 89, 73, 91, 87],
}
df = pd.DataFrame(data)
print("\nStudent Data:")
print(df)

df["Average"] = df[["Maths", "Python", "AI"]].mean(axis=1)
print("\nData with Average:")
print(df)

print("\nSummary Statistics:")
print(df.describe())

print("\n========== MATPLOTLIB ==========")
plt.figure(figsize=(8, 5))
plt.bar(df["Student"], df["Average"])
plt.xlabel("Students")
plt.ylabel("Average Marks")
plt.title("Average Marks of Students")
plt.show()

print("\n========== SEABORN ==========")
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x="Maths",
    y="Python",
    data=df,
    s=100,
)
plt.title("Maths Marks vs Python Marks")
plt.xlabel("Maths Marks")
plt.ylabel("Python Marks")
plt.show()

plt.figure(figsize=(7, 5))
correlation = df[["Maths", "Python", "AI"]].corr()
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation Between Subjects")
plt.show()