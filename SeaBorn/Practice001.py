import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


tips = sns.load_dataset("tips")
# print(tips.head())
# sns.relplot(x="total_bill", y="tip", data=tips, col="smoker", hue="sex", size="size")
# plt.show()

# here 
# 1. col = it is used to split the data based on a specific variable. 
# 2. hue = it is used to color the points based on a specific variable. 
# 3. size = it is used to size the points based on a specific variable.

iris = sns.load_dataset("iris")
# print(iris.head())

sns.relplot(x="sepal_length", y="sepal_width", data=iris, hue="species")
# plt.show() 


titanic = sns.load_dataset("titanic")
# print(titanic.head())

sns.countplot(x="class", data=titanic)
plt.show()
