# 1️⃣ Dataset

data = {
    'Area': [1200, 1500, 1800, 2000],
    'Bedrooms': [2, 3, 3, 4],
    'Price': [200, 250, 275, 300]
}
# this data is known as dataset

import pandas as pd
df = pd.DataFrame(data)
print(df)
# As Area and bedrooms increase, the price increases


# 2️⃣Features : independent variables which are used to predict the dependent variable
features = df[['Area', 'Bedrooms']]
print(features)

# 3️⃣Target or label : dependent variable which is the value we want to predict based on the features
target = df['Price']
print(target)
