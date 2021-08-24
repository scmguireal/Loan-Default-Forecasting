import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# this is the input dataframe
df = pd.read_csv("data.csv")

# define individual transformers in a pipeline
categorical_preprocessing = Pipeline([('ohe', OneHotEncoder())])
numerical_preprocessing = Pipeline([('imputation', SimpleImputer())])

# define which transformer applies to which columns
preprocess = ColumnTransformer([
    ('categorical_preprocessing', categorical_preprocessing, ['favorite_color']),
    ('numerical_preprocessing', numerical_preprocessing, ['age'])
])

# create the final pipeline with preprocessing steps and 
# the final classifier step
pipeline = Pipeline([
    ('preprocess', preprocess),
    ('clf', DecisionTreeClassifier())
])

# now fit the pipeline using the whole dataframe
df_features = df[['favorite_color','age']]
df_target = df['target']

# call fit on the dataframes
pipeline.fit(df_features, df_target)