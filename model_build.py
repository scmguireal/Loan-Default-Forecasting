import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error as mse

from math import sqrt
import pickle

### Load cleaned dataset
dataset = pd.read_csv('clean_df.csv')
data = dataset.copy()




#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# own class that can be inserted to pipeline as any other sklearn object.
class RawFeats:
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)


    def fit(self, X, y=None, **fit_params):
        return self
    
##### Create functions that perform transformations
def col_drop(input_df):
    return(input_df.drop(['Loan_ID','Unnamed: 0'],axis=1,inplace=True))
def bool_tran(input_df):
    return (input_df.replace(['Y','Yes','Male','Graduate','N','No','Female', 'Not Graduate'],[True,True,True,True,False,False,False,False],inplace=True))
def married_nan(input_df):
    return dfs.dropna(subset=['Married'],inplace=True)

##### Pipeline Builds
# Transformation Pipeline
trans_pipe = Pipeline([
    ('drop', col_drop()),
    ('to_bool',bool_tran()),
    ('null_replace', null_value_replace())
])

# Model Pipelines
# models used
lr = LogisticRegression()
lrcv = LogisticRegressionCV()

# set up our parameters grid
params = [{'penalty': ['l1','l2','elasticnet'],
          'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
          'verbose': [0,1]},
          {'cv':[1,3,5],
           'penalty': ['l1','l2','elasticnet'],
          'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
          'verbose': [0,1]}]

scalers_to_test = [StandardScaler(), RobustScaler()]

# Logistic Regression Pipeline
logreg_pipe = Pipeline([
    ("scalers", scalers_to_test),
    ("lr", lr),
    ("params", params[0])
])

logregcv_pipe = Pipeline([
    ("scalers", scalers_to_test),
    ("lrcv", lrcv),
    ("params", params[1])
])

##### FeatureUnion will run these in parallel
##### Putting them in just one pipeline run them in order
main_pipe = FeatureUnion([
    ("logreg_pipe", logreg_pipe),
    ("logregcv_pipe", logregcv_pipe)
])

#pipeline = Pipeline([features, model, parameters])
##### GridSearch
# create a Grid Search object
grid_search = GridSearchCV(trans_pipe,logreg_pipe, logregcv_pipe, params).fit()    

# fit the model and tune parameters
grid_search.fit(X, y)

print('Final score is: ', grid_search.score(X, y))

print(grid_search.best_params_)
#pickle.dump( grid_search, open( "model.p", "wb" ) )