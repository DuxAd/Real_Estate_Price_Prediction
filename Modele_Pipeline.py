import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestRegressor
import MyFunction

X = pd.read_csv('train.csv')
y = X['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--------------------------------------")
print("Score d'asymétrie Skew")
df_numerique = X.select_dtypes(include=np.number)
df_numerique.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
for i in df_numerique:
    print(i,(10-len(i))*" ", f"\t skew : {df_numerique[i].skew():.5f}", )

    
print("Amelioration du score d'asymétrie Skew apres l'application d'une fonction log")
Candidat_Log = ['MSSubClass','LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF',
                '1stFlrSF', 'GrLivArea', 'OpenPorchSF', 'WoodDeckSF', 'LotFrontage', 'SalePrice'
                ]
df_numerique[Candidat_Log].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
df_numerique_log = np.log1p(df_numerique[Candidat_Log])
df_numerique_log.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);

for i in df_numerique_log:
    print(i, (10-len(i))*" ", f"\t skew : {df_numerique[i].skew():.5f}  =>  {df_numerique_log[i].skew():.5f}")

df_numerique = X_train.select_dtypes(include=np.number)
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
print("\n--------------------------------------")
# Correlation 
# Weak : 0.1 - 0.3
# Average : 0.3 - 0.7
# Strong : 0.7 - 1 
num_features = len(df_numerique.keys())
n_cols = 6
n_rows = (num_features + n_cols - 1) // n_cols
print("\nCorrelation of numerical features : ")
plt.figure(figsize=(n_cols * 5, n_rows * 4))
for i, feature in enumerate(df_numerique) :
    plt.subplot(n_rows, n_cols, i+1)
    plt.scatter(df_numerique[feature], y_train, alpha=0.6)
    plt.title(f'{feature} vs Y')
    plt.xlabel(feature)
    plt.ylabel('Target_y')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if feature in Candidat_Log:
        correlation_pandas = df_numerique_log[feature].corr(y_train, method="pearson")
    else:
        correlation_pandas = df_numerique[feature].corr(y_train, method="pearson")
    print(feature,  (10-len(feature))*" ", f"\t Correlation : {correlation_pandas:.5f}")

print("--------------------------------------\n")
print("Chi2 Score for categorical variables :")

# p-value > 0.05 random
# p-value < 0.05 statistical relationship
y_train_binned = pd.qcut(y_train, q=50, labels=False)
df_string = X_train.select_dtypes(exclude=np.number)
for i in df_string.keys():
    contingence = pd.crosstab(y_train_binned, df_string[i])
    chi2, p, dof, expected = chi2_contingency(contingence)
    print(i, (15-len(i))*" ", f"\tChi2 : {chi2:.2f}\t, p-value : {p:.10f}")


print("--------------------------------------")

## Data Cleaning
Key = X_train.keys()
print('Missing Data per category before preprocessing:')
for i in Key :
    if X_train[i].isna().sum() != 0 :
        print( i,  (10-len(i))*" ", "\t Missing : ", X_train[i].isna().sum())
print("Total Data:", len(X_train[X_train.keys()[0]]))
print("--------------------------------------")


# Features selection
Features = ['YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','TotalBsmtSF','1stFlrSF','2ndFlrSF',
            'FullBath','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
            'GrLivArea','OverallQual','LotFrontage','MSZoning',  'Alley', 'LotShape', 'LandContour',
            'LotConfig', 'Neighborhood','Condition1', 'BldgType' , 'HouseStyle', 'RoofStyle',
            'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual','ExterCond' , 'Foundation', 
            'BsmtQual', 'BsmtCond', 'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC',
            'CentralAir','Electrical', 'KitchenQual','Functional', 'FireplaceQu', 'GarageType', 
            'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'Fence', 'SaleType', 'SaleCondition', 
            'LotArea', 'HalfBath', 'GarageYrBlt', 'Street', 'LandSlope']

X_train = X_train[Features]
X_test = X_test[Features]
X = X[Features]


# Create a class that impute the median of a specified group
class GroupedMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, group_column):
        self.target_column = target_column
        self.group_column = group_column
        self.medians = {}

    def fit(self, X, y=None):
        self.medians = X.groupby(self.group_column)[self.target_column].median().to_dict()
        return self

    def transform(self, X):        
        X_copy = X.copy() 

        X_copy['__temp_group_median__'] = X_copy[self.group_column].map(self.medians).fillna(X_copy[self.target_column].median())

        nan_mask = X_copy[self.target_column].isna()
        X_copy.loc[nan_mask, self.target_column] = X_copy.loc[nan_mask, '__temp_group_median__']
        
        # Delete temporary feature
        X_copy = X_copy.drop(columns=['__temp_group_median__'])
        
        return X_copy

def log_transform(X):
    return np.log1p(X)

def inverse_log_transform(X_transformed):
    return np.expm1(X_transformed)

#########################################################
############ Imputer ###########
# Selection of columns of imputation
None_columns = [i for i in ['GarageFinish', 'GarageQual','GarageCond', 'PoolQC',
                'Fence', 'MiscFeature', 'BsmtFinType1', 'BsmtFinType2',
                'FireplaceQu', 'GarageType', 'MasVnrType', 'Alley'] if i in Features]
Na_columns = [i for i in ['BsmtQual', 'BsmtCond', 'BsmtExposure'] if i in Features]
Zero_columns = [i for i in ['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', '2ndFlrSF', 'FullBath',
       'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
       'OverallQual', 'HalfBath'] if i in Features]
Frequency_columns = [i for i in ['Electrical'] if i in Features]
log_columns = [i for i in Candidat_Log if i in Features]

# Definition of imputers
None_imputer = SimpleImputer(strategy='constant', fill_value='None')
Na_imputer = SimpleImputer(strategy='constant', fill_value='Na')
Zero_imputer = SimpleImputer(strategy='constant', fill_value=0)
Frequency_imputer = SimpleImputer(strategy='most_frequent')

# Selection of the rest of the categorical features
X_string = X_train.select_dtypes(exclude=np.number)
Key_string = X_string.keys()
Key_missing = list(set(Key_string) - (set(None_columns) | set(Na_columns) | set(Frequency_columns)) )

# Preparing the preprocessot pipeline
preprocessor = ColumnTransformer(
    transformers=[
        # Impute 0 and apply a log function
        ('Num_Pipe', Pipeline([
            ('Impute_0', Zero_imputer),
            ('Log', FunctionTransformer(func=log_transform, inverse_func=inverse_log_transform, validate=True))
        ]), log_columns),
        
        # Impute None and apply a OneHotEncoder
        ('num_none_pipeline', Pipeline([
            ('num_none_imputer', None_imputer),
            ('ohe_A', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), None_columns), 
        
        # Impute Na and apply a OneHotEncoder
        ('num_na_pipeline', Pipeline([
            ('num_na_imputer', Na_imputer),
            ('ohe_B', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), Na_columns), 
        
        #Impute using the most frequent values and apply a OnehotEncoder
        ('num_frequency_pipeline', Pipeline([
            ('num_frequency_imputer', Frequency_imputer),
            ('ohe_C', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), Frequency_columns), 

        # Imput 0
        ('num_zero_imputer', Zero_imputer, Zero_columns),

        # Apply OneHotEncoder
        ('Encodage_One_Hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), Key_missing)
        
    ],
    remainder='passthrough'
)

Process_pipeline = Pipeline([('GroupMediane', GroupedMedianImputer(target_column='LotFrontage', group_column='Neighborhood')),
                          ('preprocessor', preprocessor) ,
                          ('scaler_final', StandardScaler(with_mean=False))                                                
                          ])


## Getting the columns name from the pipeline
Process_pipeline.fit(X_train)
preprocessor_obj = Process_pipeline.named_steps['preprocessor']
Columns_name = np.concatenate([
    np.array(log_columns),
    preprocessor_obj.named_transformers_['num_none_pipeline']['ohe_A'].get_feature_names_out(),
    preprocessor_obj.named_transformers_['num_na_pipeline']['ohe_B'].get_feature_names_out(),
    preprocessor_obj.named_transformers_['num_frequency_pipeline']['ohe_C'].get_feature_names_out(),
    np.array(Zero_columns),
    preprocessor_obj.named_transformers_['Encodage_One_Hot'].get_feature_names_out()#,
    #preprocessor_obj.named_transformers_['remainder'].get_feature_names_out()
])

X_train_transformed = Process_pipeline.transform(X_train)
print("Valeurs manquantes après prétraitement (X_train) :", np.isnan(X_train_transformed).sum())
X_test_transformed = Process_pipeline.transform(X_test)
print("Valeurs manquantes après prétraitement (X_test) :", np.isnan(X_test_transformed).sum())

#########################################################
####### Creating of the model #######
# Linear Regression
model_pipeline_logi = Pipeline(steps=[
    ('preprocessor', Process_pipeline),
    ('Regression', LinearRegression())
])

model_pipeline_logi.fit(X_train, y_train)
MyFunction.Affichage(X_train, y_train, X_test, y_test, model_pipeline_logi)
    
#########################################################
# Random Forest
model_pipeline_RF = Pipeline(steps=[
    ('preprocessor', Process_pipeline),
    ('RandomForest', RandomForestRegressor(random_state=42))
])

# Grid Search
param_grid_RF = {
    'RandomForest__n_estimators':[90,100,110],
     'RandomForest__max_features':[40,50,60]
     }

model_pipeline_RF = MyFunction.GridSearch(model_pipeline_RF, param_grid_RF, X_train, y_train, Columns_name)
MyFunction.Affichage(X_train, y_train, X_test, y_test, model_pipeline_RF)

#########################################################
## GradientBoosting
from sklearn.ensemble import GradientBoostingRegressor

model_pipeline_gb = Pipeline(steps=[
    ('preprocessor', Process_pipeline),
    ('GradientBoosting', GradientBoostingRegressor(random_state=42))
])

# GridSearch
param_grid_gb = {
    'GradientBoosting__n_estimators': [150, 200, 250],
    'GradientBoosting__learning_rate': [0.01, 0.1],
    'GradientBoosting__max_depth': [3, 5]
}
model_pipeline_gb = MyFunction.GridSearch(model_pipeline_gb, param_grid_gb, X_train, y_train, Columns_name)
MyFunction.Affichage(X_train, y_train, X_test, y_test, model_pipeline_gb)
