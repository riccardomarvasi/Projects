import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
import os

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

if __name__ == "__main__":

    if(len(sys.argv)<4):
        print("ERROR! Usage: python scriptName.py fileCSV targetN modelloML\n")
              
        sys.exit(1)
    nome_script, pathCSV, targId, mlModel = sys.argv

    targetId = int(targId)

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    back = ['Artistic', 'Scientific']
    pos = 1
    ds = 'S'
    if (pathCSV == 'datasetArtisticBackground.csv'):
        pos = 0
        ds = 'A'


    dataset = pd.read_csv(pathCSV, sep=';')

    index_target= dataset.iloc[:,-7:]
    list_ind_t = index_target.columns.values.tolist()
    targetN = list_ind_t[targetId]

    X = dataset[['timeDuration', 'nMovements', 'movementsDifficulty', 'AItechnique', 'robotSpeech', 'acrobaticMovements', 'movementsRepetition', 'musicGenre', 'movementsTransitionsDuration', 'humanMovements', 'balance', 'speed', 'bodyPartsCombination', 'musicBPM', 'sameStartEndPositionPlace', 'headMovement', 'armsMovement', 'handsMovement', 'legsMovement', 'feetMovement']]
    y = dataset[targetN]

    categorical_features = ['AItechnique', 'musicGenre']

    categorical_transformer = Pipeline(steps=[
                                          ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numeric_features = ['timeDuration', 'nMovements', 'movementsDifficulty', 'robotSpeech', 'acrobaticMovements', 'movementsRepetition', 'movementsTransitionsDuration', 'humanMovements', 'balance', 'speed', 'bodyPartsCombination', 'musicBPM', 'sameStartEndPositionPlace', 'headMovement', 'armsMovement', 'handsMovement', 'legsMovement', 'feetMovement']
    numeric_transformer = Pipeline(steps=[
                                      ('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
                                 transformers=[
                                               ('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)])
    
    ############# ML MODELS ###############

    model_reg = ['lr',
                'dt',
                'rf',
                'gbr']

    param_lr = [{'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}]

    param_dt = [{'max_depth': [5,10,20]}]

    param_rf = [{'bootstrap': [True, False],
                 'max_depth': [10, 20],
                 'max_features': ['auto', 'sqrt'],
                 'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2],}]

    param_gbr = [{'learning_rate': [0.01,0.03],
                'subsample'    : [0.5, 0.2],
                'n_estimators' : [100,200],
                'max_depth'    : [4,8]}]

    models_regression = {
        'lr': {'name': 'Linear Regression',
               'estimator': LinearRegression(),
               'param': param_lr,
              },
        'dt': {'name': 'Decision Tree',
               'estimator': DecisionTreeRegressor(random_state=42),
               'param': param_dt,
              },
        'rf': {'name': 'Random Forest',
               'estimator': RandomForestRegressor(random_state=42),
               'param': param_rf,
              },

        'gbr': {'name': 'Gradient Boosting Regressor',
                'estimator': GradientBoostingRegressor(random_state=42),
                'param': param_gbr
                },
    }

    k = 10
    kf = KFold(n_splits=k, random_state=None)
    mod_grid = GridSearchCV(models_regression[mlModel]['estimator'], models_regression[mlModel]['param'], cv=5, return_train_score = False, scoring='neg_mean_squared_error', n_jobs = 8)

    for train_index , test_index in kf.split(X):
        data_train , data_test = X.iloc[train_index,:],X.iloc[test_index,:]
        target_train , target_test = y[train_index] , y[test_index]

        model = Pipeline(steps=[('preprocessor', preprocessor),
                ('regressor', mod_grid)])


        _ = model.fit(data_train, target_train)

        target_pred = model.predict(data_test)
    

    lf = ['t', 'n', 'md', 'rs', 'am', 'mr', 'mtd', 'h', 'b', 's', 'bc', 'bpm', 'pp', 'hm', 'arm', 'hdm', 'lm', 'fm', 'AIc', 'AIp', 'AIs', 'mEl', 'mFol', 'mInd', 'mPop', 'mRap', 'mRock']


    ####################### DiCE #############################
    import dice_ml
    from sklearn.model_selection import train_test_split
    Ncount=30

    X = preprocessor.fit_transform(X)

    feature_cat_names = model['preprocessor'].transformers_[1][1]['onehot'].get_feature_names(categorical_features)
    l= feature_cat_names.tolist()
    ltot = numeric_features + l

    constraints={}
    
    X = pd.DataFrame(X, columns=ltot)

    desc=X.describe()
    print(desc)
    for i in numeric_features:
        constraints[i]=[desc[i]['min'],desc[i]['max']]
    X['output'] = y

    X_train, X_test = train_test_split(X,test_size=0.2,random_state=42,stratify=X['output'])

    dice_train = dice_ml.Data(dataframe=X_train,
                 continuous_features=numeric_features,
                 outcome_name='output')
    
    m = dice_ml.Model(model=mod_grid.best_estimator_,backend='sklearn', model_type='regressor',func=None)
    exp = dice_ml.Dice(dice_train,m)

    query_instance = X_test.drop(columns="output")
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=Ncount,desired_range=[1,5],permitted_range=constraints)

    if not Path.cwd().joinpath("dice_results").exists():
        Path.cwd().joinpath("dice_results").mkdir(parents=True)
    data = []
    for cf_example in dice_exp.cf_examples_list:
        data.append(cf_example.final_cfs_df)

    df_combined = pd.concat(data, ignore_index=True)
    for i in range(len(df_combined)):
        df_combined.iloc[i] = df_combined.iloc[i] - X_test.iloc[i//Ncount]
    df_combined.to_csv(path_or_buf=f'dice_results/{targetId}_{mlModel}_{ds}_counterfactuals.csv', index=False, sep=',')
    df_combined.dtypes
    df_filtered = df_combined[df_combined['output'] != 0]
    count_per_column = df_filtered.apply(lambda x: (x != 0).sum())
    diff_per_column = df_filtered.apply(lambda x: (abs(x)*abs(df_filtered['output'])).sum())

    #print(count_per_column)
    #print(diff_per_column['output'])

    correlation_matrix = df_combined.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Correlation")

    plt.savefig(f'dice_results/{targetId}_{mlModel}_{ds}_correlations.png')

    original_stdout = sys.stdout
    with open(f'dice_results/{targetId}_{mlModel}_{ds}_count.txt', 'w') as f:
        sys.stdout = f
        print('\n--------------------- Counterfactual absolute counts:-------------------------')
        print(diff_per_column)
        print('\n--------------------- Counterfactual relative counts:-------------------------')
        print(diff_per_column/count_per_column)
            
        
    sys.stdout = original_stdout
