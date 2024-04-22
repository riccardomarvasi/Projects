import numpy as np
from sklearn.linear_model import Lasso
from scipy.stats import f
from scipy import stats
from statsmodels.formula.api import ols


def P_anova(True_labels, Predicted_labels, acc=1):
 
    #True_mean = np.mean(True_labels)
    Predicted_mean = np.mean(Predicted_labels)
    #True_mean = np.mean(True_labels)

    # Total sum of squares (TSS) - Total variability in the dependent variables (y)
    #SST = np.sum((True_labels - True_mean)**2)
    # Regression sum of squares (SSR) - Variability in y explained by the model
    SSR = np.sum((Predicted_labels - Predicted_mean)**2)
    # Residual sum of squares (RSS) - Variability in y unexplained by the model
    SSE = np.sum((True_labels - Predicted_labels)**2)
    #print(f"SST:{SST}")
    #print(f"SSR:{SSR}")
    #print(f"SSE:{SSE}")



    # Degrees of freedom
    n_samples = len(True_labels)
    df_total = n_samples - 1
    df_regression = acc # Number of predictors
    df_residual = df_total - df_regression - 1 #total dof number
    # Mean squares
    MSR = SSR / df_regression
    MSE = SSE / df_residual

    # F-statistic and p-value
    F_statistic = MSR / MSE
    p_value = 1 - f.cdf(F_statistic, df_regression, df_residual)
    return p_value

def create_dict(names, values, lf):
    D = {}
    Results = []
    for i in range(len(names)):
        D[names[i]] = values[i]
    
    for name in lf:
        if name in D:
            print(f"{name} : {D[name]}")
            Results.append(f"{name} : {D[name]}")
        else:
            print(f"{name} : Irrelevant")
            Results.append(f"{name} : Irrelevant")
    return Results

def Conditions(model, y, X):
    # Conditions for reliable anova results
    residuals = y - model.predict(X)
    # Check normality of residuals
    _, normality_p_value = stats.normaltest(residuals)
    if normality_p_value < 0.05:
        print("Residuals are not normally distributed.")

    # Check homoscedasticity
    _, homoscedasticity_p_value = stats.levene(model.predict(X), residuals)
    if homoscedasticity_p_value < 0.05:
        print("Residuals exhibit heteroscedasticity.")
        

def Anova_Decomposition(model, X, y, lf, Alpha=0.10, r=2, train_idx=500):

    #model = models_grid.best_estimator_
    #X_preprocessed = preprocessor.fit_transform(X)
    #X_preprocessed = np.array(X_preprocessed[:,:20])
    Conditions(model, y, X)
    #medians = X.median()
    X = np.array(X) 

    # ANOVA 
    One_way_intr = []
    Two_way_intr = [[0]*X.shape[1]]*X.shape[1]
    True_labels = np.array(y)[train_idx:]

    # PART 1 - Oneway interactions 
    for i in range(X.shape[1]):
        # Consider splitting in train / test
        model.fit(X[:train_idx,i].reshape(-1,1), y[:train_idx]) 
        Predicted_labels = model.predict(X[train_idx:,i].reshape(-1,1))
        One_way_intr.append(P_anova(True_labels, Predicted_labels))

    # PART 2 - Twoway interactions
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i<j:
               model.fit(X[:train_idx,[i, j]], y[:train_idx])
               Predicted_labels = model.predict(X[train_idx:,[i,j]])
               Two_way_intr[i][j]=P_anova(True_labels, Predicted_labels, acc=2)
               Two_way_intr[j][i]=Two_way_intr[i][j]

    # PART 3 - Treshold, what we do is checking if p_value is less than alpha
    Relevant_one = [False]*X.shape[1]
    Relevant = [False]*X.shape[1]
    #Relevant_two = [[False]*X.shape[1]]*X.shape[1]

    # Main treshold
    for i in range(len(One_way_intr)):
        #print(One_way_intr[i])
        if One_way_intr[i] <= Alpha:
            Relevant_one[i] = True

    # Recall treshold
    if r==2:
        for i in range(len(Two_way_intr)):
            for j in range(len(Two_way_intr[i])):
                if (Two_way_intr[i][j]<=Alpha) and (i!=j) and (Relevant_one[i]==False) and (Relevant_one[j]==False):
                    Relevant[i] = True
                    Relevant[j] = True

    for i in range(len(Relevant_one)):
        if Relevant_one[i]:
            Relevant[i]=True

    for i in Two_way_intr:
        print(i)

    #print(Relevant_one)
    #print(Relevant_two)

    idx = np.array(Relevant) == True
    X_filtered = X[:,idx]
    names = np.array(lf)[idx]
    
    # Lasso 
    lasso_model = Lasso(alpha=0.01)
    lasso_model.fit(X_filtered, y)
    lasso_coefficents = lasso_model.coef_
    Results = create_dict(names, lasso_coefficents,lf)
    return Results