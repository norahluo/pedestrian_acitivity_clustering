import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from numba import jit, cuda

import statsmodels.api as sm
from hyperopt import fmin, tpe, hp, Trials


# divide years into groups
def year_group(year):
    """
    Categorizes a given year into predefined groups based on ranges.
    
    Parameters:
        year (int): The year to be categorized
    
    Return:
        str: A string representing the group the year belongs to
    """
    if year <= 2014:
        return '1314'
    elif year in [2015, 2016]:
        return '1516'
    elif year in [2017, 2018, 2019]:
        return '171819'
    elif year in [2020, 2021]:
        return '2021'
    else:
        return '2223'

def data_preparation(sites_df, landuse_df, climate_df = None, trail_df = None):
    """
    Given sites_df, landuse_df, climate_df, trail_df,
    
    calculate percentage of businesses within certain buffer
    extract climate information for corresponding year
    create year/covid indicator
    
    Return the prepared dataframe
    """
    
    # get trail/nontrai information
    if trail_df is not None:
        sites_df = sites_df.merge(trail_df[['site_id', 'Classification']], on = 'site_id')
        sites_df['trail'] = sites_df['Classification'].apply(lambda x: 1 if x == 'Trail' else 0)
        sites_df.drop(['Classification'], axis = 1, inplace = True)
    else:
        sites_df['trail'] = 0
    
    # Get the landuse information for sample locations
    data = sites_df.merge(landuse_df, left_on = 'site_id', right_on = 'id', how = 'left')
    # calculate the percentage of a certain business type among the total POIs
    for suffix in ['2', '4', '6', '8', '10']:
        df = data.loc[:,'colleges_'+suffix:'travel & accommodation_'+suffix].div(data.loc[:, 'total_'+suffix], axis = 0)
        df.rename(columns = dict(zip(df.columns, ['per_' + s for s in df.columns])), inplace = True)
        data = pd.concat([data, df], axis = 1)
        data['log_total_'+suffix] = np.log(data['total_'+suffix] + 1e-6)
        
    # for locations where total pois = 0, the percentage calculation will be NA, replace NA with 0
    data.fillna(0, inplace = True)
    
    # get the climate info for the corresponding year
    if climate_df is not None:
        var = climate_df.columns[climate_df.columns.str.contains('^t_|^snow_|^precip_')].to_list()
        data = data.merge(climate_df[['id'] + var], left_on = 'site_id', right_on = 'id', how = 'left').drop(['site_id', 'id_x', 'id_y', 'latitude', 'longitude'], axis = 1)
        data['temp90'] = data.apply(lambda x: x['t_'+str(int(max(x.year, 2013)))], axis = 1)
        data['snow'] = data.apply(lambda x: x['snow_'+str(int(max(x.year, 2013)))], axis = 1)
        data['precip_5in'] = data.apply(lambda x: x['precip_p5in_'+str(int(max(x.year, 2013)))], axis = 1)
        data['precip_1in'] = data.apply(lambda x: x['precip_1in_'+str(int(max(x.year, 2013)))], axis = 1)
        data.drop(var, axis = 1, inplace = True)
    else:
        data = data.drop(['site_id', 'id', 'latitude', 'longitude'], axis = 1)
    
    # divide year into pre-covid (2013-2019), covid (2020-2021), post-covid (2022)
    data['covid'] = data.apply(lambda x: 'pre-covid' if x.year <= 2019 else ('post-covid' if x.year >= 2022 else 'covid'), axis = 1)
    data['year_group'] = data.apply(lambda x: year_group(x.year), axis = 1)
    
    cat_columns = ['covid', 'year_group', 'year']
    encoder = OneHotEncoder(sparse = False)
    one_hot_encoded = encoder.fit_transform(data[cat_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns = encoder.get_feature_names_out(cat_columns))
    data_encoded = pd.concat([data, one_hot_df], axis = 1)
    data_encoded.drop(cat_columns, axis = 1, inplace = True)
    
    return data_encoded


def check_dist(data, var):
    """
    Create boxplots that compare the distribution of landuse variable cross groups
    """
    fig, axes = plt.subplots(3, 2, figsize = (10, 12), constrained_layout = True)
    
    suffixes = ['_2', '_4', '_6', '_8', '_10']
    
    for i in range(5):
        
        data.boxplot(column = var+suffixes[i], by = 'label', ax = axes[i//2, i%2])
        axes[i//2, i%2].set_ylim([-data[var+suffixes[i]].mean()*5, data[var+suffixes[i]].mean()*5])
        axes[i//2, i%2].set_xlabel('group')
    
    axes[2,1].axis('off')
    fig.suptitle(f'Distribution of "{var}" across groups', fontsize = 16)
    plt.show()
    
    
def pca(data):
    
    # prepare label, feature dataset
    y, X = data['label'], data.drop(['label'], axis = 1)
    
    pca = PCA(n_components = 10)
    pca.fit(X)

    # plot the explained variance ratio
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker = 'o', linestyle = '--')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')

    # Annotate each explained variance
    for i, variance in enumerate(pca.explained_variance_ratio_):
        plt.text(i+1, variance, f'{variance:.2f}', ha = 'center', va = 'bottom')
    
    return pca

# @jit(target_backend='cuda') # run this function on GPU
def classifier_acc_cv(clf, params, random_state = None, cv = None, X = None, Y = None, verbose = None):
    
    """
    Cross-validation accuracy for a Classifier with given hyperparameters.

    Parameters:
    - params (dict): Hyperparameters to evaluate.
    - random_state (int or None): Random state for reproducibility.
    - cv (cross-validation generator): Cross-validation strategy (e.g., KFold).
    - X (array-like): Training features.
    - Y (array-like): Training labels.
    - verbose (bool): Whether to print the parameters being tested.

    Returns:
    - float: Mean accuracy from cross-validation.
    """    
    
    # Initialize the model
    model = clf(random_state = random_state, **params)
    
    # Perform cross validation and compute mean accuracy
    acc = cross_val_score(model, X, Y, cv = cv, scoring = 'balanced_accuracy').mean()
    
    return -acc

def optimize(clf, space, n_iter=100, random_state = None, cv = None, X = None, Y = None):
    
   
    """
    Perform hyperparameter optimization using Hyperopt's Tree-structured Parzen Estimator (TPE).
    
    Parameters:
        clf (callable): A classifier or model function to be optimized.
        space (dict): Search space for hyperparameters.
        n_iter (int, optional): Number of iterations for the optimization. Default is 100.
        random_state (int or None, optional): Random state for reproducibility. Default is None.
        cv (cross-validation splitter, optional): Cross-validation strategy (e.g., KFold or StratifiedKFold).
        X (array-like): Feature matrix.
        Y (array-like): Target vector.
    
    Returns:
        dict: Best hyperparameters found during optimization.
    """

    # Use Trials object to store optimization results
    trial = Trials()
    
    # Run the optimization
    best = fmin(fn = lambda p: classifier_acc_cv(clf, p, random_state = random_state, cv = cv, X = X, Y = Y),
                space = space, 
                algo = tpe.suggest, 
                trials = trial, 
                max_evals = n_iter, 
                rstate = np.random.default_rng(random_state))
    
    return best

def model_performance(X_train, y_train, X_test, y_test, model, cv, params, random_state):
    
    """
    Evaluate the performance of a machine learning model.

    Parameters:
        X_train (array-like): Training feature set.
        y_train (array-like): Training target labels.
        X_test (array-like): Test feature set.
        y_test (array-like): Test target labels.
        model (class): Scikit-learn model class (e.g., RandomForestClassifier).
        cv (object): Cross-validation splitter (e.g., KFold or StratifiedKFold).
        params (dict): Dictionary of hyperparameters for the model.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Training accuracy, cross-validation accuracy, and test accuracy.
    """
    
    # get prediction accuracy on training data
    clf = model(random_state = random_state, **params)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    ac_tr = sum(y_train_pred == y_train)/len(y_train)
    print('The prediction accuracy of train set is {:.1f}%'.format(ac_tr*100))
    cm = confusion_matrix(y_train, y_train_pred)
    print('The confusion matrix is:')
    print(cm)
    print('-----------------------------------------')
    
    # get prediction accuracy of cross validation
    model_ = model(random_state = random_state, **params)
    ac_cv = cross_val_score(model_, X_train, y_train, cv = cv, scoring = 'balanced_accuracy').mean()
    print('The prediction accuracy of {} fold cross-validation is {:.1f}%'.format(cv.n_splits, ac_cv*100))
    print('-----------------------------------------')
    
    # get prediction accuracy on test data
    y_test_pred = clf.predict(X_test)
    ac_tt = sum(y_test_pred == y_test)/len(y_test)
    print('The prediction accuracy of test set is {:.1f}%'.format(ac_tt*100))
    cm = confusion_matrix(y_test, y_test_pred)
    print('The confusion matrix is:')
    print(cm)
    
    return ac_tr, ac_cv, ac_tt

class SMWrapperMNL(BaseEstimator, RegressorMixin):
    """
    A universal sklearn-style wrapper for statsmodels multinomial logit regressors.
    
    Parameters:
        random_state (int or None): Random state for reproducibility. Currently unused.
        alpha (float): Regularization parameter for the L1-penalized estimation.
        fit_intercept (bool): Whether to add a constant term (intercept) to the model.
    
    Methods:
        fit(X, y):
            Fits the multinomial logit model to the data.
        
        predict(X):
            Predicts the class labels for the input data.
        
        summary():
            Returns the summary of the fitted model.
    """
    def __init__(self, random_state, alpha, fit_intercept = True):
        
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.alpha = alpha
        
        
    def fit(self, X, y):
            
        # add constant                
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant = 'add')
        
        # fit model
        self.model_ = sm.MNLogit(y, X)
        self.results_ = self.model_.fit_regularized(method = 'l1', alpha = self.alpha, maxiter = 500, 
                                                    trim_mode = 'size',
                                                    random_state = self.random_state, qc_verbose = False)
        return self
    
    def predict(self, X):
        
        """
        Predict class labels for input data.
        
        Parameters:
            X (DataFrame or array-like): Feature matrix.
        
        Returns:
            Series: Predicted class labels.
        """
        
        if not hasattr(self, "results_"):
            raise ValueError("The model must be fitted before calling predict().")    
            
        # add constant
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant = 'add')
           
        # prediction
        y_pred = self.results_.predict(X).idxmax(axis = 1)

        return y_pred
    
    def summary(self):
        
        """
        Returns the summary of the fitted model.
        
        Returns:
            Summary: Model summary object.
        """
    
        if not hasattr(self, "results_"):
            raise ValueError("The model must be fitted before calling summary().")
            
        return self.results_.summary()
    
    
    
    