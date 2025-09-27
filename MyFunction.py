import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def Affichage(X_train, y_train, X_test, y_test_log, pipeline):
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test = np.expm1(y_test_log)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test_log, y_pred_log)
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_CrossVal = []
    for train, val in kfold.split(X_train):
        X_tr, X_val = X_train.iloc[train], X_train.iloc[val]
        y_tr, y_val_cv_log = y_train.iloc[train], y_train.iloc[val]
        pipeline.fit(X_tr, y_tr)
        y_pred_cv_log = pipeline.predict(X_val)
        y_pred_cv = np.expm1(y_pred_cv_log)
        y_val_cv = np.expm1(y_val_cv_log)
        rmse_CrossVal.append(np.sqrt(mean_squared_error(y_val_cv, y_pred_cv)))

    scores = [f"{x:.2f}" for x in rmse_CrossVal]

    print('\n--------------------------------------------\n')
    print("\nModel Results (log)")
    print('Score of model :', pipeline.steps[-1][0])
    print(f'MSE : {mse:.3f}, sqrt(MSE) :{np.sqrt(mse):.3f}')
    print(f'R² : {r2:.5f}')
    print(f'Cross Validation scores (RMSE) : {scores}')
    print(f"Average Accuracy : {np.mean(rmse_CrossVal):.3f}")
    print(f"Std-deviation : {np.std(rmse_CrossVal):.3f}")
    print('\n--------------------------------------------\n')
   
    plt.figure()
    plt.title(f'Predictions vs True Values ({pipeline.named_steps[pipeline.steps[-1][0]].__class__.__name__})')
    plt.scatter(y_pred, y_test, alpha=0.6, s = 10)
    plt.plot([0, max(y_test)], [0, max(y_test)], marker=' ', linestyle='-')
    plt.xlabel('Predicted SalePrice (dollars)')
    plt.ylabel('True SalePrice (dollars)')
    
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, s = 10)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted SalePrice (dollars)')
    plt.ylabel('Residuals (dollars)')
    plt.title(f'Residuals Plot ({pipeline.named_steps[pipeline.steps[-1][0]].__class__.__name__})')
    plt.show()
    
def GridSearch(model_pipeline, param_grid, X_train, y_train, Columns_name):
    
    
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("Best parameters :", grid_search.best_params_)

    cvres = grid_search.cv_results_
    print('\n--------------------------------------------\n')
    print('Mean score for Grid_search')
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(f"{np.sqrt(-mean_score):.4f}", params)
    print('\n--------------------------------------------\n')
    
    feature_importances = grid_search.best_estimator_.named_steps[model_pipeline.steps[-1][0]].feature_importances_

    import heapq
    Top5 = heapq.nlargest(5, zip(feature_importances, Columns_name), key=lambda x: x[0])
    print("######## Features importance scores ########")
    for s, f in Top5:
        print('Importance score of feature ', f, f' : {s:.4f}')

    return grid_search.best_estimator_