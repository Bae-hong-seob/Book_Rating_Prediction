from catboost import *

def CatBoost(args):
    return CatBoostRegressor(random_state=args.random_state,
                             verbose=100,
                             iterations=args.iterations,
                             learning_rate=args.lr,
                             eval_metric='RMSE',
                             l2_leaf_reg=args.l2_leaf_reg)