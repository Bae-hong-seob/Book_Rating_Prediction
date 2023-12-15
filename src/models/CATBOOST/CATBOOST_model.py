from catboost import *

def CatBoost(args):
    return CatBoostRegressor(random_state=args.random_state,
                             verbose=100,
                             iterations=args.iterations,
                             learning_rate=args.lr)