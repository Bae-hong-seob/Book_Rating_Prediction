from xgboost import XGBRegressor

def XGB(args):
    return XGBRegressor(n_estimators=args.n_estimators,
                        learning_rate=args.lr,
                        max_depth=args.max_depth,
                        gamma=args.gamma,
                        colsample_bytree=args.colsample_bytree,
                        random_state=args.random_state)