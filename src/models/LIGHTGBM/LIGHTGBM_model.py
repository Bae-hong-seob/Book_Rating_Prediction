import lightgbm as lgb

def LIGTHGBM(args):
    return lgb.LGBMRegressor(num_leaves=args.num_leaves,
                            num_iterations=args.num_iterations,
                            max_depth=args.lightgbm_max_depth,
                            learning_rate=args.lr,
                            early_stopping_round=args.early_stopping_rounds,
                            verbose=1000
                            )