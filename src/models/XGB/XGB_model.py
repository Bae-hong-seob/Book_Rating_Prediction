from xgboost import XGBRegressor

# class eXtraGredientBoost():
#     def __init__(self, args):
#         self.n_estimators = args.n_estimators
#         self.learning_rate = args.lr
#         self.max_depth = args.max_depth
#         self.gamma = args.gamma
#         self.colsample_bytree = args.colsample_bytree
#         self.random_state = args.random_state
        
#     def forward(self):
#         return XGBRegressor(n_estimators=self.n_estimators,
#                             learning_rate=self.lr,
#                             max_depth=self.max_depth,
#                             gamma=self.gamma,
#                             colsample_bytree=self.colsample_bytree,
#                             random_state=self.random_state)
        
def eXtraGredientBoost(args):
    return XGBRegressor(n_estimators=args.n_estimators,
                        learning_rate=args.lr,
                        max_depth=args.max_depth,
                        gamma=args.gamma,
                        colsample_bytree=args.colsample_bytree,
                        random_state=args.random_state,
                        )