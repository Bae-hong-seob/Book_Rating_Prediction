import os
import tqdm
import pickle
import torch
import torch.nn as nn
import lightgbm as lgb
from torch.nn import MSELoss
from torch.optim import SGD, Adam


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss

import optuna
import numpy as np
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import StratifiedKFold

def objective(trial, args, dataloader, model, setting):
    param_grid = {
        #"device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
    }
    
    X_train, X_test = dataloader['X_train'], dataloader['X_valid']
    y_train, y_test = dataloader['y_train'], dataloader['y_valid']
    
    model = lgb.LGBMRegressor(objective='mse', **param_grid)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[
            LightGBMPruningCallback(trial, "rmse")
        ],  # Add a pruning callback
    )
    
    with open(f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    y_hat = model.predict(X_test)
    y, y_hat = torch.Tensor(y_test), torch.Tensor(y_hat)
    loss_fn = RMSELoss()
    loss = loss_fn(y, y_hat)

    return loss

def multi_train(args, model, context_dataloader, image_dataloader, text_dataloader, logger, setting):
    minimum_loss = 999999999
    if args.loss_fn == 'MSE':
        loss_fn = MSELoss()
    elif args.loss_fn == 'RMSE':
        loss_fn = RMSELoss()
    else:
        pass
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'ADAM':
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        pass

    for epoch in tqdm.tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        batch = 0

        for idx, (context_data, image_data, text_data) in enumerate(zip(context_dataloader['train_dataloader'],image_dataloader['train_dataloader'],text_dataloader['train_dataloader'])):
            context_x, context_y = context_data[0].to(args.device), context_data[1].to(args.device)
            image_x, image_y = [image_data['user_isbn_vector'].to(args.device), image_data['img_vector'].to(args.device)], image_data['label'].to(args.device)
            text_x, text_y = [text_data['user_isbn_vector'].to(args.device), text_data['user_summary_merge_vector'].to(args.device), text_data['item_summary_vector'].to(args.device)], text_data['label'].to(args.device)
            
            y_hat = model(context_x, image_x, text_x)
            loss = loss_fn(context_y.float(), y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch +=1
            
        valid_loss = valid(args, model, context_dataloader, image_dataloader, text_dataloader, loss_fn)
        print(f'Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
        logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
        if minimum_loss > valid_loss:
            minimum_loss = valid_loss
            os.makedirs(args.saved_model_path, exist_ok=True)
            torch.save(model.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pt')
    logger.close()
    return model

def train(args, model, dataloader, logger, setting):
    minimum_loss = 999999999
    if args.loss_fn == 'MSE':
        loss_fn = MSELoss()
    elif args.loss_fn == 'RMSE':
        loss_fn = RMSELoss()
    else:
        pass
    
    if args.model in ('XGB', 'CATBOOST'):
        x, y = dataloader['X_train'], dataloader['y_train']
        model.fit(x,y, eval_set=[(dataloader['X_valid'], dataloader['y_valid'])])
        valid_loss = valid(args, model, dataloader, loss_fn)
        
        if minimum_loss > valid_loss:
            minimum_loss = valid_loss
            os.makedirs(args.saved_model_path, exist_ok=True)
            model.save_model(f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.json')
                
        print(f'valid_loss: {valid_loss:.3f}')
        
    elif args.model in ('LIGHTGBM'):
        if args.optuna == True:
            study = optuna.create_study(direction='minimize', study_name='LGBM Regressor')
            func = lambda trial : objective(trial, args, dataloader, model, setting)
            study.optimize(func, n_trials=20)
            
            print(f"\tBest value (rmse): {study.best_value:.5f}")
            print(f"\tBest params:")

            for key, value in study.best_params.items():
                print(f"\t\t{key}: {value}")
                
        else:
            x, y = dataloader['X_train'], dataloader['y_train']
            model.fit(x,y, eval_set=[(dataloader['X_valid'], dataloader['y_valid'])], eval_metric='rmse')
            valid_loss = valid(args, model, dataloader, loss_fn)
            
            if minimum_loss > valid_loss:
                minimum_loss = valid_loss
                os.makedirs(args.saved_model_path, exist_ok=True)
                with open(f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
                    
            print(f'valid_loss: {valid_loss:.3f}')
        
    else:
        if args.optimizer == 'SGD':
            optimizer = SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'ADAM':
            optimizer = Adam(model.parameters(), lr=args.lr)
        else:
            pass
        
        for epoch in tqdm.tqdm(range(args.epochs)):
            model.train()
            total_loss = 0
            batch = 0

            for idx, data in enumerate(dataloader['train_dataloader']):
                if args.model == 'CNN_FM':
                    x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
                elif args.model == 'DeepCoNN':
                    x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
                else:
                    x, y = data[0].to(args.device), data[1].to(args.device)
                y_hat = model(x)
                loss = loss_fn(y.float(), y_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch +=1
            valid_loss = valid(args, model, dataloader, loss_fn)
            print(f'Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}')
            logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
            if minimum_loss > valid_loss:
                minimum_loss = valid_loss
                os.makedirs(args.saved_model_path, exist_ok=True)
                torch.save(model.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pt')
        logger.close()
        
    return model

def valid(args, model, dataloader, loss_fn):
    if args.model in ('XGB', 'LIGHTGBM', 'CATBOOST'):
        x,y = dataloader['X_valid'], dataloader['y_valid']
        y_hat = model.predict(x)
        y, y_hat = torch.Tensor(y.values), torch.Tensor(y_hat)
        loss = loss_fn(y, y_hat)
        valid_loss = loss
        
    else:
        model.eval()
        total_loss = 0
        batch = 0

        for idx, data in enumerate(dataloader['valid_dataloader']):
            if args.model == 'CNN_FM':
                x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
            elif args.model == 'DeepCoNN':
                x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
            else:
                x, y = data[0].to(args.device), data[1].to(args.device)
                
            if args.model in ('XGB', 'LIGHTGBM', 'CATBOOST'):
                y_hat = model.predict(x)
                y_hat = torch.Tensor(y_hat)
            else:
                y_hat = model(x)
                
            loss = loss_fn(y.float(), y_hat)
            total_loss += loss.item()
            batch +=1
        valid_loss = total_loss/batch
        
    return valid_loss

def multi_valid(args, model, context_dataloader, image_dataloader, text_dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0

    for idx, (context_data, image_data, text_data) in enumerate(zip(context_dataloader['valid_dataloader'],image_dataloader['valid_dataloader'],text_dataloader['valid_dataloader'])):
        context_x, context_y = context_data[0].to(args.device), context_data[1].to(args.device)
        image_x, image_y = [image_data['user_isbn_vector'].to(args.device), image_data['img_vector'].to(args.device)], image_data['label'].to(args.device)
        text_x, text_y = [text_data['user_isbn_vector'].to(args.device), text_data['user_summary_merge_vector'].to(args.device), text_data['item_summary_vector'].to(args.device)], text_data['label'].to(args.device)
        
        y_hat = model(context_x, image_x, text_x)
        loss = loss_fn(context_y.float(), y_hat)
        total_loss += loss.item()
        batch +=1
    valid_loss = total_loss/batch
    return valid_loss

def test(args, model, dataloader, setting):
    predicts = list()
    if args.use_best_model == True:
        if args.model in ('XGB', 'CATBOOST'):
            model.load_model(f'./saved_models/{setting.save_time}_{args.model}_model.json')
            
        elif args.model in ('LIGHTGBM'):
            if args.optuna==True:
                with open(f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pkl', 'rb') as f:
                    model = pickle.load(f)
            else:
                with open(f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                
        else:
            model.load_state_dict(torch.load(f'./saved_models/{setting.save_time}_{args.model}_model.pt'))
            
    if args.model in ('XGB', 'LIGHTGBM', 'CATBOOST'):
        x = dataloader['test']
        y_hat = model.predict(x)
        predicts.extend(y_hat.tolist())
        
    else:
        model.eval()
        for idx, data in enumerate(dataloader['test_dataloader']):
            if args.model in ('XGB', 'LIGHTGBM', 'CATBOOST'):
                x = data[0]
            elif args.model == 'CNN_FM':
                x, _ = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
            elif args.model == 'DeepCoNN':
                x, _ = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
            else:
                x = data[0].to(args.device)
                
            y_hat = model(x)
            predicts.extend(y_hat.tolist())
        
    return predicts

def multi_test(args, model, context_dataloader, image_dataloader, text_dataloader, setting):
    predicts = list()
    if args.use_best_model == True:
        model.load_state_dict(torch.load(f'./saved_models/{setting.save_time}_{args.model}_model.pt'))
    else:
        pass
    model.eval()
    for idx, (context_data, image_data, text_data) in enumerate(zip(context_dataloader['test_dataloader'],image_dataloader['test_dataloader'],text_dataloader['test_dataloader'])):
        context_x = context_data[0].to(args.device), context_data[1].to(args.device)
        image_x, _ = [image_data['user_isbn_vector'].to(args.device), image_data['img_vector'].to(args.device)], image_data['label'].to(args.device)
        text_x, _ = [text_data['user_isbn_vector'].to(args.device), text_data['user_summary_merge_vector'].to(args.device), text_data['item_summary_vector'].to(args.device)], text_data['label'].to(args.device)
            
        y_hat = model(context_x, image_x, text_x)
        predicts.extend(y_hat.tolist())
    return predicts
