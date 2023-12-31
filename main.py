import time
import argparse
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

from src.utils import Logger, Setting, models_load, multi_load
from src.data_preprocess.ml_data import ml_data_load, ml_data_split
from src.data_preprocess import context_data_load, context_data_split, context_data_loader
from src.data_preprocess import image_data_load, image_data_split, image_data_loader
from src.data_preprocess import text_data_load, text_data_split, text_data_loader

from src.train import train, test, multi_train, multi_test


def main(args):
    Setting.seed_everything(args.seed)


    ######################## DATA LOAD
    print(f'--------------- {args.model} Load Data ---------------')
    if args.model == 'Multi':
        context_data = context_data_load(args)
        image_data = image_data_load(args)
        import nltk
        nltk.download('punkt')
        text_data = text_data_load(args)
        
    elif args.model in ('XGB, LIGHTGBM, CATBOOST'):
        data = ml_data_load(args)
        
    # elif args.model in ('FM', 'FFM'):
    #     data = context_data_load(args)
    # elif args.model in ('NCF', 'WDN', 'DCN'):
    #     data = dl_data_load(args)
    # elif args.model == 'CNN_FM':
    #     data = image_data_load(args)
    # elif args.model == 'DeepCoNN':
    #     import nltk
    #     nltk.download('punkt')
    #     data = text_data_load(args)
    else:
        pass


    ######################## Train/Valid Split
    print(f'--------------- {args.model} Train/Valid Split ---------------')
    if args.model=='Multi':
        context_data = context_data_split(args, context_data)
        context_data = context_data_loader(args, context_data)
        image_data = image_data_split(args, image_data)
        image_data = image_data_loader(args, image_data)
        text_data = text_data_split(args, text_data)
        text_data = text_data_loader(args, text_data)
        
    elif args.model in ('XGB, LIGHTGBM, CATBOOST'):
        data = ml_data_split(args, data)
                
    # elif args.model in ('FM', 'FFM'):
    #     data = context_data_split(args, data)
    #     data = context_data_loader(args, data)

    # elif args.model in ('NCF', 'WDN', 'DCN'):
    #     data = dl_data_split(args, data)
    #     data = dl_data_loader(args, data)

    # elif args.model=='CNN_FM':
    #     data = image_data_split(args, data)
    #     data = image_data_loader(args, data)

    # elif args.model=='DeepCoNN':
    #     data = text_data_split(args, data)
    #     data = text_data_loader(args, data)
    else:
        pass

    ####################### Setting for Log
    setting = Setting()

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)

    logger = Logger(args, log_path)
    logger.save_args()


    ######################## Model
    print(f'--------------- INIT {args.model} ---------------')
    if args.model=='Multi':
        Autoint_model = torch.load('/data/ephemeral/home/Book_Rating_Prediction/saved_models/20231220_200029_AutoInt_model.pt')
        CNN_FM_model = torch.load('/data/ephemeral/home/Book_Rating_Prediction/saved_models/20231221_041454_CNN_FM_model.pt')
        DeepCoNN_model = torch.load('/data/ephemeral/home/Book_Rating_Prediction/saved_models/20231221_052629_DeepCoNN_model.pt')
        
        model = multi_load(args, context_data,image_data,text_data)
    else:
        model = models_load(args,data)


    ######################## TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    if args.model=='Multi':
        model = multi_train(args, model, Autoint_model, context_data, CNN_FM_model, image_data, DeepCoNN_model, text_data, logger, setting)
    else:
        model = train(args, model, data, logger, setting)


    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    if args.model=='Multi':
        predicts = multi_test(args, model, Autoint_model, context_data, CNN_FM_model, image_data, DeepCoNN_model, text_data, setting)
    else:
        predicts = test(args, model, data, setting)


    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    if args.model in ('XGB', 'LIGHTGBM', 'CATBOOST'):
        submission['rating'] = predicts
    elif args.model in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'Multi'):
        submission['rating'] = predicts
    else:
        pass
    
    submission['rating'] = submission['rating'].apply(lambda x: 1 if x < 1 else (10 if x > 10 else x))

    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)
    print('make csv file ... ', filename)


if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument


    ############### BASIC OPTION
    arg('--data_path', type=str, default='data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, choices=['XGB', 'LIGHTGBM','CATBOOST','FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'Multi'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')


    ############### TRAINING OPTION
    arg('--batch_size', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--epochs', type=int, default=30, help='Epoch 수를 조정할 수 있습니다.')
    arg('--lr', type=float, default=0.003, help='Learning Rate를 조정할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE', choices=['MSE', 'RMSE'], help='손실 함수를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')
    arg('--optuna', type=bool, default=False, help='하이퍼 파라미터 자동 최적화 설정입니다.')


    ############### GPU
    arg('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')


    ############### XGB OPTION
    arg('--n_estimators', type=int, default=3000, help='XGB 학습 모델 수, 생성 트리 개수 입니다.')
    arg('--max_depth', type=int, default=6, help='최대 트리 탐색 깊이입니다.')
    arg('--gamma', type=int, default=0, help='감마 곡선 정도, 클수록 과적합 방지 효과가 있습니다.')
    arg('--colsample_bytree', type=float, default=0.5, help='max_features 비율을 의미. 피처가 많을 때 과적합 조절을 위해 사용합니다.')
    arg('--random_state', type=int, default=4, help='모델 파라미터 재현을 위한 설정입니다.')
    arg('--early_stopping_rounds', type=int, default=100, help='early stop size 설정입니다.')
    
    
    ############### LightGBM OPTION
    arg('--num_iterations', type=int, default=10000, help='전체 트리 반복 학습 횟수')
    arg('--num_leaves', type=int, default=1000, help='전체 트리의 leave 수')
    arg('--lightgbm_max_depth', type=int, default=-1, help='최대 트리 탐색 깊이입니다.')
    
    
    
    ############### CatBoost OPTION
    arg('--iterations', type=int, default=5000, help='boost계열 모델 업데이트 횟수입니다.')
    arg('--l2_leaf_reg', type=int, default=5, help='가중치에 대한 L2 정규화 용어입니다. 큰 가중치에 불이익을 주어 과적합을 방지하는 데 도움이 됩니다.')
    
    
    ############### FM, FFM, NCF, WDN, DCN Common OPTION
    arg('--embed_dim', type=int, default=16, help='FM, FFM, NCF, WDN, DCN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--dropout', type=float, default=0.2, help='NCF, WDN, DCN에서 Dropout rate를 조정할 수 있습니다.')
    arg('--mlp_dims', type=list, default=(16, 16), help='NCF, WDN, DCN에서 MLP Network의 차원을 조정할 수 있습니다.')


    ############### DCN
    arg('--num_layers', type=int, default=3, help='에서 Cross Network의 레이어 수를 조정할 수 있습니다.')


    ############### CNN_FM
    arg('--cnn_embed_dim', type=int, default=64, help='CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--cnn_latent_dim', type=int, default=12, help='CNN_FM에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')


    ############### DeepCoNN
    arg('--vector_create', type=bool, default=False, help='DEEP_CONN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.')
    arg('--deepconn_embed_dim', type=int, default=32, help='DEEP_CONN에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--deepconn_latent_dim', type=int, default=10, help='DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')
    arg('--conv_1d_out_dim', type=int, default=50, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')
    arg('--kernel_size', type=int, default=3, help='DEEP_CONN에서 1D conv의 kernel 크기를 조정할 수 있습니다.')
    arg('--word_dim', type=int, default=768, help='DEEP_CONN에서 1D conv의 입력 크기를 조정할 수 있습니다.')
    arg('--out_dim', type=int, default=32, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')


    args = parser.parse_args()
    main(args)