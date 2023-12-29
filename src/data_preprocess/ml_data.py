import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def mean_std_var(data: dict) -> dict:
    data['train']['mean'] = data['train'].groupby('user_id')['rating'].transform('mean')
    data['train']['var'] = data['train'].groupby('user_id')['rating'].transform('var')
    data['train']['std'] = data['train'].groupby('user_id')['rating'].transform('std')
    
		# test에서 NaN 값으로 채운 경우 valid loss는 더 낮았지만 test loss는 오히려 높았음. 
    temp = data['train'].drop_duplicates('user_id')[['user_id', 'mean', 'var', 'std']]
    data['test'] = pd.merge(data['test'], temp, how='left', on='user_id')
    
		# unique 혹은 dropna 중 고르기 
		# baseline FM에서는 dropna가 validation에서 더 나은 성능을 보임
    new_fields = (
        len(data['train']['mean'].dropna()),
        len(data['train']['var'].dropna()),
        len(data['train']['std'].dropna())
        )
        
    data['field_dims'] = np.append(data['field_dims'], new_fields)

    return data

def process_ml_data(users, books, ratings1, ratings2):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    """
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '', regex=True) # 특수문자 제거
    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)
    
    # print('----start books preprocess ...')
    # print('before publichser preprocess : ',books['publisher'].nunique()) # 수정전 항목 수를 확인합니다.
    
    # publisher_dict=(books['publisher'].value_counts()).to_dict()
    # publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])
    # publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)
    # modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values
    # print(len(modify_list))
    # for publisher in modify_list:
    #     try:
    #         number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
    #         right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
    #         books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
    #     except:
    #         pass
    # print('after publichser preprocess : ',books['publisher'].nunique()) # 수정전 항목 수를 확인합니다.

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    ml_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(ml_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(ml_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(ml_df['location_country'].unique())}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age_category'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age_category'] = test_df['age'].apply(age_map)
    
    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(ml_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(ml_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(ml_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(ml_df['book_author'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df


def ml_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, ml_train, ml_test = process_ml_data(users, books, train, test)
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    data = {
            'train':ml_train,
            'test':ml_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }
    
    return data


def ml_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    X_train = X_train.dropna()
    common_indices = X_train.index.intersection(y_train.index)
    y_train = y_train.loc[common_indices]
    print(X_train.head())
    print(y_train)
    y_train = np.float64(y_train)

    
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    print(f'train size : x,y: {X_train.shape},{y_train.shape}')
    print(f'valid size : x,y: {X_valid.shape},{y_valid.shape}')
    print(X_train)
    return data
