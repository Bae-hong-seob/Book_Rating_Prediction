import streamlit as st
import pandas as pd
import argparse
from predict import load_model, get_prediction

def main(args):
    st.title('사용자 평점 예측')
    st.caption('사용자 데이터를 입력하세요 :')

    model = load_model(args)
    model.eval()

    # 측정 결과들 모아두는 df
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame({
            'user_id': [],
            'isbn' : [],
            'age' : [],
            'location_city' : [],
            'location_state' : [],
            'location_country' : [],
            'category' : [],
            'publisher' : [],
            'language' : [],
            'book_author' : [],
            'age_category' : [],
            'rating' : []
        })

    with st.form(key='문장입력 form'):
        user_id = st.text_input("Enter user_id:")
        isbn = st.text_input("Enter isbn:")
        age = st.text_input("Enter age:")
        location_city = st.text_input("Enter location city:")
        location_state = st.text_input("Enter location state:")
        location_country = st.text_input("Enter location country:")
        category = st.text_input("Enter category:")
        publisher = st.text_input("Enter publisher:")
        language = st.text_input("Enter language:")
        book_author = st.text_input("Enter book_author:")
        age_category = st.text_input("Enter age_category:")

        form_submitted = st.form_submit_button('유사도 측정')

    if form_submitted:
        if all(user_id, isbn, age, location_city, location_state, location_country, category, publisher, language, book_author, age_category):

            # 새로운 데이터를 기존 df에 합치기 
            new_data = pd.DataFrame({
                'user_id': [user_id],
                'isbn' : [isbn],
                'age' : [age],
                'location_city' : [location_city],
                'location_state' : [location_state],
                'location_country' : [location_country],
                'category' : [category],
                'publisher' : [publisher],
                'language' : [language],
                'book_author' : [book_author],
                'age_category' : [age_category],
            })
            rating = get_prediction(model, new_data)
            new_data[rating] = rating
            
            st.session_state.df = pd.concat([st.session_state.df, new_data])
            
            # similarity 기준으로 순위 매기기
            st.session_state.df = st.session_state.df.sort_values(by='rating', ascending=False).reset_index(drop=True)
            
            # rank 컬럼 추가
            st.session_state.df['rank'] = st.session_state.df.index + 1
            
            st.write(f"사용자의 예측 평점 : {rating}")
            st.success('성공!')
        else:
            st.write("Please enter both sentences.")
            st.error('다시 한번 생각해보세요!')

    st.divider()
    col1, col2, col3 = st.columns(3)
    
    # df 크기 조절
    col1.checkbox("창 크기조절", value=True, key="use_container_width")

    # df 리셋 버튼
    if col2.button("데이터 리셋하기"):
        st.session_state.df = pd.DataFrame({
            'user_id': [],
            'isbn' : [],
            'age' : [],
            'location_city' : [],
            'location_state' : [],
            'location_country' : [],
            'category' : [],
            'publisher' : [],
            'language' : [],
            'book_author' : [],
            'age_category' : [],
            'rating' : []
        })

    # df csv로 다운로드
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False, header=True).encode('cp949')
    csv = convert_df(st.session_state.df)
    col3.download_button(
        label="CSV로 다운받기",
        data=csv,
        file_name='sts_data_outputs.csv',
        mime='text/csv',
    )

    st.dataframe(st.session_state.df, use_container_width=st.session_state.use_container_width)

if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument


    ############### BASIC OPTION
    arg('--data_path', type=str, default='data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, default='CATBOOST', choices=['XGB', 'LIGHTGBM','CATBOOST','FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'Multi'],
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