머신러닝 앙상블
  - bagging, voting, boosting
  
  - 랜덤 포레스트 (random forest)
      랜덤 포레스트는 결정 트리(decision tree)를 기반하여 만들어졌다. 여러 개의 결정 트리 classifier가 생성되고 데이터를 각자 방식으로 sampling하여 개별적인 학습
      최종적으로  voting을 통해 데이터에 대한 예측을 수행
      
      ![image](https://user-images.githubusercontent.com/37740450/120281113-82aab880-c2f3-11eb-813d-0c16432e3f7e.png)
      
      
      랜덤 포레스트는 각각의 classifier를 가지고 훈련 하지만 학습하는 dataset은 original dataset 에서 sampling하여 가지고 온다. 이 과정을 부트스트래핑(bootstraping)이라고 한다
      각 bootstrap은 데이터가 중복될 수 있으며 내부적으로도 데이터가 중복 될 수 있다.
      
      scikit learn의 ensemble패키지로 이용
      
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot asplt
        import seaborn as sns
      
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
      
      
      
      kaggle의 타이타닉 활용
        
        y = titanic_df['Survived']
        x = titanic_df.drop['Survived', axis = 1]
        
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 10)
        rf.fit(x_train, y_train)
        pred = rf.predict(x_test)
        print("정확도 : {0:.3f}".format(accuracy_score(y_test, pred)))
        
        
      랜덤 포레스트의 하이퍼 파라미터
      
      n_estimator : 결정 트리의 개수. default = 10
      max_features : 데이터의 feature를 참조할 비율, 개수를 뜻한다. default = auto
      max_depth : 트리의 깊이
      min_samples_leaf : 리프노드가 되기 위한 최소한의 샘플 데이터 개수
      min_samples_split : 노드를 분할하기 위한 최소한의 데이터 개수
      
      grid search는 key, value 쌍의 데이터를 입력 받아 key값에 해당하는 파라미터의 값 튜닝
      rf_param_grid = {
      'n_estimators' : [100, 200],
      'max_depth' : [6,8,10,12] ,
      'min_samples_leaf' : [3,5,7,10]
      'min_samples_split' : [2,3,5,10]
      }
      
      rf_grid = GridSearchcv(rf, param_grid = rf_param_grid, scoring = "accuracy", n_jobs = -1, verbose = 1)\
      rf_grid.fit(x_train, y_train)
      
      rf_grid.best_score_  #최고의 평균 정확도
      rf_grid.best_params_ #최고의 파라미터
      
      cv_result_df.sort_values(by = ['rank_test_score'], inplace = True)
      cv_result_df[['params', 'mean_test_score', 'rank_test_score']].head(10)
      
      feature_importances = model.feature_importances_
      ft_importances = pd.Series(feature_importances, index = x_train.columns)
      ft_importances = ft_importances.sort_values(ascending = False)
      
      plt.figure(figsize = (12,10))
      plt.title("feature importances")
      sns.barplot(x = ft_importances, y = x_train.columns)
      plt.show()
      
      
   - 앙상블 부스팅 (ensemble boosting)
    배깅은 여러개의 단일 모델을 만든후 boostrap 과정으로 데이터를 랜덤으로 추출한 뒤 모델 훈련 그리고 voting과정을 통해 데이터 예측
    부스팅은 가중치를 부여하여 틀린 부분을 더 잘 맞출 수 있도록 하는 것
    부스팅은 배깅과 유사하게 초기 샘플 데이터를 뽑아내고 다수의 분류기를 생성한다, 훈련 과정에서 앞 모델이 틀렸던 부분을 가중치 부여하며 진행
    약검출기(week classifier)들을 여러개 모아 강검출기(strong classifier)생성
    다음 단계의 week classifier는 이전 단계의 weak classifier의 영향을 받게 되고 이전의 양상을 본 뒤 잘 맞출 수 있는 방향으로 다음 단계 진행하며 weight 업데이트
    최종적으로 strong classifier 생성
    
    
   ![image](https://user-images.githubusercontent.com/37740450/120288697-79bde500-c2fb-11eb-82bd-63cded191d29.png)
    
    
   - ensemble boosting - Adaboost
    Adaboost는 Adaptive Boosting으로서 이전 분류기가 틀린 부분을 adaptive하게 바꾸어가며 잘못 분류되는 데이터에 집중하도록 한다.
      
   ![image](https://user-images.githubusercontent.com/37740450/120288875-a7a32980-c2fb-11eb-9ac2-dee580826d50.png)
   
   
   첫 번째 dataset에서 classifier가 분류를 진행한다. 틀린 부분에 가중치를 부여 +가 두번째 그림에서 커짐(가중치 부여) 다른 classifier가 예측을 했을 때
   틀리면 또 가중치 부여 Adaboost 이것을 반복
   
   
   - Gradient boosting (Adaboost와 비슷한 메커니즘)
     여기서는  Graidient Descent 사용
     
   - 파이썬 사이킷런에서의 Gradient Boosting
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
   
    gb = GradientBoostingClassifier(random_state = 0)
    gb.fit(x_train, y_train)
    pred = gb.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    
    파라미터
    n_estimator : 결정 트리 개수 default = 10
    max_features : 데이터의 feautre를 참조할 비율, 개수
    max_depth : 트리의 깊이
    min_samples_leaf : 리프노드가 되기 위한 최소한의 샘플 데이터 개수
    min_samples_split : 노드를 분할하기 위한 최소한의 데이터 개수
    
    (랜덤 포레스트와 gradinet boosing의 파라미터 같은 거)
    
    
    gradient boosting에서 사용되는 하이퍼 파라미터
    loss : gradient descent에서 사용할 비용함수.
    learning_rate : 학습률
    subsample : weak learner가 학습에 사용하는 데이터 샘플링 비율. 기본 값은 1이며 전체 학습 데이터를 기반 0.7 = 70%
    
    gradient boosting부터는 하이퍼 파라미터가 매우 많기 때문에 튜닝 과정에서 시간 소모가 크게 작용된다
    
    gb_param_grid = {
    'n_estimators' : [100, 200],
    'max_depth' : [6, 8, 10, 12],
    'min_samples_leaf' : [3, 5, 7 , 10],
    'min_samples_split' : [2, 3, 5, 10],
    'learning_rate' : [0.05, 0.1, 0.2]
    }
    
    gb_grid = GridSearchCV(gb, param_grid = gb_param_grid, scoring = "accuracy", n_jobs = -1, verbose = 1)
    gb_grid.fit(x_train, y_train)
     
    cv_result_df.sort_values(by = ['rank_test_score'], inplace = True)
    cv_result_df[['params', 'mean_test_score', 'rank_test_score']].head(10)
    
    model = gb_grid.best_estimator_          #best_estimator_를 통해 최고의 모델 추출
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)

    feature의 importances를 마찬가지로 체크 할 수 있다.
    
    feature_importances = model.feature_importances_
    
    ft_importances = pd.Series(feature_importances, index = x_train.columns)
    ft_iomportances = ft_)importances.sort_values(ascending = False)
    
    plt.figure(figsize = (12,10))
    plt.title("feature importances")
    sns.barplot(x = ft.importances, y = x_train.columns)
    plt.show()
    
      
      
      
  Gradient Boosting의 문제점
    - 부스팅은 강력한 모델이지만 느리고 overfitting 문제가 있다.
    
    
  Xgboost
    - gbm보다 빠르다.
    - overfitting 방지가 가능한 규제가 포함
    - CART (Classification And Regression Tree)를 기반
    
    - 조기 종료(early stopping) 제공
    - Gradient Boost 기반
        앙상블 부스팅의 특징인 가중치 부여를 경사하강법으로 한다.
      
      
      
    from xgboost import plot_importance
    from xgboost import XGBClassifier
    
    xgboost의 하이퍼파라미터
    - n_estimators(or num_boost_round) : 결정 트리의 개수
    - max_depth : 트리의 깊이
    - colsample_bytree : 컬럼의 샘플링 비율 (random forest의 max_features와 비슷)
    - subsample : weak learner가 학습에 사용하는 데이터 샘플링 비율
    - learning_rate : 학습률
    - min_split_loss : 리프 노드를 추가적으로 나눌지 결정하는 값
    - reg_lambda : L2 규제
    - reg_alpha : L1 규제
    
    xgb = XGBClassifier()
    
    xgb_param_grid = {
    
    
    
    }
      
    xgb_grid = GridSearchCV(grid, param_grid = xgb_param_grid, scoring = "accuracy", n_jobs = -1, verbose = 1)
    xgb_grid.fit(x_train, y_train)
    
    xgb_grid.best_score_  #최고 평균 정확도
    xgb_grid.best_params_ #최고의 파라미터
      
    cv_result_df = pdDataFrame(xgb_grid.cv_results_)
    cv_result_df.sort_values(by = ['rank_test_score'], inplace = True)
    
    cv_result_df[['params', 'mean_test_score', 'rank_test_score']].head(10)
    
    
    
    - xgboost는 조기 종료 (early stopping)기능을 제공
    성능이 좋아지지 않는 모델을 불필요하게 학습 시키는 것을 방지하고 최고의 모델을 뽑도록
    
    xgb = XGBClassifier(n_estimators = 400, learning_rate = 0.1, max_depth = 3)
    evals = [(x_test, y_test)]
    xgb.fit(x_train, y_train, early_stopping_rounds = 100, eval_metric = "logloss", eval_set = evals, verbose = 1)
    
    fig, ax = plt.subplots()
    plot_importance(xgb, ax = ax)
    
    
   LightGBM 
   - boosting 알고리즘인 xgboost는 굉장히 좋은 성능을 보여주었지만 여전히 학습시간이 느리다.
   - 더불어 하이퍼파라미터도 많다.
   - grid search로 튜닝하면 시간은 더 오래 걸림
   - LightGBM은 대용량 데이터 처리가 가능하고, 다른 모델들보다 더 적은 자원을 사용하며 빠르다, GPU도 지원
   - 너무 적은 수 사용하면 overfitting 문제

   ![image](https://user-images.githubusercontent.com/37740450/120297148-c73e5000-c303-11eb-9a88-e370d29fdd76.png)
    
   - lightgbm은 leaf wise(리프 중심) 트리 분할 사용
   - 기존 트리들은 tree depth 줄이기 위해 level wise(균형 트리) 분할 사용.
   - level-wise 트리 분석은 균형을 잡아ㅏ주어야 하기 때문에 tree의 depth가 줄어든다. 균형을 잡아주기 위한 연산이 추가된다.
   - lightgbm 트리의 균형은 맞추지 않고 리프 노드를 지속적으로 분할하면서 진행
   - 이 리프 노드를 max delta loss 값을 가지는 리프 노드를 계속 분할해간다. 비대칭적이고 깊은 트리가 생성되지만 동일한 leaf를 생성할 때 leaf-wise는
   - level-wise 보다 손실을 줄일 수 있다.
   
     n_estimators : 반복하려는 트리의 개수
     learning_rate : 학습률
     max_depth : 트리의 최대 깊이
     min_child_samples: 리프 노드가 되기 위한 최소한의 샘플 데이터 개수
     num_leaves : 하나의 트리가 가질 수 있는 최대 리프 개수
     feature_fraction :트리를 학습할 때마다 선택하는 feature의 비율
     reg_lambda : L2 regularization
     reg_alpha : L1 regularization
      
   
    from lightgbm import LGBMClassifier, plot_importance
     lgb = LGBMClassifier(n_estimators = 400)
     lgb.fit(x_train, y_train)
     lgb_pred = lgb.predict(x_test)
     metrics(y_test, lgb_pred)

     lgb = LGBMClassifier(n_estimators = 400)
     evals = [(x_test, y_test)]
     lgb.fit(x_train, y_train, early_stopping_rounds = 100, eval_metric = "logloss", eval_set = evals, verbose =True)

     fig, ax = plt.subplots(figsize = (10, 6))
     plot_importance(lgb, ax = ax)
      
      
      
      
      
      
      
