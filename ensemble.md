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
    
    
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      