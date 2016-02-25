# Homesite - 31st solution
import numpy as np
import pandas as pd
import xgboost as xgb

FOLDER = './data/'
FILE1 = 'train.csv'
FILE2 = 'test.csv'
FILE3 = 'train_numeric.csv'
FILE4 = 'test_numeric.csv'


class HomesiteModel():
    
    def __init__(self, model_ind, strategy, params, folder=FOLDER):
        
        self.model_ind = model_ind
        self.strategy = strategy
        self.params = params
        self.folder = folder
        
        # extra variables
        self._train = None
        self._test = None
        self._clf = None
    
    def _feature_cleaning(self, train, test):
        
        if self.strategy is 'numeric':
            train = train.drop(['PropertyField6_pctg', 'GeographicField10A'], axis=1)
            test = test.drop(['PropertyField6_pctg', 'GeographicField10A'], axis=1)
        else:
            train = train.drop(['PropertyField6', 'GeographicField10A'], axis=1)
            test = test.drop(['PropertyField6', 'GeographicField10A'], axis=1)
                     
        return train, test
    
    def _fill_na(self, train, test):
        
        train = train.fillna(value=-1)
        test = test.fillna(value=-1)
        
        return train, test
    
    def _ohe(self, train, test):
        
        # only convert variables with dtype 'object' to dummies
        important = ['QuoteNumber', 'Original_Quote_Date', 
                     'QuoteConversion_Flag']
        df = pd.concat([train, test])
        df.reset_index(inplace=True)
        df.drop(['index'], axis=1, inplace=True)
        reordered = important + [c for c in df.columns if c not in important]
        df = pd.DataFrame(df[reordered])
        
        # perform the conversion
        for col in df:
            if df[col].dtype == 'object':
                df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
                df.drop([col], axis=1, inplace=True)
        
        # split the train and test again
        train = pd.DataFrame(df[df['QuoteConversion_Flag'].notnull()])
        test = pd.DataFrame(df[df['QuoteConversion_Flag'].isnull()])
        train.reset_index(inplace=True)
        test.reset_index(inplace=True)
        train.drop(['index'], axis=1, inplace=True)
        test.drop(['index', 'QuoteConversion_Flag'], axis=1, inplace=True)
        
        return train, test

    def _numeric_encoding_helper(self, train, test):
        
        return train, test
        
    def _extra_features(self, train, test):
        
        # convert date to year, month, day, day of week, and week of year
        for df in [train, test]:
            df['Year'] = df.Original_Quote_Date.apply(lambda x: x.year)
            df['Month'] = df.Original_Quote_Date.apply(lambda x: x.month)
            df['Day'] = df.Original_Quote_Date.apply(lambda x: x.day)
            df['WeekOfYear'] = df.Original_Quote_Date.apply(lambda x:\
                                                            x.weekofyear)
            df['DayOfWeek'] = df.Original_Quote_Date.apply(lambda x:\
                                                           x.dayofweek)
        # add engineered features
        # credits go to Vincent.Y
        # https://www.kaggle.com/yangnanhai
        golden_feature = [("CoverageField1B","PropertyField21B"),
                          ("GeographicField6A","GeographicField8A"),
                          ("GeographicField6A","GeographicField13A"),
                          ("GeographicField8A","GeographicField13A"),
                          ("GeographicField11A","GeographicField13A"),
                          ("GeographicField8A","GeographicField11A")]
                        
        for featureA, featureB in golden_feature:
            train["_".join([featureA, featureB, "diff"])] = train[featureA] - train[featureB]
            test["_".join([featureA, featureB, "diff"])] = test[featureA] - test[featureB]
        
        return train, test
        
    def _pre_processing(self, train, test):

        # get rid of redundant features
        train, test = self._feature_cleaning(train, test)
        
        # fill the null values
        train, test = self._fill_na(train, test)
        
        # perform strategy (choose between ohe or numeric encoding)
        choices = {'ohe': self._ohe, 'numeric': self._numeric_encoding_helper}
        train, test = choices[self.strategy](train, test)
        
        # add extra features
        self._train, self._test = self._extra_features(train, test)
        
    def _fit(self):
        
        # prepare X and y
        y = self._train['QuoteConversion_Flag'].values
        X = self._train.drop(['QuoteNumber', 'Original_Quote_Date',
                              'QuoteConversion_Flag'], axis=1).values

        # load model parameters
        plst, num_rounds = self.params[0], self.params[1]
        
        # train the model
        train_xgb = xgb.DMatrix(X, label=y)
        self._clf = xgb.train(plst, train_xgb, num_rounds)
        
    def _predict(self):
        
        # prepare X_test
        X_test = self._test.drop(['QuoteNumber', 'Original_Quote_Date'], 
                                 axis=1).values
        
        # predict for test
        test_xgb = xgb.DMatrix(X_test)
        y_pred = self._clf.predict(test_xgb)
        
        # save to file
        test_id = self._test['QuoteNumber'].astype(int).values
        df_result = pd.DataFrame({'QuoteNumber': test_id, 
                                  'QuoteConversion_Flag': y_pred})
        df_result.set_index('QuoteNumber', inplace=True)
        df_result.to_csv(self.folder+'Model{}.csv'.format(self.model_ind))
        
    def build(self, train, test):
        
        self._pre_processing(train, test)
        self._fit()
        self._predict()


def ensemble_results(y_list, w):
	
    # Calculate ensemble results given predictions and weights
    combined = np.empty([y_list[0].shape[0], len(y_list)])
    for col, y in enumerate(y_list):
        combined[:, col] = y

    return np.average(combined, axis=1, weights=w)
    
    
def main():
    
    # read in raw data
    df_train = pd.read_csv(FOLDER+FILE1, parse_dates=[1])
    df_test = pd.read_csv(FOLDER+FILE2, parse_dates=[1])
    df_train_numeric = pd.read_csv(FOLDER+FILE3, parse_dates=[1])
    df_test_numeric = pd.read_csv(FOLDER+FILE4, parse_dates=[1])
    
    # train and predict for the first model
    params = {'objective': 'binary:logistic',
              'eta': 0.0075,
              'subsample': 0.9,
              'max_depth': 9,
              'min_child_weight': 1,
              'colsample_bytree': 0.7,
              'gamma': 5,
              'silent': 1
              }
    num_rounds = 7500
    hm1 = HomesiteModel(1, 'ohe', [list(params.items()), num_rounds])
    hm1.build(df_train, df_test)
    print "model 1 is completed!"
    
    # train and predict for the second model
    params = {'objective': 'binary:logistic',
              'eta': 0.005,
              'subsample': 0.9,
              'max_depth': 12,
              'min_child_weight': 1,
              'colsample_bytree': 0.7,
              'gamma': 5,
              'silent': 1
              }
    num_rounds = 10000
    hm2 = HomesiteModel(2, 'numeric', [list(params.items()), num_rounds])
    hm2.build(df_train_numeric, df_test_numeric)
    print "model 2 is completed!"
    
    # train and predict for the third model
    params = {'objective': 'binary:logistic',
              'eta': 0.005,
              'subsample': 0.9,
              'max_depth': 12,
              'min_child_weight': 1,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
    num_rounds = 10000
    hm3 = HomesiteModel(3, 'numeric', [list(params.items()), num_rounds])
    hm3.build(df_train_numeric, df_test_numeric)
    print "model 3 is completed!"
    
    # ensemble the results
    w = [1, 1, 1]
    y_list = []
    for model_idx in range(1, 4):
        df = pd.read_csv(FOLDER+'Model{}.csv'.format(model_idx))
        y_list.append(df['QuoteConversion_Flag'].values)
    y_pred = ensemble_results(y_list, w)
    print "\nEnsemble Completed!\n"
    
    # save to the submission file
    test_id = df_test['QuoteNumber'].astype(int).values
    df_submission = pd.DataFrame({'QuoteNumber': test_id, 
                                  'QuoteConversion_Flag': y_pred})
    df_submission.set_index('QuoteNumber', inplace=True)
    df_submission.to_csv(FOLDER+'submission.csv')
    
    
if __name__ == '__main__':
    
    main()
