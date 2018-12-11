from const import REDD_DIR, TRAIN_END


class RegressionModeler(object):

    def __init__(self, house_id, AR_terms):
        self.house_id = house_id
        self.AR_terms = AR_terms
        self.X_train = None
        self.X_test = None
        self.train_targets = None
        self.test_targets = None
        self.apps = None

    def prepare_train_test_sets(self):

        house_data = pd.read_csv(os.path.join(REDD_DIR, 'building_{0}.csv'.format(self.house_id)))
        house_data = house_data.set_index(pd.DatetimeIndex(house_data['time'])).drop('time', axis=1)

        apps = house_data.columns.values
        apps = apps[apps != 'Main']

        train_data = house_data[:TRAIN_END]
        test_data = house_data[TRAIN_END:]

        # construct X_train predictor matrix using autoregressive terms
        ar_list = []
        for i in range(self.AR_terms + 1):
            ar_list.append(train_data.Main.shift(i))

        X_train = pd.concat(ar_list, axis=1)
        X_train.columns = ['Main'] + ['AR{0}'.format(x) for x in range(1, self.AR_terms+1)]
        X_train = X_train[self.AR_terms:]

        # construct X_test predictor matrix using autoregressive terms
        ar_list = []
        for i in range(self.AR_terms + 1):
            ar_list.append(test_data.Main.shift(i))

        X_test = pd.concat(ar_list, axis=1)
        X_test.columns = ['Main'] + ['AR{0}'.format(x) for x in range(1, self.AR_terms+1)]
        X_test = X_test[self.AR_terms:]


        # construct target variables. Because of autoregression 'cost', must throw
        # out self.AR_terms rows of the data
        train_targets = {}
        test_targets = {}
        for item in apps:
            train_targets[item] = train_data[item][self.AR_terms:]
            test_targets[item] = test_data[item][self.AR_terms:]

        self.X_train = X_train
        self.X_test = X_test
        self.train_targets = train_targets
        self.test_targets = test_targets
        self.apps = apps

    def fit_model(self, model):

        ### Prediction ###
        app_scores = []
        for target_app in self.apps:

            y_train = self.train_targets[target_app].values
            y_test = self.test_targets[target_app].values

            model.fit(self.X_train, y_train)

            preds = model.predict(self.X_test)
            app_scores.append(rmse(preds, y_test))

        return app_scores

    def fit_multitask_model(self, model):

        y_train = np.hstack([train_targets[app].values.reshape(-1,1) for app in apps])
        y_test = np.hstack([test_targets[app].values.reshape(-1,1) for app in apps])

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        app_scores = []
        for i in range(len(self.apps)):
            app_scores.append(rmse(preds[:,i], y_test[:,i]))

        return app_scores



def rmse(pred, target):
    return np.sqrt(np.mean((pred - target)**2))



def main():
    house_id = 1
    AR_terms = 48

    rmd = RegressionModeler(house_id, AR_terms)
    rmd.prepare_train_test_sets()

    model = LinearRegression()
    print(rmd.fit_model(model))

    model = ElasticNetCV()
    print(rmd.fit_model(model))

    model = RandomForestRegressor()
    print(rmd.fit_model(model))

    model = SVR()
    print(rmd.fit_model(model))

    model = MultiTaskElasticNetCV()
    print(rmd.fit_multitask_model(model))

if __name__ == '__main__':
    main()
