from os import join
import pandas as pd
import numpy as np

from sklearn.decomposition import SparseCoder
from sklearn.decomposition import sparse_encode

from const import REDD_DIR, TRAIN_END


def main():
    house_id = 1

    house_data = pd.read_csv(os.path.join(REDD_DIR, 'building_{0}.csv'.format(house_id)))
    house_data = house_data.set_index('time')

    train_data = house_data[:TRAIN_END]
    dev_data = house_data[TRAIN_END:]


    X_train, y_train = train_data.drop('main', axis=1), train_data.main
    X_dev, y_dev = dev_data.drop('main', axis=1), dev_data.main

    sc = SparseCoding()
    sc.fit(X_train, y_train)


class SparseCoding(object):

    def __init__(self):
        pass

    def fit(self, X, y):
        pass





if __name__ == '__main__':
    main()
