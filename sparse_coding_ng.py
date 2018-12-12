import os
from os.path import join
import pandas as pd
import numpy as np

from sklearn.decomposition import SparseCoder
from sklearn.decomposition import sparse_encode
from sklearn.decomposition import DictionaryLearning

import matplotlib.pyplot as plt

from const import REDD_DIR, TRAIN_END

house_id = 1
path = os.path.join(REDD_DIR, 'building_{0}.csv'.format(house_id))


class SparseCoding(object):
    def __init__(self, n, transform_algorithm='lars'):
        self.n = n
        self.net = DictionaryLearning(n_components=n, alpha=0.8, max_iter=1000)
        self.net.set_params(transform_algorithm=transform_algorithm)


    def plot_B(self, B):
        plt.figure(figsize=(4.2, 4))
        for i, comp in enumerate(B[:self.n]):
            plt.subplot(10, 10, i + 1)
            plt.imshow(comp, cmap=plt.cm.gray_r,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())

        plt.suptitle('Dictionary learned from time series\n' +
                     'Train time %.1fs on %d patches' % (dt, len(data)),
                     fontsize=16)

        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    def _init(self):
        a = np.random.random((self.n,self.m))
        b = np.random.random((self.T,self.n))
        b /= sum(b)
        return a,b

    def init_weights(self, X_mat):
        B, A , recon = [], [], []
        for app in X_mat:
            data = X_mat[app].reshape(1, -1)
            B_i = self.net.fit(data).components_
            A_i = self.net.transform(data)
            X_hat = np.dot(A_i, B_i)

            B.append(B_i)
            A.append(A_i)
            recon.append(X_hat)

            print("MSE Error: ", np.mean((data - X_hat) ** 2))

        return A, B, recon


    def DiscriminativeDisaggregation(self, appliances, B, A):

        x = np.array([appliances[app] for app in appliances])
        x = x.T

        A_star = np.vstack(A)
        B_cat = np.hstack(B)
        change = 1
        t = 0

        print(A_star.shape)
        print(B_cat.shape)

        while t <= self.steps and self.epsilon <= change:
            B_cat_p = B_cat
            acts = self.F(x, B_cat, A=A_star)
            B_cat = (B_cat - self.alpha * ((x - B_cat.dot(acts)).dot(acts.T) - (x - B_cat.dot(A_star)).dot(A_star.T)))
            B_cat = self._pos_constraint(B_cat)
            B_cat /= sum(B_cat)

            change = np.linalg.norm(B_cat - B_cat_p)
            t += 1
            print("Change is {} and step is {} ".format(change, t))

        return B_cat

    def F(self,x,B,x_train=None,A=None,rp_tep=False,rp_gl=False):
        B = np.asarray(B)
        A = np.asarray(A)
        coder = SparseCoder(dictionary=B.T, transform_alpha=self.rp,transform_algorithm='lasso_cd')

        comps, acts = librosa.decompose.decompose(x,transformer=coder)
        acts = self._pos_constraint(acts)
        return acts

    def predict(self,A,B):
        print(A.shape)
        print(B.shape)

        return B.dot(A)


def train(app_matrix, net):
    A, B, recon = net.init_weights(app_matrix)
    dictionary = net.DiscriminativeDisaggregation(app_matrix, B, A)
    x = np.array([app_matrix[app] for app in app_matrix])
    acts = net.F(x.T, B_cat)
    preds = net.predict(acts, B_cat)

    visualize_piechart(preds, x, app_matrix)



house_data = pd.read_csv(path)
house_data = house_data.set_index(pd.DatetimeIndex(house_data['time'])).drop('time', axis=1)

apps = house_data.columns.values
apps = apps[apps != 'Main']
train_data = house_data[:TRAIN_END]
dev_data = house_data[TRAIN_END:]

net = SparseCoding(n = 10)
app_matrix = { app : train_data[app].values for app in apps }

train(app_matrix, net)
