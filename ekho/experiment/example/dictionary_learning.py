import os
from os.path import join
import pandas as pd
import numpy as np

from sklearn.decomposition import SparseCoder
from sklearn.decomposition import sparse_encode
from sklearn.decomposition import DictionaryLearning

import librosa
import matplotlib.pyplot as plt

from const import REDD_DIR, TRAIN_END


class DisaggregationNet(object):

    def __init__(self, n, m, T):
        self.n = n
        self.m = m
        self.T = T

        self.steps = 500
        self.alpha = 0.1
        self.rp = 0.1
        self.epsilon = 0.01

    ################################################################
    def _initialization(self):
        a = np.random.random((self.n,self.m))
        b = np.random.random((self.T,self.n))
        b /= sum(b)
        return a,b

    #################################################################
    def pre_training(self,x):
        A_list,B_list = self.NonNegativeSparseCoding(x)
        return A_list,B_list

    ################################################################
    @staticmethod
    def _pos_constraint(a):
        indices = np.where(a < 0.0)
        a[indices] = 0.0
        return a

    ##################################################################
    def predict(self,A,B):
        print(A.shape)
        print(B.shape)

        return B.dot(A)


        # x = map(lambda x,y: x.dot(y),B,A)
        # return x

    ##################################################################
    def F(self,x,B,x_train=None,A=None,rp_tep=False,rp_gl=False):
        B = np.asarray(B)
        A = np.asarray(A)
        coder = SparseCoder(dictionary=B.T, transform_alpha=self.rp,transform_algorithm='lasso_cd')

        comps, acts = librosa.decompose.decompose(x,transformer=coder)
        acts = self._pos_constraint(acts)
        return acts

    ###############################################################
    def NonNegativeSparseCoding(self,appliances):
    # ’’’
    # Method as in NNSC from nonnegative sparse coding finland.
    # from P.Hoyer
    # TODO : (ericle@kth.se)
    # ’’’
        epsilon = 0.01
        A_list = []
        B_list = []
        for app in appliances:
            x = appliances[app].reshape(-1, 1)
            A,B = self._initialization()
            Ap = A
            Bp = B
            Ap1 = Ap
            Bp1 = Bp
            t = 0
            change = 1
            while t <= self.steps and self.epsilon <= change:
                # 2a
                Bp = Bp - self.alpha*np.dot((np.dot(Bp,Ap) - x),Ap.T)
                # 2b
                Bp = self._pos_constraint(Bp)
                # 2c
                Bp /= sum(Bp)
                # element wise division
                dot2 = np.divide(np.dot(Bp.T,x),(np.dot(np.dot(Bp.T,Bp),Ap) + self.rp))
                # 2d
                Ap = np.multiply(Ap,dot2)

                change = np.linalg.norm(Ap - Ap1)
                Ap1 = Ap
                Bp1 = Bp
                t += 1

            print("Completed optimization for {}".format(app))
            A_list.append(Ap)
            B_list.append(Bp)

        return A_list,B_list
    ################################################################
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

    ########################

def visualize_piechart(preds, x, app_matrix):

    truth_label = app_matrix.keys()
    truth_val = x.sum(axis=0)
    pred_val = preds.sum(axis=1)

    pie, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    patches, txt = ax1.pie(truth_val, autopct=None)
    ax1.axis('equal')
    ax1.set_title('True usage')

    patches, txt = ax2.pie(pred_val, autopct=None)
    ax2.axis('equal')
    ax2.set_title('Predicted usage')

    plt.legend(patches, truth_label, bbox_to_anchor=(1, 0.2), ncol=4)


    #draw circle
    centre_circle = plt.Circle((0,0),0.6,fc='white')
    ax1.add_artist(centre_circle)
    centre_circle = plt.Circle((0,0),0.6,fc='white')
    ax2.add_artist(centre_circle)


    pie.set_size_inches(4.5, 8)
    pie.savefig('pie1.png', dpi=300)
    # plt.show()


def visualize_predictions(preds, X, app_matrix, train_data, house_id=1):
    print(preds.shape)
    print(X.shape)

    fig, ax = plt.subplots(len(app_matrix) + 1, 1, sharex=True)

    ma = ax[0].plot_date(x=train_data.index.values, y=X.sum(axis=1),
                            color='k', linewidth=1, fmt='-')

    ax[0].set_ylabel('Total energy')
    ax[0].set_title('Building {0}'.format(house_id))

    for i, app in enumerate(app_matrix.keys()):
        tr, = ax[i+1].plot_date(x=train_data.index.values, y=X[:,i].T,
                                color='k', label='Truth', linewidth=2, fmt='-')
                                
        pr, = ax[i+1].plot_date(x=train_data.index.values, y=preds[:,i].T,
                                color='r', label='Predicted', linewidth=1, fmt='-')
        ax[i+1].set_ylabel(app)


    for j in range(len(app_matrix) + 1):
        ax[j].spines['top'].set_visible(False)
        ax[j].spines['right'].set_visible(False)

    ax[0].legend(handles=[tr,pr])

    fig.align_ylabels(ax)
    plt.show()
    return fig


def train(app_matrix, app_test, net, train_data):
    A, B = net.pre_training(app_matrix)

    print(len(A))
    print(len(B))
    print(A[0].shape)
    print(B[0].shape)

    B_cat = net.DiscriminativeDisaggregation(app_matrix, B, A)
    print(B_cat.shape)

    x = np.array([app_matrix[app] for app in app_matrix])
    acts = net.F(x.T, B_cat)
    preds = net.predict(acts, B_cat)

    # visualize_piechart(preds.T, x.T, app_matrix)
    visualize_predictions(preds, x.T, app_matrix, train_data)



house_id = 1
path = os.path.join(REDD_DIR, 'building_{0}.csv'.format(house_id))

house_data = pd.read_csv(path)
house_data = house_data.set_index(pd.DatetimeIndex(house_data['time'])).drop('time', axis=1)

apps = house_data.columns.values
apps = apps[apps != 'Main']
train_data = house_data[:TRAIN_END]
dev_data = house_data[TRAIN_END:]

net = DisaggregationNet(n=1, m=6, T=len(train_data))
app_matrix = { app : train_data[app].values for app in apps }
app_test = { app : dev_data[app].values for app in apps }

train(app_matrix, app_test, net, train_data)
