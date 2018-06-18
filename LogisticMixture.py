from numpy import exp
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

class LogisticFunc(object):

    def __init__(self, beta, offset, prior):
        """1/(1 + exp(beta*(x - offset)))"""
        self.beta = beta
        self.offset = offset
        self.prior = prior

    def __call__(self, x):
        return 1./(1 + exp(self.beta*(x - self.offset)))

    def __str__(self):
        return "(1 + exp(%.2f*(x %.2f)))**-1\nclass prior prob: %.2f" %(self.beta, -self.offset, self.prior)

    def __repr__(self):
        return self.__str__()

def EM(x, y, bidask, pb, beta_b, beta_s, offset_b, offset_s, max_iter=10):
    ps = 1. - pb
    num_buy = np.sum(y==-1)
    num_sell = np.sum(y==1)
    x_doneaway = x[y==0]
    xbuy_doneaway = x_doneaway + bidask/2.
    xsell_doneaway = x_doneaway - bidask/2.
    x_buy = np.concatenate((x[y==-1] + bidask/2., xbuy_doneaway))
    x_sell = np.concatenate((x[y==1] - bidask/2., xsell_doneaway))
    buy_weights = np.ones(len(x_buy))
    sell_weights = np.ones(len(x_sell))
    y_buy = [True]*num_buy + [False]*len(x_doneaway)
    y_sell = [True]*num_sell + [False]*len(x_doneaway)
    lr_buy = LogisticRegression(intercept_scaling=100., C=1000.)
    lr_buy.coef_ = np.array([[beta_b]])
    lr_buy.intercept_ = np.array([-beta_b * offset_b])
    lr_sell = LogisticRegression(intercept_scaling=100., C=1000.)
    lr_sell.coef_ = np.array([[beta_s]])
    lr_sell.intercept_ = np.array([-beta_s * offset_s])

    for i in xrange(max_iter):
        p_buydoneway = pb * lr_buy.predict_proba(xbuy_doneaway[:,np.newaxis])[:,0]
        p_selldoneaway = ps * lr_sell.predict_proba(xsell_doneaway[:,np.newaxis])[:,0]
        normalize_factor = (p_buydoneway + p_selldoneaway)
        buy_weights[-len(x_doneaway):] = p_buydoneway / normalize_factor
        sell_weights[-len(x_doneaway):] = p_selldoneaway / normalize_factor
        lr_buy.fit(x_buy[:, np.newaxis], y_buy, buy_weights)
        lr_sell.fit(x_sell[:, np.newaxis], y_sell, sell_weights)

        p_buydoneway = pb * lr_buy.predict_proba(xbuy_doneaway[:,np.newaxis])[:,0]
        p_selldoneaway = ps * lr_sell.predict_proba(xsell_doneaway[:,np.newaxis])[:,0]
        expected_num_buyer = num_buy + np.sum(p_buydoneway/(p_buydoneway + p_selldoneaway))
        pb = expected_num_buyer / len(x)
        ps = 1 - pb

        print('step %d' %i)
        print('prob buyer: %f' %pb)
        print('buyer mid: %f' %(-lr_buy.intercept_/lr_buy.coef_)[0])
        print('seller mid: %f' %(-lr_sell.intercept_/lr_sell.coef_)[0])
        print('\n')

        # print lr_buy.predict_proba(90)[0,1], lr_buy.predict_proba(110)[0,1]
        # print lr_sell.predict_proba(90)[0,1], lr_sell.predict_proba(110)[0,1]
        # q_buy = lr_buy.predict_proba(xbuy_doneaway[:,np.newaxis])[:,1]
        # q_sell = lr_sell.predict_proba(xsell_doneaway[:,np.newaxis])[:,1]
        # pb_iter_f = lambda pb_prior: num_buy / (num_sell/(1-pb_prior) -
        #                                         np.sum((q_buy-q_sell) / (pb_prior*(q_buy-q_sell) + q_sell)))
        # for j in xrange(max_iter):
        #     pb_prev = pb
        #     pb = pb_iter_f(pb_prev)
        #     if np.abs(pb_prev - pb) < 0.001:
        #         break
        if (abs(offset_b + lr_buy.intercept_[0]/lr_buy.coef_[0,0]) < 0.01 and
            abs(offset_s + lr_sell.intercept_[0]/lr_sell.coef_[0,0]) < 0.01 ):
            print 'terminate iteration at step', i
            break
        else:
            offset_b = -lr_buy.intercept_[0]/lr_buy.coef_[0,0]
            offset_s = -lr_sell.intercept_[0]/lr_sell.coef_[0,0]


    logit_b = LogisticFunc(-lr_buy.coef_[0,0], -lr_buy.intercept_[0]/lr_buy.coef_[0,0], pb)
    logit_s = LogisticFunc(-lr_sell.coef_[0,0], -lr_sell.intercept_[0]/lr_sell.coef_[0,0], ps)
    return logit_b, logit_s

if __name__ == '__main__':
    N = int(1e4)
    x = np.zeros(N)
    y = np.zeros(N)

    from sim import Client
    from scipy.stats import norm
    buyer = Client('buyer', 90., 10.)
    seller = Client('seller', 110., 10.)

    for i in xrange(N):
        quote = np.random.normal(100., 20.)
        if np.random.uniform(0., 1.,) < 0.4:
            order = buyer(quote, quote)
        else:
            order = seller(quote, quote)
        y[i] = order
        x[i] = quote

    b, s = EM(x, y, 0, pb=0.5, beta_b=0.1, beta_s=0.1, offset_b=80., offset_s=105., max_iter=100)

    px_grid = np.linspace(50, 150, 200)
    buyer_cdf = 1-norm.cdf(px_grid, loc=90., scale=10.)
    seller_cdf = norm.cdf(px_grid, loc=110., scale=10.)
    plt.figure()
    plt.plot(px_grid, buyer_cdf * 0.4, 'b-', label='buyer')
    plt.plot(px_grid, seller_cdf * 0.6, 'r-', label='seller')
    plt.plot(px_grid, b(px_grid)*b.prior, 'b--', label='est. buyer')
    plt.plot(px_grid, s(px_grid)*s.prior, 'r--', label='est. buyer')
    plt.legend()

    import pandas as pd
    summary = pd.DataFrame()
    summary['buyer'] = buyer_cdf
    summary['seller'] = seller_cdf
    summary['est. buyer'] = b(px_grid)*b.prior
    summary['est. seller'] = s(px_grid)*s.prior
    summary.index = px_grid
    summary.index.name = 'Price'
    summary.to_csv('result_N=1e4.csv')
