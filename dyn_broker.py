import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_chisquare
import matplotlib.pyplot as plt
from collections import deque

class Broker(object):

    class OrderStats(object):

        def __init__(self):
            self.buy = 0
            self.sell = 0
            self.doneaway = 0
            self.ave_mid = 0

        def add_order(self, order, mid):
            self.ave_mid = (self.total_orders*self.ave_mid + mid) / (self.total_orders + 1)
            if order == 0:
                self.doneaway += 1
            elif order == -1:
                self.sell += 1
            else:
                self.buy += 1

        @property
        def total_orders(self):
            return self.buy + self.sell + self.doneaway

    def __init__(self, curr_mid, significance_level=0.1, max_inventory=10., bid_ask_spread=1./8):
        self.order_history = []
        self.quote_history = []
        self.broker_mid = []
        self.cashflow_history = []
        self.inventory = 0.
        self.curr_mid_stats = Broker.OrderStats()
        self.curr_mid_bounds = [curr_mid - 15., curr_mid + 15.]  # min and max bounds
        self.significance_level = significance_level
        self.MAX_INVENTORY = max_inventory
        self.BID_ASK_SPREAD = bid_ask_spread

    def quote(self, client_callback):
        if len(self.quote_history) > 100:
            mid = np.mean(self.quote_history[-100:])
        else:
            mid = (self.curr_mid_bounds[0] + self.curr_mid_bounds[1]) / 2.
        bidaskspread = self.curr_mid_bounds[1] - self.curr_mid_bounds[0]
        self.curr_mid_bounds[1], self.curr_mid_bounds[0] = mid + bidaskspread/2, mid - bidaskspread/2
        skew = (self.curr_mid_bounds[1] - self.curr_mid_bounds[0]) / 2. * np.tanh(-self.inventory / self.MAX_INVENTORY)
        client_mid = mid + skew
        order = client_callback(client_mid - self.BID_ASK_SPREAD/2., client_mid + self.BID_ASK_SPREAD/2.)
        self.quote_history.append(client_mid)
        self.broker_mid.append(mid)
        self.order_history.append(order)
        self.curr_mid_stats.add_order(order, client_mid)
        self.inventory += order
        self.cashflow_history.append(-client_mid*order + self.BID_ASK_SPREAD/2. * np.abs(order))

        pbuy = float(self.curr_mid_stats.buy) / self.curr_mid_stats.total_orders
        psell = float(self.curr_mid_stats.sell) / self.curr_mid_stats.total_orders
        pval = proportions_chisquare(
            np.array([self.curr_mid_stats.buy, self.curr_mid_stats.sell]),
            self.curr_mid_stats.total_orders,
            np.array([(pbuy+psell)/2, (pbuy+psell)/2]))[1]
        if pval < self.significance_level:
            ave_mid = self.curr_mid_stats.ave_mid
            if self.curr_mid_stats.buy < self.curr_mid_stats.sell:
                self.curr_mid_bounds[1] += (ave_mid - self.curr_mid_bounds[0]) / 2.
                self.curr_mid_bounds[0] += (ave_mid - self.curr_mid_bounds[0]) / 2.
            else:
                self.curr_mid_bounds[0] -= (self.curr_mid_bounds[1] - ave_mid) / 2.
                self.curr_mid_bounds[1] -= (self.curr_mid_bounds[1] - ave_mid) / 2.
            self.curr_mid_stats = Broker.OrderStats()
            print('ave mid: %f, bound %f %f'  %((ave_mid,) + tuple(self.curr_mid_bounds)))

class Client(object):

    def __init__(self, client_type, mean, std):
        assert(client_type in ('buyer', 'seller'))
        self.client_type = client_type
        self.mean = mean
        self.mean_history = []
        self.std = std

    def __call__(self, bid, ask):
        self.mean_history.append(self.mean)
        draw = np.random.normal(self.mean, self.std)
        if self.client_type == 'buyer':
            if ask < draw:
                return -1
        else:
            if bid > draw:
                return 1
        return 0


if __name__ == '__main__':
    from scipy.stats import norm
    from scipy.optimize import bisect
    np.random.seed(1)
    broker = Broker(95., significance_level=0.5, max_inventory=100., bid_ask_spread=2.)
    buyer = Client('buyer', 90., 10.)
    seller = Client('seller', 110., 10.)
    buyer_proportion = 0.3


    real_mean_history = []
    for i in range(200000):
        dW = np.random.normal(0, 5e-2)
        #dW = 0.
        buyer.mean += dW
        seller.mean += dW

        p_buyer_minus_seller = lambda px: ((1-norm.cdf(px, buyer.mean, 10))*buyer_proportion -
                                       norm.cdf(px, seller.mean, 10)*(1-buyer_proportion))
        actual_mid = bisect(p_buyer_minus_seller, 10, 200)
        real_mean_history.append(actual_mid)
        if np.random.uniform(0, 1) < buyer_proportion:
            broker.quote(buyer)
        else:
            broker.quote(seller)

    quote_history = np.array(broker.quote_history)
    order_history = np.array(broker.order_history)
    cashflow = np.array(broker.cashflow_history)
    broker_mid = np.array(broker.broker_mid)
    inventory_history = np.cumsum(order_history)
    inventory_mtm = inventory_history * quote_history
    cash = np.cumsum(cashflow)

    plt.figure()
    plt.plot(cash+inventory_mtm, label='cumulative PnL')
    plt.legend()

    plt.figure()
    plt.plot(quote_history, label='quote')
    #plt.plot(broker_mid, label='mid')
    plt.plot(real_mean_history, label='actual mid')
    plt.legend()

    plt.figure()
    plt.plot(inventory_history, label='inventory')
    plt.legend()
    plt.show()



    # from LogisticMixture import EM
    # pb, ps = EM(quote_history[-10000:], order_history[-10000:], 1./8, 0.5,
    #             beta_b=1., beta_s=-1., offset_b=98., offset_s=103.)
    # print pb
    # print ps
    #
    #
    #
    # px_grid = np.linspace(80, 120, 200)
    # buyer_cdf = 1-norm.cdf(px_grid, loc=95., scale=5.)
    # seller_cdf = norm.cdf(px_grid, loc=105., scale=5.)
    # plt.figure()
    # plt.plot(px_grid, buyer_cdf * 0.3, 'b-', label='buyer')
    # plt.plot(px_grid, seller_cdf * 0.7, 'r-', label='seller')
    # plt.plot(px_grid, pb(px_grid)*pb.prior, 'b--', label='est. buyer')
    # plt.plot(px_grid, ps(px_grid)*ps.prior, 'r--', label='est. buyer')
    # plt.plot(px_grid, ps(px_grid)*ps.prior - pb(px_grid)*pb.prior, 'r--', label='est. buyer')
    # plt.legend()
