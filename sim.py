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


    def __init__(self, curr_mid, significance_level=0.1, max_inventory=10.):
        self.order_history = []
        self.quote_history = []
        self.broker_mid = []
        self.cashflow_history = []
        self.inventory = 0.
        self.curr_mid_stats = Broker.OrderStats()
        self.curr_mid_bounds = [curr_mid - 1., curr_mid + 1.]  # min and max bounds
        self.significance_level = significance_level
        self.MAX_INVENTORY = max_inventory
        self.BID_ASK_SPREAD = 1./8

    def quote(self, client_callback):
        mid = (self.curr_mid_bounds[0] + self.curr_mid_bounds[1]) / 2.
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
    broker = Broker(98.5, significance_level=0.5)
    buyer = Client('buyer', 98., 2.)
    seller = Client('seller', 102., 2.)
    buyer_proportion = 0.5

    real_mean_history = []
    for i in range(10000):
        #dW = np.random.normal(0, 1e-3)
        dW = 0.
        buyer.mean += dW
        seller.mean += dW
        real_mean_history.append((buyer.mean + seller.mean) / 2.)
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
    plt.plot(cash+inventory_mtm)

    plt.figure()
    plt.plot(quote_history, label='quote')
    plt.plot(broker_mid, label='mid')
    plt.plot(real_mean_history, label='actual mid')
    plt.legend()

    plt.figure()
    plt.plot(inventory_history)
    plt.show()

    from LogisticMixture import EM
    pb, ps = EM(quote_history[-1000:], order_history[-1000:], 1./8, 0.5,
                beta_b=1., beta_s=-1., offset_b=98., offset_s=102.)
