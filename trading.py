import backtrader as bt
import datetime

class CommisionScheme(bt.CommInfoBase):
    params = (
    ('stocklike', False),  # Futures
    ('commtype', bt.CommInfoBase.COMM_PERC),  # Apply % Commission
    ('percabs', True),  # pass perc as 0.xx
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * self.p.commission

class TestStrategy(bt.Strategy):
    params = (
        ("threshold", 0.025),
    )

    def log(self, txt, dt=None):
        """ Logging function fot this strategy"""
        dt = dt or self.data.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        print("%s, %s" % (dt.date(), txt))

    def notify_order(self, order):
        """ Triggered upon changes to orders. """
        # Suppress notification if it is just a submitted order.
        if order.status == order.Submitted:
            return

        # Print out the date, security name, order number and status.
        type = "Buy" if order.isbuy() else "Sell"
        self.log(
            f"{order.data._name:<6} Order: {order.ref:3d} "
            f"Type: {type:<5}\tStatus"
            f" {order.getstatusname():<8} \t"
            f"Size: {order.created.size:9.4f} Price: {order.created.price:9.4f} "
            f"Position: {self.getposition(order.data).size:5.2f}"
        )
        if order.status == order.Margin:
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            self.log(
                f"{order.data._name:<6} {('BUY' if order.isbuy() else 'SELL'):<5} "
                # f"EXECUTED for: {dn} "
                f"Price: {order.executed.price:6.2f} "
                f"Cost: {order.executed.value:6.2f} "
                f"Comm: {order.executed.comm:4.2f} "
                f"Size: {order.created.size:9.4f} "
            )

    def notify_trade(self, trade):
        """Provides notification of closed trades."""
        if trade.isclosed:
            self.log(
                "{} Closed: PnL Gross {}, Net {},".format(
                    trade.data._name,
                    round(trade.pnl, 2),
                    round(trade.pnlcomm, 1),
                )
            )

    def __init__(self):
        for d in self.datas:
            d.target = {
                datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").date(): allocation
                for date, allocation in d.target.items()
            }

    '''
    3-label: 
    Trade with constatnt intial weight 
    1. If pred_0 = 1, (will rise in the coming interval), close until next -1 
    2. If pred_0 = 0, (will be stationary): pass
    3. If pred_0 = -1, close until next 1 
    '''

    '''
    5-label: 
    2: more wegiht, 1: less weight 
    1. If 2: 
        1.1. 2->2 
        1.2. 2->1 
        1.3. 2->0 
        1.4. 2->(-) Close 
    Similar 
    '''

    '''
    regression: 
    weight -proportional-> (the regression result - current price) 
    '''

    def next(self):
        date = self.data.datetime.date()
        track_trades = dict()
        total_value = self.broker.get_value() 


        for d in self.datas:
            if date not in d.target:
                if self.getposition(d):
                    self.close(d)
                continue
            
            ''''''''''''''''''''''''
            ' params: (For each crypto currency)'
            ' track_trades = {date:{units: units_to_trade, threshold: 0 or 1} }'
            ' target_allocation: target_weight (in %) '
            ' current_allocation: current position / portfolio_value '
            ''''''''''''''''''''''''

            
            track_trades[d] = dict()
            target_allocation = d.target[date] 
            ## ?? the position of self.datas[i]
            value = self.broker.get_value(datas=[d])
            current_allocation = value / total_value
            net_allocation = target_allocation - current_allocation
            units_to_trade = (
                (net_allocation) * total_value / d.close[0]
            )
            track_trades[d]["units"] = units_to_trade
            # Can check to make sure there is enough distance away from ideal to trade.
            track_trades[d]["threshold"] = abs(net_allocation) > self.p.threshold

        rebalance = False
        for values in track_trades.values():
            if values["threshold"]:
                rebalance = True

        if not rebalance:
            return

        # Sell shares first
        for d, value in track_trades.items():
            if value["units"] < 0:
                self.sell(d, size=value["units"])

        # Buy shares second
        for d, value in track_trades.items():
            if value["units"] > 0:
                self.buy(d, size=value["units"])

