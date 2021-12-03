import backtrader as bt
import datetime


class CommisionScheme(bt.CommInfoBase):
    params = (
        ('stocklike', False),  # Futures
        ('commtype', bt.CommInfoBase.COMM_PERC),  # Apply % Commission
        ('percabs', True)  # pass perc as 0.xx
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * (self.p.commission)



class Strategy(bt.Strategy):

    def __init__(self):
        d = self.datas[0]
        d.target = list(d.target)[0]
        d.target = {
            datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").replace(microsecond=0): allocation
            for date, allocation in d.target.items()
        }
    
    def log(self, txt, dt=None):
        """ Logging function fot this strategy"""
        dt = dt or self.data.datetime
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
                    round(trade.pnl, 4),
                    round(trade.pnlcomm, 3),
                )
            )

   

    def next(self):
        date = self.data.datetime.datetime()
        total_value = self.broker.get_value()

        d = self.datas[0]
        if date not in d.target:
            pass
        else:
            target_allocation = d.target.get(date)

            if target_allocation == 0:
                pass
            else:
                cur_pos_size = self.getposition(d).size 
                print(target_allocation)
                trade_units = target_allocation  / d.close[0] - cur_pos_size

                if trade_units < 0:
                    self.sell(d, size=abs(trade_units)) 

                elif trade_units > 0:
                    self.buy(d, size=abs(trade_units))

        print(total_value)