import pandas as pd
from tqsdk import TqApi, TqAuth
from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)

from vnpy.trader.constant import Interval, Direction, Offset, Exchange
import numpy as np
from scipy import stats


class MarketMicrostructureStrategy(CtaTemplate):
    """市场微观结构策略"""

    author = "Strategy Developer"

    # 参数
    volume_ratio_threshold = 2.0  # 成交量比率阈值
    price_impact_window = 20  # 价格影响窗口
    order_imbalance_window = 30  # 订单不平衡窗口
    volatility_lookback = 50  # 波动率回看期
    correlation_period = 100  # 相关性计算周期
    trade_size = 1  # 交易手数

    # 变量
    volume_ratio = 0
    price_impact = 0
    order_imbalance = 0
    realized_volatility = 0
    market_correlation = 0

    quant_user =""
    quant_password=""

    parameters = [
        "quant_user",
        "quant_password",
        "volume_ratio_threshold",
        "price_impact_window",
        "order_imbalance_window",
        "volatility_lookback",
        "correlation_period",
        "trade_size"
    ]

    variables = [
        "volume_ratio",
        "price_impact",
        "order_imbalance",
        "realized_volatility",
        "market_correlation"
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.setting = setting
        self.vt_symbol = vt_symbol
        vt_symbol_code = self.vt_symbol.split(".")

        self.symbol = f"{vt_symbol_code[1]}.{vt_symbol_code[0]}"
        self.bg = BarGenerator(self.on_bar, 7, self.on_1min_bar, Interval.TICK)
        self.am = ArrayManager(size=20)

        # 存储原始数据
        self.tick_prices = []
        self.tick_volumes = []
        self.bid_ask_spreads = []
        self.trade_imbalances = []

        # 存储滚动数据
        self.rolling_volumes = []
        self.rolling_high_low = []

        # 相关品种（可根据实际调整）
        self.correlated_symbols = ["IF888", "IC888"]  # 股指期货示例

    def on_init(self):
        """策略初始化"""
        self.write_log("策略初始化")

    def on_start(self):
        """策略启动"""
        self.write_log("策略启动")
        self.trading = True
        self.put_event()
        # 创建api实例，设置web_gui=True生成图形化界面
        api = TqApi(web_gui=True, auth=TqAuth(self.setting["quant_user"], self.setting["quant_password"]))
        # 订阅 ni2010 合约的10秒线
        klines = api.get_tick_serial(self.symbol , 10)
        while True:
            symbol = klines.iloc[-1].symbol
            symbol_code = symbol.split(".")
            datetime = klines.iloc[-1].datetime
            ask_price1 = klines.iloc[-1].ask_price1
            ask_volume1 = klines.iloc[-1].ask_volume1
            bid_price1 = klines.iloc[-1].bid_price1
            bid_volume1 = klines.iloc[-1].bid_volume1
            last_price = klines.iloc[-1].last_price
            volume = klines.iloc[-1].volume
            highest = klines.iloc[-1].highest
            lowest = klines.iloc[-1].lowest
            open_interest = klines.iloc[-1].open_interest

            # pandas的纳秒时间戳
            numpy_ts = np.float64(klines.iloc[-1].datetime)
            # 转换为pandas.Timestamp，再转datetime
            pd_timestamp = pd.Timestamp(numpy_ts)
            datetime = pd_timestamp.to_pydatetime()

            tick = TickData(
                gateway_name='ctp',
                symbol=symbol_code[1],
                exchange=Exchange(str(symbol_code[0])),
                datetime=datetime,
                name=symbol,
                last_price=float(last_price),
                volume=float(volume),
                bid_price_1=float(bid_price1),
                bid_volume_1=float(bid_volume1),
                ask_price_1=float(ask_price1),
                ask_volume_1=float(ask_volume1),
                open_interest=float(open_interest),
                high_price=float(highest),
                low_price=float(lowest),
            )
            self.write_log(f" MarketMicrostructureStrategy Tick s 更新  {tick}")
            self.update_tick(tick)
            # 通过wait_update刷新数据
            api.wait_update()


    def on_stop(self):
        """策略停止"""
        self.write_log("策略停止")
        self.put_event()

    def update_tick(self, tick: TickData):
        self.bg.update_tick(tick)

        # 收集tick数据
        if tick.last_price > 0:
            self.tick_prices.append(tick.last_price)
            self.tick_volumes.append(tick.volume)

            # 计算买卖价差
            if tick.bid_price_1 > 0 and tick.ask_price_1 > 0:
                spread = tick.ask_price_1 - tick.bid_price_1
                self.bid_ask_spreads.append(spread)

            # 计算交易不平衡
            if tick.bid_volume_1 > 0 and tick.ask_volume_1 > 0:
                imbalance = (tick.bid_volume_1 - tick.ask_volume_1) / (tick.bid_volume_1 + tick.ask_volume_1)
                self.trade_imbalances.append(imbalance)

            # 保持数据长度
            if len(self.tick_prices) > 1000:
                self.tick_prices.pop(0)
                self.tick_volumes.pop(0)
            if len(self.bid_ask_spreads) > 1000:
                self.bid_ask_spreads.pop(0)
            if len(self.trade_imbalances) > 1000:
                self.trade_imbalances.pop(0)


    def on_tick(self, tick: TickData):
        """Tick更新"""
        self.write_log(f" MarketMicrostructureStrategy Tick更新  {tick}")

    def on_bar(self, bar: BarData):
        self.write_log(f" MarketMicrostructureStrategy on_bar  {bar}")
        self.bg.update_bar(bar)

        self.put_event()


    def on_1min_bar(self, bar: BarData):
        """1分钟K线"""
        self.write_log(f" MarketMicrostructureStrategy 1分钟K线更新  {bar}")
        self.am.update_bar(bar)

        # 1. 成交量异常检测
        self.calculate_volume_anomaly(bar)

        # 2. 价格影响分析
        self.calculate_price_impact()

        # 3. 订单流不平衡
        self.calculate_order_imbalance()

        # 4. 已实现波动率
        self.calculate_realized_volatility()

        # 生成交易信号
        self.generate_trading_signals(bar)

        self.put_event()

    def calculate_volume_anomaly(self, bar: BarData):
        """计算成交量异常"""
        if len(self.rolling_volumes) >= 4:
            recent_volume = np.mean(self.rolling_volumes[-1:])  # 最近5分钟平均
            historical_volume = np.mean(self.rolling_volumes[-3:])  # 历史20分钟平均

            if historical_volume > 0:
                self.volume_ratio = recent_volume / historical_volume
            else:
                self.volume_ratio = 1

        self.rolling_volumes.append(bar.volume)
        if len(self.rolling_volumes) > 100:
            self.rolling_volumes.pop(0)

    def calculate_price_impact(self):
        """计算价格冲击"""
        if len(self.tick_prices) >= self.price_impact_window:
            # 使用分位数回归计算价格冲击
            prices = np.array(self.tick_prices[-self.price_impact_window:])
            volumes = np.array(self.tick_volumes[-self.price_impact_window:])

            if len(prices) > 10 and np.std(volumes) > 0:
                # 计算价格变化与成交量的关系
                price_changes = np.diff(prices)
                log_volumes = np.log(volumes[1:] + 1)

                # 简单线性回归计算冲击系数
                if len(price_changes) == len(log_volumes):
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            log_volumes, price_changes
                        )
                        self.price_impact = slope * 100  # 转换为百分比
                    except:
                        self.price_impact = 0

    def calculate_order_imbalance(self):
        """计算订单不平衡"""
        if len(self.trade_imbalances) >= self.order_imbalance_window:
            recent_imbalance = np.mean(self.trade_imbalances[-10:])
            historical_imbalance = np.mean(self.trade_imbalances[-self.order_imbalance_window:])

            self.order_imbalance = recent_imbalance - historical_imbalance

    def calculate_realized_volatility(self):
        """计算已实现波动率"""
        if len(self.tick_prices) >= self.volatility_lookback:
            returns = np.diff(np.log(self.tick_prices[-self.volatility_lookback:]))
            self.realized_volatility = np.std(returns) * np.sqrt(252 * 240)  # 年化波动率



    def generate_trading_signals(self, bar: BarData):
        """生成交易信号"""

        # 多条件信号生成
        buy_signal_score = 0
        sell_signal_score = 0

        # 1. 成交量激增信号
        if self.volume_ratio > self.volume_ratio_threshold:
            buy_signal_score += 1 if bar.close_price > bar.open_price else 0
            sell_signal_score += 1 if bar.close_price < bar.open_price else 0

        # 2. 价格冲击信号
        if self.price_impact > 0.5:  # 强正冲击
            buy_signal_score += 1
        elif self.price_impact < -0.5:  # 强负冲击
            sell_signal_score += 1

        # 3. 订单流信号
        if self.order_imbalance > 0.3:  # 买方占优
            buy_signal_score += 1
        elif self.order_imbalance < -0.3:  # 卖方占优
            sell_signal_score += 1

        # 4. 波动率过滤
        volatility_ok = 0.05 < self.realized_volatility < 0.5

        # 执行交易逻辑
        if buy_signal_score >= 1 and volatility_ok and not self.pos:
            self.write_log(f" MarketMicrostructureStrategy buy  {bar.close_price}")
            self.buy_action(bar.close_price, self.trade_size)

        elif sell_signal_score >= 1 and volatility_ok and not self.pos:
            self.write_log(f" MarketMicrostructureStrategy short  {bar.close_price}")
            self.short_action(bar.close_price, self.trade_size)

        elif self.pos > 0 and sell_signal_score >= 3:
            self.write_log(f" MarketMicrostructureStrategy sell  {bar.close_price}")
            self.sell(bar.close_price, abs(self.pos))

        elif self.pos < 0 and buy_signal_score >= 3:
            self.write_log(f" MarketMicrostructureStrategy cover  {bar.close_price}")
            self.cover(bar.close_price, abs(self.pos))

    def buy_action(self, price, volume):
        """开多仓"""
        self.cancel_all()

        order = self.buy(price, volume)




    def short_action(self, price, volume):
        """开空仓"""
        self.cancel_all()

        order = self.short(price, volume)




    def on_order(self, order: OrderData):
        """订单更新"""
        super().on_order(order)

    def on_trade(self, trade: TradeData):
        """成交更新"""
        super().on_trade(trade)

    def on_stop_order(self, stop_order: StopOrder):
        """停止单更新"""
        pass