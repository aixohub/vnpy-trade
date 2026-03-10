#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from vnpy.trader.constant import Direction, Offset, Interval


class FractalTrendStrategy(CtaTemplate):
    """基于分形的趋势突破策略"""

    author = "Chaos"
    vt_symbol: str = ''
    # 策略参数
    kline_period = 6  # K线周期（分钟）
    fractal_window = 2  # 分形窗口大小
    short_ma_period = 2  # 短期MA周期
    long_ma_period = 9  # 长期MA周期
    volume_period = 5  # 成交量平均周期
    atr_period = 10  # ATR计算周期
    atr_multiplier = 1.2  # 止损ATR乘数
    tp_ratio = 1.8  # 止盈比例
    position_size = 1  # 交易手数

    # 策略变量
    current_direction = 0  # 当前持仓方向：1=多头，-1=空头，0=空仓
    entry_price = 0.0  # 开仓价格
    stop_loss_price = 0.0  # 止损价格
    take_profit_price = 0.0  # 止盈价格
    last_bull_fractal_idx = 0  # 最近牛市分形索引
    last_bull_fractal_low = 0.0  # 最近牛市分形低点
    last_bull_fractal_high = 0.0  # 最近牛市分形高点
    last_bear_fractal_idx = 0  # 最近熊市分形索引
    last_bear_fractal_low = 0.0  # 最近熊市分形低点
    last_bear_fractal_high = 0.0  # 最近熊市分形高点

    short_ma = 0.0
    long_ma = 0.0
    current_atr = 0.0
    volume_avg = 0.0

    parameters = [
        "vt_symbol",
        "kline_period",
        "fractal_window",
        "short_ma_period",
        "long_ma_period",
        "volume_period",
        "atr_period",
        "atr_multiplier",
        "tp_ratio",
        "position_size"
    ]

    variables = [
        "current_direction",
        "entry_price",
        "stop_loss_price",
        "take_profit_price",
        "short_ma",
        "long_ma",
        "current_atr"
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """构造函数"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.vt_symbol: str = vt_symbol
        # K线生成器
        self.bg = BarGenerator(
            self.on_bar,
            window=self.kline_period,
            on_window_bar=self.on_window_bar,
            interval=Interval.TICK
        )

        # 数组管理器
        self.am = ArrayManager(size=12)

    def on_init(self):
        """策略初始化"""
        self.write_log("策略初始化")
        self.pos = 0

    def on_start(self):
        """策略启动"""
        self.write_log("策略启动")

    def on_stop(self):
        """策略停止"""
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """Tick数据推送"""
        self.write_log(f"FractalTrendStrategy on_tick == {tick.__str__()}")
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """1分钟Bar数据推送"""
        # self.write_log(f"FractalTrendStrategy on_bar ===== {bar}")
        self.bg.update_bar(bar)




    def on_window_bar(self, bar: BarData):
        """周期K线数据推送"""
        self.am.update_bar(bar)

        if not self.am.inited:
            return
        self.write_log(f"FractalTrendStrategy on_window_bar ===== {bar}")
        # 计算技术指标
        """计算技术指标"""
        # 计算移动平均线
        self.short_ma = self.am.sma(self.short_ma_period)
        self.long_ma = self.am.sma(self.long_ma_period)

        # 计算ATR
        self.current_atr = self.am.atr(self.atr_period)

        # 计算成交量平均值
        volume_array = self.am.volume[-self.volume_period:]
        self.volume_avg = volume_array.mean()

        # 识别分形
        self.identify_fractals()

        # 获取当前价格
        current_price = bar.close_price

        # 确定当前趋势
        uptrend = (self.short_ma - self.long_ma) > 2
        downtrend = True
        if abs(self.short_ma - self.long_ma) < 2:
            downtrend = False

        # 输出当前状态
        self.write_log("=" * 50)
        self.write_log(f"时间: {bar.datetime}")
        self.write_log(f"价格: {current_price:.2f}, ATR: {self.current_atr:.2f}")
        self.write_log(f"短期MA: {self.short_ma:.2f}, 长期MA: {self.long_ma:.2f}")
        self.write_log(f"趋势: {'上升' if uptrend else '下降' if downtrend else '盘整'}")
        self.write_log(f"持仓方向: {self.current_direction}")

        # 执行交易逻辑
        # self.execute_trading_logic(current_price, uptrend, downtrend)
        """执行交易逻辑"""
        current_pos = self.pos

        # 空仓状态 - 寻找开仓机会
        if current_pos == 0:
            # 多头入场信号
            if (self.last_bull_fractal_idx > 0 and
                    self.last_bull_fractal_idx < self.am.count - 1 and
                    current_price > self.last_bull_fractal_high and
                    uptrend):

                self.current_direction = 1
                self.entry_price = current_price

                # 设置止损和止盈
                self.stop_loss_price = (
                        self.last_bull_fractal_low -
                        self.current_atr * self.atr_multiplier
                )
                self.take_profit_price = (
                        self.entry_price +
                        (self.entry_price - self.stop_loss_price) * self.tp_ratio
                )

                self.buy(current_price, self.position_size)
                self.write_log(
                    f"多头开仓: 价格={self.entry_price:.2f}, "
                    f"止损={self.stop_loss_price:.2f}, "
                    f"止盈={self.take_profit_price:.2f}"
                )

            # 空头入场信号
            elif (self.last_bear_fractal_idx > 0 and
                  self.last_bear_fractal_idx < self.am.count - 1 and
                  current_price < self.last_bear_fractal_low and
                  downtrend):

                self.current_direction = -1
                self.entry_price = current_price

                # 设置止损和止盈
                self.stop_loss_price = (
                        self.last_bear_fractal_high +
                        self.current_atr * self.atr_multiplier
                )
                self.take_profit_price = (
                        self.entry_price -
                        (self.stop_loss_price - self.entry_price) * self.tp_ratio
                )

                self.short(current_price, self.position_size)
                self.write_log(
                    f"空头开仓: 价格={self.entry_price:.2f}, "
                    f"止损={self.stop_loss_price:.2f}, "
                    f"止盈={self.take_profit_price:.2f}"
                )

        # 多头持仓 - 检查平仓条件
        elif current_pos > 0:
            should_close = False
            close_reason = ""

            # 1. 止损条件
            if current_price <= self.stop_loss_price:
                should_close = True
                close_reason = "止损"

            # 2. 止盈条件
            elif current_price >= self.take_profit_price:
                should_close = True
                close_reason = "止盈"

            # 3. 价格跌破短期MA
            elif current_price < self.short_ma:
                should_close = True
                close_reason = "MA反转"

            # 4. 出现新的熊市分形且价格跌破该分形的低点
            elif (self.last_bear_fractal_idx > self.am.count - 10 and
                  current_price < self.last_bear_fractal_low):
                should_close = True
                close_reason = "分形反转"

            if should_close:
                self.sell(current_price, abs(current_pos))
                self.current_direction = 0
                self.write_log(
                    f"多头{close_reason}平仓: 价格={current_price:.2f}, "
                    f"entry_price= {self.entry_price:.2f}%"
                )

        # 空头持仓 - 检查平仓条件
        elif current_pos < 0:
            should_close = False
            close_reason = ""

            # 1. 止损条件
            if current_price >= self.stop_loss_price:
                should_close = True
                close_reason = "止损"

            # 2. 止盈条件
            elif current_price <= self.take_profit_price:
                should_close = True
                close_reason = "止盈"

            # 3. 价格突破短期MA
            elif current_price > self.short_ma:
                should_close = True
                close_reason = "MA反转"

            # 4. 出现新的牛市分形且价格突破该分形的高点
            elif (self.last_bull_fractal_idx > self.am.count - 10 and
                  current_price > self.last_bull_fractal_high):
                should_close = True
                close_reason = "分形反转"

            if should_close:
                self.cover(current_price, abs(current_pos))
                self.current_direction = 0
                self.write_log(
                    f"空头{close_reason}平仓: 价格={current_price:.2f}, "
                    f"entry_price= {self.entry_price:.2f}%"
                )
        # 更新图形界面
        self.put_event()

    def calculate_indicators(self):
        """计算技术指标"""
        # 计算移动平均线
        self.short_ma = self.am.sma(self.short_ma_period)
        self.long_ma = self.am.sma(self.long_ma_period)

        # 计算ATR
        self.current_atr = self.am.atr(self.atr_period)

        # 计算成交量平均值
        volume_array = self.am.volume[-self.volume_period:]
        self.volume_avg = volume_array.mean()


    def identify_fractals(self):
        """识别牛市和熊市分形"""
        if self.am.count < 2 * self.fractal_window + 1:
            return

        # 获取价格数组
        high_array = self.am.high_array
        low_array = self.am.low_array

        # 检查范围（只检查最近的K线）
        start_idx = max(self.fractal_window, len(high_array) - 20)
        end_idx = len(high_array) - self.fractal_window

        for i in range(start_idx, end_idx):
            # 识别牛市分形（低点低于两侧）
            is_bull_fractal = True
            for j in range(1, self.fractal_window + 1):
                if low_array[i] >= low_array[i - j] or low_array[i] >= low_array[i + j]:
                    is_bull_fractal = False
                    break

            if is_bull_fractal:
                self.last_bull_fractal_idx = i
                self.last_bull_fractal_low = low_array[i]
                self.last_bull_fractal_high = high_array[i]
                self.write_log(
                    f"发现牛市分形: 索引={i}, "
                    f"低点={self.last_bull_fractal_low:.2f}, "
                    f"高点={self.last_bull_fractal_high:.2f}"
                )

            # 识别熊市分形（高点高于两侧）
            is_bear_fractal = True
            for j in range(1, self.fractal_window + 1):
                if high_array[i] <= high_array[i - j] or high_array[i] <= high_array[i + j]:
                    is_bear_fractal = False
                    break

            if is_bear_fractal:
                self.last_bear_fractal_idx = i
                self.last_bear_fractal_low = low_array[i]
                self.last_bear_fractal_high = high_array[i]
                self.write_log(
                    f"发现熊市分形: 索引={i}, "
                    f"低点={self.last_bear_fractal_low:.2f}, "
                    f"高点={self.last_bear_fractal_high:.2f}"
                )

    def execute_trading_logic(self, current_price, uptrend, downtrend):
        """执行交易逻辑"""
        current_pos = self.pos

        # 空仓状态 - 寻找开仓机会
        if current_pos == 0:
            # 多头入场信号
            if (self.last_bull_fractal_idx > 0 and
                    self.last_bull_fractal_idx < self.am.count - 1 and
                    current_price > self.last_bull_fractal_high and
                    uptrend):

                self.current_direction = 1
                self.entry_price = current_price

                # 设置止损和止盈
                self.stop_loss_price = (
                        self.last_bull_fractal_low -
                        self.current_atr * self.atr_multiplier
                )
                self.take_profit_price = (
                        self.entry_price +
                        (self.entry_price - self.stop_loss_price) * self.tp_ratio
                )

                self.buy(current_price, self.position_size)
                self.write_log(
                    f"多头开仓: 价格={self.entry_price:.2f}, "
                    f"止损={self.stop_loss_price:.2f}, "
                    f"止盈={self.take_profit_price:.2f}"
                )

            # 空头入场信号
            elif (self.last_bear_fractal_idx > 0 and
                  self.last_bear_fractal_idx < self.am.count - 1 and
                  current_price < self.last_bear_fractal_low and
                  downtrend):

                self.current_direction = -1
                self.entry_price = current_price

                # 设置止损和止盈
                self.stop_loss_price = (
                        self.last_bear_fractal_high +
                        self.current_atr * self.atr_multiplier
                )
                self.take_profit_price = (
                        self.entry_price -
                        (self.stop_loss_price - self.entry_price) * self.tp_ratio
                )

                self.short(current_price, self.position_size)
                self.write_log(
                    f"空头开仓: 价格={self.entry_price:.2f}, "
                    f"止损={self.stop_loss_price:.2f}, "
                    f"止盈={self.take_profit_price:.2f}"
                )

        # 多头持仓 - 检查平仓条件
        elif current_pos > 0:
            should_close = False
            close_reason = ""

            # 1. 止损条件
            if current_price <= self.stop_loss_price:
                should_close = True
                close_reason = "止损"

            # 2. 止盈条件
            elif current_price >= self.take_profit_price:
                should_close = True
                close_reason = "止盈"

            # 3. 价格跌破短期MA
            elif current_price < self.short_ma:
                should_close = True
                close_reason = "MA反转"

            # 4. 出现新的熊市分形且价格跌破该分形的低点
            elif (self.last_bear_fractal_idx > self.am.count - 10 and
                  current_price < self.last_bear_fractal_low):
                should_close = True
                close_reason = "分形反转"

            if should_close:
                self.sell(current_price, abs(current_pos))
                self.current_direction = 0
                self.write_log(
                    f"多头{close_reason}平仓: 价格={current_price:.2f}, "
                    f"entry_price= {self.entry_price:.2f}%"
                )

        # 空头持仓 - 检查平仓条件
        elif current_pos < 0:
            should_close = False
            close_reason = ""

            # 1. 止损条件
            if current_price >= self.stop_loss_price:
                should_close = True
                close_reason = "止损"

            # 2. 止盈条件
            elif current_price <= self.take_profit_price:
                should_close = True
                close_reason = "止盈"

            # 3. 价格突破短期MA
            elif current_price > self.short_ma:
                should_close = True
                close_reason = "MA反转"

            # 4. 出现新的牛市分形且价格突破该分形的高点
            elif (self.last_bull_fractal_idx > self.am.count - 10 and
                  current_price > self.last_bull_fractal_high):
                should_close = True
                close_reason = "分形反转"

            if should_close:
                self.cover(current_price, abs(current_pos))
                self.current_direction = 0
                self.write_log(
                    f"空头{close_reason}平仓: 价格={current_price:.2f}, "
                    f"entry_price= {self.entry_price:.2f}%"
                )

    def on_order(self, order: OrderData):
        """委托回报"""
        pass

    def on_trade(self, trade: TradeData):
        """成交回报"""
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """停止单回报"""
        pass
