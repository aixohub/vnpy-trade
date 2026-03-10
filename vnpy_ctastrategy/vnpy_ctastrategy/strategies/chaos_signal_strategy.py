"""
混沌与信号理论驱动的VnPy日内期货量化交易策略
==============================================
架构：
  状态识别层: Hurst指数 / 分形维数 / 近似熵
  信号处理层: 小波变换 / Kalman滤波 / 希尔伯特变换
  交易决策层: 状态路由 / 动态风险控制 / 执行管理

依赖安装：
  pip install vnpy vnpy_ctastrategy pywavelets numpy scipy
"""

import numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress
from typing import Optional, List
from collections import deque
import pywt

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

# ══════════════════════════════════════════════════
#  第一层：混沌状态识别工具函数
# ══════════════════════════════════════════════════

def compute_hurst_exponent(series: np.ndarray, min_chunk: int = 8) -> float:
    """
    计算Hurst指数
    H > 0.6: 趋势持续 (persistent)
    H ~ 0.5: 随机游走 (random)
    H < 0.4: 均值回归 (anti-persistent)
    """
    n = len(series)
    if n < 20:
        return 0.5

    lags = []
    rs_values = []

    chunk_size = min_chunk
    while chunk_size <= n // 2:
        chunks = [series[i:i + chunk_size] for i in range(0, n - chunk_size + 1, chunk_size)]
        rs_list = []
        for chunk in chunks:
            mean = np.mean(chunk)
            deviations = np.cumsum(chunk - mean)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            lags.append(np.log(chunk_size))
            rs_values.append(np.log(np.mean(rs_list)))
        chunk_size *= 2

    if len(lags) < 2:
        return 0.5

    slope, _, _, _, _ = linregress(lags, rs_values)
    return max(0.0, min(1.0, slope))


def compute_fractal_dimension(series: np.ndarray) -> float:
    """
    计算分形维数 (Higuchi方法)
    FD接近1: 趋势明显, 接近2: 高度复杂/随机
    """
    n = len(series)
    if n < 10:
        return 1.5

    kmax = min(8, n // 4)
    lk = []
    k_vals = []

    for k in range(1, kmax + 1):
        lengths = []
        for m in range(1, k + 1):
            indices = range(m - 1, n, k)
            idx_list = list(indices)
            if len(idx_list) < 2:
                continue
            sub = series[idx_list]
            length = np.sum(np.abs(np.diff(sub))) * (n - 1) / (k * k * len(idx_list))
            lengths.append(length)
        if lengths:
            lk.append(np.log(np.mean(lengths)))
            k_vals.append(np.log(1.0 / k))

    if len(k_vals) < 2:
        return 1.5

    slope, _, _, _, _ = linregress(k_vals, lk)
    return max(1.0, min(2.0, slope))


def compute_approximate_entropy(series: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    计算近似熵 (ApEn)
    值越小: 规律性强 (可预测)
    值越大: 随机性强 (不可预测)
    """
    n = len(series)
    if n < 10:
        return 1.0

    r = r_factor * np.std(series)
    if r == 0:
        return 0.0

    def phi(m_val):
        count = 0
        templates = np.array([series[i:i + m_val] for i in range(n - m_val + 1)])
        for template in templates:
            diffs = np.max(np.abs(templates - template), axis=1)
            count += np.sum(diffs <= r)
        return np.log(count / (n - m_val + 1)) / (n - m_val + 1)

    return abs(phi(m) - phi(m + 1))


# ══════════════════════════════════════════════════
#  第二层：信号处理工具
# ══════════════════════════════════════════════════

def wavelet_denoise(series: np.ndarray, wavelet: str = 'db4', level: int = 3) -> np.ndarray:
    """
    小波变换去噪 + 趋势提取
    返回去噪后的信号 (低频近似分量)
    """
    if len(series) < 2 ** level:
        return series.copy()

    coeffs = pywt.wavedec(series, wavelet, level=level)
    # 软阈值去噪
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(series)))

    denoised_coeffs = [coeffs[0]]  # 保留近似系数
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode='soft'))

    return pywt.waverec(denoised_coeffs, wavelet)[:len(series)]


def wavelet_decompose(series: np.ndarray, wavelet: str = 'db4', level: int = 3):
    """
    小波多尺度分解，返回 (趋势, 波动, 噪声)
    """
    if len(series) < 2 ** level:
        return series.copy(), np.zeros_like(series), np.zeros_like(series)

    coeffs = pywt.wavedec(series, wavelet, level=level)

    # 趋势 = 近似分量重构
    trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    trend = pywt.waverec(trend_coeffs, wavelet)[:len(series)]

    # 波动 = 中间细节分量
    swing_coeffs = [np.zeros_like(coeffs[0])] + [coeffs[i] if i < level else np.zeros_like(coeffs[i]) for i in range(1, len(coeffs))]
    swing = pywt.waverec(swing_coeffs, wavelet)[:len(series)]

    # 噪声 = 原始 - 趋势 - 波动
    noise = series - trend - swing

    return trend, swing, noise


class KalmanFilter1D:
    """
    一维Kalman滤波器，用于价格噪声过滤与状态估计
    """

    def __init__(self, process_variance: float = 1e-5, measurement_variance: float = 1e-3):
        self.Q = process_variance       # 过程噪声
        self.R = measurement_variance   # 观测噪声
        self.P = 1.0                    # 估计误差协方差
        self.x = None                   # 状态估计

    def update(self, measurement: float) -> float:
        if self.x is None:
            self.x = measurement
            return self.x

        # 预测
        P_pred = self.P + self.Q

        # 更新 (Kalman增益)
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * P_pred

        return self.x

    def reset(self):
        self.x = None
        self.P = 1.0


def hilbert_transform_analysis(series: np.ndarray):
    """
    希尔伯特变换相位分析
    返回 (瞬时振幅, 瞬时相位, 瞬时频率)
    """
    if len(series) < 4:
        return np.zeros_like(series), np.zeros_like(series), np.zeros_like(series)

    analytic = hilbert(series)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    frequency = np.diff(phase) / (2.0 * np.pi)
    frequency = np.append(frequency, frequency[-1])  # 补齐长度

    return amplitude, phase, frequency


# ══════════════════════════════════════════════════
#  市场状态枚举
# ══════════════════════════════════════════════════

class MarketState:
    TRENDING_UP = "trending_up"       # 上升趋势 (Hurst高, ApEn低)
    TRENDING_DOWN = "trending_down"   # 下降趋势
    MEAN_REVERT = "mean_revert"       # 均值回归 (Hurst低)
    CHAOTIC = "chaotic"               # 混沌/随机 (ApEn高, FD高)
    TRANSITION = "transition"         # 状态转换中


# ══════════════════════════════════════════════════
#  核心策略
# ══════════════════════════════════════════════════

class ChaosSignalStrategy(CtaTemplate):
    """
    混沌信号策略
    
    参数说明：
    - fast_window: 快速均线周期
    - slow_window: 慢速均线周期
    - chaos_window: 混沌指标计算窗口
    - hurst_threshold_trend: Hurst趋势判定阈值 (>此值为趋势)
    - hurst_threshold_revert: Hurst均值回归阈值 (<此值为回归)
    - apen_threshold: 近似熵阈值 (>此值为混沌)
    - atr_multiplier: ATR止损倍数
    - max_position: 最大持仓
    """

    author = "ChaosSignal"
    vt_symbol: str = ''
    # 策略参数
    fast_window: int = 5
    slow_window: int = 20
    chaos_window: int = 60       # 混沌指标计算周期
    signal_window: int = 30      # 信号处理窗口

    hurst_threshold_trend: float = 0.6
    hurst_threshold_revert: float = 0.4
    apen_threshold: float = 0.8
    fd_threshold: float = 1.7

    atr_period: int = 14
    atr_multiplier: float = 2.0
    max_position: int = 1

    # 内部变量
    hurst: float = 0.5
    fractal_dim: float = 1.5
    approx_entropy: float = 1.0
    market_state: str = MarketState.TRANSITION
    kalman_price: float = 0.0
    hilbert_phase: float = 0.0

    parameters = [
        "vt_symbol",
        "fast_window", "slow_window", "chaos_window", "signal_window",
        "hurst_threshold_trend", "hurst_threshold_revert",
        "apen_threshold", "fd_threshold",
        "atr_period", "atr_multiplier", "max_position"
    ]

    variables = [
        "hurst", "fractal_dim", "approx_entropy",
        "market_state", "kalman_price", "hilbert_phase"
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # K线生成器 (1分钟 -> 5分钟)
        self.bg = BarGenerator(self.on_bar, 5, self.on_5min_bar)
        self.am = ArrayManager(size=200)

        # 混沌指标缓冲
        self.close_buf: deque = deque(maxlen=max(self.chaos_window, self.signal_window) + 10)

        # 信号处理组件
        self.kalman = KalmanFilter1D(process_variance=1e-5, measurement_variance=1e-3)
        self.kalman_prices: deque = deque(maxlen=self.signal_window)

        # 交易状态
        self.long_stop: float = 0.0
        self.short_stop: float = 0.0
        self.entry_price: float = 0.0

        # 当日统计
        self.daily_trades: int = 0
        self.daily_pnl: float = 0.0
        self.max_daily_drawdown: float = 0.02  # 最大日亏损2%停止交易

        # 订单管理
        self.order_ids: list = []

    def on_init(self):
        self.write_log("策略初始化 - 混沌信号策略 ChaosSignalStrategy")

    def on_start(self):
        self.write_log("策略启动 ChaosSignalStrategy")

    def on_stop(self):
        self.write_log("策略停止 ChaosSignalStrategy")

    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        self.bg.update_bar(bar)

    def on_5min_bar(self, bar: BarData):
        """5分钟Bar主处理逻辑"""
        self.cancel_all()

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return

        # 收盘价入缓冲
        self.close_buf.append(bar.close_price)

        # ── Kalman滤波更新 ──
        kp = self.kalman.update(bar.close_price)
        self.kalman_price = kp
        self.kalman_prices.append(kp)

        # 数据不足时跳过
        if len(self.close_buf) < self.chaos_window:
            return

        closes = np.array(list(self.close_buf))
        kprices = np.array(list(self.kalman_prices)) if len(self.kalman_prices) >= 10 else closes[-10:]

        # ══ 第一层：状态识别 ══
        self._update_chaos_indicators(closes)

        # ══ 第二层：信号处理 ══
        trend_signal, swing_signal, phase_signal = self._process_signals(closes, kprices)

        # ══ 第三层：交易决策 ══
        self._make_trading_decision(bar, trend_signal, swing_signal, phase_signal)

    # ──────────────────────────────────────────
    # 第一层实现：混沌状态识别
    # ──────────────────────────────────────────
    def _update_chaos_indicators(self, closes: np.ndarray):
        """更新混沌指标，确定市场状态"""
        # 计算三个混沌指标
        self.hurst = compute_hurst_exponent(closes[-self.chaos_window:])
        self.fractal_dim = compute_fractal_dimension(closes[-min(40, len(closes)):])
        self.approx_entropy = compute_approximate_entropy(closes[-min(50, len(closes)):])

        # 状态路由
        if self.hurst > self.hurst_threshold_trend:
            # 趋势市场：判断方向
            recent = closes[-10:]
            if recent[-1] > recent[0]:
                self.market_state = MarketState.TRENDING_UP
            else:
                self.market_state = MarketState.TRENDING_DOWN
        elif self.hurst < self.hurst_threshold_revert and self.approx_entropy < self.apen_threshold:
            self.market_state = MarketState.MEAN_REVERT
        elif self.approx_entropy > self.apen_threshold or self.fractal_dim > self.fd_threshold:
            self.market_state = MarketState.CHAOTIC
        else:
            self.market_state = MarketState.TRANSITION

    # ──────────────────────────────────────────
    # 第二层实现：信号处理
    # ──────────────────────────────────────────
    def _process_signals(self, closes: np.ndarray, kprices: np.ndarray):
        """
        多信号处理，返回综合信号
        trend_signal: [-1, 0, 1] 趋势方向
        swing_signal: [-1, 0, 1] 摆动信号
        phase_signal: float 希尔伯特相位信号
        """
        # ── 1. 小波趋势信号 ──
        trend_signal = 0
        if len(closes) >= 16:
            trend, swing, noise = wavelet_decompose(closes, level=3)
            # 趋势方向：近期趋势斜率
            if len(trend) >= 5:
                trend_slope = (trend[-1] - trend[-5]) / (trend[-5] + 1e-10)
                if trend_slope > 0.001:
                    trend_signal = 1
                elif trend_slope < -0.001:
                    trend_signal = -1

            # 小波摆动信号：判断局部超买超卖
            if len(swing) >= 3:
                swing_mean = np.mean(swing[-10:]) if len(swing) >= 10 else np.mean(swing)
                swing_std = np.std(swing[-10:]) if len(swing) >= 10 else np.std(swing)
                swing_z = (swing[-1] - swing_mean) / (swing_std + 1e-10)
                if swing_z > 1.5:
                    swing_signal = -1  # 摆动过高，均值回归做空
                elif swing_z < -1.5:
                    swing_signal = 1   # 摆动过低，均值回归做多
                else:
                    swing_signal = 0
            else:
                swing_signal = 0
        else:
            swing_signal = 0

        # ── 2. Kalman趋势信号 ──
        kalman_signal = 0
        if len(kprices) >= 10:
            kslope = (kprices[-1] - kprices[-5]) / (kprices[-5] + 1e-10) if len(kprices) >= 5 else 0
            if kslope > 0.0005:
                kalman_signal = 1
            elif kslope < -0.0005:
                kalman_signal = -1

        # 综合趋势信号（加权）
        final_trend = np.sign(trend_signal * 0.5 + kalman_signal * 0.5)

        # ── 3. 希尔伯特相位信号 ──
        phase_signal = 0.0
        if len(closes) >= 32:
            # 对Kalman滤波后价格做希尔伯特变换
            detrended = kprices - np.mean(kprices)
            amplitude, phase, frequency = hilbert_transform_analysis(detrended)
            self.hilbert_phase = float(phase[-1])

            # 相位信号：相位在特定区间时产生买卖信号
            # 将相位归一化到 [-π, π]
            norm_phase = (phase[-1] % (2 * np.pi)) - np.pi
            if -0.3 < norm_phase < 0.3:
                phase_signal = 1.0   # 相位谷底，做多
            elif abs(norm_phase) > 2.8:
                phase_signal = -1.0  # 相位顶部，做空

        return int(final_trend), swing_signal, phase_signal

    # ──────────────────────────────────────────
    # 第三层实现：交易决策
    # ──────────────────────────────────────────
    def _make_trading_decision(self, bar: BarData, trend_signal: int,
                                swing_signal: int, phase_signal: float):
        """状态路由 -> 策略选择 -> 风险控制 -> 执行"""

        # 日内风控：亏损过多停止交易
        if self.daily_pnl < -self.max_daily_drawdown:
            self.write_log(f"日内亏损超限 {self.daily_pnl:.4f}，停止交易")
            return

        am = self.am
        atr = am.atr(self.atr_period)
        if atr == 0:
            return

        # 动态仓位：根据混沌指标调整
        position_size = self._calculate_position_size(atr, bar.close_price)

        # ── 状态路由：按市场状态选择不同逻辑 ──
        if self.market_state == MarketState.TRENDING_UP:
            self._strategy_trend_follow(bar, trend_signal, phase_signal, atr, position_size, direction=1)

        elif self.market_state == MarketState.TRENDING_DOWN:
            self._strategy_trend_follow(bar, trend_signal, phase_signal, atr, position_size, direction=-1)

        elif self.market_state == MarketState.MEAN_REVERT:
            self._strategy_mean_reversion(bar, swing_signal, phase_signal, atr, position_size)

        elif self.market_state == MarketState.CHAOTIC:
            # 混沌状态：仅管理现有仓位，不开新仓
            self._strategy_chaos_management(bar, atr)

        else:  # TRANSITION
            # 过渡状态：缩小仓位，等待信号明确
            self._manage_stops(bar, atr * 1.5)

    def _calculate_position_size(self, atr: float, price: float) -> int:
        """
        动态仓位计算
        - Hurst越高（趋势越强）：仓位越大
        - 近似熵越高（越混沌）：仓位越小
        - 分形维数越低（越规律）：仓位越大
        """
        # 基础仓位
        base = self.max_position

        # Hurst调整：趋势越强仓位越大
        hurst_factor = 2 * self.hurst - 1  # [-1, 1]，H=0.5时为0
        hurst_factor = max(0.0, hurst_factor)  # 只在趋势时加仓

        # ApEn调整：熵越低仓位越大
        apen_factor = 1.0 - min(1.0, self.approx_entropy)

        # 综合因子 [0.3, 1.5]
        factor = 0.3 + 1.2 * hurst_factor * apen_factor
        size = max(1, min(self.max_position, round(base * factor)))

        return size

    def _strategy_trend_follow(self, bar: BarData, trend_signal: int,
                                phase_signal: float, atr: float,
                                position_size: int, direction: int):
        """趋势跟踪策略"""
        am = self.am
        fast_ma = am.sma(self.fast_window)
        slow_ma = am.sma(self.slow_window)

        pos = self.pos

        if direction == 1:  # 上升趋势
            # 开多：MA金叉 + 趋势信号确认 + 相位辅助
            if fast_ma > slow_ma and trend_signal >= 0 and pos == 0:
                # 希尔伯特相位辅助：相位谷底更佳入场时机
                if phase_signal >= 0:  # 相位没有顶部信号
                    stop_price = bar.close_price - self.atr_multiplier * atr
                    self.buy(bar.close_price + 1, position_size)
                    self.long_stop = stop_price
                    self.entry_price = bar.close_price

            # 移动止盈止损
            elif pos > 0:
                new_stop = bar.close_price - self.atr_multiplier * atr
                self.long_stop = max(self.long_stop, new_stop)
                if bar.close_price <= self.long_stop:
                    self.sell(bar.close_price - 1, abs(pos))

        else:  # 下降趋势
            if fast_ma < slow_ma and trend_signal <= 0 and pos == 0:
                if phase_signal <= 0:
                    stop_price = bar.close_price + self.atr_multiplier * atr
                    self.short(bar.close_price - 1, position_size)
                    self.short_stop = stop_price
                    self.entry_price = bar.close_price

            elif pos < 0:
                new_stop = bar.close_price + self.atr_multiplier * atr
                self.short_stop = min(self.short_stop, new_stop)
                if bar.close_price >= self.short_stop:
                    self.cover(bar.close_price + 1, abs(pos))

    def _strategy_mean_reversion(self, bar: BarData, swing_signal: int,
                                  phase_signal: float, atr: float, position_size: int):
        """均值回归策略"""
        am = self.am
        # 计算布林带
        upper, middle, lower = self._bollinger_bands(am, period=20, dev=2.0)
        pos = self.pos

        # 做多信号：价格触及下轨 + 摆动信号做多
        if swing_signal == 1 or bar.close_price < lower:
            if pos == 0 and phase_signal >= -0.5:
                self.buy(bar.close_price + 1, position_size)
                self.long_stop = bar.close_price - 1.5 * atr
                self.entry_price = bar.close_price

        # 做空信号：价格触及上轨 + 摆动信号做空
        elif swing_signal == -1 or bar.close_price > upper:
            if pos == 0 and phase_signal <= 0.5:
                self.short(bar.close_price - 1, position_size)
                self.short_stop = bar.close_price + 1.5 * atr
                self.entry_price = bar.close_price

        # 均值回归平仓：回到中轨附近
        if pos > 0 and bar.close_price >= middle:
            self.sell(bar.close_price - 1, abs(pos))
        elif pos < 0 and bar.close_price <= middle:
            self.cover(bar.close_price + 1, abs(pos))

        # 止损
        self._manage_stops(bar, atr)

    def _strategy_chaos_management(self, bar: BarData, atr: float):
        """混沌状态：收紧止损，保护利润"""
        pos = self.pos
        tight_atr = atr * 1.0  # 使用更紧的止损

        if pos > 0:
            new_stop = bar.close_price - tight_atr
            self.long_stop = max(self.long_stop, new_stop)
            if bar.close_price <= self.long_stop:
                self.sell(bar.close_price - 1, abs(pos))

        elif pos < 0:
            new_stop = bar.close_price + tight_atr
            self.short_stop = min(self.short_stop, new_stop)
            if bar.close_price >= self.short_stop:
                self.cover(bar.close_price + 1, abs(pos))

    def _manage_stops(self, bar: BarData, atr: float):
        """通用止损管理"""
        pos = self.pos
        if pos > 0 and self.long_stop > 0:
            if bar.close_price <= self.long_stop:
                self.sell(bar.close_price - 1, abs(pos))
        elif pos < 0 and self.short_stop > 0:
            if bar.close_price >= self.short_stop:
                self.cover(bar.close_price + 1, abs(pos))

    def _bollinger_bands(self, am: ArrayManager, period: int = 20, dev: float = 2.0):
        """布林带计算"""
        closes = am.close[-period:]
        middle = np.mean(closes)
        std = np.std(closes)
        return middle + dev * std, middle, middle - dev * std

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        """成交回调：更新日内统计"""
        self.daily_trades += 1
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        pass


# ══════════════════════════════════════════════════
#  回测入口（独立运行）
# ══════════════════════════════════════════════════

def run_backtest():
    """
    使用VnPy CTA回测引擎运行策略
    需要先配置数据库和历史数据
    """
    from vnpy_ctastrategy.backtesting import BacktestingEngine, OptimizationSetting
    from datetime import datetime

    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol="rb2501.SHFE",           # 螺纹钢期货
        interval="1m",                      # 1分钟K线
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
        rate=0.00003,                       # 手续费率
        slippage=1,                         # 滑点（1跳）
        size=10,                            # 合约乘数
        pricetick=1,                        # 最小价格变动
        capital=500000,                     # 初始资金50万
    )

    engine.add_strategy(
        ChaosSignalStrategy,
        {
            "fast_window": 5,
            "slow_window": 20,
            "chaos_window": 60,
            "signal_window": 30,
            "hurst_threshold_trend": 0.6,
            "hurst_threshold_revert": 0.4,
            "apen_threshold": 0.8,
            "atr_multiplier": 2.0,
            "max_position": 1,
        }
    )

    engine.load_data()
    engine.run_backtesting()

    df = engine.calculate_result()
    stats = engine.calculate_statistics()

    print("\n" + "=" * 60)
    print("混沌信号策略回测结果")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    engine.show_chart()
    return df, stats


# ══════════════════════════════════════════════════
#  参数优化入口
# ══════════════════════════════════════════════════

def run_optimization():
    """参数优化"""
    from vnpy_ctastrategy.backtesting import BacktestingEngine, OptimizationSetting
    from datetime import datetime

    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol="rb2501.SHFE",
        interval="1m",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
        rate=0.00003,
        slippage=1,
        size=10,
        pricetick=1,
        capital=500000,
    )
    engine.add_strategy(ChaosSignalStrategy, {})

    setting = OptimizationSetting()
    setting.set_target("sharpe_ratio")
    setting.add_parameter("hurst_threshold_trend", 0.55, 0.70, 0.05)
    setting.add_parameter("hurst_threshold_revert", 0.35, 0.45, 0.05)
    setting.add_parameter("atr_multiplier", 1.5, 3.0, 0.5)
    setting.add_parameter("chaos_window", 40, 80, 20)

    result = engine.run_ga_optimization(setting)
    print("\n最优参数组合:")
    for item in result[:5]:
        print(item)

    return result


# ══════════════════════════════════════════════════
#  工具：实时监控指标（可接入Dashboard）
# ══════════════════════════════════════════════════

def analyze_market_state(prices: List[float]) -> dict:
    """
    独立分析函数，可用于实时监控面板
    输入：价格列表
    输出：混沌指标字典
    """
    arr = np.array(prices)
    hurst = compute_hurst_exponent(arr)
    fd = compute_fractal_dimension(arr)
    apen = compute_approximate_entropy(arr)

    # 小波分解
    if len(arr) >= 16:
        trend, swing, noise = wavelet_decompose(arr)
        trend_strength = abs(trend[-1] - trend[-5]) / (np.std(trend) + 1e-10) if len(trend) >= 5 else 0
    else:
        trend_strength = 0

    # 状态判断
    if hurst > 0.6:
        state = "趋势市" + ("↑" if arr[-1] > arr[-5] else "↓")
    elif hurst < 0.4:
        state = "震荡/均值回归"
    elif apen > 0.8:
        state = "混沌/随机"
    else:
        state = "过渡状态"

    return {
        "市场状态": state,
        "Hurst指数": round(hurst, 4),
        "分形维数": round(fd, 4),
        "近似熵": round(apen, 4),
        "趋势强度": round(trend_strength, 4),
        "可交易性": "高" if (hurst > 0.6 or hurst < 0.4) and apen < 0.8 else "低"
    }


if __name__ == "__main__":
    # 示例：单独测试混沌指标
    print("混沌指标测试")
    print("-" * 40)
    
    # 生成测试价格序列（模拟趋势 + 噪声）
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    trend_prices = 3500 + 50 * t + 20 * np.random.randn(100)  # 趋势价格
    random_prices = 3500 + 200 * np.cumsum(np.random.randn(100) * 0.1)  # 随机游走

    print("趋势价格序列分析：")
    result = analyze_market_state(list(trend_prices))
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n随机游走价格序列分析：")
    result = analyze_market_state(list(random_prices))
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n如需运行回测，请调用 run_backtest() 函数")
    print("如需参数优化，请调用 run_optimization() 函数")
