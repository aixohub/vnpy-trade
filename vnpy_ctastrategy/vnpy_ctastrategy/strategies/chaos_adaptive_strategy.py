"""
混沌理论期货量化交易系统
基于vnpy 3.0+框架
"""
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

import numpy as np
from scipy import stats
import pywt
from scipy.signal import hilbert
from scipy.stats import linregress
from scipy.optimize import curve_fit
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ==================== 混沌理论核心指标 ====================

class ChaosTheoryIndicator:
    """混沌理论指标计算器"""

    @staticmethod
    def hurst_exponent(ts: np.ndarray, max_lag: int = 50) -> float:
        """
        计算Hurst指数
        返回:
            H < 0.5: 均值回归
            H = 0.5: 随机游走
            H > 0.5: 趋势持续
        """
        lags = range(2, min(max_lag, len(ts) // 2))
        tau = []
        for lag in lags:
            tau.append(np.std(np.subtract(ts[lag:], ts[:-lag])))

        if len(tau) < 2:
            return 0.5

        # 线性拟合
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]

    @staticmethod
    def fractal_dimension(ts: np.ndarray, scale_range: Tuple[float, float] = (0.1, 1.0)) -> float:
        """
        计算分形维数 (盒计数法)
        反映价格序列的复杂度和粗糙程度
        """
        scales = np.linspace(scale_range[0], scale_range[1], 20)
        counts = []

        for scale in scales:
            # 重新缩放时间序列
            scaled_ts = (ts - np.min(ts)) / (np.max(ts) - np.min(ts) + 1e-10)
            # 计算盒计数
            num_boxes = int(np.ceil(len(scaled_ts) * scale))
            if num_boxes == 0:
                continue
            box_size = len(scaled_ts) / num_boxes

            count = 0
            for i in range(num_boxes):
                start = int(i * box_size)
                end = min(int((i + 1) * box_size), len(scaled_ts))
                if start >= end:
                    continue
                box_min = np.min(scaled_ts[start:end])
                box_max = np.max(scaled_ts[start:end])
                count += (box_max - box_min) / scale + 1

            counts.append(count)

        if len(counts) < 2:
            return 1.5

        # 线性拟合
        poly = np.polyfit(np.log(1 / scales[:len(counts)]), np.log(counts), 1)
        return poly[0]

    @staticmethod
    def approximate_entropy(ts: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        计算近似熵 (衡量序列随机性)
        值越小表示越规则，值越大表示越随机
        """

        def _phi(m):
            N = len(ts)
            C = []
            for i in range(N - m + 1):
                pattern = ts[i:i + m]
                count = 0
                for j in range(N - m + 1):
                    if np.max(np.abs(pattern - ts[j:j + m])) <= r * np.std(ts):
                        count += 1
                C.append(count / (N - m + 1))
            return np.sum(np.log(C)) / (N - m + 1)

        if len(ts) < m + 1:
            return 0

        return abs(_phi(m) - _phi(m + 1))

    @staticmethod
    def lyapunov_exponent(ts: np.ndarray, embedding_dim: int = 3, tau: int = 1) -> float:
        """
        计算最大Lyapunov指数 (衡量混沌特性)
        正值表示混沌系统
        """
        N = len(ts)
        if N < embedding_dim * 10:
            return 0

        # 相空间重构
        M = N - (embedding_dim - 1) * tau
        if M <= embedding_dim:
            return 0

        Y = np.zeros((M, embedding_dim))
        for i in range(embedding_dim):
            Y[:, i] = ts[i * tau:i * tau + M]

        # 计算Lyapunov指数
        dists = []
        for i in range(M - 1):
            distances = np.sqrt(np.sum((Y[i + 1:] - Y[i]) ** 2, axis=1))
            if len(distances) > 0:
                min_dist = np.min(distances)
                if min_dist > 0:
                    dists.append(np.log(min_dist))

        if len(dists) < 2:
            return 0

        # 线性拟合斜率
        x = np.arange(len(dists))
        slope, _, _, _, _ = linregress(x, dists)
        return slope / tau


# ==================== 信号处理层 ====================

class SignalProcessor:
    """混沌信号处理器"""

    @staticmethod
    def wavelet_transform(ts: np.ndarray, wavelet: str = 'db4', level: int = 5) -> Dict:
        """
        小波变换进行多尺度分解
        返回不同尺度下的信号
        """
        coeffs = pywt.wavedec(ts, wavelet, level=min(level, pywt.dwt_max_level(len(ts), pywt.Wavelet(wavelet))))

        result = {
            'approx': coeffs[0],  # 近似系数 (低频趋势)
            'details': coeffs[1:]  # 细节系数 (高频噪声)
        }

        # 重构各尺度信号
        reconstructed = []
        for i in range(len(coeffs)):
            coeff_list = [np.zeros_like(c) for c in coeffs]
            coeff_list[i] = coeffs[i]
            rec = pywt.waverec(coeff_list, wavelet)
            reconstructed.append(rec[:len(ts)])

        result['reconstructed'] = reconstructed
        return result

    @staticmethod
    def kalman_filter(ts: np.ndarray,
                      Q: float = 1e-5,
                      R: float = 0.01 ** 2) -> np.ndarray:
        """
        Kalman滤波进行噪声过滤
        """
        n_iter = len(ts)
        xhat = np.zeros(n_iter)  # 状态估计
        P = np.zeros(n_iter)  # 估计误差协方差
        xhatminus = np.zeros(n_iter)  # 先验估计
        Pminus = np.zeros(n_iter)  # 先验误差协方差
        K = np.zeros(n_iter)  # Kalman增益

        # 初始值
        xhat[0] = ts[0]
        P[0] = 1.0

        for k in range(1, n_iter):
            # 预测
            xhatminus[k] = xhat[k - 1]
            Pminus[k] = P[k - 1] + Q

            # 更新
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (ts[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]

        return xhat

    @staticmethod
    def hilbert_transform(ts: np.ndarray) -> Dict:
        """
        希尔伯特变换进行相位分析
        返回瞬时幅度和相位
        """
        analytic_signal = hilbert(ts)
        amplitude = np.abs(analytic_signal)
        phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_freq = np.diff(phase) / (2.0 * np.pi)

        # 对齐长度
        instantaneous_freq = np.append(instantaneous_freq, instantaneous_freq[-1])

        return {
            'amplitude': amplitude,
            'phase': phase,
            'instantaneous_freq': instantaneous_freq,
            'analytic_signal': analytic_signal
        }

    @staticmethod
    def noise_reduction(ts: np.ndarray, method: str = 'wavelet') -> np.ndarray:
        """噪声降低"""
        if method == 'wavelet':
            # 小波去噪
            coeffs = pywt.wavedec(ts, 'db4', level=4)
            # 阈值处理
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(ts)))
            coeffs[1:] = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]]
            return pywt.waverec(coeffs, 'db4')[:len(ts)]
        else:
            return SignalProcessor.kalman_filter(ts)


# ==================== 市场状态识别 ====================

class MarketState(Enum):
    """市场状态枚举"""
    TRENDING_UP = "上升趋势"
    TRENDING_DOWN = "下降趋势"
    MEAN_REVERTING = "均值回归"
    CHAOTIC = "混沌状态"
    RANDOM = "随机波动"
    LOW_VOLATILITY = "低波动"
    HIGH_VOLATILITY = "高波动"


@dataclass
class MarketStatus:
    """市场状态数据类"""
    state: MarketState
    confidence: float
    hurst: float
    fractal_dim: float
    approx_entropy: float
    lyapunov: float
    volatility: float
    trend_strength: float


class MarketStateAnalyzer:
    """市场状态分析器"""

    def __init__(self,
                 hurst_threshold: float = 0.05,
                 entropy_threshold: float = 0.3,
                 lyapunov_threshold: float = 0.001):
        self.hurst_threshold = hurst_threshold
        self.entropy_threshold = entropy_threshold
        self.lyapunov_threshold = lyapunov_threshold
        self.chaos_indicator = ChaosTheoryIndicator()

    def analyze(self, prices: np.ndarray, returns: np.ndarray) -> MarketStatus:
        """分析市场状态"""
        # 计算基本指标
        hurst = self.chaos_indicator.hurst_exponent(prices[-100:]) if len(prices) >= 100 else 0.5
        fractal_dim = self.chaos_indicator.fractal_dimension(prices[-100:]) if len(prices) >= 100 else 1.5
        approx_entropy = self.chaos_indicator.approximate_entropy(returns[-50:]) if len(returns) >= 50 else 0.5
        lyapunov = self.chaos_indicator.lyapunov_exponent(prices[-100:]) if len(prices) >= 100 else 0

        # 计算波动率和趋势强度
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
        if len(prices) >= 20:
            x = np.arange(len(prices[-20:]))
            slope, _, r_value, _, _ = linregress(x, prices[-20:])
            trend_strength = abs(r_value) * np.sign(slope)
        else:
            trend_strength = 0

        # 状态识别逻辑
        state = self._identify_state(
            hurst, fractal_dim, approx_entropy,
            lyapunov, volatility, trend_strength
        )

        # 计算置信度
        confidence = self._calculate_confidence(
            hurst, fractal_dim, approx_entropy,
            lyapunov, volatility, trend_strength
        )

        return MarketStatus(
            state=state,
            confidence=confidence,
            hurst=hurst,
            fractal_dim=fractal_dim,
            approx_entropy=approx_entropy,
            lyapunov=lyapunov,
            volatility=volatility,
            trend_strength=trend_strength
        )

    def _identify_state(self, hurst: float, fractal_dim: float,
                        approx_entropy: float, lyapunov: float,
                        volatility: float, trend_strength: float) -> MarketState:
        """识别市场状态"""

        # 判断趋势状态
        if hurst > 0.5 + self.hurst_threshold and trend_strength > 0.1:
            return MarketState.TRENDING_UP
        elif hurst > 0.5 + self.hurst_threshold and trend_strength < -0.1:
            return MarketState.TRENDING_DOWN
        elif hurst < 0.5 - self.hurst_threshold:
            return MarketState.MEAN_REVERTING

        # 判断混沌状态
        if lyapunov > self.lyapunov_threshold and approx_entropy > self.entropy_threshold:
            return MarketState.CHAOTIC

        # 判断随机状态
        if abs(hurst - 0.5) < self.hurst_threshold and approx_entropy > 0.5:
            return MarketState.RANDOM

        # 判断波动状态
        if volatility < 0.01:
            return MarketState.LOW_VOLATILITY
        elif volatility > 0.05:
            return MarketState.HIGH_VOLATILITY

        return MarketState.RANDOM

    def _calculate_confidence(self, *args) -> float:
        """计算状态识别置信度"""
        # 基于各指标的一致性计算置信度
        weights = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]  # 指标权重
        scores = []

        # 这里简化处理，实际应根据各指标的一致性计算
        for i, val in enumerate(args):
            if i == 0:  # hurst
                scores.append(1 - abs(val - 0.5) * 2)
            elif i == 4:  # volatility
                scores.append(min(val * 10, 1.0))
            else:
                scores.append(min(abs(val), 1.0))

        confidence = np.average(scores, weights=weights[:len(scores)])
        return float(np.clip(confidence, 0, 1))


# ==================== 交易策略 ====================

class TradingStrategy:
    """交易策略基类"""

    def __init__(self, name: str):
        self.name = name

    def generate_signals(self, data: pd.DataFrame, status: MarketStatus) -> pd.DataFrame:
        """生成交易信号"""
        raise NotImplementedError


class TrendFollowingStrategy(TradingStrategy):
    """趋势跟踪策略"""

    def __init__(self,
                 fast_period: int = 10,
                 slow_period: int = 30,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0):
        super().__init__("趋势跟踪")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, data: pd.DataFrame, status: MarketStatus) -> pd.DataFrame:
        df = data.copy()

        # 计算指标
        df['fast_ma'] = df['close'].rolling(self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(self.slow_period).mean()

        # ATR计算
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(self.atr_period).mean()

        # 生成信号
        df['position'] = 0

        # 趋势条件
        trend_condition = (
                (df['fast_ma'] > df['slow_ma']) &
                (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1))
        )
        df.loc[trend_condition, 'position'] = 1

        # 反转条件
        reverse_condition = (
                (df['fast_ma'] < df['slow_ma']) &
                (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1))
        )
        df.loc[reverse_condition, 'position'] = -1

        # 止损
        if 'position' in df.columns:
            df['stop_loss'] = np.nan
            long_positions = df['position'] == 1
            short_positions = df['position'] == -1

            df.loc[long_positions, 'stop_loss'] = (
                    df.loc[long_positions, 'close'] -
                    df.loc[long_positions, 'atr'] * self.atr_multiplier
            )
            df.loc[short_positions, 'stop_loss'] = (
                    df.loc[short_positions, 'close'] +
                    df.loc[short_positions, 'atr'] * self.atr_multiplier
            )

        return df


class MeanReversionStrategy(TradingStrategy):
    """均值回归策略"""

    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30):
        super().__init__("均值回归")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def generate_signals(self, data: pd.DataFrame, status: MarketStatus) -> pd.DataFrame:
        df = data.copy()

        # 布林带
        df['bb_middle'] = df['close'].rolling(self.bb_period).mean()
        bb_std = df['close'].rolling(self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + bb_std * self.bb_std
        df['bb_lower'] = df['bb_middle'] - bb_std * self.bb_std

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 生成信号
        df['position'] = 0

        # 超卖买入
        buy_condition = (
                (df['close'] < df['bb_lower']) &
                (df['rsi'] < self.rsi_oversold)
        )
        df.loc[buy_condition, 'position'] = 1

        # 超卖卖出
        sell_condition = (
                (df['close'] > df['bb_upper']) &
                (df['rsi'] > self.rsi_overbought)
        )
        df.loc[sell_condition, 'position'] = -1

        # 平仓条件：价格回归中轨
        close_long = (df['position'].shift(1) == 1) & (df['close'] >= df['bb_middle'])
        close_short = (df['position'].shift(1) == -1) & (df['close'] <= df['bb_middle'])
        df.loc[close_long | close_short, 'position'] = 0

        return df


class ChaosAdaptiveStrategy(TradingStrategy):
    """混沌自适应策略"""

    def __init__(self):
        super().__init__("混沌自适应")
        self.signal_processor = SignalProcessor()

    def generate_signals(self, data: pd.DataFrame, status: MarketStatus) -> pd.DataFrame:
        df = data.copy()

        # 使用希尔伯特变换分析相位
        hilbert_result = self.signal_processor.hilbert_transform(df['close'].values)

        df['hilbert_phase'] = hilbert_result['phase']
        df['hilbert_amp'] = hilbert_result['amplitude']

        # 相位变化率作为动量指标
        df['phase_change'] = df['hilbert_phase'].diff()
        df['phase_momentum'] = df['phase_change'].rolling(5).mean()

        # 振幅变化作为波动率指标
        df['amp_change'] = df['hilbert_amp'].pct_change()
        df['volatility_score'] = df['amp_change'].rolling(10).std()

        # 生成自适应信号
        df['position'] = 0

        # 基于混沌状态的动态规则
        if status.state == MarketState.TRENDING_UP:
            # 趋势向上：跟随趋势
            trend_signal = (df['phase_momentum'] > 0) & (df['volatility_score'] < 0.02)
            df.loc[trend_signal, 'position'] = 1

        elif status.state == MarketState.TRENDING_DOWN:
            # 趋势向下：跟随趋势
            trend_signal = (df['phase_momentum'] < 0) & (df['volatility_score'] < 0.02)
            df.loc[trend_signal, 'position'] = -1

        elif status.state == MarketState.MEAN_REVERTING:
            # 均值回归：反转交易
            overbought = (df['close'] > df['close'].rolling(20).mean() * 1.02)
            oversold = (df['close'] < df['close'].rolling(20).mean() * 0.98)
            df.loc[oversold, 'position'] = 1
            df.loc[overbought, 'position'] = -1

        elif status.state == MarketState.CHAOTIC:
            # 混沌状态：减小仓位或离场
            df['position'] = 0  # 混沌市场不交易或极小仓位

        elif status.state == MarketState.LOW_VOLATILITY:
            # 低波动：均值回归策略
            bb_middle = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df.loc[df['close'] < bb_middle - bb_std, 'position'] = 0.5  # 小仓位
            df.loc[df['close'] > bb_middle + bb_std, 'position'] = -0.5

        # 动态止损
        if 'position' in df.columns:
            atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            df['stop_loss'] = np.nan

            long_positions = df['position'] > 0
            short_positions = df['position'] < 0

            # 根据市场状态调整止损幅度
            if status.state in [MarketState.CHAOTIC, MarketState.HIGH_VOLATILITY]:
                multiplier = 3.0
            elif status.state == MarketState.LOW_VOLATILITY:
                multiplier = 1.0
            else:
                multiplier = 2.0

            df.loc[long_positions, 'stop_loss'] = (
                    df.loc[long_positions, 'close'] - atr * multiplier
            )
            df.loc[short_positions, 'stop_loss'] = (
                    df.loc[short_positions, 'close'] + atr * multiplier
            )

        return df


# ==================== VNPY策略集成 ====================


from vnpy.trader.constant import Interval, Direction, Offset
# from vnpy.trader.object import GridPositionCalculator


class ChaosTradingStrategy(CtaTemplate):
    """混沌理论交易策略 - VNPY集成版"""

    author = "Chaos Trading System"

    # 参数设置
    hurst_threshold = 0.05
    entropy_threshold = 0.3
    lyapunov_threshold = 0.001

    # 仓位管理
    fixed_size = 1
    max_pos = 3

    # 策略参数
    fast_window = 10
    slow_window = 30
    bb_window = 20
    bb_std = 2.0
    rsi_window = 14

    parameters = [
        "hurst_threshold", "entropy_threshold", "lyapunov_threshold",
        "fixed_size", "max_pos",
        "fast_window", "slow_window", "bb_window", "bb_std", "rsi_window"
    ]

    variables = [
        "pos", "market_state", "hurst", "fractal_dim",
        "approx_entropy", "lyapunov", "volatility"
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 初始化组件
        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager(size=200)

        # 混沌理论组件
        self.chaos_indicator = ChaosTheoryIndicator()
        self.signal_processor = SignalProcessor()
        self.state_analyzer = MarketStateAnalyzer(
            hurst_threshold=self.hurst_threshold,
            entropy_threshold=self.entropy_threshold,
            lyapunov_threshold=self.lyapunov_threshold
        )

        # 策略集合
        self.strategies = {
            "trend": TrendFollowingStrategy(
                fast_period=self.fast_window,
                slow_period=self.slow_window
            ),
            "mean_reversion": MeanReversionStrategy(
                bb_period=self.bb_window,
                bb_std=self.bb_std,
                rsi_period=self.rsi_window
            ),
            "chaos_adaptive": ChaosAdaptiveStrategy()
        }

        # 状态变量
        self.current_strategy = "chaos_adaptive"
        self.market_state = None
        self.hurst = 0.5
        self.fractal_dim = 1.5
        self.approx_entropy = 0.5
        self.lyapunov = 0
        self.volatility = 0
        self.trend_strength = 0

        # 仓位计算器

        # 信号缓存
        self.signals = pd.DataFrame()

    def on_init(self):
        """初始化策略"""
        self.write_log("混沌理论交易策略初始化")
        self.load_bar(30)  # 加载30天数据

    def on_start(self):
        """策略启动"""
        self.write_log("策略启动")
        self.put_event()

    def on_stop(self):
        """策略停止"""
        self.write_log("策略停止")
        self.put_event()

    def on_tick(self, tick: TickData):
        """Tick更新"""
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """K线更新"""
        # 更新数组管理器
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        # 收集数据
        closes = self.am.close_array
        highs = self.am.high_array
        lows = self.am.low_array
        volumes = self.am.volume_array

        # 计算收益率
        returns = np.diff(closes) / closes[:-1]

        # 分析市场状态
        market_status = self.state_analyzer.analyze(closes, returns)

        # 更新状态变量
        self.market_state = market_status.state.value
        self.hurst = market_status.hurst
        self.fractal_dim = market_status.fractal_dim
        self.approx_entropy = market_status.approx_entropy
        self.lyapunov = market_status.lyapunov
        self.volatility = market_status.volatility
        self.trend_strength = market_status.trend_strength

        # 选择策略
        self._select_strategy(market_status)

        # 准备数据DataFrame
        data_len = len(closes)
        data = pd.DataFrame({
            'open': self.am.open_array[-data_len:],
            'high': highs[-data_len:],
            'low': lows[-data_len:],
            'close': closes[-data_len:],
            'volume': volumes[-data_len:] if len(volumes) >= data_len else np.zeros(data_len)
        })

        # 生成交易信号
        strategy = self.strategies[self.current_strategy]
        signals = strategy.generate_signals(data, market_status)

        # 获取最新信号
        if len(signals) > 0:
            latest_signal = signals.iloc[-1]

            # 执行交易
            self._execute_trading(latest_signal, bar)

        # 更新信号缓存
        self.signals = signals

        # 风险控制
        self._risk_management(bar)

        # 更新界面
        self.put_event()

    def _select_strategy(self, status: MarketStatus):
        """根据市场状态选择策略"""
        if status.state == MarketState.TRENDING_UP or status.state == MarketState.TRENDING_DOWN:
            self.current_strategy = "trend"
            self.write_log(f"切换到趋势跟踪策略，市场状态：{status.state.value}")

        elif status.state == MarketState.MEAN_REVERTING:
            self.current_strategy = "mean_reversion"
            self.write_log(f"切换到均值回归策略，市场状态：{status.state.value}")

        elif status.state == MarketState.CHAOTIC:
            self.current_strategy = "chaos_adaptive"
            self.write_log(f"切换到混沌自适应策略，市场状态：{status.state.value}")

        elif status.state == MarketState.LOW_VOLATILITY:
            self.current_strategy = "mean_reversion"
            self.write_log(f"切换到均值回归策略，市场状态：{status.state.value}")

        else:
            self.current_strategy = "chaos_adaptive"

    def _execute_trading(self, signal: pd.Series, bar: BarData):
        """执行交易"""
        position = signal.get('position', 0)
        stop_loss = signal.get('stop_loss', None)

        # 计算目标仓位
        target_pos = int(position * self.max_pos)

        # 调整仓位
        if target_pos > self.pos:
            # 需要买入
            volume = target_pos - self.pos
            price = bar.close_price * 1.005  # 加价买入

            self.buy(price, volume)

        elif target_pos < self.pos:
            # 需要卖出
            volume = self.pos - target_pos
            price = bar.close_price * 0.995  # 减价卖出

            self.sell(price, volume)

        # 设置止损
        if stop_loss is not None and not np.isnan(stop_loss):
            if self.pos > 0 and bar.close_price < stop_loss:
                self.write_log(f"触发多头止损：{bar.close_price} < {stop_loss}")
                self.sell(bar.close_price * 0.99, abs(self.pos))
            elif self.pos < 0 and bar.close_price > stop_loss:
                self.write_log(f"触发空头止损：{bar.close_price} > {stop_loss}")
                self.cover(bar.close_price * 1.01, abs(self.pos))

    def _risk_management(self, bar: BarData):
        """风险管理"""
        # 最大回撤控制
        if self.volatility > 0.1:  # 高波动率
            self.max_pos = 1
        elif self.volatility < 0.02:  # 低波动率
            self.max_pos = 3
        else:
            self.max_pos = 2

        # 根据混沌指标调整仓位
        if self.approx_entropy > 0.7:  # 高随机性
            self.max_pos = max(1, self.max_pos - 1)

        if self.lyapunov > 0.01:  # 强混沌特性
            self.max_pos = 1

    def on_order(self, order: OrderData):
        """委托更新"""
        pass

    def on_trade(self, trade: TradeData):
        """成交更新"""
        self.pos += trade.volume if trade.direction == Direction.LONG else -trade.volume
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """停止单更新"""
        pass


# ==================== 回测和优化 ====================

class BacktestOptimizer:
    """回测优化器"""

    def __init__(self, strategy_class):
        self.strategy_class = strategy_class

    def optimize_parameters(self, data: pd.DataFrame,
                            param_ranges: Dict) -> Dict:
        """优化策略参数"""
        best_params = {}
        best_performance = -np.inf

        # 这里简化处理，实际应使用网格搜索或优化算法
        # 示例：优化Hurst阈值
        for hurst_threshold in np.linspace(0.01, 0.1, 10):
            params = {'hurst_threshold': hurst_threshold}
            performance = self._evaluate_strategy(data, params)

            if performance > best_performance:
                best_performance = performance
                best_params = params

        return best_params

    def _evaluate_strategy(self, data: pd.DataFrame, params: Dict) -> float:
        """评估策略性能"""
        # 这里简化处理，实际应进行完整回测
        # 返回夏普比率或总收益率

        # 模拟计算
        returns = data['close'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

        return sharpe_ratio


# ==================== 主程序 ====================

def main():
    """主函数"""
    print("混沌理论期货量化交易系统")
    print("=" * 50)

    # 示例：使用混沌指标分析市场
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    prices = np.cumsum(np.random.randn(n_samples) * 0.01) + 100
    returns = np.diff(prices) / prices[:-1]

    # 计算混沌指标
    indicator = ChaosTheoryIndicator()
    hurst = indicator.hurst_exponent(prices)
    fractal_dim = indicator.fractal_dimension(prices)
    approx_entropy = indicator.approximate_entropy(returns)
    lyapunov = indicator.lyapunov_exponent(prices)

    print(f"Hurst指数: {hurst:.3f}")
    print(f"分形维数: {fractal_dim:.3f}")
    print(f"近似熵: {approx_entropy:.3f}")
    print(f"Lyapunov指数: {lyapunov:.3f}")

    # 分析市场状态
    analyzer = MarketStateAnalyzer()
    status = analyzer.analyze(prices, returns)

    print(f"\n市场状态: {status.state.value}")
    print(f"置信度: {status.confidence:.2%}")
    print(f"波动率: {status.volatility:.4f}")
    print(f"趋势强度: {status.trend_strength:.3f}")


if __name__ == "__main__":
    main()