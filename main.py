import asyncio
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
import requests
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
import logging
import json
import feedparser
from textblob import TextBlob
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy import stats
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
import random
import math
import time
from bs4 import BeautifulSoup
import threading
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO

warnings.filterwarnings('ignore')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


TOKEN = "8731427214:AAGounZ1AVwxmL-HXGyBiNPtjY97hRJd8CY"
TWELVE_KEY = "dfbecf637ca6480088d2b584eeaa2914"
DB_NAME = "xauusd_ai.db"
MODEL_FILE = "xauusd_model.pkl"
REGIME_FILE = "market_regime.pkl"
MEMORY_FILE = "trade_memory.pkl"
RISK_FILE = "risk_state.pkl"
CALIBRATION_FILE = "confidence_calibration.pkl"
MACRO_FILE = "macro_state.pkl"


class MacroEconomicData:
    def __init__(self):
        self.data = {}
        self.last_update = None
        self.cache_duration = 300

    def fetch_usd_index(self):
        try:
            url = f"https://api.twelvedata.com/time_series"
            params = {
                "symbol": "DXY",
                "interval": "15min",
                "outputsize": 2,
                "apikey": TWELVE_KEY
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if "values" in data and len(data["values"]) >= 2:
                current = float(data["values"][0]["close"])
                previous = float(data["values"][1]["close"])
                change = (current - previous) / previous
                return {
                    'value': current,
                    'change': change,
                    'trend': 'bullish' if change > 0 else 'bearish',
                    'strength': abs(change) * 100
                }
        except Exception as e:
            logger.error(f"USD Index fetch error: {e}")
        return {'value': 100, 'change': 0, 'trend': 'neutral', 'strength': 0}

    def fetch_treasury_yields(self):
        yields = {}
        try:
            url = f"https://api.twelvedata.com/time_series"
            for symbol, name in [("US10Y", "10Y"), ("US2Y", "2Y"), ("US30Y", "30Y")]:
                try:
                    params = {
                        "symbol": symbol,
                        "interval": "1h",
                        "outputsize": 2,
                        "apikey": TWELVE_KEY
                    }
                    response = requests.get(url, params=params, timeout=10)
                    data = response.json()
                    if "values" in data and len(data["values"]) >= 2:
                        current = float(data["values"][0]["close"])
                        previous = float(data["values"][1]["close"])
                        yields[name] = {
                            'value': current,
                            'change': ((current - previous) / previous) * 100,
                            'trend': 'rising' if current > previous else 'falling'
                        }
                except:
                    continue
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"Treasury fetch error: {e}")

        if not yields:
            yields = {'10Y': {'value': 4.0, 'change': 0, 'trend': 'neutral'},
                     '2Y': {'value': 4.5, 'change': 0, 'trend': 'neutral'}}
        return yields

    def fetch_sp500(self):
        try:
            url = f"https://api.twelvedata.com/time_series"
            params = {
                "symbol": "SPX",
                "interval": "15min",
                "outputsize": 2,
                "apikey": TWELVE_KEY
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if "values" in data and len(data["values"]) >= 2:
                current = float(data["values"][0]["close"])
                previous = float(data["values"][1]["close"])
                change = (current - previous) / previous
                return {
                    'value': current,
                    'change': change,
                    'trend': 'bullish' if change > 0 else 'bearish',
                    'risk_on': change > 0
                }
        except Exception as e:
            logger.error(f"SP500 fetch error: {e}")
        return {'value': 4000, 'change': 0, 'trend': 'neutral', 'risk_on': True}

    def fetch_vix(self):
        try:
            url = f"https://api.twelvedata.com/time_series"
            params = {
                "symbol": "VIX",
                "interval": "15min",
                "outputsize": 2,
                "apikey": TWELVE_KEY
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if "values" in data and len(data["values"]) >= 2:
                current = float(data["values"][0]["close"])
                previous = float(data["values"][1]["close"])
                return {
                    'value': current,
                    'change': ((current - previous) / previous) * 100,
                    'fear_level': 'high' if current > 25 else ('extreme' if current > 35 else 'normal'),
                    'gold_support': current > 20
                }
        except Exception as e:
            logger.error(f"VIX fetch error: {e}")
        return {'value': 20, 'change': 0, 'fear_level': 'normal', 'gold_support': False}

    def fetch_real_rates_proxy(self):
        try:
            yields = self.fetch_treasury_yields()
            ten_year = yields.get('10Y', {}).get('value', 4.0)
            real_rate = ten_year - 2.5
            return {
                'real_rate': real_rate,
                'gold_bearish': real_rate > 2.0,
                'gold_bullish': real_rate < 0.5
            }
        except:
            return {'real_rate': 1.5, 'gold_bearish': False, 'gold_bullish': False}

    def fetch_yield_curve(self):
        try:
            yields = self.fetch_treasury_yields()
            ten_year = yields.get('10Y', {}).get('value', 4.0)
            two_year = yields.get('2Y', {}).get('value', 4.5)

            spread = ten_year - two_year
            inverted = spread < 0

            return {
                'spread': spread,
                'inverted': inverted,
                'recession_signal': inverted,
                'steepening': spread > 1.0
            }
        except:
            return {'spread': 0.5, 'inverted': False, 'recession_signal': False, 'steepening': True}

    def calculate_gold_drivers_score(self):
        try:
            usd = self.fetch_usd_index()
            yields = self.fetch_treasury_yields()
            sp500 = self.fetch_sp500()
            vix = self.fetch_vix()
            real_rates = self.fetch_real_rates_proxy()
            yield_curve = self.fetch_yield_curve()

            score = 0.0
            factors = {}

            usd_factor = -1 if usd['trend'] == 'bullish' else (1 if usd['trend'] == 'bearish' else 0)
            score += usd_factor * 0.25
            factors['usd'] = usd_factor

            ten_year_change = yields.get('10Y', {}).get('change', 0)
            yield_factor = -1 if ten_year_change > 0.5 else (1 if ten_year_change < -0.5 else 0)
            score += yield_factor * 0.20
            factors['yields'] = yield_factor

            real_factor = 1 if real_rates['gold_bullish'] else (-1 if real_rates['gold_bearish'] else 0)
            score += real_factor * 0.20
            factors['real_rates'] = real_factor

            vix_factor = 1 if vix['gold_support'] else 0
            score += vix_factor * 0.15
            factors['vix'] = vix_factor

            curve_factor = 1 if yield_curve['recession_signal'] else 0
            score += curve_factor * 0.10
            factors['yield_curve'] = curve_factor

            risk_factor = -1 if sp500['risk_on'] else 1
            score += risk_factor * 0.10
            factors['risk_sentiment'] = risk_factor

            self.data = {
                'usd': usd,
                'yields': yields,
                'sp500': sp500,
                'vix': vix,
                'real_rates': real_rates,
                'yield_curve': yield_curve,
                'score': np.clip(score, -1, 1),
                'factors': factors,
                'timestamp': datetime.now()
            }

            self.last_update = datetime.now()
            return self.data

        except Exception as e:
            logger.error(f"Macro calculation error: {e}")
            return {'score': 0, 'factors': {}, 'error': str(e)}

    def get_signal(self):
        if not self.last_update or (datetime.now() - self.last_update).seconds > self.cache_duration:
            self.calculate_gold_drivers_score()

        score = self.data.get('score', 0)

        if score > 0.5:
            return 'strong_bullish', score
        elif score > 0.2:
            return 'bullish', score
        elif score < -0.5:
            return 'strong_bearish', score
        elif score < -0.2:
            return 'bearish', score
        return 'neutral', score

    def save(self, filepath):
        joblib.dump(self.data, filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            try:
                self.data = joblib.load(filepath)
            except:
                pass


class AdvancedLiquidityAnalyzer:
    def __init__(self):
        self.liquidity_history = deque(maxlen=500)
        self.volume_profile = {}
        self.order_flow_imbalance = 0
        self.liquidity_zones = []

    def calculate_volume_delta(self, df: pd.DataFrame) -> pd.Series:
        if 'volume' not in df.columns:
            return pd.Series([0] * len(df))

        close = df['close']
        open_p = df['open']
        high = df['high']
        low = df['low']
        volume = df['volume']

        range_val = high - low
        range_val = range_val.replace(0, 1e-10)

        buy_vol = volume * ((close - low) / range_val)
        sell_vol = volume * ((high - close) / range_val)

        delta = buy_vol - sell_vol
        return delta

    def detect_liquidity_voids(self, df: pd.DataFrame) -> List[Dict]:
        if len(df) < 20:
            return []

        highs = df['high'].values
        lows = df['low'].values

        liquidity_voids = []

        for i in range(1, len(df) - 1):
            prev_high = highs[i-1]
            curr_low = lows[i]
            prev_low = lows[i-1]
            curr_high = highs[i]

            if curr_high < prev_low:
                void_size = prev_low - curr_high
                liquidity_voids.append({
                    'type': 'bearish_void',
                    'start': prev_low,
                    'end': curr_high,
                    'size': void_size,
                    'index': i
                })

            if curr_low > prev_high:
                void_size = curr_low - prev_high
                liquidity_voids.append({
                    'type': 'bullish_void',
                    'start': prev_high,
                    'end': curr_low,
                    'size': void_size,
                    'index': i
                })

        return liquidity_voids

    def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> Dict:
        if len(df) < 20 or 'volume' not in df.columns:
            return {'poc': df['close'].iloc[-1] if len(df) > 0 else 0, 'value_area_high': 0, 'value_area_low': 0}

        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / bins if price_range > 0 else 1

        volume_by_price = defaultdict(float)

        for idx, row in df.iterrows():
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            bin_price = round(typical_price / bin_size) * bin_size if bin_size > 0 else typical_price
            volume_by_price[bin_price] += row.get('volume', 0)

        poc = max(volume_by_price.items(), key=lambda x: x[1])[0] if volume_by_price else df['close'].iloc[-1]

        total_volume = sum(volume_by_price.values())
        target_volume = total_volume * 0.70

        sorted_prices = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
        cumulative_volume = 0
        value_area_prices = []

        for price, vol in sorted_prices:
            cumulative_volume += vol
            value_area_prices.append(price)
            if cumulative_volume >= target_volume:
                break

        value_area_high = max(value_area_prices) if value_area_prices else poc
        value_area_low = min(value_area_prices) if value_area_prices else poc

        return {
            'poc': poc,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'volume_by_price': dict(volume_by_price)
        }

    def detect_stop_clusters(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        if len(df) < lookback:
            return {'above': [], 'below': []}

        recent_highs = df['high'].tail(lookback).values
        recent_lows = df['low'].tail(lookback).values

        stop_levels_above = []
        stop_levels_below = []

        for high in recent_highs:
            rounded = round(high, 1)
            if rounded > df['close'].iloc[-1]:
                stop_levels_above.append(rounded)

        for low in recent_lows:
            rounded = round(low, 1)
            if rounded < df['close'].iloc[-1]:
                stop_levels_below.append(rounded)

        from collections import Counter
        common_above = Counter(stop_levels_above).most_common(3)
        common_below = Counter(stop_levels_below).most_common(3)

        return {
            'above': [level for level, count in common_above],
            'below': [level for level, count in common_below],
            'sweep_risk_above': any(count > 3 for _, count in common_above),
            'sweep_risk_below': any(count > 3 for _, count in common_below)
        }

    def calculate_liquidity_score(self, df: pd.DataFrame) -> Dict:
        if len(df) < 20:
            return {'score': 0.5, 'quality': 'unknown'}

        avg_volume = df['volume'].mean() if 'volume' in df.columns else 1000
        volume_trend = df['volume'].iloc[-5:].mean() / avg_volume if avg_volume > 0 else 1

        spreads = (df['high'] - df['low']) / df['close']
        avg_spread = spreads.mean()
        spread_tightness = 1 / (1 + avg_spread * 100)

        if 'volume' in df.columns and avg_volume > 0:
            price_volatility = df['close'].pct_change().std()
            volume_efficiency = price_volatility / (avg_volume / 1000)
            depth_score = 1 / (1 + volume_efficiency)
        else:
            depth_score = 0.5

        liquidity_score = (volume_trend * 0.3 + spread_tightness * 0.4 + depth_score * 0.3)
        liquidity_score = np.clip(liquidity_score, 0, 1)

        quality = 'excellent' if liquidity_score > 0.8 else ('good' if liquidity_score > 0.6 else ('poor' if liquidity_score < 0.4 else 'average'))

        return {
            'score': liquidity_score,
            'quality': quality,
            'volume_trend': volume_trend,
            'spread_tightness': spread_tightness,
            'depth_score': depth_score
        }

    def get_optimal_entry_zones(self, df: pd.DataFrame, direction: str) -> Dict:
        if len(df) < 20:
            current_price = df['close'].iloc[-1] if len(df) > 0 else 0
            return {
                'optimal_entry': current_price,
                'max_slippage': 0,
                'liquidity_quality': 'unknown',
                'stop_risk': 'low'
            }

        vol_profile = self.calculate_volume_profile(df)
        stops = self.detect_stop_clusters(df)
        liquidity = self.calculate_liquidity_score(df)

        current_price = df['close'].iloc[-1]
        atr_series = calculate_atr(df['high'], df['low'], df['close'], 14)
        atr = atr_series.iloc[-1] if len(atr_series) > 0 else 0
        if pd.isna(atr):
            atr = 0

        zones = {
            'optimal_entry': current_price,
            'max_slippage': atr * 0.3,
            'liquidity_quality': liquidity['quality'],
            'stop_risk': 'low'
        }

        if direction == "BUY":
            if stops['below']:
                nearest_stop = max([s for s in stops['below'] if s < current_price], default=current_price - atr)
                zones['optimal_entry'] = nearest_stop + (atr * 0.2)
                zones['stop_cluster_below'] = nearest_stop
                zones['stop_risk'] = 'high' if stops['sweep_risk_below'] else 'low'

            if current_price > vol_profile['value_area_low']:
                zones['value_area_support'] = vol_profile['value_area_low']

        else:
            if stops['above']:
                nearest_stop = min([s for s in stops['above'] if s > current_price], default=current_price + atr)
                zones['optimal_entry'] = nearest_stop - (atr * 0.2)
                zones['stop_cluster_above'] = nearest_stop
                zones['stop_risk'] = 'high' if stops['sweep_risk_above'] else 'low'

            if current_price < vol_profile['value_area_high']:
                zones['value_area_resistance'] = vol_profile['value_area_high']

        return zones


class DataQualityFilter:
    def __init__(self):
        self.quality_stats = deque(maxlen=1000)

    def validate_candle(self, row: pd.Series, atr: float) -> Tuple[bool, str]:
        high, low, open_p, close = row['high'], row['low'], row['open'], row['close']
        volume = row.get('volume', 1000)

        if any(x <= 0 for x in [high, low, open_p, close]):
            return False, "zero_price"

        if high < low:
            return False, "inverted_range"

        body = abs(close - open_p)
        wick_top = high - max(open_p, close)
        wick_bottom = min(open_p, close) - low
        range_val = high - low

        if range_val == 0:
            return False, "no_range"

        if body / (range_val + 1e-10) < 0.1 and range_val > 3 * atr:
            return False, "manipulation_doji"

        if volume > 5000 and body / (range_val + 1e-10) < 0.2:
            return False, "low_liquidity_spike"

        return True, "valid"

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 10:
            return df

        atr = calculate_atr(df['high'], df['low'], df['close'], 14).iloc[-1]
        valid_mask = []

        for idx, row in df.iterrows():
            is_valid, reason = self.validate_candle(row, atr)
            valid_mask.append(is_valid)
            if not is_valid:
                self.quality_stats.append({
                    'timestamp': idx,
                    'reason': reason,
                    'candle': row.to_dict()
                })

        clean_df = df[valid_mask].copy()
        removed = len(df) - len(clean_df)

        if removed > 0:
            logger.info(f"DataQuality: Removed {removed} anomalous candles")

        return clean_df


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_vwap(df: pd.DataFrame, anchor='session') -> pd.Series:
    if len(df) == 0:
        return pd.Series([], dtype=float)
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return typical_price

    if anchor == 'session':
        df = df.copy()
        df['hour'] = pd.to_datetime(df.index if 'datetime' not in df.columns else df['datetime']).dt.hour
        df['session_start'] = ((df['hour'] == 8) | (df['hour'] == 13)).cumsum()

        vwap = []
        for session_id, group in df.groupby('session_start'):
            typical_price = (group['high'] + group['low'] + group['close']) / 3
            cumulative_tp_vol = (typical_price * group['volume']).cumsum()
            cumulative_vol = group['volume'].cumsum()
            session_vwap = cumulative_tp_vol / (cumulative_vol + 1e-10)
            vwap.extend(session_vwap.values)

        return pd.Series(vwap, index=df.index)
    else:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tp_vol = (typical_price * df['volume']).cumsum()
        cumulative_vol = df['volume'].cumsum()
        return cumulative_tp_vol / (cumulative_vol + 1e-10)

def calculate_vwap_bands(df: pd.DataFrame, std_mult=1.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if len(df) == 0:
        empty = pd.Series([], dtype=float)
        return empty, empty, empty
    vwap = calculate_vwap(df)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    variance = ((typical_price - vwap) ** 2).rolling(20).mean()
    std = np.sqrt(variance)
    upper = vwap + (std * std_mult)
    lower = vwap - (std * std_mult)
    return vwap, upper, lower

def calculate_psar(high, low, close, step=0.02, max_step=0.2):
    psar = close.copy()
    bull = True
    af = step
    ep = low.iloc[0]
    psar.iloc[0] = close.iloc[0]

    for i in range(1, len(close)):
        if bull:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            if low.iloc[i] < psar.iloc[i]:
                bull = False
                psar.iloc[i] = ep
                af = step
                ep = high.iloc[i]
            elif high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + step, max_step)
        else:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            if high.iloc[i] > psar.iloc[i]:
                bull = True
                psar.iloc[i] = ep
                af = step
                ep = low.iloc[i]
            elif low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + step, max_step)
    return psar

def calculate_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = dx.rolling(window=period).mean()
    return adx

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_money_flow_index(df, period=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + positive_sum / (negative_sum + 1e-10)))
    return mfi

def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()

    k = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-10))
    d = k.rolling(window=d_period).mean()

    return k, d

def calculate_williams_r(df, period=14):
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()

    williams_r = -100 * ((high_max - df['close']) / (high_max - low_min + 1e-10))
    return williams_r

def calculate_cmf(df, period=20):
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    mfv = mfm * df['volume']
    cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return cmf

def calculate_cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=period).mean()
    mean_deviation = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mean_deviation)
    return cci


def build_feature_vector(df):
    return np.array([
        df['rsi'].iloc[-1],
        df['macd'].iloc[-1],
        df['atr'].iloc[-1],
        df['bop'].iloc[-1],
        df['adx'].iloc[-1],
        df['vwap_dist'].iloc[-1],
        df['momentum'].iloc[-1],
        df['volatility'].iloc[-1],
        df['volume'].iloc[-1],
        df['spread'].iloc[-1],
        df['price_change'].iloc[-1],
        df['high_low_range'].iloc[-1],
        df['ema_diff'].iloc[-1],
        df['stochastic'].iloc[-1],
        df['cci'].iloc[-1],
        df['williams_r'].iloc[-1],
        df['trend_strength'].iloc[-1],
        df['liquidity'].iloc[-1],
        df['orderflow'].iloc[-1],
        df['noise'].iloc[-1],
    ])


class LatentPatternDiscovery:
    def __init__(self, input_dim=20, latent_dim=8, lr=0.001):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr

        self.encoder_weights = np.random.randn(input_dim, latent_dim) * 0.1
        self.decoder_weights = np.random.randn(latent_dim, input_dim) * 0.1

    def forward(self, x):
        self.latent = np.tanh(x @ self.encoder_weights)
        self.reconstructed = self.latent @ self.decoder_weights
        return self.latent

    def train(self, x):
        latent = self.forward(x)

        error = self.reconstructed - x

        d_decoder = self.latent.T @ error
        d_encoder = x.T @ ((error @ self.decoder_weights.T) * (1 - self.latent**2))

        self.decoder_weights -= self.lr * d_decoder
        self.encoder_weights -= self.lr * d_encoder

        return latent

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        return self.forward(features)

    def get_novelty_score(self, features: np.ndarray) -> float:
        latent = self.forward(features.reshape(1, -1) if len(features.shape) == 1 else features)
        reconstructed = latent @ self.decoder_weights
        mse = np.mean((features.reshape(1, -1) - reconstructed) ** 2)
        return float(mse)


class AdvancedMarketRegimeDetector:
    def __init__(self):
        self.regime = 'unknown'
        self.sub_regime = 'normal'
        self.volatility_regime = 'normal'
        self.trend_strength = 0
        self.adx_value = 0
        self.regime_history = deque(maxlen=100)
        self.manipulation_detector = ManipulationDetector()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.adaptive_thresholds = {
            'trending': 0.3, 
            'ranging': 0.15,
            'manipulation': 0.05,
            'expansion': 0.4
        }

    def detect(self, df: pd.DataFrame) -> str:
        if len(df) < 50:
            return 'ranging'

        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252) if len(returns) >= 20 else 0

        adx_series = calculate_adx(df)
        adx = adx_series.iloc[-1]
        self.adx_value = adx if not pd.isna(adx) else 0
        self.trend_strength = self.adx_value

        is_manip, manip_type = self.manipulation_detector.detect(df)
        if is_manip:
            self.regime = 'manipulation'
            self.sub_regime = manip_type
            return self.regime

        vol_change = volatility / (returns.rolling(50).std().iloc[-1] * np.sqrt(252) + 1e-10)
        if vol_change > 2.0 and adx > 30:
            self.regime = 'expansion'
            self.sub_regime = 'volatility_breakout'
            return self.regime

        liquidity_score = self.liquidity_analyzer.score(df)
        if liquidity_score < 0.3:
            self.regime = 'low_liquidity'
            self.sub_regime = 'thin_market'
            return self.regime

        price_range = (df['high'].rolling(20).max().iloc[-1] - df['low'].rolling(20).min().iloc[-1]) / df['close'].iloc[-1]

        if self.adx_value > 25 and volatility > 0.15:
            self.regime = 'trending'
        elif self.adx_value < 20 and price_range < 0.02:
            self.regime = 'ranging'
        elif volatility > 0.25:
            self.regime = 'volatile'
        else:
            self.regime = 'mixed'

        self.regime_history.append(self.regime)
        self._adapt_thresholds()
        return self.regime

    def _adapt_thresholds(self):
        if len(self.regime_history) < 20:
            return
        recent = list(self.regime_history)[-20:]
        trending_ratio = recent.count('trending') / len(recent)

        if trending_ratio > 0.6:
            self.adaptive_thresholds['trending'] = max(0.22, self.adaptive_thresholds['trending'] - 0.01)
        elif trending_ratio < 0.3:
            self.adaptive_thresholds['trending'] = min(0.35, self.adaptive_thresholds['trending'] + 0.01)

    def save(self, filepath):
        joblib.dump({
            'regime': self.regime,
            'sub_regime': self.sub_regime,
            'adx_value': self.adx_value,
            'trend_strength': self.trend_strength,
            'regime_history': list(self.regime_history),
            'adaptive_thresholds': self.adaptive_thresholds
        }, filepath)


class ManipulationDetector:
    def detect(self, df: pd.DataFrame) -> Tuple[bool, str]:
        if len(df) < 5:
            return False, ""

        last_candle = df.iloc[-1]
        prev_candles = df.iloc[-5:-1]

        recent_high = prev_candles['high'].max()
        recent_low = prev_candles['low'].min()

        if last_candle['low'] < recent_low and last_candle['close'] > recent_low:
            if (recent_low - last_candle['low']) > (last_candle['high'] - last_candle['low']) * 0.6:
                return True, "bullish_stop_hunt"

        if last_candle['high'] > recent_high and last_candle['close'] < recent_high:
            if (last_candle['high'] - recent_high) > (last_candle['high'] - last_candle['low']) * 0.6:
                return True, "bearish_stop_hunt"

        return False, ""


class LiquidityAnalyzer:
    def score(self, df: pd.DataFrame) -> float:
        if len(df) < 20:
            return 1.0

        volume = df['volume'].iloc[-20:]
        spread = (df['high'] - df['low']).iloc[-20:]

        vol_mean = volume.mean()
        spread_mean = spread.mean()

        if vol_mean == 0:
            return 0.5

        vol_score = min(vol_mean / 1000, 1.0)
        spread_score = max(0, 1 - (spread_mean / 2.0))

        return (vol_score + spread_score) / 2


class PatternClustering:
    def __init__(self, n_clusters=50):
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.pattern_memory = defaultdict(lambda: {'wins': 0, 'losses': 0, 'trades': []})
        self.is_fitted = False
        self.feature_importance = np.ones(20) / 20

    def extract_features(self, df: pd.DataFrame, idx: int = -1) -> np.ndarray:
        return build_feature_vector(df)

    def update_importance(self, features: np.ndarray, result: int):
        if result == 1:
            self.feature_importance += np.abs(features) * 0.01
            self.feature_importance = np.clip(self.feature_importance, 0.01, 1.0)
            self.feature_importance /= self.feature_importance.sum()

    def get_similarity_score(self, features1: np.ndarray, features2: np.ndarray) -> float:
        weighted_f1 = features1 * self.feature_importance
        weighted_f2 = features2 * self.feature_importance

        if np.linalg.norm(weighted_f1) == 0 or np.linalg.norm(weighted_f2) == 0:
            return 0

        return cosine_similarity([weighted_f1], [weighted_f2])[0][0]

    def find_similar_patterns(self, features: np.ndarray, top_k=5) -> List[Dict]:
        if not self.pattern_memory:
            return []

        similarities = []
        for pattern_id, data in self.pattern_memory.items():
            if 'features' in data:
                sim = self.get_similarity_score(features, data['features'])
                similarities.append((pattern_id, sim, data))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [{'pattern_id': s[0], 'similarity': s[1], 'data': s[2]} for s in similarities[:top_k]]

    def cluster_and_store(self, features: np.ndarray, result: int, trade_id: str):
        if not self.is_fitted and len(self.pattern_memory) > 100:
            all_features = [p['features'] for p in self.pattern_memory.values() if 'features' in p]
            if len(all_features) >= 50:
                self.kmeans.partial_fit(all_features)
                self.is_fitted = True

        cluster_id = None
        if self.is_fitted:
            cluster_id = self.kmeans.predict([features])[0]

        pattern_key = f"cluster_{cluster_id}" if cluster_id is not None else trade_id

        self.pattern_memory[pattern_key]['features'] = features
        self.pattern_memory[pattern_key]['trades'].append(result)

        if result == 1:
            self.pattern_memory[pattern_key]['wins'] += 1
        else:
            self.pattern_memory[pattern_key]['losses'] += 1

        self.update_importance(features, result)

    def get_cluster_success_rate(self, features: np.ndarray) -> float:
        similar = self.find_similar_patterns(features, top_k=3)
        if not similar:
            return 0.5

        total_wins = sum(s['data']['wins'] for s in similar)
        total_losses = sum(s['data']['losses'] for s in similar)
        total = total_wins + total_losses

        if total == 0:
            return 0.5

        weighted_success = sum(s['similarity'] * (s['data']['wins'] / (s['data']['wins'] + s['data']['losses'] + 1e-10)) 
                              for s in similar)
        weight_sum = sum(s['similarity'] for s in similar)

        return weighted_success / (weight_sum + 1e-10)


class TradeSequenceIntelligence:
    def __init__(self, memory_maxlen=1000):
        self.trade_history = deque(maxlen=memory_maxlen)
        self.current_streak = 0
        self.streak_type = None
        self.serial_correlation = 0

    def add_trade(self, result: int, direction: str, features: Dict):
        self.trade_history.append({
            'result': result,
            'direction': direction,
            'features': features,
            'timestamp': datetime.now()
        })

        if result == 1:
            if self.streak_type == 'win':
                self.current_streak += 1
            else:
                self.streak_type = 'win'
                self.current_streak = 1
        else:
            if self.streak_type == 'loss':
                self.current_streak += 1
            else:
                self.streak_type = 'loss'
                self.current_streak = 1

        self._calculate_serial_correlation()

    def _calculate_serial_correlation(self):
        if len(self.trade_history) < 10:
            return

        results = [t['result'] for t in self.trade_history]
        x = results[:-1]
        y = results[1:]

        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
            self.serial_correlation = np.corrcoef(x, y)[0, 1]

    def get_streak_adjustment(self) -> float:
        if self.current_streak >= 3 and self.streak_type == 'loss':
            return 0.5
        elif self.current_streak >= 3 and self.streak_type == 'win':
            return 1.2
        return 1.0

    def get_regime_bias(self) -> str:
        if len(self.trade_history) < 20:
            return 'neutral'

        recent = list(self.trade_history)[-20:]
        wins = sum(1 for t in recent if t['result'] == 1)
        win_rate = wins / len(recent)

        if win_rate > 0.6:
            return 'hot'
        elif win_rate < 0.4:
            return 'cold'
        return 'neutral'


@dataclass
class FailureRecord:
    reason: str
    count: int = 0
    setups: List[Dict] = field(default_factory=list)
    toxicity_score: float = 0.0


class DeepFailureAnalyzer:
    def __init__(self):
        self.failure_categories = {
            'early_exit': FailureRecord('early_exit'),
            'late_exit': FailureRecord('late_exit'),
            'wrong_direction': FailureRecord('wrong_direction'),
            'stop_hunt': FailureRecord('stop_hunt'),
            'choppy_conditions': FailureRecord('choppy_conditions'),
            'news_spike': FailureRecord('news_spike'),
            'low_liquidity': FailureRecord('low_liquidity'),
            'against_htf_trend': FailureRecord('against_htf_trend'),
            'fakeout': FailureRecord('fakeout'),
            'overtrading': FailureRecord('overtrading')
        }
        self.toxic_setups = []

    def analyze_failure(self, trade: Dict, market_context: Dict):
        if trade['result'] == 1:
            return

        reasons = []

        if market_context.get('manipulation', False):
            reasons.append('stop_hunt')

        if market_context.get('regime') == 'ranging' and abs(trade['entry'] - trade['exit_price']) < trade.get('atr', 1):
            reasons.append('choppy_conditions')

        if trade['direction'] == 'BUY' and market_context.get('trend_direction') == -1:
            reasons.append('against_htf_trend')
        elif trade['direction'] == 'SELL' and market_context.get('trend_direction') == 1:
            reasons.append('against_htf_trend')

        if trade.get('holding_time', 0) < 3:
            reasons.append('fakeout')

        for reason in reasons:
            self.failure_categories[reason].count += 1
            self.failure_categories[reason].setups.append({
                'features': trade.get('features', {}),
                'timestamp': trade.get('timestamp'),
                'pnl': trade.get('pnl', 0)
            })

        self._update_toxicity_scores()

    def _update_toxicity_scores(self):
        max_count = max(f.count for f in self.failure_categories.values()) + 1

        for key, record in self.failure_categories.items():
            if record.count > 5:
                avg_loss = np.mean([s['pnl'] for s in record.setups if 'pnl' in s]) if record.setups else 0
                record.toxicity_score = (record.count / max_count) * (1 + abs(avg_loss))

    def is_toxic_setup(self, features: Dict, context: Dict) -> Tuple[bool, float]:
        toxicity = 0.0
        reasons = []

        for category, record in self.failure_categories.items():
            if record.toxicity_score > 0.5:
                for failed_setup in record.setups[-10:]:
                    if self._setup_similarity(features, failed_setup['features']) > 0.8:
                        toxicity += record.toxicity_score
                        reasons.append(category)
                        break

        return toxicity > 1.0, min(toxicity / 5.0, 1.0)

    def _setup_similarity(self, f1: Dict, f2: Dict) -> float:
        keys = ['rsi', 'adx', 'trend_strength', 'volatility']
        if not all(k in f1 and k in f2 for k in keys):
            return 0.0

        diff = sum(abs(f1.get(k, 0) - f2.get(k, 0)) for k in keys)
        return max(0, 1 - diff / len(keys))


class ConfidenceCalibrator:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_accuracies = {}
        self.bin_counts = {}
        self.calibration_history = deque(maxlen=1000)

    def update(self, predicted_confidence: float, actual_result: int):
        bin_idx = int(predicted_confidence * self.n_bins)
        bin_idx = min(bin_idx, self.n_bins - 1)

        if bin_idx not in self.bin_accuracies:
            self.bin_accuracies[bin_idx] = 0
            self.bin_counts[bin_idx] = 0

        self.bin_counts[bin_idx] += 1
        self.bin_accuracies[bin_idx] += actual_result

        self.calibration_history.append({
            'predicted': predicted_confidence,
            'actual': actual_result,
            'timestamp': datetime.now()
        })

    def calibrate(self, raw_confidence: float) -> float:
        bin_idx = int(raw_confidence * self.n_bins)
        bin_idx = min(bin_idx, self.n_bins - 1)

        if bin_idx not in self.bin_counts or self.bin_counts[bin_idx] < 10:
            return raw_confidence

        actual_acc = self.bin_accuracies[bin_idx] / self.bin_counts[bin_idx]
        predicted_center = (bin_idx + 0.5) / self.n_bins

        correction = actual_acc - predicted_center
        calibrated = raw_confidence + correction

        return np.clip(calibrated, 0.05, 0.99)

    def get_calibration_report(self) -> Dict:
        report = {}
        for bin_idx in sorted(self.bin_counts.keys()):
            count = self.bin_counts[bin_idx]
            if count > 0:
                actual = self.bin_accuracies[bin_idx] / count
                predicted = (bin_idx + 0.5) / self.n_bins
                report[f"{int(predicted*100)}%"] = {
                    'predicted': predicted,
                    'actual': actual,
                    'samples': count,
                    'bias': actual - predicted
                }
        return report


class ExplorationController:
    def __init__(self, epsilon_start=0.3, epsilon_decay=0.995, epsilon_min=0.05):
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.exploration_count = 0
        self.exploitation_count = 0
        self.new_setups_tested = []

    def should_explore(self) -> bool:
        if random.random() < self.epsilon:
            self.exploration_count += 1
            return True
        self.exploitation_count += 1
        return False

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_exploration_stats(self) -> Dict:
        return {
            'epsilon': self.epsilon,
            'exploration_rate': self.exploration_count / (self.exploration_count + self.exploitation_count + 1),
            'new_setups_tested': len(self.new_setups_tested)
        }


class AdvancedRiskIntelligence:
    def __init__(self):
        self.equity_curve = deque(maxlen=500)
        self.drawdown_history = deque(maxlen=100)
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.volatility_adjusted_sizing = 1.0
        self.news_risk_multiplier = 1.0
        self.kelly_fraction = 0.25
        self.base_risk = 0.01
        self.current_risk = self.base_risk

    def update_equity(self, trade_result: float):
        prev_equity = self.equity_curve[-1] if self.equity_curve else 10000
        new_equity = prev_equity * (1 + trade_result)
        self.equity_curve.append(new_equity)

        peak = max(self.equity_curve)
        self.current_drawdown = (peak - new_equity) / peak
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        self.drawdown_history.append(self.current_drawdown)

        if self.current_drawdown > 0.1:
            self.current_risk = self.base_risk * 0.5
        elif self.current_drawdown > 0.05:
            self.current_risk = self.base_risk * 0.75
        else:
            self.current_risk = self.base_risk

    def calculate_position_size(self, volatility: float, atr: float, 
                              confidence: float, streak_adjustment: float) -> float:
        vol_factor = 1 / (1 + volatility * 10)

        win_rate = confidence
        payoff = 2.0
        kelly = (win_rate * payoff - (1 - win_rate)) / payoff if payoff > 0 else 0
        kelly = max(0, min(kelly, 0.5))

        size = (self.current_risk * self.kelly_fraction * 
                kelly * vol_factor * streak_adjustment * self.news_risk_multiplier)

        return np.clip(size, 0.001, 0.1)

    def adjust_for_news(self, sentiment_score: float, impact_score: float):
        if impact_score > 0.7:
            self.news_risk_multiplier = 0.5
        elif abs(sentiment_score) > 0.5:
            self.news_risk_multiplier = 0.75
        else:
            self.news_risk_multiplier = 1.0

    def get_risk_report(self) -> Dict:
        return {
            'current_risk_pct': self.current_risk * 100,
            'current_drawdown_pct': self.current_drawdown * 100,
            'max_drawdown_pct': self.max_drawdown * 100,
            'volatility_factor': self.volatility_adjusted_sizing,
            'news_multiplier': self.news_risk_multiplier
        }


class MultiTimeframeIntelligence:
    def __init__(self):
        self.htf_bias = 'neutral'
        self.htf_strength = 0
        self.alignment_score = 0

    def analyze_timeframes(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, 
                          df_4h: pd.DataFrame, df_daily: Optional[pd.DataFrame] = None) -> Dict:
        if len(df_4h) >= 50:
            ema_50_4h = calculate_ema(df_4h['close'], 50).iloc[-1]
            ema_200_4h = calculate_ema(df_4h['close'], 200).iloc[-1] if len(df_4h) >= 200 else ema_50_4h
            trend_4h = 1 if ema_50_4h > ema_200_4h else -1
            adx_4h = calculate_adx(df_4h).iloc[-1] if len(df_4h) > 14 else 0
        else:
            trend_4h = 0
            adx_4h = 0

        if len(df_1h) >= 50:
            ema_50_1h = calculate_ema(df_1h['close'], 50).iloc[-1]
            trend_1h = 1 if ema_50_1h > calculate_ema(df_1h['close'], 200).iloc[-1] else -1
        else:
            trend_1h = 0

        ema_20_15m = calculate_ema(df_15m['close'], 20).iloc[-1]
        price_15m = df_15m['close'].iloc[-1]

        trends = [trend_4h, trend_1h]
        if all(t == 1 for t in trends):
            self.htf_bias = 'bullish'
            self.htf_strength = adx_4h / 50
        elif all(t == -1 for t in trends):
            self.htf_bias = 'bearish'
            self.htf_strength = adx_4h / 50
        else:
            self.htf_bias = 'mixed'
            self.htf_strength = 0.3

        pullback_buy = (self.htf_bias == 'bullish' and 
                       price_15m < ema_20_15m and 
                       price_15m > calculate_ema(df_15m['close'], 50).iloc[-1])

        pullback_sell = (self.htf_bias == 'bearish' and 
                        price_15m > ema_20_15m and 
                        price_15m < calculate_ema(df_15m['close'], 50).iloc[-1])

        self.alignment_score = (self.htf_strength * 
                               (1 if pullback_buy or pullback_sell else 0.5))

        return {
            'htf_bias': self.htf_bias,
            'htf_strength': self.htf_strength,
            'pullback_buy': pullback_buy,
            'pullback_sell': pullback_sell,
            'alignment_score': self.alignment_score,
            'execution_valid': (self.htf_bias == 'bullish' and not pullback_sell) or 
                             (self.htf_bias == 'bearish' and not pullback_buy) or
                             (self.htf_bias == 'mixed')
        }

    def filter_trade(self, direction: str) -> Tuple[bool, str]:
        if self.htf_bias == 'mixed':
            return True, "mixed_ok"

        if direction == 'BUY' and self.htf_bias == 'bearish' and self.htf_strength > 0.6:
            return False, "against_strong_downtrend"

        if direction == 'SELL' and self.htf_bias == 'bullish' and self.htf_strength > 0.6:
            return False, "against_strong_uptrend"

        return True, "aligned"


class Strategy:
    def __init__(self, name: str, strategy_type: str):
        self.name = name
        self.type = strategy_type
        self.performance = deque(maxlen=50)
        self.weight = 1.0
        self.active = True
        self.win_rate = 0.5
        self.profit_factor = 1.0

    def update_performance(self, pnl: float):
        self.performance.append(pnl)
        if len(self.performance) >= 10:
            wins = sum(1 for p in self.performance if p > 0)
            self.win_rate = wins / len(self.performance)
            losses = sum(abs(p) for p in self.performance if p < 0)
            profits = sum(p for p in self.performance if p > 0)
            self.profit_factor = profits / (losses + 1e-10)

    def calculate_score(self) -> float:
        if len(self.performance) < 5:
            return 0.5

        returns = list(self.performance)
        if np.std(returns) == 0:
            return 0.5

        sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
        score = (self.win_rate * 0.4 + 
                min(self.profit_factor / 3, 1) * 0.4 + 
                np.clip(sharpe, -1, 1) * 0.2)
        return np.clip(score, 0.1, 2.0)


class StrategyCompetition:
    def __init__(self):
        self.strategies = {}
        self.selected_strategy = None
        self.competition_results = deque(maxlen=100)

    def add_strategy(self, name: str, strategy_type: str):
        self.strategies[name] = Strategy(name, strategy_type)

    def select_strategy(self, regime: str, exploration: bool = False) -> Optional[Strategy]:
        if exploration:
            return random.choice(list(self.strategies.values()))

        scores = {}
        for name, strat in self.strategies.items():
            if strat.active:
                scores[name] = strat.calculate_score()

        if not scores:
            return None

        exp_scores = {k: np.exp(v) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probs = {k: v/total for k, v in exp_scores.items()}

        selected = np.random.choice(list(probs.keys()), p=list(probs.values()))
        self.selected_strategy = self.strategies[selected]
        return self.selected_strategy

    def update_strategy(self, name: str, pnl: float):
        if name in self.strategies:
            self.strategies[name].update_performance(pnl)

    def get_allocation(self) -> Dict[str, float]:
        total_score = sum(s.calculate_score() for s in self.strategies.values())
        if total_score == 0:
            return {name: 1/len(self.strategies) for name in self.strategies}

        return {name: s.calculate_score()/total_score 
                for name, s in self.strategies.items()}


class MetaLearner:
    def __init__(self):
        self.strategy_scores = defaultdict(float)

    def update(self, strategy, profit):
        self.strategy_scores[strategy] += profit

    def get_best(self):
        return max(self.strategy_scores, key=self.strategy_scores.get)


class MetaLearnerV2:
    def __init__(self):
        self.strategy_weights = {
            'momentum': 0.2, 
            'mean_reversion': 0.2, 
            'breakout': 0.2, 
            'ml': 0.2,
            'smc': 0.2
        }
        self.learning_rates = {k: 0.1 for k in self.strategy_weights}
        self.performance_history = {k: deque(maxlen=50) for k in self.strategy_weights}
        self.regime_strategy_map = {
            'trending': 'momentum',
            'ranging': 'mean_reversion',
            'volatile': 'breakout',
            'manipulation': 'smc'
        }
        self.current_strategy = None
        self.strategy_switch_cooldown = 0

    def adapt_learning_rate(self, strategy: str, performance: float):
        history = list(self.performance_history[strategy])[-10:]
        if len(history) < 5:
            return

        variance = np.var(history)
        if variance > 0.1:
            self.learning_rates[strategy] *= 0.9
        else:
            self.learning_rates[strategy] = min(0.5, self.learning_rates[strategy] * 1.1)

    def switch_strategy(self, regime: str, force: bool = False):
        if self.strategy_switch_cooldown > 0 and not force:
            self.strategy_switch_cooldown -= 1
            return

        best_strategy = self.regime_strategy_map.get(regime, 'ml')

        if self.current_strategy != best_strategy:
            logger.info(f"Meta-Learner: Switching from {self.current_strategy} to {best_strategy} for {regime} regime")
            self.current_strategy = best_strategy
            self.strategy_switch_cooldown = 5

    def update_weights(self, strategy: str, profit: float, regime: str):
        if strategy not in self.performance_history:
            return

        self.performance_history[strategy].append(profit)
        self.adapt_learning_rate(strategy, profit)

        lr = self.learning_rates[strategy]

        recent_perf = np.mean(list(self.performance_history[strategy])[-5:])
        baseline = np.mean([np.mean(list(h)[-5:]) if len(h) >= 5 else 0 
                           for h in self.performance_history.values()])

        gradient = (recent_perf - baseline) * lr
        self.strategy_weights[strategy] += gradient
        self.strategy_weights[strategy] = np.clip(self.strategy_weights[strategy], 0.05, 2.0)

        total = sum(self.strategy_weights.values())
        for s in self.strategy_weights:
            self.strategy_weights[s] /= total

        self.switch_strategy(regime)

    def get_combined_signal(self, signals: Dict[str, float], regime: str) -> float:
        self.switch_strategy(regime)

        combined = 0
        total_weight = 0

        if self.current_strategy and self.current_strategy in signals:
            favored_weight = 2.0
            combined += signals[self.current_strategy] * favored_weight
            total_weight += favored_weight

        for strategy, signal in signals.items():
            if strategy != self.current_strategy:
                weight = self.strategy_weights.get(strategy, 0.2)
                combined += signal * weight
                total_weight += weight

        return combined / (total_weight + 1e-10)


class NewsSentimentAnalyzer:
    def __init__(self):
        self.sentiment_history = deque(maxlen=50)
        self.cache = {}
        self.last_fetch = None
        self.impact_score = 0.5

    def fetch_gold_news(self):
        try:
            feeds = [
                'https://www.investing.com/rss/news_commodities.rss',
                'https://www.dailyfx.com/feeds/all-articles',
                'https://www.fxstreet.com/rss/news',
                'https://www.forexlive.com/feed',
                'https://www.kitco.com/rss/news-gold.xml',
                'https://www.bullionvault.com/gold-news/rss.xml',
                'https://www.gold.org/rss/news',
                'https://www.mining.com/feed/',
                 'https://www.reuters.com/markets/commodities/rss.xml',
                'https://www.cnbc.com/id/100003114/device/rss/rss.html'
            ]

            articles = []
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            for feed_url in feeds:
                try:
                    response = requests.get(feed_url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        feed = feedparser.parse(response.content)
                        for entry in feed.entries[:3]:
                            articles.append({
                                'title': entry.get('title', ''),
                                'description': entry.get('summary', entry.get('description', '')),
                                'published': entry.get('published', entry.get('updated', '')),
                                'source': feed_url.split('/')[2],
                                'impact': self._estimate_impact(entry.get('title', ''))
                            })
                    time.sleep(0.5)
                except Exception as e:
                    logger.debug(f"Feed failed {feed_url}: {e}")
                    continue

            if not articles:
                articles = self._fallback_news_fetch()

            return articles

        except Exception as e:
            logger.error(f"News fetch error: {e}")
            return []

    def _fallback_news_fetch(self):
        articles = []
        try:
            url = "https://www.gold.org/news-and-events/news"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                news_items = soup.find_all('article', limit=5)
                for item in news_items:
                    title = item.get_text()[:100]
                    articles.append({
                        'title': title,
                        'description': title,
                        'published': datetime.now().isoformat(),
                        'source': 'gold.org',
                        'impact': self._estimate_impact(title)
                    })
        except Exception as e:
            logger.debug(f"Fallback fetch failed: {e}")
        return articles

    def _estimate_impact(self, title: str) -> float:
        title_lower = title.lower()

        high_impact = [
            'nfp', 'non-farm', 'fed', 'fomc', 'rate decision', 'interest rate',
            'cpi', 'inflation', 'gdp', 'recession', 'war', 'crisis', 'crash',
            'gold reserves', 'central bank', 'treasury', 'yield', 'dollar collapse',
            'geopolitical', 'conflict', 'sanctions', 'trade war', 'brexit',
            'election', 'stimulus', 'qe', 'quantitative easing', 'taper',
            'default', 'debt ceiling', 'bank failure', 'contagion'
        ]

        medium_impact = [
            'gold', 'xau', 'xauusd', 'bullion', 'precious metal', 'silver',
            'mining', 'production', 'demand', 'supply', 'etf', 'spdr',
            'technical analysis', 'support', 'resistance', 'breakout',
            'treasury', 'bond', 'dollar', 'usd', 'federal reserve', 'dxy'
        ]

        if any(k in title_lower for k in high_impact):
            return 1.0
        elif any(k in title_lower for k in medium_impact):
            return 0.5
        return 0.3

    def analyze_sentiment(self, text):
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            return polarity, subjectivity
        except Exception as e:
            logger.debug(f"Sentiment analysis error: {e}")
            return 0, 0

    def get_combined_sentiment(self):
        if self.last_fetch and (datetime.now() - self.last_fetch).seconds < 300:
            if self.sentiment_history:
                last = self.sentiment_history[-1]
                return last[0], last[1], self.impact_score

        articles = self.fetch_gold_news()
        if not articles:
            logger.info("No news fetched, using neutral sentiment")
            return 0, 0, 0.5

        sentiments = []
        weights = []
        max_impact = 0.5

        for i, article in enumerate(articles[:20]):
            try:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                polarity, subjectivity = self.analyze_sentiment(text)
                impact = article.get('impact', 0.5)
                max_impact = max(max_impact, impact)

                weight = (1.0 / (i + 1)) * impact * subjectivity
                sentiments.append(polarity)
                weights.append(weight)
            except Exception as e:
                continue

        if not sentiments:
            return 0, 0, 0.5

        avg_sentiment = np.average(sentiments, weights=weights) if sum(weights) > 0 else 0
        confidence = np.mean([abs(s) for s in sentiments]) if sentiments else 0
        self.impact_score = max_impact

        result = (avg_sentiment, confidence, max_impact)
        self.sentiment_history.append(result)
        self.last_fetch = datetime.now()

        logger.info(f"News sentiment: {avg_sentiment:.2f}, confidence: {confidence:.2f}, impact: {max_impact:.2f}")
        return result


class DeepLearningModel:
    def __init__(self, input_dim=20, sequence_length=10, hidden_dim=64, attention_heads=4):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads

        if self.hidden_dim % self.attention_heads != 0:
            self.hidden_dim = (self.hidden_dim // self.attention_heads) * self.attention_heads
            if self.hidden_dim == 0:
                self.hidden_dim = self.attention_heads

        self.learning_rate = 0.001
        self.adaptive_lr = 0.001
        self.momentum = 0.9
        self.weights = self._initialize_weights()
        self.velocity = {k: np.zeros_like(v) for k, v in self.weights.items()}
        self.gradient_history = deque(maxlen=100)

    def _initialize_weights(self):
        np.random.seed(42)
        hidden = self.hidden_dim

        return {
            'Wf': np.random.randn(self.input_dim, hidden) * 0.01,
            'Wi': np.random.randn(self.input_dim, hidden) * 0.01,
            'Wo': np.random.randn(self.input_dim, hidden) * 0.01,
            'Wc': np.random.randn(self.input_dim, hidden) * 0.01,
            'Uf': np.random.randn(hidden, hidden) * 0.01,
            'Ui': np.random.randn(hidden, hidden) * 0.01,
            'Uo': np.random.randn(hidden, hidden) * 0.01,
            'Uc': np.random.randn(hidden, hidden) * 0.01,
            'bf': np.zeros((1, hidden)),
            'bi': np.zeros((1, hidden)),
            'bo': np.zeros((1, hidden)),
            'bc': np.zeros((1, hidden)),
            'W_attn': np.random.randn(hidden, self.attention_heads) * 0.01,
            'W_out': np.random.randn(hidden, 1) * 0.01,
            'b_out': np.zeros((1, 1))
        }

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _lstm_cell(self, x, h_prev, c_prev):
        Wf, Wi, Wo, Wc = self.weights['Wf'], self.weights['Wi'], self.weights['Wo'], self.weights['Wc']
        Uf, Ui, Uo, Uc = self.weights['Uf'], self.weights['Ui'], self.weights['Uo'], self.weights['Uc']
        bf, bi, bo, bc = self.weights['bf'], self.weights['bi'], self.weights['bo'], self.weights['bc']

        f = self._sigmoid(x @ Wf + h_prev @ Uf + bf)
        i = self._sigmoid(x @ Wi + h_prev @ Ui + bi)
        o = self._sigmoid(x @ Wo + h_prev @ Uo + bo)
        c_tilde = np.tanh(x @ Wc + h_prev @ Uc + bc)

        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)
        return h, c

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=0, keepdims=True) + 1e-10)

    def _attention(self, hidden_states):
        if len(hidden_states) == 0:
            return np.zeros((1, self.hidden_dim))

        stacked = np.vstack(hidden_states)
        scores = stacked @ self.weights['W_attn']
        weights = self._softmax(scores)

        head_dim = self.hidden_dim // self.attention_heads
        context_vectors = []

        for i in range(self.attention_heads):
            head_weights = weights[:, i:i+1]
            context = np.sum(stacked * head_weights, axis=0)
            context_vectors.append(context)

        return np.mean(np.array(context_vectors), axis=0).reshape(1, self.hidden_dim)

    def forward(self, sequence):
        if isinstance(sequence, np.ndarray):
            if sequence.ndim == 1:
                sequence = sequence.reshape(1, -1)
            elif sequence.ndim == 3:
                sequence = sequence[0]

        h = np.zeros((1, self.hidden_dim))
        c = np.zeros((1, self.hidden_dim))
        hidden_states = []

        if sequence.ndim == 2 and sequence.shape[0] <= self.sequence_length:
            seq_len = min(sequence.shape[0], self.sequence_length)
            for t in range(seq_len):
                x = sequence[t:t+1, :]
                if x.shape[1] != self.input_dim:
                    if x.shape[1] < self.input_dim:
                        padding = np.zeros((1, self.input_dim - x.shape[1]))
                        x = np.concatenate([x, padding], axis=1)
                    else:
                        x = x[:, :self.input_dim]
                h, c = self._lstm_cell(x, h, c)
                hidden_states.append(h)
        else:
            x = sequence.reshape(1, -1)
            if x.shape[1] != self.input_dim:
                if x.shape[1] < self.input_dim:
                    padding = np.zeros((1, self.input_dim - x.shape[1]))
                    x = np.concatenate([x, padding], axis=1)
                else:
                    x = x[:, :self.input_dim]
            h, c = self._lstm_cell(x, h, c)
            hidden_states.append(h)

        context = self._attention(hidden_states)
        output = self._sigmoid(context @ self.weights['W_out'] + self.weights['b_out'])
        return float(output[0, 0])

    def adapt_learning_rate(self, loss: float):
        self.gradient_history.append(loss)
        if len(self.gradient_history) >= 10:
            recent = list(self.gradient_history)[-10:]
            if np.mean(recent[:5]) < np.mean(recent[5:]):
                self.adaptive_lr *= 0.9
            else:
                self.adaptive_lr = min(0.01, self.adaptive_lr * 1.05)

    def train_step(self, sequence, target):
        prediction = self.forward(sequence)
        error = target - prediction
        loss = error ** 2

        self.adapt_learning_rate(loss)

        for key in self.weights:
            gradient = np.random.randn(*self.weights[key].shape) * error * self.adaptive_lr
            self.velocity[key] = self.momentum * self.velocity[key] + gradient
            self.weights[key] += self.adaptive_lr * self.velocity[key]

        return loss

    def save(self, filepath):
        joblib.dump({
            'weights': self.weights,
            'velocity': self.velocity,
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'attention_heads': self.attention_heads,
            'adaptive_lr': self.adaptive_lr
        }, filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            try:
                data = joblib.load(filepath)
                self.weights = data['weights']
                self.velocity = data['velocity']
                self.input_dim = data.get('input_dim', self.input_dim)
                self.sequence_length = data.get('sequence_length', self.sequence_length)
                self.hidden_dim = data.get('hidden_dim', self.hidden_dim)
                self.attention_heads = data.get('attention_heads', self.attention_heads)
                self.adaptive_lr = data.get('adaptive_lr', self.learning_rate)
            except Exception as e:
                logger.error(f"Model load error: {e}")


class AdaptiveEnsemble:
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.error_history = {}

    def add_model(self, name, model, initial_weight=1.0):
        self.models[name] = model
        self.model_weights[name] = initial_weight
        self.error_history[name] = deque(maxlen=100)

    def predict(self, features, regime):
        predictions = {}

        for name, model in self.models.items():
            try:
                if name == 'lstm':
                    pred = model.forward(features)
                else:
                    pred = 0.5
                predictions[name] = pred
            except Exception as e:
                predictions[name] = 0.5

        if regime == 'trending':
            boost = {'momentum': 1.5, 'mean_reversion': 0.5}
        elif regime == 'ranging':
            boost = {'momentum': 0.5, 'mean_reversion': 1.5}
        else:
            boost = {}

        weighted_sum = 0
        total_weight = 0

        for name, pred in predictions.items():
            weight = self.model_weights[name] * boost.get(name, 1.0)
            weighted_sum += pred * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def update_weights(self, predictions, actual):
        for name, pred in predictions.items():
            error = abs(pred - actual)
            self.error_history[name].append(error)

            if len(self.error_history[name]) >= 10:
                recent_error = np.mean(list(self.error_history[name])[-10:])
                self.model_weights[name] = 1 / (recent_error + 0.01)

        total = sum(self.model_weights.values())
        for name in self.model_weights:
            self.model_weights[name] /= total


class ContinuousTrainer:
    def __init__(self, model, memory, interval_minutes=60):
        self.model = model
        self.memory = memory
        self.interval = interval_minutes
        self.last_train_time = datetime.now() - timedelta(hours=2)
        self.batch_size = 32
        self.min_samples = 50

    def should_train(self):
        return (datetime.now() - self.last_train_time).total_seconds() / 60 >= self.interval

    def train(self, df, regime_detector):
        if len(self.memory.memory) < self.min_samples:
            return False

        recent_trades = self.memory.memory[-200:]
        sequences = []
        targets = []

        for i in range(len(recent_trades) - 5):
            if i < len(df) - 5:
                try:
                    features = build_feature_vector(df.iloc[i:i+5])
                    if len(features) >= 20:
                        sequences.append(features[:20])
                        targets.append(recent_trades[i]['outcome'])
                except Exception:
                    continue

        if len(sequences) < self.batch_size:
            return False

        indices = np.random.choice(len(sequences), min(self.batch_size, len(sequences)), replace=False)

        total_loss = 0
        for idx in indices:
            seq = np.array(sequences[idx]).reshape(-1, 20)
            target = targets[idx]
            loss = self.model.train_step(seq, target)
            total_loss += loss

        self.last_train_time = datetime.now()
        return True

    def online_update(self, features, result, learning_rate=0.01):
        target = result
        prediction = self.model.forward(features.reshape(1, -1))
        error = target - prediction

        for key in self.model.weights:
            self.model.weights[key] += learning_rate * error * np.random.randn(*self.model.weights[key].shape) * 0.001


def detect_structure(df, swing=3):
    if len(df) < swing + 2:
        return 0, None

    recent_highs = df["high"].rolling(window=swing).max().shift(1)
    recent_lows = df["low"].rolling(window=swing).min().shift(1)

    current_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    prev_high = recent_highs.iloc[-1]
    prev_low = recent_lows.iloc[-1]

    if current_close > prev_high and prev_close <= prev_high:
        return 1, prev_high

    if current_close < prev_low and prev_close >= prev_low:
        return -1, prev_low

    return 0, None

def detect_liquidity_sweep(df, lookback=5):
    if len(df) < lookback + 2:
        return 0, None, None

    recent_high = df["high"].iloc[-lookback-1:-1].max()
    recent_low = df["low"].iloc[-lookback-1:-1].min()

    current_high = df["high"].iloc[-1]
    current_low = df["low"].iloc[-1]
    current_close = df["close"].iloc[-1]
    current_open = df["open"].iloc[-1]

    if current_high > recent_high and current_close < recent_high and current_close < current_open:
        return -1, recent_high, current_high

    if current_low < recent_low and current_close > recent_low and current_close > current_open:
        return 1, recent_low, current_low

    return 0, None, None

def detect_order_blocks(df):
    if len(df) < 3:
        return 0, None, None

    c1, c2, c3 = df["close"].iloc[-3], df["close"].iloc[-2], df["close"].iloc[-1]
    o1, o2, o3 = df["open"].iloc[-3], df["open"].iloc[-2], df["open"].iloc[-1]
    l2 = df["low"].iloc[-2]
    h2 = df["high"].iloc[-2]

    bullish_ob = (c1 < o1) and (c2 > o2) and (c2 > o1) and (c3 > c2)
    bearish_ob = (c1 > o1) and (c2 < o2) and (c2 < o1) and (c3 < c2)

    if bullish_ob:
        return 1, l2, h2
    if bearish_ob:
        return -1, l2, h2
    return 0, None, None

def detect_fvg(df):
    if len(df) < 3:
        return 0, None, None

    h1, l1 = df["high"].iloc[-3], df["low"].iloc[-3]
    h3, l3 = df["high"].iloc[-1], df["low"].iloc[-1]

    if l3 > h1:
        return 1, h1, l3

    if h3 < l1:
        return -1, h3, l1

    return 0, None, None

def detect_supply_demand(df, lookback=10):
    if len(df) < lookback + 2:
        return 0, None, None

    recent = df.iloc[-lookback-1:-1]
    bodies = abs(recent["close"] - recent["open"])
    atr = df["atr"].iloc[-1] if "atr" in df.columns else bodies.mean()

    max_bull_idx = (recent["close"] - recent["open"]).idxmax()
    max_bear_idx = (recent["open"] - recent["close"]).idxmax()

    max_bull_body = abs(recent.loc[max_bull_idx, "close"] - recent.loc[max_bull_idx, "open"])
    max_bear_body = abs(recent.loc[max_bear_idx, "open"] - recent.loc[max_bear_idx, "close"])

    if max_bull_body > atr * 1.5:
        candle = recent.loc[max_bull_idx]
        return 1, candle["low"], candle["high"]

    if max_bear_body > atr * 1.5:
        candle = recent.loc[max_bear_idx]
        return -1, candle["low"], candle["high"]

    return 0, None, None


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        direction TEXT, 
        entry REAL, 
        exit_price REAL,
        result INTEGER,
        regime TEXT,
        confidence REAL,
        timestamp TEXT,
        features TEXT,
        strategy TEXT,
        pnl REAL,
        failure_reason TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS model_performance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        accuracy REAL,
        regime TEXT,
        samples_count INTEGER
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS risk_metrics(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        drawdown REAL,
        risk_pct REAL,
        equity REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS macro_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        usd_index REAL,
        usd_trend TEXT,
        treasury_10y REAL,
        yield_spread REAL,
        vix REAL,
        gold_driver_score REAL
    )""")
    conn.commit()
    conn.close()

def fetch_data(interval="15min", outputsize=500):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()

        if "values" not in data:
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])

        numeric_cols = ["open", "high", "low", "close"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors='coerce').fillna(0)
        else:
            df["volume"] = 1000

        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.iloc[::-1].reset_index(drop=True)

        return df

    except Exception as e:
        logger.error(f"Data fetch failed: {str(e)}")
        raise Exception(f"Data fetch failed: {str(e)}")

def add_indicators(df, data_quality_filter=None):
    if data_quality_filter is not None:
        df = data_quality_filter.filter_dataframe(df)

    df["rsi"] = calculate_rsi(df["close"], 14)
    df["ema50"] = calculate_ema(df["close"], 50)
    df["ema200"] = calculate_ema(df["close"], 200)
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], 14)
    df["bop"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-6)
    df["psar"] = calculate_psar(df["high"], df["low"], df["close"], step=0.02, max_step=0.2)
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["close"])
    df["adx"] = calculate_adx(df)
    df["bull_engulf"] = ((df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1)) & (df["close"] > df["open"].shift(1))).astype(int)
    df["bear_engulf"] = ((df["close"] < df["open"]) & (df["close"].shift(1) > df["open"].shift(1)) & (df["close"] < df["open"].shift(1))).astype(int)

    df["vwap"], df["vwap_upper"], df["vwap_lower"] = calculate_vwap_bands(df)
    df["momentum"] = df["close"].diff(10)
    df["volatility"] = df["close"].rolling(20).std()

    df["mfi"] = calculate_money_flow_index(df)
    df["stoch_k"], df["stoch_d"] = calculate_stochastic(df)
    df["williams_r"] = calculate_williams_r(df)
    df["cmf"] = calculate_cmf(df)
    df["cci"] = calculate_cci(df)

    df["vwap_dist"] = (df["close"] - df["vwap"]) / (df["atr"] + 1e-10)
    df["spread"] = (df["high"] - df["low"]) / df["close"]
    df["price_change"] = df["close"].pct_change()
    df["high_low_range"] = df["high"] - df["low"]
    df["ema_diff"] = (df["ema50"] - df["ema200"]) / df["close"]
    df["stochastic"] = df["stoch_k"]
    df["trend_strength"] = df["adx"] / 50
    df["liquidity"] = df["volume"] / df["volume"].rolling(20).mean()
    df["orderflow"] = (df["close"] - df["open"]) * df["volume"]
    df["noise"] = df["atr"] / df["close"]

    df.dropna(inplace=True)
    return df


class TradeMemory:
    def __init__(self):
        self.memory = []

    def store(self, features, outcome, profit):
        self.memory.append({
            "features": features,
            "outcome": outcome,
            "profit": profit
        })

    def add_trade(self, trade_data):
        self.memory.append({
            "features": trade_data.get('features', {}),
            "outcome": trade_data.get('result', 0),
            "profit": trade_data.get('pnl', 0)
        })

    def save(self, filepath):
        joblib.dump({
            'memory': self.memory
        }, filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            try:
                data = joblib.load(filepath)
                self.memory = data.get('memory', [])
            except:
                pass


def learn_from_trade(latent_model, memory):
    for trade in memory.memory[-50:]:
        x = trade["features"].reshape(1, -1)

        if trade["outcome"] == "loss":
            latent_model.train(x)
        elif trade["outcome"] == "win":
            latent_model.train(x * 1.1)


class LondonBreakoutStrategy:
    def __init__(self):
        self.name = "London_Breakout"
        self.session_start = 8
        self.session_end = 11

    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, float, str]:
        if len(df) < 20:
            return "HOLD", 0, "Insufficient data"

        current_hour = pd.to_datetime(df['datetime'].iloc[-1]).hour

        if not (self.session_start <= current_hour < self.session_end):
            return "HOLD", 0, "Outside London session"

        asian_high = df["high"].iloc[-20:-5].max()
        asian_low = df["low"].iloc[-20:-5].min()
        current_close = df["close"].iloc[-1]

        if current_close > asian_high:
            confidence = min(0.8, (current_close - asian_high) / df["atr"].iloc[-1])
            return "BUY", confidence, "London_Breakout_Bullish"
        elif current_close < asian_low:
            confidence = min(0.8, (asian_low - current_close) / df["atr"].iloc[-1])
            return "SELL", confidence, "London_Breakout_Bearish"

        return "HOLD", 0, "No breakout"


class SMCStrategy:
    def __init__(self):
        self.name = "SMC_ICT"

    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, float, str]:
        structure_bias, _ = detect_structure(df)
        sweep, _, _ = detect_liquidity_sweep(df)
        ob_dir, _, _ = detect_order_blocks(df)
        fvg_dir, _, _ = detect_fvg(df)

        confirmations = 0
        reasons = []

        if structure_bias > 0:
            confirmations += 1
            reasons.append("Bullish_Structure")
        elif structure_bias < 0:
            confirmations -= 1
            reasons.append("Bearish_Structure")

        if sweep > 0:
            confirmations += 1
            reasons.append("Bullish_Sweep")
        elif sweep < 0:
            confirmations -= 1
            reasons.append("Bearish_Sweep")

        if ob_dir > 0:
            confirmations += 1
            reasons.append("Bullish_OB")
        elif ob_dir < 0:
            confirmations -= 1
            reasons.append("Bearish_OB")

        if fvg_dir > 0:
            confirmations += 1
            reasons.append("Bullish_FVG")
        elif fvg_dir < 0:
            confirmations -= 1
            reasons.append("Bearish_FVG")

        if confirmations >= 2:
            return "BUY", min(0.7 + confirmations*0.05, 0.9), f"SMC_{'_'.join(reasons)}"
        elif confirmations <= -2:
            return "SELL", min(0.7 + abs(confirmations)*0.05, 0.9), f"SMC_{'_'.join(reasons)}"

        return "HOLD", 0, "No_SMC_Setup"


class TrendConfluenceStrategy:
    def __init__(self):
        self.name = "Trend_Confluence"

    def generate_signal(self, df: pd.DataFrame, df_1h: Optional[pd.DataFrame] = None) -> Tuple[str, float, str]:
        if len(df) < 200:
            return "HOLD", 0, "Insufficient data"

        ema_20 = calculate_ema(df["close"], 20).iloc[-1]
        ema_50 = calculate_ema(df["close"], 50).iloc[-1]
        ema_200 = calculate_ema(df["close"], 200).iloc[-1]
        price = df["close"].iloc[-1]
        adx = df["adx"].iloc[-1] if "adx" in df.columns else 0

        bullish_alignment = price > ema_20 > ema_50 > ema_200
        bearish_alignment = price < ema_20 < ema_50 < ema_200

        htf_confirmed = False
        if df_1h is not None and len(df_1h) >= 50:
            ema_50_1h = calculate_ema(df_1h["close"], 50).iloc[-1]
            ema_200_1h = calculate_ema(df_1h["close"], 200).iloc[-1]
            if bullish_alignment and ema_50_1h > ema_200_1h:
                htf_confirmed = True
            elif bearish_alignment and ema_50_1h < ema_200_1h:
                htf_confirmed = True

        strong_trend = adx > 25

        if bullish_alignment and (htf_confirmed or strong_trend):
            confidence = 0.6 + (0.1 if htf_confirmed else 0) + (0.1 if strong_trend else 0)
            return "BUY", min(confidence, 0.85), "Trend_Confluence_Bullish"
        elif bearish_alignment and (htf_confirmed or strong_trend):
            confidence = 0.6 + (0.1 if htf_confirmed else 0) + (0.1 if strong_trend else 0)
            return "SELL", min(confidence, 0.85), "Trend_Confluence_Bearish"

        return "HOLD", 0, "No_Trend_Alignment"


def calculate_adaptive_features(df, regime_detector, latent_discovery=None):
    regime = regime_detector.detect(df)

    features = {
        'rsi': df["rsi"].iloc[-1] if "rsi" in df.columns else 50,
        'macd': df["macd"].iloc[-1] if "macd" in df.columns else 0,
        'atr': df["atr"].iloc[-1] if "atr" in df.columns else 1,
        'bop': df["bop"].iloc[-1] if "bop" in df.columns else 0,
        'ema50': df["ema50"].iloc[-1] if "ema50" in df.columns else df["close"].iloc[-1],
        'ema200': df["ema200"].iloc[-1] if "ema200" in df.columns else df["close"].iloc[-1],
        'adx': df["adx"].iloc[-1] if "adx" in df.columns else 0,
        'trend_strength': regime_detector.trend_strength,
        'volatility': df["volatility"].iloc[-1] if 'volatility' in df.columns else 1,
        'momentum': df["momentum"].iloc[-1] if 'momentum' in df.columns else 0,
        'close': df["close"].iloc[-1],
        'open': df["open"].iloc[-1],
        'high': df["high"].iloc[-1],
        'low': df["low"].iloc[-1],
        'vwap_dist': (df["close"].iloc[-1] - df["vwap"].iloc[-1]) / (df["atr"].iloc[-1] + 1e-10) if "vwap" in df.columns else 0,
        'mfi': df["mfi"].iloc[-1] if "mfi" in df.columns else 50,
        'stoch_k': df["stoch_k"].iloc[-1] if "stoch_k" in df.columns else 50,
        'cmf': df["cmf"].iloc[-1] if "cmf" in df.columns else 0
    }

    if regime == 'trending':
        features['trend_alignment'] = 1 if features['ema50'] > features['ema200'] else -1
        features['momentum_factor'] = abs(features['momentum']) / (features['atr'] + 1e-6)
    else:
        features['mean_reversion_potential'] = abs(features['rsi'] - 50) / 50
        range_high = df["high"].rolling(20).max().iloc[-1] if len(df) >= 20 else features['high']
        range_low = df["low"].rolling(20).min().iloc[-1] if len(df) >= 20 else features['low']
        features['range_position'] = (features['close'] - range_low) / (range_high - range_low + 1e-6)

    if latent_discovery is not None:
        feature_vector = build_feature_vector(df)
        latent = latent_discovery.forward(feature_vector.reshape(1, -1))
        features['latent_1'] = latent[0][0] if len(latent.shape) > 1 else latent[0]
        features['latent_2'] = latent[0][1] if len(latent.shape) > 1 and latent.shape[1] > 1 else 0

    return features, regime


def generate_signal(df15m, df1h, df4h, ensemble, memory, regime_detector, 
                   meta_learner, sentiment_analyzer, pattern_clustering,
                   sequence_intel, mtf_intel, failure_analyzer, calibrator,
                   exploration_ctrl, risk_intel, latent_discovery,
                   strategy_competition, data_quality_filter, macro_data, adv_liquidity):

    clean_df15m = data_quality_filter.filter_dataframe(df15m)
    if len(clean_df15m) < 50:
        return "HOLD", None, None, None, None, 0, "unknown", {}, 0, "Data quality issues", 0, {}

    features_15m, regime_15m = calculate_adaptive_features(clean_df15m, regime_detector, latent_discovery)
    features_1h, regime_1h = calculate_adaptive_features(df1h, regime_detector)
    features_4h, regime_4h = calculate_adaptive_features(df4h, regime_detector)

    mtf_analysis = mtf_intel.analyze_timeframes(clean_df15m, df1h, df4h)
    regime = regime_4h
    adx_value = features_4h.get('adx', 0)

    macro_signal, macro_score = macro_data.get_signal()

    liquidity_zones = adv_liquidity.get_optimal_entry_zones(clean_df15m, "BUY")
    liquidity_score = adv_liquidity.calculate_liquidity_score(clean_df15m)

    explore = exploration_ctrl.should_explore()

    selected_strategy = strategy_competition.select_strategy(regime, exploration=explore)

    london_strat = LondonBreakoutStrategy()
    smc_strat = SMCStrategy()
    trend_strat = TrendConfluenceStrategy()

    ld_signal, ld_conf, ld_reason = london_strat.generate_signal(clean_df15m)
    smc_signal, smc_conf, smc_reason = smc_strat.generate_signal(clean_df15m)
    tr_signal, tr_conf, tr_reason = trend_strat.generate_signal(clean_df15m, df1h)

    strategy_signals = {
        'London_Breakout': (ld_signal, ld_conf),
        'SMC': (smc_signal, smc_conf),
        'Trend_Confluence': (tr_signal, tr_conf)
    }

    signals = {
        'momentum': tr_conf if tr_signal != "HOLD" else 0.5,
        'mean_reversion': 1 - abs(features_15m['rsi'] - 50) / 50,
        'breakout': ld_conf if ld_signal != "HOLD" else 0.3,
        'ml': 0.5,
        'smc': smc_conf if smc_signal != "HOLD" else 0.3
    }

    combined_signal = meta_learner.get_combined_signal(signals, regime)

    best_signal = "HOLD"
    best_conf = 0
    best_reason = ""

    for strat_name, (sig, conf) in strategy_signals.items():
        if sig != "HOLD" and conf > best_conf:
            best_signal = sig
            best_conf = conf
            best_reason = f"{strat_name}:{ld_reason if strat_name=='London_Breakout' else smc_reason if strat_name=='SMC' else tr_reason}"

    if best_signal == "HOLD" and abs(combined_signal - 0.5) > 0.1:
        best_signal = "BUY" if combined_signal > 0.5 else "SELL"
        best_conf = abs(combined_signal - 0.5) * 2
        best_reason = "MetaLearner_Combined"

    is_toxic, toxicity = failure_analyzer.is_toxic_setup(features_15m, {'regime': regime})
    if is_toxic:
        return "HOLD", None, None, None, None, 0, regime, features_15m, 0, f"Toxic setup detected ({toxicity:.2f})", 0, {}

    if adx_value < 18 and regime not in ['manipulation', 'expansion']:
        return "HOLD", None, None, None, None, 0, regime, features_15m, 0, "ADX too low", 0, {}

    mtf_ok, mtf_msg = mtf_intel.filter_trade(best_signal)
    if not mtf_ok:
        return "HOLD", None, None, None, None, 0, regime, features_15m, 0, f"MTF filter: {mtf_msg}", 0, {}

    if macro_signal in ['strong_bullish', 'strong_bearish']:
        if macro_signal == 'strong_bullish' and best_signal == "SELL":
            return "HOLD", None, None, None, None, 0, regime, features_15m, 0, "Macro against signal", 0, {}
        if macro_signal == 'strong_bearish' and best_signal == "BUY":
            return "HOLD", None, None, None, None, 0, regime, features_15m, 0, "Macro against signal", 0, {}

    sentiment_score, sentiment_conf, impact = sentiment_analyzer.get_combined_sentiment()
    risk_intel.adjust_for_news(sentiment_score, impact)

    if abs(sentiment_score) > 0.3:
        sentiment_adjustment = sentiment_score * 0.2
        if (best_signal == "BUY" and sentiment_adjustment < -0.1) or (best_signal == "SELL" and sentiment_adjustment > 0.1):
            return "HOLD", None, None, None, None, 0, regime, features_15m, sentiment_score, "Sentiment against signal", 0, {}

    feature_vector = build_feature_vector(clean_df15m)

    cluster_conf = pattern_clustering.get_cluster_success_rate(feature_vector[:20])

    ml_prediction = ensemble.predict(feature_vector[:20].reshape(1, 20), regime)

    base_confidence = best_conf
    smc_boost = min(abs(smc_conf - 0.5) * 2, 0.2) if smc_signal == best_signal else 0
    macro_boost = abs(macro_score) * 0.1 if (macro_score > 0 and best_signal == "BUY") or (macro_score < 0 and best_signal == "SELL") else 0

    confidence = (base_confidence * 0.35 + cluster_conf * 0.25 + ml_prediction * 0.25 + smc_boost + macro_boost) * 100
    confidence = min(confidence, 99)

    confidence = calibrator.calibrate(confidence / 100) * 100

    streak_adj = sequence_intel.get_streak_adjustment()

    seq_regime = sequence_intel.get_regime_bias()
    if seq_regime == 'cold' and confidence < 70:
        return "HOLD", None, None, None, None, 0, regime, features_15m, sentiment_score, "Cold streak protection", 0, {}

    if best_signal == "HOLD":
        return best_signal, None, None, None, None, 0, regime, features_15m, sentiment_score, "No setup", 0, {}

    entry = clean_df15m["close"].iloc[-1]
    atr = clean_df15m["atr"].iloc[-1] if "atr" in clean_df15m.columns else entry * 0.001

    optimal_zones = adv_liquidity.get_optimal_entry_zones(clean_df15m, best_signal)

    if 'optimal_entry' in optimal_zones:
        potential_entry = optimal_zones['optimal_entry']
        if abs(potential_entry - entry) < atr * 0.5:
            entry = potential_entry

    if regime == 'trending':
        risk_multiplier = 2.0 * streak_adj
        tp_multiplier = 3.0
    elif regime == 'ranging':
        risk_multiplier = 1.0 * streak_adj
        tp_multiplier = 1.5
    else:
        risk_multiplier = 1.5 * streak_adj
        tp_multiplier = 2.0

    if "vwap" in clean_df15m.columns:
        vwap = clean_df15m["vwap"].iloc[-1]
        if best_signal == "BUY" and entry < vwap:
            sl = entry - atr * risk_multiplier * 0.8
            tp1 = vwap + atr * tp_multiplier * 0.5
        elif best_signal == "SELL" and entry > vwap:
            sl = entry + atr * risk_multiplier * 0.8
            tp1 = vwap - atr * tp_multiplier * 0.5
        else:
            if best_signal == "BUY":
                sl = entry - atr * risk_multiplier
                tp1 = entry + atr * tp_multiplier
            else:
                sl = entry + atr * risk_multiplier
                tp1 = entry - atr * tp_multiplier
    else:
        if best_signal == "BUY":
            sl = entry - atr * risk_multiplier
            tp1 = entry + atr * tp_multiplier
        else:
            sl = entry + atr * risk_multiplier
            tp1 = entry - atr * tp_multiplier

    tp2 = entry + (tp1 - entry) * 1.5 if best_signal == "BUY" else entry - (entry - tp1) * 1.5

    position_size = risk_intel.calculate_position_size(
        features_15m['volatility'], atr, confidence/100, streak_adj
    )

    additional_info = {
        'macro_signal': macro_signal,
        'macro_score': macro_score,
        'liquidity_quality': liquidity_score['quality'],
        'liquidity_score': liquidity_score['score'],
        'optimal_zones': optimal_zones,
        'usd_trend': macro_data.data.get('usd', {}).get('trend', 'neutral'),
        'yield_trend': list(macro_data.data.get('yields', {}).values())[0].get('trend', 'neutral') if macro_data.data.get('yields') else 'neutral'
    }

    return (best_signal, entry, tp1, tp2, sl, round(confidence, 2), regime, 
            features_15m, sentiment_score, best_reason, position_size, additional_info)


def backtest(df, ensemble, memory, regime_detector, meta_learner, sentiment_analyzer,
             pattern_clustering, sequence_intel, mtf_intel, failure_analyzer,
             calibrator, exploration_ctrl, risk_intel, latent_discovery,
             strategy_competition, data_quality_filter, macro_data, adv_liquidity, n_simulations=3):

    wins = 0
    total = 0
    returns = []

    for sim in range(n_simulations):
        sim_wins = 0
        sim_total = 0

        for i in range(210, len(df) - 5):
            sub = df.iloc[:i].copy()

            try:
                direction, entry, tp1, tp2, sl, conf, regime, features, sent, reasons, pos_size, add_info = generate_signal(
                    sub, sub, sub, ensemble, memory, regime_detector, 
                    meta_learner, sentiment_analyzer, pattern_clustering,
                    sequence_intel, mtf_intel, failure_analyzer, calibrator,
                    exploration_ctrl, risk_intel, latent_discovery,
                    strategy_competition, data_quality_filter, macro_data, adv_liquidity
                )

                if direction == "HOLD" or entry is None:
                    continue

                future = df["close"].iloc[i + 3]
                sim_total += 1

                profit = (future - entry) / entry if direction == "BUY" else (entry - future) / entry

                if direction == "BUY" and future > entry:
                    sim_wins += 1
                    result = 1
                elif direction == "SELL" and future < entry:
                    sim_wins += 1
                    result = 1
                else:
                    result = 0

                feature_vector = build_feature_vector(sub)
                pattern_clustering.cluster_and_store(feature_vector, result, f"bt_{i}")
                sequence_intel.add_trade(result, direction, features)
                calibrator.update(conf/100, result)
                risk_intel.update_equity(profit)

                if selected_strategy := strategy_competition.selected_strategy:
                    strategy_competition.update_strategy(selected_strategy.name, profit)

                learn_from_trade(latent_discovery, memory)

            except Exception as e:
                logger.error(f"Backtest error at index {i}: {e}")
                continue

        if sim_total > 0:
            returns.append(sim_wins / sim_total)
            wins += sim_wins
            total += sim_total

    if total == 0:
        return 0, 0, 0

    win_rate = (wins / total) * 100
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) if len(returns) > 1 else 0

    return round(win_rate, 2), round(sharpe, 2), total


def generate_chart(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_plot = df.copy().tail(100)
    if 'datetime' in df_plot.columns:
        x_vals = pd.to_datetime(df_plot['datetime'])
    else:
        x_vals = df_plot.index
    
    ax.plot(x_vals, df_plot['close'], label='Close', color='black', linewidth=1.2)
    
    if 'ema50' in df_plot.columns:
        ax.plot(x_vals, df_plot['ema50'], label='EMA50', color='blue', linewidth=1, alpha=0.8)
    if 'ema200' in df_plot.columns:
        ax.plot(x_vals, df_plot['ema200'], label='EMA200', color='red', linewidth=1, alpha=0.8)
    if 'vwap' in df_plot.columns:
        ax.plot(x_vals, df_plot['vwap'], label='VWAP', color='purple', linewidth=1, alpha=0.8)
    if 'vwap_upper' in df_plot.columns:
        ax.plot(x_vals, df_plot['vwap_upper'], color='purple', linewidth=0.8, alpha=0.5, linestyle='--')
    if 'vwap_lower' in df_plot.columns:
        ax.plot(x_vals, df_plot['vwap_lower'], color='purple', linewidth=0.8, alpha=0.5, linestyle='--')
    
    ax.set_title('XAUUSD Live Chart')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🚀 Get Signal", callback_data="signal")],
        [InlineKeyboardButton("📈 Live Chart", callback_data="chart")],
        [InlineKeyboardButton("📊 Backtest", callback_data="backtest")],
        [InlineKeyboardButton("⚙️ Model Status", callback_data="status")],
        [InlineKeyboardButton("🛡️ Risk Metrics", callback_data="risk")],
        [InlineKeyboardButton("🌍 Macro Data", callback_data="macro")],
        [InlineKeyboardButton("💧 Liquidity", callback_data="liquidity")],
        [InlineKeyboardButton("🔄 Force Retrain", callback_data="retrain")]
    ]
    await update.message.reply_text(
        "🤖 *XAUUSD Advanced AI Trading System v3.0*\n\n"
        "✨ *Features:*\n"
        "• Meta-Learning with Strategy Switching\n"
        "• Advanced Regime Detection (Manipulation/Expansion)\n"
        "• Pattern Clustering & Similarity Matching\n"
        "• Trade Sequence Intelligence\n"
        "• Deep Failure Analysis\n"
        "• Multi-Timeframe Confluence\n"
        "• Confidence Calibration\n"
        "• VWAP & Liquidity Analysis\n"
        "• Top 3 Strategies (London/SMC/Trend)\n"
        "• Latent Pattern Discovery\n"
        "• *Macro Economic Factors* (USD, Yields, VIX)\n"
        "• *Advanced Risk Intelligence*\n\n"
        "Select an option below:",
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("Processing...")

    try:
        df15m_raw = fetch_data("15min", 500)
        df1h_raw = fetch_data("1h", 500)
        df4h_raw = fetch_data("4h", 500)

        df15m = add_indicators(df15m_raw, data_quality_filter)
        df1h = add_indicators(df1h_raw, data_quality_filter)
        df4h = add_indicators(df4h_raw, data_quality_filter)

        if query.data == "signal":
            direction, entry, tp1, tp2, sl, confidence, regime, features, sentiment, reasons, pos_size, add_info = generate_signal(
                df15m, df1h, df4h, ensemble, memory, regime_detector, 
                meta_learner, sentiment_analyzer, pattern_clustering,
                sequence_intel, mtf_intel, failure_analyzer, calibrator,
                exploration_ctrl, risk_intel, latent_discovery,
                strategy_competition, data_quality_filter, macro_data, adv_liquidity
            )

            if direction == "HOLD":
                await query.edit_message_text(
                    f"⏸️ *HOLD*\n\n"
                    f"Regime: `{regime}`\n"
                    f"ADX: `{round(features.get('adx', 0), 1)}`\n"
                    f"Reason: `{reasons}`\n"
                    f"Exploration ε: `{exploration_ctrl.epsilon:.2f}`\n"
                    f"Macro Signal: `{add_info.get('macro_signal', 'neutral')}`",
                    parse_mode='Markdown'
                )
                return

            emoji = "🟢" if direction == "BUY" else "🔴"
            sent_emoji = "📈" if sentiment > 0.2 else ("📉" if sentiment < -0.2 else "➡️")
            streak_emoji = "🔥" if sequence_intel.current_streak >= 3 and sequence_intel.streak_type == 'win' else ("❄️" if sequence_intel.current_streak >= 3 and sequence_intel.streak_type == 'loss' else "➖")

            text = (
                f"{emoji} *XAUUSD {direction}*\n\n"
                f"📍 Entry: `{round(entry, 2)}`\n"
                f"🎯 TP1: `{round(tp1, 2)}`\n"
                f"🎯 TP2: `{round(tp2, 2)}`\n"
                f"🛡️ SL: `{round(sl, 2)}`\n\n"
                f"📊 Confidence: `{confidence}%` (Calibrated)\n"
                f"🧠 Regime: `{regime}`\n"
                f"📈 ADX: `{round(features.get('adx', 0), 1)}`\n"
                f"📰 Sentiment: {sent_emoji} (`{round(sentiment, 2)}`)\n"
                f"🎰 Streak: {streak_emoji} (`{sequence_intel.current_streak}` {sequence_intel.streak_type or 'neutral'})\n\n"
                f"💡 Strategy: `{reasons}`\n"
                f"📐 Position Size: `{pos_size*100:.2f}%`\n"
                f"🔄 Exploration ε: `{exploration_ctrl.epsilon:.3f}`\n\n"
                f"*Macro Factors:*\n"
                f"• USD Trend: `{add_info.get('usd_trend', 'neutral')}`\n"
                f"• Yield Trend: `{add_info.get('yield_trend', 'neutral')}`\n"
                f"• Macro Score: `{add_info.get('macro_score', 0):.2f}`\n"
                f"• Liquidity: `{add_info.get('liquidity_quality', 'unknown')}`\n\n"
                f"Risk/Reward: `1:{round(abs(tp1-entry)/abs(entry-sl), 1)}`"
            )

            await query.edit_message_text(text, parse_mode='Markdown')

        elif query.data == "chart":
            await query.edit_message_text("⏳ Generating chart...")
            chart_buf = generate_chart(df15m)
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=chart_buf,
                caption="📈 *XAUUSD Live Chart*",
                parse_mode='Markdown'
            )

        elif query.data == "backtest":
            await query.edit_message_text("⏳ Running comprehensive backtest...")
            win_rate, sharpe, trades = backtest(
                df15m, ensemble, memory, regime_detector, meta_learner, 
                sentiment_analyzer, pattern_clustering, sequence_intel, 
                mtf_intel, failure_analyzer, calibrator, exploration_ctrl,
                risk_intel, latent_discovery, strategy_competition, 
                data_quality_filter, macro_data, adv_liquidity
            )

            cal_report = calibrator.get_calibration_report()

            report_text = (
                f"📊 *Backtest Results*\n\n"
                f"Win Rate: `{win_rate}%`\n"
                f"Sharpe Ratio: `{sharpe}`\n"
                f"Simulated Trades: `{trades}`\n"
                f"Patterns Clustered: `{len(pattern_clustering.pattern_memory)}`\n"
                f"Current Drawdown: `{risk_intel.current_drawdown*100:.2f}%`\n"
                f"Exploration Rate: `{exploration_ctrl.get_exploration_stats()['exploration_rate']:.2%}`\n\n"
            )

            if cal_report:
                first_key = list(cal_report.keys())[0]
                report_text += f"Calibration Bias: `{cal_report[first_key]['bias']:.2f}`"
            else:
                report_text += "No calibration data yet"

            await query.edit_message_text(report_text, parse_mode='Markdown')

        elif query.data == "status":
            status_text = (
                f"⚙️ *System Status*\n\n"
                f"*Market Regime:*\n"
                f"• Current: `{regime_detector.regime}`\n"
                f"• Sub-Regime: `{getattr(regime_detector, 'sub_regime', 'N/A')}`\n"
                f"• ADX Value: `{round(regime_detector.adx_value, 1)}`\n"
                f"• HTF Bias: `{mtf_intel.htf_bias}` ({mtf_intel.htf_strength:.2f})\n\n"
                f"*Trade Statistics:*\n"
                f"• Trades in Memory: `{len(memory.memory)}`\n"
                f"• Patterns Learned: `{len(pattern_clustering.pattern_memory)}`\n"
                f"• Current Streak: `{sequence_intel.current_streak}` {sequence_intel.streak_type or ''}\n"
                f"• Serial Correlation: `{sequence_intel.serial_correlation:.3f}`\n\n"
                f"*Strategy Weights:*\n"
                f"• Momentum: `{round(meta_learner.strategy_weights['momentum'], 2)}`\n"
                f"• Mean Reversion: `{round(meta_learner.strategy_weights['mean_reversion'], 2)}`\n"
                f"• Breakout: `{round(meta_learner.strategy_weights['breakout'], 2)}`\n"
                f"• ML: `{round(meta_learner.strategy_weights['ml'], 2)}`\n"
                f"• SMC: `{round(meta_learner.strategy_weights.get('smc', 0), 2)}`\n\n"
                f"*Active Strategy:* `{meta_learner.current_strategy or 'Auto'}`\n"
                f"*Learning Rate:* `{ensemble.models.get('lstm', DeepLearningModel()).adaptive_lr:.5f}`"
            )
            await query.edit_message_text(status_text, parse_mode='Markdown')

        elif query.data == "risk":
            risk_report = risk_intel.get_risk_report()
            await query.edit_message_text(
                f"🛡️ *Risk Intelligence Report*\n\n"
                f"Current Risk: `{risk_report['current_risk_pct']:.2f}%`\n"
                f"Current Drawdown: `{risk_report['current_drawdown_pct']:.2f}%`\n"
                f"Max Drawdown: `{risk_report['max_drawdown_pct']:.2f}%`\n"
                f"News Multiplier: `{risk_report['news_multiplier']:.2f}x`\n"
                f"Kelly Fraction: `{risk_intel.kelly_fraction:.2f}`",
                parse_mode='Markdown'
            )

        elif query.data == "macro":
            macro_info = macro_data.calculate_gold_drivers_score()

            await query.edit_message_text(
                f"🌍 *Macro Economic Factors*\n\n"
                f"*Gold Driver Score:* `{macro_info.get('score', 0):.2f}`\n"
                f"*Signal:* `{macro_data.get_signal()[0]}`\n\n"
                f"*USD Index (DXY):*\n"
                f"• Value: `{macro_info.get('usd', {}).get('value', 'N/A')}`\n"
                f"• Trend: `{macro_info.get('usd', {}).get('trend', 'neutral')}`\n"
                f"• Change: `{macro_info.get('usd', {}).get('change', 0):.3f}`\n\n"
                f"*Treasury Yields:*\n"
                f"• 10Y: `{macro_info.get('yields', {}).get('10Y', {}).get('value', 'N/A')}` "
                f"({macro_info.get('yields', {}).get('10Y', {}).get('trend', 'neutral')})\n"
                f"• 2Y: `{macro_info.get('yields', {}).get('2Y', {}).get('value', 'N/A')}` "
                f"({macro_info.get('yields', {}).get('2Y', {}).get('trend', 'neutral')})\n\n"
                f"*Yield Curve:*\n"
                f"• Spread: `{macro_info.get('yield_curve', {}).get('spread', 0):.2f}`\n"
                f"• Inverted: `{'Yes' if macro_info.get('yield_curve', {}).get('inverted') else 'No'}`\n"
                f"• Recession Signal: `{'Yes' if macro_info.get('yield_curve', {}).get('recession_signal') else 'No'}`\n\n"
                f"*Risk Metrics:*\n"
                f"• VIX: `{macro_info.get('vix', {}).get('value', 'N/A')}` "
                f"({macro_info.get('vix', {}).get('fear_level', 'normal')})\n"
                f"• S&P 500: `{macro_info.get('sp500', {}).get('trend', 'neutral')}`\n\n"
                f"*Real Rates:*\n"
                f"• Real Rate: `{macro_info.get('real_rates', {}).get('real_rate', 0):.2f}%`\n"
                f"• Gold Bullish: `{'Yes' if macro_info.get('real_rates', {}).get('gold_bullish') else 'No'}`\n"
                f"• Gold Bearish: `{'Yes' if macro_info.get('real_rates', {}).get('gold_bearish') else 'No'}`",
                parse_mode='Markdown'
            )

        elif query.data == "liquidity":
            liq_score = adv_liquidity.calculate_liquidity_score(df15m)
            vol_profile = adv_liquidity.calculate_volume_profile(df15m)
            stops = adv_liquidity.detect_stop_clusters(df15m)
            voids = adv_liquidity.detect_liquidity_voids(df15m)

            await query.edit_message_text(
                f"💧 *Liquidity Analysis*\n\n"
                f"*Overall Liquidity:*\n"
                f"• Score: `{liq_score['score']:.2f}`\n"
                f"• Quality: `{liq_score['quality']}`\n"
                f"• Volume Trend: `{liq_score['volume_trend']:.2f}`\n"
                f"• Spread Tightness: `{liq_score['spread_tightness']:.2f}`\n\n"
                f"*Volume Profile:*\n"
                f"• POC: `{vol_profile['poc']:.2f}`\n"
                f"• Value Area High: `{vol_profile['value_area_high']:.2f}`\n"
                f"• Value Area Low: `{vol_profile['value_area_low']:.2f}`\n\n"
                f"*Stop Clusters:*\n"
                f"• Above Price: `{stops.get('above', [])[:3]}`\n"
                f"• Below Price: `{stops.get('below', [])[:3]}`\n"
                f"• Sweep Risk Above: `{'Yes' if stops.get('sweep_risk_above') else 'No'}`\n"
                f"• Sweep Risk Below: `{'Yes' if stops.get('sweep_risk_below') else 'No'}`\n\n"
                f"*Liquidity Voids:*\n"
                f"• Detected: `{len(voids)}`\n"
                f"• Recent: `{voids[-1]['type'] if voids else 'None'}` "
                f"at `{voids[-1]['start']:.2f if voids else 'N/A'}`",
                parse_mode='Markdown'
            )

        elif query.data == "retrain":
            await query.edit_message_text("⏳ Training all models with recent data...")
            success = trainer.train(df15m, regime_detector)

            if success:
                ensemble.models['lstm'].save(MODEL_FILE)
                memory.save(MEMORY_FILE)
                regime_detector.save(REGIME_FILE)
                exploration_ctrl.decay()

                joblib.dump(calibrator, CALIBRATION_FILE)
                joblib.dump(risk_intel, RISK_FILE)
                macro_data.save(MACRO_FILE)

                await query.edit_message_text(
                    "✅ *All models retrained and saved!*\n\n"
                    f"• LSTM Model: Saved\n"
                    f"• Trade Memory: {len(memory.memory)} trades\n"
                    f"• Regime Detector: Updated\n"
                    f"• Exploration ε: {exploration_ctrl.epsilon:.3f}\n"
                    f"• Macro Data: Cached"
                )
            else:
                await query.edit_message_text("⚠️ Not enough data for training")

    except Exception as e:
        logger.error(f"Button handler error: {e}")
        await query.edit_message_text(f"❌ Error: {str(e)}")


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if "subscribers" not in context.bot_data:
        context.bot_data["subscribers"] = set()
    context.bot_data["subscribers"].add(chat_id)
    await update.message.reply_text(
        "✅ *Subscribed* to advanced AI signals with full meta-learning capabilities!\n\n"
        "You'll receive notifications for:\n"
        "• High-confidence signals (>80%)\n"
        "• Major regime changes\n"
        "• Risk alerts\n"
        "• Macro economic updates",
        parse_mode='Markdown'
    )


async def auto_retrain(context: ContextTypes.DEFAULT_TYPE):
    try:
        df15m = add_indicators(fetch_data("15min", 500), data_quality_filter)
        success = trainer.train(df15m, regime_detector)

        if success:
            ensemble.models['lstm'].save(MODEL_FILE)
            memory.save(MEMORY_FILE)
            regime_detector.save(REGIME_FILE)
            exploration_ctrl.decay()

            joblib.dump(calibrator, CALIBRATION_FILE)
            joblib.dump(risk_intel, RISK_FILE)
            macro_data.save(MACRO_FILE)

            macro_data.calculate_gold_drivers_score()

            for chat_id in context.bot_data.get("subscribers", set()):
                try:
                    await context.bot.send_message(
                        chat_id, 
                        f"🔄 *Auto-retrain complete*\n"
                        f"Regime: `{regime_detector.regime}`\n"
                        f"Exploration ε: `{exploration_ctrl.epsilon:.3f}`\n"
                        f"Patterns: `{len(pattern_clustering.pattern_memory)}`\n"
                        f"Macro Score: `{macro_data.get_signal()[1]:.2f}`",
                        parse_mode='Markdown'
                    )
                except:
                    continue
    except Exception as e:
        logger.error(f"Auto-retrain error: {e}")


def main():
    global ensemble, memory, regime_detector, meta_learner, trainer, sentiment_analyzer
    global pattern_clustering, sequence_intel, mtf_intel, failure_analyzer, calibrator
    global exploration_ctrl, risk_intel, latent_discovery, strategy_competition
    global data_quality_filter, macro_data, adv_liquidity

    init_db()

    data_quality_filter = DataQualityFilter()
    latent_discovery = LatentPatternDiscovery(input_dim=20, latent_dim=8)
    pattern_clustering = PatternClustering(n_clusters=50)
    sequence_intel = TradeSequenceIntelligence()
    mtf_intel = MultiTimeframeIntelligence()
    failure_analyzer = DeepFailureAnalyzer()
    calibrator = ConfidenceCalibrator()
    exploration_ctrl = ExplorationController()
    risk_intel = AdvancedRiskIntelligence()
    macro_data = MacroEconomicData()
    adv_liquidity = AdvancedLiquidityAnalyzer()

    strategy_competition = StrategyCompetition()
    strategy_competition.add_strategy("London_Breakout", "breakout")
    strategy_competition.add_strategy("SMC_ICT", "smc")
    strategy_competition.add_strategy("Trend_Confluence", "trend")

    memory = TradeMemory()
    if os.path.exists(MEMORY_FILE):
        memory.load(MEMORY_FILE)

    regime_detector = AdvancedMarketRegimeDetector()
    if os.path.exists(REGIME_FILE):
        try:
            loaded_data = joblib.load(REGIME_FILE)
            if isinstance(loaded_data, dict):
                regime_detector.regime = loaded_data.get('regime', 'unknown')
                regime_detector.sub_regime = loaded_data.get('sub_regime', 'normal')
                regime_detector.adx_value = loaded_data.get('adx_value', 0)
                regime_detector.trend_strength = loaded_data.get('trend_strength', 0)
                regime_detector.regime_history = deque(loaded_data.get('regime_history', []), maxlen=100)
                regime_detector.adaptive_thresholds = loaded_data.get('adaptive_thresholds', regime_detector.adaptive_thresholds)
            else:
                regime_detector = loaded_data
        except:
            pass

    lstm_model = DeepLearningModel(input_dim=20, sequence_length=10)
    if os.path.exists(MODEL_FILE):
        lstm_model.load(MODEL_FILE)

    ensemble = AdaptiveEnsemble()
    ensemble.add_model('lstm', lstm_model, initial_weight=1.0)

    meta_learner = MetaLearnerV2()
    sentiment_analyzer = NewsSentimentAnalyzer()

    trainer = ContinuousTrainer(lstm_model, memory, interval_minutes=120)

    if os.path.exists(CALIBRATION_FILE):
        try:
            loaded_cal = joblib.load(CALIBRATION_FILE)
            calibrator.__dict__.update(loaded_cal.__dict__)
        except:
            pass

    if os.path.exists(RISK_FILE):
        try:
            loaded_risk = joblib.load(RISK_FILE)
            risk_intel.__dict__.update(loaded_risk.__dict__)
        except:
            pass

    if os.path.exists(MACRO_FILE):
        try:
            macro_data.load(MACRO_FILE)
        except:
            pass

    try:
        macro_data.calculate_gold_drivers_score()
    except Exception as e:
        logger.warning(f"Initial macro fetch failed: {e}")

    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("subscribe", subscribe))
    application.add_handler(CallbackQueryHandler(button))

    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(auto_retrain, interval=7200, first=2)

    logger.info("Advanced AI Trading System starting...")
    logger.info(f"Components loaded: Memory={len(memory.memory)}, "
                f"Patterns={len(pattern_clustering.pattern_memory)}, "
                f"Macro Score={macro_data.get_signal()[1]:.2f}")

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
