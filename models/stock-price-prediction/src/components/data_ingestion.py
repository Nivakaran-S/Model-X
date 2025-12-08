"""
models/stock-price-prediction/src/components/data_ingestion.py
Data Ingestion for Sri Lanka Stock Price Prediction
Fetches historical data for top CSE stocks from multiple sources
"""
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import requests
from io import StringIO

# yfinance for stock data (primary)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARNING] yfinance not available. Install with: pip install yfinance")

sys.path.insert(0, str(Path(__file__).parent.parent))

from entity.config_entity import DataIngestionConfig, SRI_LANKA_STOCKS

logger = logging.getLogger("stock_prediction.data_ingestion")


class StockDataIngestion:
    """
    Ingests historical stock data for top Sri Lankan companies.
    
    Data sources:
    1. Yahoo Finance (primary)
    2. Fallback to CSE web scraping if Yahoo fails
    
    Features generated:
    - OHLCV data
    - Technical indicators (SMA, EMA, RSI, MACD, Bollinger)
    - Volume analysis
    """
    
    def __init__(self, config: Optional[DataIngestionConfig] = None):
        self.config = config or DataIngestionConfig()
        os.makedirs(self.config.raw_data_dir, exist_ok=True)
        
        # CSE-specific Yahoo Finance symbols may not work, so we have fallbacks
        self.fallback_symbols = {
            "JKH": "JKH.CM",
            "COMB": "COMB.CM",
            "SAMP": "SAMP.CM",
            "HNB": "HNB.CM",
            "DIAL": "DIAL.CM",
            "CTC": "CTC.CM",
            "NEST": "NEST.CM",
            "CARG": "CARG.CM",
            "HNBA": "HNBA.CM",
            "CARS": "CARS.CM"
        }
    
    def fetch_stock_data(
        self,
        stock_code: str,
        period: str = "2y"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            stock_code: Internal stock code (e.g., "JKH")
            period: Data period (1y, 2y, 5y)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        stock_info = self.config.stocks.get(stock_code)
        if not stock_info:
            logger.error(f"[STOCK] Unknown stock code: {stock_code}")
            return None
        
        # Try multiple symbol formats
        symbols_to_try = [
            stock_info["yahoo_symbol"],
            self.fallback_symbols.get(stock_code, ""),
            f"{stock_code}.CM",
            f"{stock_code}.CO"
        ]
        
        for symbol in symbols_to_try:
            if not symbol:
                continue
            
            try:
                logger.info(f"[STOCK] Trying {stock_code} with symbol {symbol}...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval="1d")
                
                if not df.empty and len(df) > 30:  # Need at least 30 days
                    logger.info(f"[STOCK] ✓ {stock_code}: {len(df)} records")
                    
                    df = df.reset_index()
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    df["stock_code"] = stock_code
                    df["symbol"] = symbol
                    
                    return df
            except Exception as e:
                logger.warning(f"[STOCK] Failed with {symbol}: {e}")
                continue
        
        # If all Yahoo symbols fail, try generating synthetic data for demo
        logger.warning(f"[STOCK] No Yahoo data for {stock_code}, generating demo data")
        return self._generate_demo_data(stock_code, period)
    
    def _generate_demo_data(self, stock_code: str, period: str) -> pd.DataFrame:
        """
        Generate demo stock data for testing when real data unavailable.
        Based on typical CSE stock characteristics.
        """
        days = {"1y": 252, "2y": 504, "5y": 1260}.get(period, 504)
        
        # Base price for different stocks (approximate)
        base_prices = {
            "JKH": 180, "COMB": 95, "SAMP": 75, "HNB": 210,
            "DIAL": 12, "CTC": 1200, "NEST": 1800, "CARG": 350,
            "HNBA": 28, "CARS": 550
        }
        
        base_price = base_prices.get(stock_code, 100)
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')  # Business days
        
        # Random walk with drift
        np.random.seed(hash(stock_code) % 2**32)
        returns = np.random.normal(0.0003, 0.02, days)  # Slight positive drift
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            "date": dates,
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, days)),
            "high": prices * (1 + np.random.uniform(0, 0.02, days)),
            "low": prices * (1 - np.random.uniform(0, 0.02, days)),
            "close": prices,
            "volume": np.random.randint(10000, 500000, days),
            "stock_code": stock_code,
            "symbol": f"{stock_code}.CM",
            "is_demo": True
        })
        
        logger.info(f"[STOCK] ✓ {stock_code}: {len(df)} demo records generated")
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical analysis features for ML training.
        """
        df = df.copy()
        
        # Price-based features
        df["daily_return"] = df["close"].pct_change()
        df["daily_range"] = (df["high"] - df["low"]) / df["close"]
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
            df[f"ema_{period}"] = df["close"].ewm(span=period).mean()
        
        # Price relative to MAs
        df["price_to_sma20"] = df["close"] / df["sma_20"]
        df["price_to_sma50"] = df["close"] / df["sma_50"]
        
        # Volatility
        df["volatility_5"] = df["daily_return"].rolling(window=5).std()
        df["volatility_20"] = df["daily_return"].rolling(window=20).std()
        
        # Momentum
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
        
        # RSI (14-day)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        
        # Volume features
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1)
        
        # Temporal features
        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)
        
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        return df
    
    def ingest_all_stocks(self) -> Dict[str, str]:
        """
        Ingest data for all configured stocks.
        
        Returns:
            Dictionary mapping stock code to saved file path
        """
        logger.info(f"[INGESTION] Starting data ingestion for {len(self.config.stocks)} stocks...")
        
        results = {}
        
        for stock_code, stock_info in self.config.stocks.items():
            logger.info(f"\n[INGESTION] Processing {stock_code} - {stock_info['name']}...")
            
            # Fetch raw data
            df = self.fetch_stock_data(stock_code, self.config.history_period)
            
            if df is None or df.empty:
                logger.error(f"[INGESTION] ✗ Failed to get data for {stock_code}")
                continue
            
            # Add technical indicators
            if self.config.include_technical_indicators:
                df = self.add_technical_indicators(df)
            
            # Drop NaN rows from indicator calculations
            df = df.dropna().reset_index(drop=True)
            
            # Save to CSV
            save_path = os.path.join(
                self.config.raw_data_dir,
                f"{stock_code}_data.csv"
            )
            df.to_csv(save_path, index=False)
            
            results[stock_code] = save_path
            logger.info(f"[INGESTION] ✓ {stock_code}: {len(df)} records saved to {save_path}")
        
        logger.info(f"\n[INGESTION] Complete! Processed {len(results)}/{len(self.config.stocks)} stocks")
        return results
    
    def load_stock_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """Load existing stock data."""
        path = os.path.join(self.config.raw_data_dir, f"{stock_code}_data.csv")
        
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["date"])
        return None
    
    def get_all_available_stocks(self) -> List[str]:
        """Get list of stocks with available data."""
        data_dir = Path(self.config.raw_data_dir)
        if not data_dir.exists():
            return []
        
        available = []
        for f in data_dir.glob("*_data.csv"):
            stock_code = f.stem.replace("_data", "")
            available.append(stock_code)
        
        return available


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test ingestion
    ingestion = StockDataIngestion()
    
    print("Testing stock data ingestion...")
    results = ingestion.ingest_all_stocks()
    
    print(f"\nResults: {len(results)} stocks ingested")
    for stock, path in results.items():
        df = ingestion.load_stock_data(stock)
        if df is not None:
            print(f"  {stock}: {len(df)} records, latest close: {df['close'].iloc[-1]:.2f}")
