"""
models/weather-prediction/src/components/data_ingestion.py
Data Ingestion component for Weather Prediction Pipeline
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tutiempo_scraper import TutiempoScraper
from entity.config_entity import DataIngestionConfig, WEATHER_STATIONS

logger = logging.getLogger("weather_prediction.data_ingestion")


class DataIngestion:
    """
    Handles ingestion of historical weather data from Tutiempo.net
    
    Ingests data for all 20 Sri Lankan weather stations and saves
    to CSV for training.
    """
    
    def __init__(self, config: Optional[DataIngestionConfig] = None):
        self.config = config or DataIngestionConfig()
        os.makedirs(self.config.raw_data_dir, exist_ok=True)
        
        self.scraper = TutiempoScraper(cache_dir=self.config.raw_data_dir)
        
    def ingest_all(self) -> str:
        """
        Ingest historical weather data for all stations.
        
        Returns:
            Path to saved CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        save_path = os.path.join(
            self.config.raw_data_dir,
            f"weather_history_{timestamp}.csv"
        )
        
        logger.info(f"[DATA_INGESTION] Starting ingestion for {len(self.config.stations)} stations")
        logger.info(f"[DATA_INGESTION] Fetching {self.config.months_to_fetch} months of history")
        
        df = self.scraper.scrape_all_stations(
            stations=self.config.stations,
            months=self.config.months_to_fetch,
            save_path=save_path
        )
        
        logger.info(f"[DATA_INGESTION] ✓ Ingested {len(df)} total records")
        return save_path
    
    def ingest_station(self, station_name: str, months: int = None) -> pd.DataFrame:
        """
        Ingest data for a single station.
        
        Args:
            station_name: Name of station (e.g., "COLOMBO")
            months: Override months to fetch
            
        Returns:
            DataFrame with station data
        """
        if station_name not in self.config.stations:
            raise ValueError(f"Unknown station: {station_name}")
        
        station_config = self.config.stations[station_name]
        months = months or self.config.months_to_fetch
        
        df = self.scraper.scrape_historical(
            station_code=station_config["code"],
            station_name=station_name,
            months=months
        )
        
        return df
    
    def load_existing(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Load existing ingested data.
        
        Args:
            path: Path to CSV file, or uses latest in data dir
            
        Returns:
            DataFrame with weather data
        """
        if path and os.path.exists(path):
            return pd.read_csv(path, parse_dates=["date"])
        
        # Find latest CSV
        data_dir = Path(self.config.raw_data_dir)
        csv_files = list(data_dir.glob("weather_history_*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No weather data found in {data_dir}")
        
        latest = max(csv_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"[DATA_INGESTION] Loading {latest}")
        
        return pd.read_csv(latest, parse_dates=["date"])
    
    def get_data_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about ingested data."""
        return {
            "total_records": len(df),
            "stations": df["station_name"].nunique() if "station_name" in df.columns else 0,
            "date_range": {
                "start": str(df["date"].min()),
                "end": str(df["date"].max())
            },
            "columns": list(df.columns),
            "missing_percentage": df.isnull().mean().to_dict()
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test ingestion
    ingestion = DataIngestion()
    
    # Test single station
    print("Testing single station ingestion...")
    df = ingestion.ingest_station("COLOMBO", months=2)
    
    print(f"\nIngested {len(df)} records for COLOMBO")
    if not df.empty:
        print("\nSample data:")
        print(df.head())
        
        print("\nStats:")
        stats = ingestion.get_data_stats(df)
        for k, v in stats.items():
            print(f"  {k}: {v}")
