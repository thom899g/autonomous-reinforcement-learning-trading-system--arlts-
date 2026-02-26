"""
Configuration management for ARLTS with type safety and validation.
Uses Pydantic for robust configuration validation.
"""
import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from pydantic import BaseSettings, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logging.warning("Pydantic not available, using basic configuration")

class TradingMode(Enum):
    """Trading operation modes"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

class DataSource(Enum):
    """Supported data sources"""
    YFINANCE = "yfinance"
    CCXT = "ccxt"
    ALPACA = "alpaca"

if PYDANTIC_AVAILABLE:
    class ARLTSConfig(BaseSettings):
        """Main configuration class with validation"""
        
        # Trading Configuration
        trading_mode: TradingMode = Field(default=TradingMode.PAPER, env="TRADING_MODE")
        data_source: DataSource = Field(default=DataSource.YFINANCE, env="DATA_SOURCE")
        symbols: List[str] = Field(default=["AAPL", "MSFT", "GOOGL"], env="SYMBOLS")
        timeframe: str = Field(default="1h", env="TIMEFRAME")
        
        # RL Configuration
        learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1, env="LEARNING_RATE")
        gamma: float = Field(default=0.99, ge=0.9, le=0.999, env="GAMMA")
        batch_size: int = Field(default=32, ge=1, le=1024, env="BATCH_SIZE")
        buffer_size: int = Field(default=10000, ge=1000, le=100000, env="BUFFER_SIZE")
        
        # Neural Network Configuration
        hidden_layers: List[int] = Field(default=[256, 128, 64], env="HIDDEN_LAYERS")
        dropout_rate: float = Field(default=0.2, ge=0.0, le=0.5, env="DROPOUT_RATE")
        
        # Risk Management
        max_position_size: float = Field(default=0.1, ge=0.01, le=1.0, env="MAX_POSITION_SIZE")
        stop_loss_pct: float = Field(default=0.02, ge=0.005, le=0.1, env="STOP_LOSS_PCT")
        take_profit_pct: float = Field(default=0.04, ge=0.01, le=0.2, env="TAKE_PROFIT_PCT")
        
        # Firebase Configuration
        firebase_project_id: Optional[str] = Field(default=None, env="FIREBASE_PROJECT_ID")
        firebase_collection: str = Field(default="trading_states", env="FIREBASE_COLLECTION")
        
        # Logging Configuration
        log_level: str = Field(default="INFO", env="LOG_LEVEL")
        log_file: str = Field(default="logs/arlts.log", env="LOG_FILE")
        
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
        
        @validator("symbols")
        def validate_symbols(cls, v):
            if not v:
                raise ValueError("Symbols list cannot be empty")
            return [s.upper() for s in v]
        
        @validator("timeframe")
        def validate_timeframe(cls, v):
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
            if v not in valid_timeframes:
                raise ValueError(f"Invalid timeframe. Must be one of {valid_timeframes}")
            return v
else:
    @dataclass
    class ARLTSConfig:
        """Fallback configuration without Pydantic"""
        trading_mode: TradingMode = TradingMode.PAPER
        data_source: DataSource = DataSource.YFINANCE
        symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
        timeframe: str = "1h"
        learning_rate: float = 0.001
        gamma: float = 0.99
        batch_size: int = 32
        buffer_size: int = 10000
        hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
        dropout_rate: float = 0.2
        max_position_size: float = 0.1
        stop_loss_pct: float = 0.02
        take_profit_pct: float = 0.04
        firebase_project_id: Optional[str] = None
        firebase_collection: str = "trading_states"
        log_level: str = "INFO"
        log_file: str = "logs/arlts.log"

def load_config(config_path: Optional[str] = None) -> ARLTSConfig:
    """
    Load configuration from file or environment
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        ARLTSConfig instance
        
    Raises:
        FileNotFoundError: If config file specified but not found
        ValueError: If configuration is invalid
    """
    try:
        if PYDANTIC_AVAILABLE:
            if config_path and os.path.exists(config_path):
                return ARLTSConfig(_env_file=config_path)
            return ARLTS