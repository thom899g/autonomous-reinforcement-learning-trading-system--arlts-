"""
Autonomous Reinforcement Learning Trading System (ARLTS)
Core package for AI-driven trading using reinforcement learning.
"""

__version__ = "1.0.0"
__author__ = "Evolution Ecosystem AGI"
__description__ = "Autonomous trading system using RL for market interactions"

# Core exports
from arlts.config import ARLTSConfig
from arlts.environment import TradingEnvironment
from arlts.agent import TradingAgent
from arlts.data_handler import MarketDataHandler
from arlts.firebase_client import FirebaseClient

__all__ = [
    'ARLTSConfig',
    'TradingEnvironment',
    'TradingAgent',
    'MarketDataHandler',
    'FirebaseClient'
]