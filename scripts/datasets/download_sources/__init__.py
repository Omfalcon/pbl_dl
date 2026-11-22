"""Modular downloaders for PhishVision AI data sources."""
from .base import BaseDownloader, DownloadResult
from .openphish import OpenPhishDownloader
from .phishtank import PhishTankDownloader
from .phreshphish import PhreshPhishDownloader
from .tranco import TrancoDownloader
from .urlhaus import URLhausDownloader
from .zenodo import ZenodoDownloader

__all__ = [
    "BaseDownloader",
    "DownloadResult",
    "OpenPhishDownloader",
    "PhishTankDownloader",
    "PhreshPhishDownloader",
    "TrancoDownloader",
    "URLhausDownloader",
    "ZenodoDownloader",
]
