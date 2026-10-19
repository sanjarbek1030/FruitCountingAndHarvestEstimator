"""Exception hierarchy for the Fruit Counting & Harvest Estimator."""


class FruitCounterError(Exception):
    """Base class for all application errors."""


class ConfigError(FruitCounterError, ValueError):
    """Raised when configuration is missing or invalid."""


class ModelError(FruitCounterError):
    """Raised when a model cannot be found, loaded, or used."""


class InputError(FruitCounterError, ValueError):
    """Raised when an input image or video is missing, unreadable, or unsafe."""


class VideoError(InputError):
    """Raised when a video cannot be opened, decoded, or written."""


class YieldEstimationError(FruitCounterError, ValueError):
    """Raised when harvest-estimation parameters are invalid."""
