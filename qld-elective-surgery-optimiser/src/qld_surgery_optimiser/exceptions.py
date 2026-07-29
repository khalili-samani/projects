"""Project-specific exception hierarchy."""


class SurgeryOptimiserError(Exception):
    """Base exception for known application failures."""


class ConfigurationError(SurgeryOptimiserError):
    """Raised when application or scenario configuration is invalid."""


class SourceDiscoveryError(SurgeryOptimiserError):
    """Raised when source resources cannot be discovered safely."""


class DownloadError(SurgeryOptimiserError):
    """Raised when a source resource cannot be downloaded or verified."""


class DataValidationError(SurgeryOptimiserError):
    """Raised when source data fails mandatory validation rules."""


class EntityResolutionError(SurgeryOptimiserError):
    """Raised when required facility or service entities cannot be resolved."""


class OptimisationError(SurgeryOptimiserError):
    """Raised when the optimisation workflow cannot complete."""


class InfeasibleModelError(OptimisationError):
    """Raised when no allocation satisfies the mandatory constraints."""