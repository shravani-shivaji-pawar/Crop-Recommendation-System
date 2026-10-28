"""Custom exception hierarchy for the AgriWise decision support system."""

class AgriWiseError(Exception):
    """Base class for all AgriWise application errors."""
    pass

class ValidationError(AgriWiseError):
    """Raised when user input fails agronomic or type constraints."""
    def __init__(self, message: str, field: str = None, value=None):
        super().__init__(message)
        self.field = field
        self.value = value

class ModelArtifactError(AgriWiseError):
    """Raised when a model or scaler artifact cannot be found or is corrupted."""
    pass

class PredictionError(AgriWiseError):
    """Raised when an error occurs during model inference."""
    pass
