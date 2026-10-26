"""
HealthGuard ML: Data Preprocessing & Validation
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional

try:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.pipeline import Pipeline
except ImportError:
    pass

def create_clinical_preprocessor(scaling: str = "standard") -> Any:
    """Returns a scikit-learn Pipeline with Median Imputation + Feature Scaling."""
    scaler = StandardScaler() if scaling == "standard" else RobustScaler()
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])

def validate_clinical_inputs(data: dict) -> Tuple[bool, Optional[str]]:
    """Validates physical bounds for physiological clinical measurements."""
    if "glucose" in data:
        val = data["glucose"]
        if val < 40 or val > 600:
            return False, f"Glucose value {val} is outside physiologically viable range (40-600 mg/dL)"
    if "age" in data:
        val = data["age"]
        if val < 1 or val > 120:
            return False, f"Age {val} is outside standard bounds (1-120 years)"
    if "systolic_bp" in data:
        val = data["systolic_bp"]
        if val < 60 or val > 260:
            return False, f"Systolic BP {val} is outside viable range (60-260 mmHg)"
    return True, None
