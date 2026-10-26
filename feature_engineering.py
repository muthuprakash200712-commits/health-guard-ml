"""
HealthGuard ML: Feature Engineering & Preprocessing Pipelines
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

def compute_derived_clinical_features(
    systolic_bp: float,
    diastolic_bp: float,
    bmi: float,
    age: int,
    glucose: float,
    cholesterol: float = 200.0,
    heart_rate: float = 75.0,
    spo2: float = 98.0
) -> Dict[str, Any]:
    """
    Computes evidence-based physiological features:
    - Pulse Pressure = Systolic BP - Diastolic BP
    - Mean Arterial Pressure (MAP) = Diastolic BP + (Pulse Pressure / 3)
    - BMI categories and obesity flag
    - Glycemic status
    - Cardiovascular & metabolic vulnerability indices
    """
    pulse_pressure = max(10.0, systolic_bp - diastolic_bp)
    mean_arterial_pressure = diastolic_bp + (pulse_pressure / 3.0)
    
    is_obese = bmi >= 30.0
    is_overweight = (bmi >= 25.0) and (bmi < 30.0)
    is_hypertensive = (systolic_bp >= 130.0) or (diastolic_bp >= 80.0)
    
    if glucose < 100.0:
        glucose_category = "Normal"
    elif glucose <= 125.0:
        glucose_category = "Prediabetic"
    else:
        glucose_category = "Diabetic"
        
    return {
        "pulse_pressure": round(pulse_pressure, 1),
        "mean_arterial_pressure": round(mean_arterial_pressure, 1),
        "is_obese": is_obese,
        "is_overweight": is_overweight,
        "is_hypertensive": is_hypertensive,
        "glucose_category": glucose_category,
        "is_low_spo2": spo2 < 95.0,
        "is_tachycardia": heart_rate > 100.0,
        "is_bradycardia": heart_rate < 55.0
    }
