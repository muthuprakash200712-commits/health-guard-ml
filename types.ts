export type UserRole = 'admin' | 'doctor' | 'patient' | 'ml_engineer';

export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  avatarUrl?: string;
  specialty?: string;
  patientId?: string;
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: 'male' | 'female' | 'other';
  bloodGroup: string;
  heightCm: number;
  weightKg: number;
  bmi: number;
  contactNumber: string;
  emergencyContact: string;
  allergies: string[];
  chronicConditions: string[];
  assignedDoctorId: string;
  assignedDoctorName: string;
  lastVisitDate: string;
  diabetesRiskScore?: number;
  diabetesRiskLevel?: 'Low' | 'Moderate' | 'High';
  heartRiskScore?: number;
  heartRiskLevel?: 'Low' | 'Moderate' | 'High';
  clinicalRiskIndex?: number;
  anomalyDetected?: boolean;
  anomalyScore?: number;
  earlyWarningStatus?: 'Normal' | 'Monitor' | 'Moderate Attention' | 'High Attention';
}

export interface VitalReading {
  id: string;
  patientId: string;
  timestamp: string;
  systolicBp: number;
  diastolicBp: number;
  heartRate: number;
  oxygenSaturation: number; // SpO2 %
  bloodGlucoseMgDl: number;
  temperatureCelsius?: number;
  respiratoryRate?: number;
  steps?: number;
  sleepHours?: number;
  isAnomaly?: boolean;
  anomalyScore?: number;
}

export interface MedicalRecord {
  id: string;
  patientId: string;
  date: string;
  diagnosis: string;
  symptoms: string[];
  treatmentPlan: string;
  doctorNotes: string;
  doctorId: string;
  doctorName: string;
}

export interface Prescription {
  id: string;
  patientId: string;
  datePrescribed: string;
  medicationName: string;
  dosage: string;
  frequency: string;
  durationDays: number;
  instructions: string;
  prescribingDoctor: string;
  status: 'active' | 'completed' | 'discontinued';
}

export interface LabReport {
  id: string;
  patientId: string;
  testName: string;
  testCategory: 'Biochemistry' | 'Hematology' | 'Lipid Panel' | 'Cardiac Markers' | 'Metabolic';
  dateConducted: string;
  results: {
    parameter: string;
    value: number;
    unit: string;
    referenceRange: string;
    isAbnormal: boolean;
  }[];
  labTechnicianNotes: string;
  status: 'normal' | 'abnormal' | 'critical';
}

export interface DerivedClinicalFeatures {
  pulsePressure: number;
  meanArterialPressure: number;
  bmiCategory: 'Underweight' | 'Normal' | 'Overweight' | 'Obese';
  isObese: boolean;
  ageCategory: 'Young' | 'Middle' | 'Senior' | 'Elderly';
  glucoseCategory: 'Normal' | 'Prediabetic' | 'Diabetic';
  isHypertensive: boolean;
  cholesterolRiskCategory: 'Desirable' | 'Borderline High' | 'High';
  isTachycardia: boolean;
  isBradycardia: boolean;
  isLowSpO2: boolean;
  abnormalVitalCount: number;
  comorbidityCount: number;
  cardiovascularRiskIndicator: number; // 0 to 1
  metabolicRiskIndicator: number; // 0 to 1
  sleepActivityRiskIndicator: number; // 0 to 1
}

export interface ShapFactor {
  feature: string;
  featureDisplayName: string;
  rawValue: number | string;
  unit?: string;
  shapValue: number; // positive increases risk, negative decreases risk
  impactPercent: number;
  direction: 'increases_risk' | 'decreases_risk' | 'neutral';
  clinicalContext: string;
}

export interface DiseaseRiskPrediction {
  disease: 'diabetes' | 'heart_disease';
  diseaseDisplayName: string;
  modelName: string;
  modelVersion: string;
  algorithm: string;
  rawProbability: number;
  calibratedProbability: number;
  riskLevel: 'Low' | 'Moderate' | 'High';
  predictionClass: 0 | 1;
  confidenceScore: number;
  topFactors: ShapFactor[];
  baseValueShap: number;
  derivedFeatures: DerivedClinicalFeatures;
  recommendationSpecialty: string;
  recommendedUrgency: 'Routine' | 'Elective' | 'Urgent' | 'Immediate Evaluation' | 'Within 1-2 Weeks' | 'Within 48 Hours';
  explanationText: string;
  timestamp: string;
}

export interface MultiDiseaseAnalysis {
  patientId: string;
  timestamp: string;
  clinicalRiskIndex: {
    score: number; // 0-100 heuristic
    category: 'Low' | 'Moderate' | 'High' | 'Critical';
    methodology: 'Clinical rule-based aggregate (non-ML heuristic)';
    contributingRules: string[];
  };
  risks: {
    diabetes: DiseaseRiskPrediction;
    heartDisease: DiseaseRiskPrediction;
  };
  anomalyDetection: {
    isAnomaly: boolean;
    anomalyScore: number;
    normalizedScore: number; // 0 to 1
    flaggedVitals: string[];
    summary: string;
    model: string;
  };
  earlyWarning: {
    status: 'Normal' | 'Monitor' | 'Moderate Attention' | 'High Attention';
    priorityScore: number; // 0-100
    mlRiskContribution: number;
    anomalyContribution: number;
    vitalSpikeContribution: number;
    clinicalAlerts: string[];
    actionItems: string[];
  };
  appointmentRecommendation: {
    recommendedSpecialty: string;
    urgencyLevel: 'Routine' | 'Within 1-2 Weeks' | 'Within 48 Hours' | 'Immediate Attention';
    clinicalRationale: string;
    suggestedScreenings: string[];
  };
}

export interface ModelMetricDetails {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  confusionMatrix: {
    trueNegative: number;
    falsePositive: number;
    falseNegative: number;
    truePositive: number;
  };
  rocCurve?: {
    fpr: number[];
    tpr: number[];
    thresholds: number[];
  };
  featureImportances: {
    feature: string;
    displayName: string;
    importance: number;
    shapMeanAbs: number;
  }[];
  hyperparameters: Record<string, any>;
  trainTestSplitRatio: string;
  trainingSamples: number;
  testSamples: number;
  trainingDurationSeconds: number;
  crossValidationFolds: number;
  cvMeanRocAuc: number;
  cvStdRocAuc: number;
}

export interface RegisteredModel {
  id: string;
  name: string;
  diseaseTarget: 'diabetes' | 'heart_disease' | 'vital_anomaly';
  version: string;
  algorithm: string;
  description: string;
  isProductionActive: boolean;
  trainedAt: string;
  datasetName: string;
  datasetSize: number;
  metrics: ModelMetricDetails;
  selectionScore: number;
  selectionRationale: string;
}

export interface ModelComparisonRow {
  modelId: string;
  algorithm: string;
  diseaseTarget: 'diabetes' | 'heart_disease';
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
  isSelectedForProduction: boolean;
  trainingTimeMs: number;
  cvScore: number;
  notes: string;
}

export interface VitalTrend {
  metric: string;
  displayName: string;
  unit: string;
  direction: 'increasing' | 'decreasing' | 'stable' | 'fluctuating';
  slope: number;
  recentAverage: number;
  baselineAverage: number;
  percentageChange: number;
  status: 'normal' | 'improving' | 'deteriorating' | 'monitor';
  clinicalInterpretation: string;
  dataPoints: {
    date: string;
    value: number;
    rollingAvg7d: number;
  }[];
}

export interface DataDriftFeatureResult {
  feature: string;
  displayName: string;
  baselineMean: number;
  baselineStd: number;
  productionMean: number;
  productionStd: number;
  psiScore: number; // Population Stability Index
  ksStatistic: number; // Kolmogorov-Smirnov
  ksPValue: number;
  driftStatus: 'Low' | 'Moderate' | 'High';
  interpretation: string;
}

export interface DataDriftReport {
  timestamp: string;
  modelTarget: 'diabetes' | 'heart_disease';
  overallStatus: 'LOW' | 'MODERATE' | 'HIGH';
  baselineSampleCount: number;
  productionSampleCount: number;
  features: DataDriftFeatureResult[];
  recommendedAction: string;
}

export interface PredictionLogEntry {
  id: string;
  patientId?: string;
  patientName?: string;
  diseaseTarget: 'diabetes' | 'heart_disease';
  modelName: string;
  modelVersion: string;
  timestamp: string;
  inputsSnapshot: Record<string, any>;
  predictedProbability: number;
  riskLevel: 'Low' | 'Moderate' | 'High';
  predictedClass: 0 | 1;
  topFactorNames: string[];
  latencyMs: number;
}
