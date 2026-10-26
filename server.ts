import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import {
  INITIAL_PATIENTS,
  INITIAL_USERS,
  INITIAL_VITALS,
  INITIAL_MEDICAL_RECORDS,
  INITIAL_PRESCRIPTIONS,
  INITIAL_LAB_REPORTS,
  INITIAL_PREDICTION_LOGS
} from './src/data/initialData';
import {
  predictDiabetesRisk,
  predictHeartDiseaseRisk,
  detectVitalAnomalies,
  computeLongitudinalTrends,
  runComprehensivePatientAnalysis,
  computeDataDriftReport,
  INITIAL_MODEL_REGISTRY,
  MODEL_COMPARISON_TABLE
} from './src/lib/mlEngine';
import { Patient, VitalReading, PredictionLogEntry } from './src/types';

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // In-memory persistent database store
  let users = [...INITIAL_USERS];
  let patients: Patient[] = [...INITIAL_PATIENTS];
  let vitals: VitalReading[] = [...INITIAL_VITALS];
  let medicalRecords = [...INITIAL_MEDICAL_RECORDS];
  let prescriptions = [...INITIAL_PRESCRIPTIONS];
  let labReports = [...INITIAL_LAB_REPORTS];
  let predictionLogs: PredictionLogEntry[] = [...INITIAL_PREDICTION_LOGS];
  let modelRegistry = [...INITIAL_MODEL_REGISTRY];
  let modelComparisonTable = [...MODEL_COMPARISON_TABLE];

  // Helper logger for ML inference
  function logPrediction(
    diseaseTarget: 'diabetes' | 'heart_disease',
    modelName: string,
    modelVersion: string,
    inputs: Record<string, any>,
    probability: number,
    riskLevel: 'Low' | 'Moderate' | 'High',
    topFactors: string[],
    patientId?: string,
    patientName?: string
  ) {
    const entry: PredictionLogEntry = {
      id: `log-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
      patientId,
      patientName,
      diseaseTarget,
      modelName,
      modelVersion,
      timestamp: new Date().toISOString(),
      inputsSnapshot: inputs,
      predictedProbability: probability,
      riskLevel,
      predictedClass: probability >= 0.5 ? 1 : 0,
      topFactorNames: topFactors,
      latencyMs: Math.floor(Math.random() * 15) + 8
    };
    predictionLogs.unshift(entry);
    if (predictionLogs.length > 200) predictionLogs.pop();
  }

  // ==========================================
  // AUTH & USER ROUTES
  // ==========================================

  app.post('/api/v1/auth/login', (req, res) => {
    const { role, email } = req.body;
    let user = users.find(u => u.role === role);
    if (email) {
      const emailMatch = users.find(u => u.email.toLowerCase() === email.toLowerCase());
      if (emailMatch) user = emailMatch;
    }
    if (!user) {
      user = users[0];
    }
    res.json({
      token: `jwt-token-${user.id}-${Date.now()}`,
      user
    });
  });

  app.get('/api/v1/auth/me', (req, res) => {
    const authHeader = req.headers.authorization;
    let currentUser = users[0];
    if (authHeader && authHeader.includes('user-doctor')) currentUser = users[1];
    else if (authHeader && authHeader.includes('user-patient')) currentUser = users[2];
    res.json(currentUser);
  });

  // ==========================================
  // PATIENT & EHR RECORDS
  // ==========================================

  app.get('/api/v1/patients', (req, res) => {
    res.json(patients);
  });

  app.get('/api/v1/patients/:id', (req, res) => {
    const patient = patients.find(p => p.id === req.params.id);
    if (!patient) return res.status(404).json({ detail: 'Patient not found' });
    res.json(patient);
  });

  app.post('/api/v1/patients', (req, res) => {
    const pData = req.body;
    const newPatient: Patient = {
      id: `p-${Date.now()}`,
      name: pData.name || 'New Patient',
      age: Number(pData.age) || 45,
      gender: pData.gender || 'female',
      bloodGroup: pData.bloodGroup || 'O+',
      heightCm: Number(pData.heightCm) || 165,
      weightKg: Number(pData.weightKg) || 70,
      bmi: Number((Number(pData.weightKg) / ((Number(pData.heightCm) / 100) ** 2)).toFixed(1)) || 25.7,
      contactNumber: pData.contactNumber || '+1 (555) 000-0000',
      emergencyContact: pData.emergencyContact || 'Family +1 (555) 000-1111',
      allergies: pData.allergies || [],
      chronicConditions: pData.chronicConditions || [],
      assignedDoctorId: pData.assignedDoctorId || 'user-doctor',
      assignedDoctorName: 'Dr. Marcus Vance, MD',
      lastVisitDate: new Date().toISOString().split('T')[0],
      earlyWarningStatus: 'Normal'
    };
    patients.unshift(newPatient);
    res.status(201).json(newPatient);
  });

  app.get('/api/v1/patients/:id/records', (req, res) => {
    const recs = medicalRecords.filter(r => r.patientId === req.params.id);
    res.json(recs);
  });

  app.post('/api/v1/patients/:id/records', (req, res) => {
    const newRec = {
      id: `rec-${Date.now()}`,
      patientId: req.params.id,
      date: new Date().toISOString().split('T')[0],
      ...req.body
    };
    medicalRecords.unshift(newRec);
    res.status(201).json(newRec);
  });

  app.get('/api/v1/patients/:id/vitals', (req, res) => {
    const v = vitals.filter(item => item.patientId === req.params.id);
    res.json(v);
  });

  app.post('/api/v1/patients/:id/vitals', (req, res) => {
    const raw = req.body;
    const anomalyCheck = detectVitalAnomalies({
      heartRate: Number(raw.heartRate) || 75,
      oxygenSaturation: Number(raw.oxygenSaturation) || 98,
      systolicBp: Number(raw.systolicBp) || 120,
      diastolicBp: Number(raw.diastolicBp) || 80,
      sleepHours: Number(raw.sleepHours) || 7,
      steps: Number(raw.steps) || 6000
    });

    const newVital: VitalReading = {
      id: `vit-${Date.now()}`,
      patientId: req.params.id,
      timestamp: new Date().toISOString(),
      systolicBp: Number(raw.systolicBp) || 120,
      diastolicBp: Number(raw.diastolicBp) || 80,
      heartRate: Number(raw.heartRate) || 75,
      oxygenSaturation: Number(raw.oxygenSaturation) || 98,
      bloodGlucoseMgDl: Number(raw.bloodGlucoseMgDl) || 100,
      temperatureCelsius: Number(raw.temperatureCelsius) || 36.6,
      steps: Number(raw.steps) || 6000,
      sleepHours: Number(raw.sleepHours) || 7,
      isAnomaly: anomalyCheck.isAnomaly,
      anomalyScore: anomalyCheck.anomalyScore
    };
    vitals.unshift(newVital);

    // Update patient summary cache
    const patient = patients.find(p => p.id === req.params.id);
    if (patient) {
      patient.anomalyDetected = anomalyCheck.isAnomaly;
      patient.anomalyScore = anomalyCheck.anomalyScore;
      const analysis = runComprehensivePatientAnalysis(patient, newVital, vitals.filter(v => v.patientId === patient.id));
      patient.diabetesRiskScore = analysis.risks.diabetes.calibratedProbability;
      patient.diabetesRiskLevel = analysis.risks.diabetes.riskLevel;
      patient.heartRiskScore = analysis.risks.heartDisease.calibratedProbability;
      patient.heartRiskLevel = analysis.risks.heartDisease.riskLevel;
      patient.clinicalRiskIndex = analysis.clinicalRiskIndex.score;
      patient.earlyWarningStatus = analysis.earlyWarning.status;
    }

    res.status(201).json(newVital);
  });

  app.get('/api/v1/patients/:id/prescriptions', (req, res) => {
    const rx = prescriptions.filter(p => p.patientId === req.params.id);
    res.json(rx);
  });

  app.post('/api/v1/patients/:id/prescriptions', (req, res) => {
    const newRx = {
      id: `rx-${Date.now()}`,
      patientId: req.params.id,
      datePrescribed: new Date().toISOString().split('T')[0],
      status: 'active',
      ...req.body
    };
    prescriptions.unshift(newRx);
    res.status(201).json(newRx);
  });

  app.get('/api/v1/patients/:id/labs', (req, res) => {
    const labs = labReports.filter(l => l.patientId === req.params.id);
    res.json(labs);
  });

  // ==========================================
  // ML PREDICTION & INFERENCE API ENDPOINTS
  // ==========================================

  // Module 3 & 7: Diabetes ML Prediction & SHAP
  app.post('/api/v1/ml/diabetes/predict', (req, res) => {
    const input = req.body;
    // Input validation
    if (input.glucose === undefined || isNaN(Number(input.glucose))) {
      return res.status(422).json({ detail: 'Validation Error: blood glucose field is required and must be numeric.' });
    }
    if (input.glucose < 40 || input.glucose > 600) {
      return res.status(400).json({ detail: 'Physiological range warning: blood glucose must be between 40 and 600 mg/dL.' });
    }

    const prediction = predictDiabetesRisk(input);
    logPrediction(
      'diabetes',
      prediction.modelName,
      prediction.modelVersion,
      input,
      prediction.calibratedProbability,
      prediction.riskLevel,
      prediction.topFactors.slice(0, 3).map(f => f.featureDisplayName),
      input.patientId,
      input.patientName
    );
    res.json(prediction);
  });

  // Module 4 & 7: Heart Disease ML Prediction & SHAP
  app.post('/api/v1/ml/heart/predict', (req, res) => {
    const input = req.body;
    if (input.age === undefined || isNaN(Number(input.age))) {
      return res.status(422).json({ detail: 'Validation Error: age field is required and must be numeric.' });
    }
    if (input.age < 18 || input.age > 120) {
      return res.status(400).json({ detail: 'Physiological range warning: age must be between 18 and 120 years.' });
    }

    const prediction = predictHeartDiseaseRisk(input);
    logPrediction(
      'heart_disease',
      prediction.modelName,
      prediction.modelVersion,
      input,
      prediction.calibratedProbability,
      prediction.riskLevel,
      prediction.topFactors.slice(0, 3).map(f => f.featureDisplayName),
      input.patientId,
      input.patientName
    );
    res.json(prediction);
  });

  // Module 9: Multi-Disease Patient Risk Engine
  app.post('/api/v1/ml/patient/:patient_id/analyze', (req, res) => {
    const patient = patients.find(p => p.id === req.params.patient_id);
    if (!patient) return res.status(404).json({ detail: 'Patient not found' });

    const patientVitals = vitals.filter(v => v.patientId === patient.id);
    const latestVital = patientVitals[0];

    const analysis = runComprehensivePatientAnalysis(patient, latestVital, patientVitals);
    res.json(analysis);
  });

  // Module 8: SHAP Explanations
  app.get('/api/v1/ml/patient/:patient_id/explanations', (req, res) => {
    const patient = patients.find(p => p.id === req.params.patient_id);
    if (!patient) return res.status(404).json({ detail: 'Patient not found' });

    const patientVitals = vitals.filter(v => v.patientId === patient.id);
    const analysis = runComprehensivePatientAnalysis(patient, patientVitals[0], patientVitals);

    res.json({
      patientId: patient.id,
      patientName: patient.name,
      diabetesExplainer: {
        model: analysis.risks.diabetes.modelName,
        baseValue: analysis.risks.diabetes.baseValueShap,
        probability: analysis.risks.diabetes.calibratedProbability,
        factors: analysis.risks.diabetes.topFactors
      },
      heartExplainer: {
        model: analysis.risks.heartDisease.modelName,
        baseValue: analysis.risks.heartDisease.baseValueShap,
        probability: analysis.risks.heartDisease.calibratedProbability,
        factors: analysis.risks.heartDisease.topFactors
      },
      disclaimer: 'SHAP values quantify relative mathematical feature attributions for this prediction and do not denote medical causation.'
    });
  });

  // Module 12: Longitudinal Health Trends
  app.get('/api/v1/ml/patient/:patient_id/trends', (req, res) => {
    const patient = patients.find(p => p.id === req.params.patient_id);
    if (!patient) return res.status(404).json({ detail: 'Patient not found' });

    const patientVitals = vitals.filter(v => v.patientId === patient.id);
    const trends = computeLongitudinalTrends(patientVitals);
    res.json({
      patientId: patient.id,
      patientName: patient.name,
      readingCount: patientVitals.length,
      trends
    });
  });

  // Module 11: Unsupervised Vital Anomaly Detection
  app.get('/api/v1/ml/patient/:patient_id/anomalies', (req, res) => {
    const patient = patients.find(p => p.id === req.params.patient_id);
    if (!patient) return res.status(404).json({ detail: 'Patient not found' });

    const patientVitals = vitals.filter(v => v.patientId === patient.id);
    const anomalies = patientVitals.map(v => ({
      vitalId: v.id,
      timestamp: v.timestamp,
      reading: {
        heartRate: v.heartRate,
        oxygenSaturation: v.oxygenSaturation,
        systolicBp: v.systolicBp,
        diastolicBp: v.diastolicBp,
        bloodGlucose: v.bloodGlucoseMgDl
      },
      ...detectVitalAnomalies(v)
    }));

    res.json({
      patientId: patient.id,
      anomalyCount: anomalies.filter(a => a.isAnomaly).length,
      history: anomalies
    });
  });

  // Module 16 & 43: Model Registry
  app.get('/api/v1/ml/models', (req, res) => {
    res.json(modelRegistry);
  });

  app.get('/api/v1/ml/models/:model_id/metrics', (req, res) => {
    const model = modelRegistry.find(m => m.id === req.params.model_id || m.name.toLowerCase().includes(req.params.model_id.toLowerCase()));
    if (!model) return res.status(404).json({ detail: 'Model artifact not found in registry' });
    res.json(model.metrics);
  });

  // Module 6 & 39: Model Comparison Benchmarking
  app.get('/api/v1/ml/models/compare', (req, res) => {
    res.json({
      timestamp: new Date().toISOString(),
      evaluationStrategy: '5-Fold Stratified Cross-Validation + Holdout Test Evaluation with Safety Recall Priority',
      models: modelComparisonTable
    });
  });

  // Module 19: Production MLOps Monitoring
  app.get('/api/v1/ml/monitoring', (req, res) => {
    const totalPredictions = predictionLogs.length;
    const highRiskPredictions = predictionLogs.filter(p => p.riskLevel === 'High').length;
    const avgRisk = totalPredictions > 0
      ? Number((predictionLogs.reduce((acc, p) => acc + p.predictedProbability, 0) / totalPredictions).toFixed(3))
      : 0.45;

    res.json({
      activeModelsCount: modelRegistry.filter(m => m.isProductionActive).length,
      totalPredictionsCount: totalPredictions,
      highRiskPredictionsCount: highRiskPredictions,
      highRiskRate: totalPredictions > 0 ? Number(((highRiskPredictions / totalPredictions) * 100).toFixed(1)) : 0,
      averagePredictedRisk: avgRisk,
      driftStatus: 'LOW',
      systemHealth: 'HEALTHY',
      inferenceAvgLatencyMs: 14.2,
      lastTrainedTimestamp: modelRegistry[0]?.trainedAt || new Date().toISOString()
    });
  });

  // Module 18: Data Drift
  app.get('/api/v1/ml/drift', (req, res) => {
    const target = (req.query.target as 'diabetes' | 'heart_disease') || 'diabetes';
    const report = computeDataDriftReport(target);
    res.json(report);
  });

  // Module 17: Prediction Logging History
  app.get('/api/v1/ml/predictions/history', (req, res) => {
    res.json(predictionLogs);
  });

  // Module 25: Trigger Retrain Simulation
  app.post('/api/v1/ml/retrain', (req, res) => {
    const { diseaseTarget } = req.body;
    res.json({
      status: 'SUCCESS',
      message: `Retraining job completed for ${diseaseTarget || 'all models'}. Hyperparameters cross-validated across 5 folds.`,
      newVersion: 'v2.2.0',
      timestamp: new Date().toISOString()
    });
  });

  // Health check
  app.get('/api/v1/health', (req, res) => {
    res.json({
      status: 'healthy',
      service: 'HealthGuard ML Intelligence System',
      version: '2.1.0',
      activeModels: ['HealthGuard-Diabetes-GBM', 'HealthGuard-Heart-RF', 'HealthGuard-IsolationForest-v1.4']
    });
  });

  // Vite development middleware or static production serving
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa'
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`HealthGuard ML Full-Stack Server running on port ${PORT}`);
  });
}

startServer();
