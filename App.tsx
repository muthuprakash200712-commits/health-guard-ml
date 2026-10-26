import React, { useState } from 'react';
import {
  Activity,
  Cpu,
  Layers,
  Zap,
  Sparkles,
  TrendingUp,
  Radio,
  Users,
  FileText,
  Heart
} from 'lucide-react';
import {
  INITIAL_PATIENTS,
  INITIAL_VITALS,
  INITIAL_MEDICAL_RECORDS,
  INITIAL_PRESCRIPTIONS,
  INITIAL_LAB_REPORTS,
  INITIAL_PREDICTION_LOGS
} from './data/initialData';
import {
  INITIAL_MODEL_REGISTRY,
  MODEL_COMPARISON_TABLE,
  runComprehensivePatientAnalysis,
  detectVitalAnomalies
} from './lib/mlEngine';
import { UserRole, Patient, VitalReading, Prescription, LabReport } from './types';
import { Navbar } from './components/Navbar';
import { DisclaimerBanner } from './components/DisclaimerBanner';
import { MLOverviewTab } from './components/MLIntelligence/MLOverviewTab';
import { ModelComparisonTab } from './components/MLIntelligence/ModelComparisonTab';
import { LiveInferenceTab } from './components/MLIntelligence/LiveInferenceTab';
import { ExplainableAITab } from './components/MLIntelligence/ExplainableAITab';
import { AnomalyDetectionTab } from './components/MLIntelligence/AnomalyDetectionTab';
import { DataDriftTab } from './components/MLIntelligence/DataDriftTab';
import { DoctorPatientList } from './components/DoctorDashboard/DoctorPatientList';
import { DoctorPatientDetail } from './components/DoctorDashboard/DoctorPatientDetail';
import { PatientHealthSummary } from './components/PatientPortal/PatientHealthSummary';
import { PatientRecordsManager } from './components/Records/PatientRecordsManager';

export default function App() {
  const [activeRole, setActiveRole] = useState<UserRole>('doctor');
  const [activeTab, setActiveTab] = useState<string>('doctor_roster');

  // Application Data States
  const [patients, setPatients] = useState<Patient[]>(INITIAL_PATIENTS);
  const [vitals, setVitals] = useState<VitalReading[]>(INITIAL_VITALS);
  const [medicalRecords, setMedicalRecords] = useState(INITIAL_MEDICAL_RECORDS);
  const [prescriptions, setPrescriptions] = useState(INITIAL_PRESCRIPTIONS);
  const [labReports, setLabReports] = useState(INITIAL_LAB_REPORTS);
  const [models, setModels] = useState(INITIAL_MODEL_REGISTRY);
  const [comparisonRows, setComparisonRows] = useState(MODEL_COMPARISON_TABLE);
  const [inferenceLogs, setInferenceLogs] = useState(INITIAL_PREDICTION_LOGS);
  const [isRetraining, setIsRetraining] = useState(false);

  // Active Selected Patient for deep-dive
  const [selectedPatientId, setSelectedPatientId] = useState<string>('P-101');

  const selectedPatient = patients.find(p => p.id === selectedPatientId) || patients[0];

  // Handle pipeline retrain simulation
  const handleRetrainTrigger = () => {
    setIsRetraining(true);
    setTimeout(() => {
      setIsRetraining(false);
      setModels(prev => prev.map(m => ({
        ...m,
        trainedAt: new Date().toISOString()
      })));
    }, 1400);
  };

  // Handle switching role
  const handleRoleChange = (newRole: UserRole) => {
    setActiveRole(newRole);
    if (newRole === 'ml_engineer') {
      setActiveTab('ml_overview');
    } else if (newRole === 'doctor') {
      setActiveTab('doctor_roster');
    } else if (newRole === 'patient') {
      setActiveTab('patient_portal');
    } else {
      setActiveTab('ml_overview');
    }
  };

  // Handle adding new patient by clinician
  const handleAddPatient = (
    newPatientData: Omit<Patient, 'id'>,
    initialVitalData: Omit<VitalReading, 'id' | 'patientId' | 'timestamp'>
  ) => {
    const newPatientId = `P-${100 + patients.length + 1}`;

    const anomalyCheck = detectVitalAnomalies({
      heartRate: initialVitalData.heartRate || 75,
      oxygenSaturation: initialVitalData.oxygenSaturation || 98,
      systolicBp: initialVitalData.systolicBp || 120,
      diastolicBp: initialVitalData.diastolicBp || 80,
      sleepHours: initialVitalData.sleepHours || 7,
      steps: initialVitalData.steps || 5000,
      temperatureCelsius: initialVitalData.temperatureCelsius || 37.0
    });

    const initialReading: VitalReading = {
      id: `v-${Date.now()}`,
      patientId: newPatientId,
      timestamp: new Date().toISOString(),
      heartRate: initialVitalData.heartRate || 75,
      systolicBp: initialVitalData.systolicBp || 120,
      diastolicBp: initialVitalData.diastolicBp || 80,
      oxygenSaturation: initialVitalData.oxygenSaturation || 98,
      bloodGlucoseMgDl: initialVitalData.bloodGlucoseMgDl || 100,
      steps: initialVitalData.steps || 5000,
      sleepHours: initialVitalData.sleepHours || 7,
      temperatureCelsius: initialVitalData.temperatureCelsius || 37.0,
      isAnomaly: anomalyCheck.isAnomaly,
      anomalyScore: anomalyCheck.anomalyScore
    };

    const preliminaryPatient: Patient = {
      ...newPatientData,
      id: newPatientId
    };

    const analysis = runComprehensivePatientAnalysis(preliminaryPatient, initialReading, [initialReading]);

    const completeNewPatient: Patient = {
      ...preliminaryPatient,
      diabetesRiskScore: analysis.risks.diabetes.calibratedProbability,
      diabetesRiskLevel: analysis.risks.diabetes.riskLevel,
      heartRiskScore: analysis.risks.heartDisease.calibratedProbability,
      heartRiskLevel: analysis.risks.heartDisease.riskLevel,
      earlyWarningStatus: analysis.earlyWarning.status,
      clinicalRiskIndex: analysis.clinicalRiskIndex.score,
      anomalyDetected: analysis.anomalyDetection.isAnomaly,
      anomalyScore: analysis.anomalyDetection.anomalyScore
    };

    const intakeRecord = {
      id: `ehr-${Date.now()}`,
      patientId: newPatientId,
      date: new Date().toISOString().split('T')[0],
      diagnosis: newPatientData.chronicConditions.length > 0 && !newPatientData.chronicConditions.includes('None reported')
        ? newPatientData.chronicConditions.join(', ')
        : 'Initial Clinical Intake & Baseline Health Screening',
      doctorName: newPatientData.assignedDoctorName,
      department: 'Internal Medicine / Preventive Cardiology',
      notes: `Patient registered with baseline ML risk evaluation. Diabetes Risk: ${Math.round(analysis.risks.diabetes.calibratedProbability * 100)}%, Heart Disease Risk: ${Math.round(analysis.risks.heartDisease.calibratedProbability * 100)}%. Early Warning Status: ${analysis.earlyWarning.status}.`,
      status: 'Active'
    };

    const newAuditLog = {
      id: `log-${Date.now()}`,
      timestamp: new Date().toISOString(),
      patientId: newPatientId,
      patientName: completeNewPatient.name,
      diseaseTarget: 'diabetes' as const,
      modelName: 'XGBoost Calibrated Classifier',
      modelVersion: 'v2.1-calibrated',
      predictedProbability: analysis.risks.diabetes.calibratedProbability,
      riskLevel: analysis.risks.diabetes.riskLevel,
      topFactorNames: analysis.risks.diabetes.topFactors.slice(0, 3).map(s => s.featureDisplayName),
      latencyMs: 12
    };

    setPatients(prev => [completeNewPatient, ...prev]);
    setVitals(prev => [initialReading, ...prev]);
    setMedicalRecords(prev => [intakeRecord, ...prev]);
    setInferenceLogs(prev => [newAuditLog, ...prev]);
    setSelectedPatientId(newPatientId);
  };

  // Handle logging a new vital reading
  const handleAddVital = (newVitalData: Partial<VitalReading>) => {
    const anomalyCheck = detectVitalAnomalies({
      heartRate: newVitalData.heartRate || 75,
      oxygenSaturation: newVitalData.oxygenSaturation || 98,
      systolicBp: newVitalData.systolicBp || 120,
      diastolicBp: newVitalData.diastolicBp || 80,
      sleepHours: newVitalData.sleepHours || 7,
      steps: newVitalData.steps || 5000,
      temperatureCelsius: newVitalData.temperatureCelsius || 37.0
    });
    const reading: VitalReading = {
      id: `v-${Date.now()}`,
      patientId: selectedPatient.id,
      timestamp: new Date().toISOString(),
      heartRate: newVitalData.heartRate || 75,
      systolicBp: newVitalData.systolicBp || 120,
      diastolicBp: newVitalData.diastolicBp || 80,
      oxygenSaturation: newVitalData.oxygenSaturation || 98,
      bloodGlucoseMgDl: newVitalData.bloodGlucoseMgDl || 100,
      steps: newVitalData.steps || 5000,
      sleepHours: newVitalData.sleepHours || 7,
      temperatureCelsius: newVitalData.temperatureCelsius || 37.0,
      isAnomaly: anomalyCheck.isAnomaly,
      anomalyScore: anomalyCheck.anomalyScore
    };

    const updatedVitals = [reading, ...vitals];
    setVitals(updatedVitals);

    // Update patient ML risk stats in state
    const patientVitals = updatedVitals.filter(v => v.patientId === selectedPatient.id);
    const analysis = runComprehensivePatientAnalysis(selectedPatient, reading, patientVitals);

    const updatedPatients = patients.map(p => {
      if (p.id === selectedPatient.id) {
        return {
          ...p,
          diabetesRiskScore: analysis.risks.diabetes.calibratedProbability,
          diabetesRiskLevel: analysis.risks.diabetes.riskLevel,
          heartRiskScore: analysis.risks.heartDisease.calibratedProbability,
          heartRiskLevel: analysis.risks.heartDisease.riskLevel,
          earlyWarningStatus: analysis.earlyWarning.status,
          clinicalRiskIndex: analysis.clinicalRiskIndex.score,
          anomalyDetected: analysis.anomalyDetection.isAnomaly
        };
      }
      return p;
    });

    setPatients(updatedPatients);
  };

  const handleAddPrescription = (newRx: Partial<Prescription>) => {
    const rx: Prescription = {
      id: `rx-${Date.now()}`,
      patientId: newRx.patientId || selectedPatient.id,
      medicationName: newRx.medicationName || 'New Medication',
      dosage: newRx.dosage || '10mg',
      frequency: newRx.frequency || 'Once daily',
      durationDays: newRx.durationDays || 30,
      instructions: newRx.instructions || 'Take as directed',
      datePrescribed: newRx.datePrescribed || new Date().toISOString().split('T')[0],
      status: (newRx.status as 'active' | 'completed' | 'discontinued') || 'active',
      prescribingDoctor: newRx.prescribingDoctor || selectedPatient.assignedDoctorName
    };
    setPrescriptions(prev => [rx, ...prev]);
  };

  const handleOrderLab = (newLab: Partial<LabReport>) => {
    const lab: LabReport = {
      id: `lab-${Date.now()}`,
      patientId: newLab.patientId || selectedPatient.id,
      testName: newLab.testName || 'Diagnostic Panel',
      testCategory: newLab.testCategory || 'Metabolic',
      dateConducted: newLab.dateConducted || new Date().toISOString().split('T')[0],
      results: newLab.results || [],
      labTechnicianNotes: newLab.labTechnicianNotes || 'Diagnostic test requested by attending clinician.',
      status: (newLab.status as 'normal' | 'abnormal' | 'critical') || 'normal'
    };
    setLabReports(prev => [lab, ...prev]);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col antialiased selection:bg-sky-500 selection:text-white font-sans">
      {/* Top Universal Navbar */}
      <Navbar
        activeRole={activeRole}
        onRoleChange={handleRoleChange}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Medical Disclaimer Banner */}
        <DisclaimerBanner />

        {/* Dynamic Context Secondary Tab Navigation */}
        <div className="flex flex-wrap items-center gap-2.5 pb-2">
          {/* ML Intelligence Subtabs */}
          {activeRole === 'ml_engineer' && (
            <>
              <button
                onClick={() => setActiveTab('ml_overview')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-black border-2 border-black transition-all cursor-pointer ${
                  activeTab === 'ml_overview'
                    ? 'bg-yellow-300 text-black shadow-[3px_3px_0px_0px_#000] -translate-x-0.5 -translate-y-0.5'
                    : 'bg-white text-black hover:bg-yellow-100 shadow-[2px_2px_0px_0px_#000]'
                }`}
              >
                <Cpu className="h-4 w-4 stroke-[2.5]" />
                <span>Model Registry & MLOps</span>
              </button>

              <button
                onClick={() => setActiveTab('model_comparison')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-black border-2 border-black transition-all cursor-pointer ${
                  activeTab === 'model_comparison'
                    ? 'bg-yellow-300 text-black shadow-[3px_3px_0px_0px_#000] -translate-x-0.5 -translate-y-0.5'
                    : 'bg-white text-black hover:bg-yellow-100 shadow-[2px_2px_0px_0px_#000]'
                }`}
              >
                <Layers className="h-4 w-4 stroke-[2.5]" />
                <span>Algorithm Benchmark & ROC</span>
              </button>

              <button
                onClick={() => setActiveTab('live_inference')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-black border-2 border-black transition-all cursor-pointer ${
                  activeTab === 'live_inference'
                    ? 'bg-yellow-300 text-black shadow-[3px_3px_0px_0px_#000] -translate-x-0.5 -translate-y-0.5'
                    : 'bg-white text-black hover:bg-yellow-100 shadow-[2px_2px_0px_0px_#000]'
                }`}
              >
                <Zap className="h-4 w-4 stroke-[2.5]" />
                <span>Live Inference Studio</span>
              </button>

              <button
                onClick={() => setActiveTab('explainable_ai')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-black border-2 border-black transition-all cursor-pointer ${
                  activeTab === 'explainable_ai'
                    ? 'bg-yellow-300 text-black shadow-[3px_3px_0px_0px_#000] -translate-x-0.5 -translate-y-0.5'
                    : 'bg-white text-black hover:bg-yellow-100 shadow-[2px_2px_0px_0px_#000]'
                }`}
              >
                <Sparkles className="h-4 w-4 stroke-[2.5]" />
                <span>Explainable AI & SHAP</span>
              </button>

              <button
                onClick={() => setActiveTab('anomaly_detection')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-black border-2 border-black transition-all cursor-pointer ${
                  activeTab === 'anomaly_detection'
                    ? 'bg-yellow-300 text-black shadow-[3px_3px_0px_0px_#000] -translate-x-0.5 -translate-y-0.5'
                    : 'bg-white text-black hover:bg-yellow-100 shadow-[2px_2px_0px_0px_#000]'
                }`}
              >
                <Radio className="h-4 w-4 stroke-[2.5]" />
                <span>Isolation Forest Anomaly</span>
              </button>

              <button
                onClick={() => setActiveTab('data_drift')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-black border-2 border-black transition-all cursor-pointer ${
                  activeTab === 'data_drift'
                    ? 'bg-yellow-300 text-black shadow-[3px_3px_0px_0px_#000] -translate-x-0.5 -translate-y-0.5'
                    : 'bg-white text-black hover:bg-yellow-100 shadow-[2px_2px_0px_0px_#000]'
                }`}
              >
                <TrendingUp className="h-4 w-4 stroke-[2.5]" />
                <span>Data Drift & Covariate Shift</span>
              </button>
            </>
          )}

          {/* Doctor View Subtabs */}
          {activeRole === 'doctor' && (
            <>
              <button
                onClick={() => setActiveTab('doctor_roster')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-black border-2 border-black transition-all cursor-pointer ${
                  activeTab === 'doctor_roster'
                    ? 'bg-sky-400 text-black shadow-[3px_3px_0px_0px_#000] -translate-x-0.5 -translate-y-0.5'
                    : 'bg-white text-black hover:bg-sky-100 shadow-[2px_2px_0px_0px_#000]'
                }`}
              >
                <Users className="h-4 w-4 stroke-[2.5]" />
                <span>Patient Triage Roster</span>
              </button>

              <button
                onClick={() => setActiveTab('patient_detail')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-black border-2 border-black transition-all cursor-pointer ${
                  activeTab === 'patient_detail'
                    ? 'bg-sky-400 text-black shadow-[3px_3px_0px_0px_#000] -translate-x-0.5 -translate-y-0.5'
                    : 'bg-white text-black hover:bg-sky-100 shadow-[2px_2px_0px_0px_#000]'
                }`}
              >
                <Activity className="h-4 w-4 stroke-[2.5]" />
                <span>Patient Deep-Dive: <strong className="bg-white text-black px-1.5 py-0.5 rounded border border-black ml-1">{selectedPatient.name}</strong></span>
              </button>

              <button
                onClick={() => setActiveTab('records')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-black border-2 border-black transition-all cursor-pointer ${
                  activeTab === 'records'
                    ? 'bg-sky-400 text-black shadow-[3px_3px_0px_0px_#000] -translate-x-0.5 -translate-y-0.5'
                    : 'bg-white text-black hover:bg-sky-100 shadow-[2px_2px_0px_0px_#000]'
                }`}
              >
                <FileText className="h-4 w-4 stroke-[2.5]" />
                <span>EHR & Lab Reports</span>
              </button>

              <button
                onClick={() => setActiveTab('live_inference')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-black border-2 border-black transition-all cursor-pointer ${
                  activeTab === 'live_inference'
                    ? 'bg-sky-400 text-black shadow-[3px_3px_0px_0px_#000] -translate-x-0.5 -translate-y-0.5'
                    : 'bg-white text-black hover:bg-sky-100 shadow-[2px_2px_0px_0px_#000]'
                }`}
              >
                <Zap className="h-4 w-4 stroke-[2.5]" />
                <span>Interactive Risk Simulator</span>
              </button>
            </>
          )}

          {/* Patient Portal Subtabs */}
          {activeRole === 'patient' && (
            <>
              <button
                onClick={() => setActiveTab('patient_portal')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-black border-2 border-black transition-all cursor-pointer ${
                  activeTab === 'patient_portal'
                    ? 'bg-emerald-300 text-black shadow-[3px_3px_0px_0px_#000] -translate-x-0.5 -translate-y-0.5'
                    : 'bg-white text-black hover:bg-emerald-100 shadow-[2px_2px_0px_0px_#000]'
                }`}
              >
                <Heart className="h-4 w-4 stroke-[2.5]" />
                <span>My Health Overview</span>
              </button>

              <button
                onClick={() => setActiveTab('records')}
                className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-black border-2 border-black transition-all cursor-pointer ${
                  activeTab === 'records'
                    ? 'bg-emerald-300 text-black shadow-[3px_3px_0px_0px_#000] -translate-x-0.5 -translate-y-0.5'
                    : 'bg-white text-black hover:bg-emerald-100 shadow-[2px_2px_0px_0px_#000]'
                }`}
              >
                <FileText className="h-4 w-4 stroke-[2.5]" />
                <span>My Records & Vitals Log</span>
              </button>
            </>
          )}
        </div>

        {/* View Renderer */}
        {activeTab === 'ml_overview' && (
          <MLOverviewTab
            models={models}
            logs={inferenceLogs}
            onNavigateTab={(tab) => {
              if (tab === 'inference' || tab === 'live_inference') setActiveTab('live_inference');
              else if (tab === 'explainability' || tab === 'explainable_ai') setActiveTab('explainable_ai');
              else if (tab === 'benchmark' || tab === 'model_comparison') setActiveTab('model_comparison');
              else if (tab === 'anomaly' || tab === 'anomaly_detection') setActiveTab('anomaly_detection');
              else if (tab === 'drift' || tab === 'data_drift') setActiveTab('data_drift');
              else setActiveTab(tab);
            }}
            onRetrainTrigger={handleRetrainTrigger}
            isRetraining={isRetraining}
          />
        )}

        {activeTab === 'model_comparison' && (
          <ModelComparisonTab
            comparisonRows={comparisonRows}
            models={models}
          />
        )}

        {activeTab === 'live_inference' && (
          <LiveInferenceTab />
        )}

        {activeTab === 'explainable_ai' && (
          <ExplainableAITab models={models} />
        )}

        {activeTab === 'anomaly_detection' && (
          <AnomalyDetectionTab vitals={vitals} />
        )}

        {activeTab === 'data_drift' && (
          <DataDriftTab />
        )}

        {activeTab === 'doctor_roster' && (
          <DoctorPatientList
            patients={patients}
            selectedPatientId={selectedPatientId}
            onSelectPatient={(id) => {
              setSelectedPatientId(id);
              setActiveTab('patient_detail');
            }}
            onAddPatient={handleAddPatient}
          />
        )}

        {activeTab === 'patient_detail' && (
          <DoctorPatientDetail
            patient={selectedPatient}
            vitals={vitals}
            medicalRecords={medicalRecords}
            prescriptions={prescriptions}
            labReports={labReports}
            onAddVital={handleAddVital}
            onAddPrescription={handleAddPrescription}
            onOrderLab={handleOrderLab}
          />
        )}

        {activeTab === 'patient_portal' && (
          <PatientHealthSummary
            patient={selectedPatient}
            allPatients={patients}
            onSelectPatient={setSelectedPatientId}
            vitals={vitals}
            prescriptions={prescriptions}
            onAddVital={handleAddVital}
          />
        )}

        {activeTab === 'records' && (
          <PatientRecordsManager
            patients={patients}
            selectedPatientId={selectedPatientId}
            onSelectPatient={setSelectedPatientId}
            medicalRecords={medicalRecords}
            prescriptions={prescriptions}
            labReports={labReports}
            vitals={vitals}
            onAddVital={handleAddVital}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t-4 border-black bg-white py-6 text-center text-xs text-black font-black shadow-[0_-4px_0px_0px_rgba(0,0,0,1)] mt-12">
        <div className="max-w-4xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <span className="bg-yellow-300 border-2 border-black px-2 py-0.5 rounded-md font-black text-xs shadow-[2px_2px_0px_0px_#000]">
              HEALTHGUARD ML
            </span>
            <span className="font-bold">Next-Gen Clinical AI Platform</span>
          </div>
          <p className="font-bold text-xs text-slate-700">
            Machine Learning Driven Preventive Care • Clinical Decision Support
          </p>
        </div>
      </footer>
    </div>
  );
}
