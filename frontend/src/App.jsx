import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomeScan from './pages/HomeScan';
import ScanProgress from './pages/ScanProgress';
import ScanResults from './pages/ScanResults';
import ModelsAbout from './pages/ModelsAbout';

export default function App() {
  const [activeTab, setActiveTab] = useState('scan');
  const [scanId, setScanId] = useState(null);
  const [targetUrl, setTargetUrl] = useState('');
  const [fastMode, setFastMode] = useState(false);
  const [report, setReport] = useState(null);

  const handleScanSubmitted = (newScanId, url, isFastMode) => {
    setScanId(newScanId);
    setTargetUrl(url);
    setFastMode(isFastMode);
    setReport(null);
    setActiveTab('progress');
  };

  const handleScanCompleted = (completedReport) => {
    setReport(completedReport);
    setActiveTab('results');
  };

  const handleNewScan = () => {
    setScanId(null);
    setTargetUrl('');
    setReport(null);
    setActiveTab('scan');
  };

  return (
    <div className="min-h-screen flex flex-col bg-background text-slate-100 font-sans">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1 pb-12">
        {activeTab === 'scan' && (
          <HomeScan onScanSubmitted={handleScanSubmitted} />
        )}

        {activeTab === 'progress' && scanId && (
          <ScanProgress 
            scanId={scanId} 
            targetUrl={targetUrl} 
            fastMode={fastMode}
            onScanCompleted={handleScanCompleted}
            onBackToScan={handleNewScan}
          />
        )}

        {activeTab === 'results' && report && (
          <ScanResults 
            report={report} 
            onNewScan={handleNewScan} 
          />
        )}

        {activeTab === 'models' && (
          <ModelsAbout />
        )}
      </main>

      <Footer />
    </div>
  );
}
