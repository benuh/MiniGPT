import React from 'react';
import { Routes, Route } from 'react-router-dom';
import TailwindLayout from './components/TailwindLayout';
import TailwindGettingStarted from './pages/TailwindGettingStarted';
import Dashboard from './pages/Dashboard';
import Training from './pages/Training';
import Chat from './pages/Chat';
import Models from './pages/Models';
import DataManagement from './pages/DataManagement';
import CoreInfrastructure from './pages/CoreInfrastructure';
import { ApiProvider } from './contexts/ApiContext';
import { NotificationProvider } from './contexts/NotificationContext';

function App() {
  return (
    <ApiProvider>
      <NotificationProvider>
        <TailwindLayout>
          <Routes>
            <Route path="/" element={<TailwindGettingStarted />} />
            <Route path="/getting-started" element={<TailwindGettingStarted />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/training" element={<Training />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/models" element={<Models />} />
            <Route path="/data" element={<DataManagement />} />
            <Route path="/core" element={<CoreInfrastructure />} />
          </Routes>
        </TailwindLayout>
      </NotificationProvider>
    </ApiProvider>
  );
}

export default App;