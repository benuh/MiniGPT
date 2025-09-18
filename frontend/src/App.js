import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box } from '@mui/material';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Training from './pages/Training';
import Chat from './pages/Chat';
import Models from './pages/Models';
import DataManagement from './pages/DataManagement';
import { ApiProvider } from './contexts/ApiContext';
import { NotificationProvider } from './contexts/NotificationContext';

function App() {
  return (
    <ApiProvider>
      <NotificationProvider>
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/training" element={<Training />} />
              <Route path="/chat" element={<Chat />} />
              <Route path="/models" element={<Models />} />
              <Route path="/data" element={<DataManagement />} />
            </Routes>
          </Layout>
        </Box>
      </NotificationProvider>
    </ApiProvider>
  );
}

export default App;