import React, { createContext, useContext, useState } from 'react';

const ApiContext = createContext();

export const useApi = () => {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};

export const ApiProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const apiCall = async (endpoint, options = {}) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Training API methods
  const startTraining = async (config) => {
    return apiCall('/api/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  };

  const stopTraining = async () => {
    return apiCall('/api/training/stop', {
      method: 'POST',
    });
  };

  const getTrainingStatus = async () => {
    return apiCall('/api/training/status');
  };

  const getTrainingProgress = async () => {
    return apiCall('/api/training/progress');
  };

  // Model API methods
  const getModels = async () => {
    return apiCall('/api/models');
  };

  const deleteModel = async (modelId) => {
    return apiCall(`/api/models/${modelId}`, {
      method: 'DELETE',
    });
  };

  const deployModel = async (modelId) => {
    return apiCall(`/api/models/${modelId}/deploy`, {
      method: 'POST',
    });
  };

  const downloadModel = async (modelId) => {
    return apiCall(`/api/models/${modelId}/download`);
  };

  // Chat API methods
  const sendMessage = async (message, config = {}) => {
    return apiCall('/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        message,
        model: config.model || 'MiniGPT-v2',
        temperature: config.temperature || 0.7,
        max_tokens: config.max_tokens || 150,
      }),
    });
  };

  const getChatHistory = async () => {
    return apiCall('/api/chat/history');
  };

  const clearChatHistory = async () => {
    return apiCall('/api/chat/clear', {
      method: 'POST',
    });
  };

  // Dataset API methods
  const getDatasets = async () => {
    return apiCall('/api/datasets');
  };

  const uploadDataset = async (file, metadata) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('metadata', JSON.stringify(metadata));

    return apiCall('/api/datasets/upload', {
      method: 'POST',
      headers: {},
      body: formData,
    });
  };

  const deleteDataset = async (datasetId) => {
    return apiCall(`/api/datasets/${datasetId}`, {
      method: 'DELETE',
    });
  };

  const downloadDataset = async (datasetId) => {
    return apiCall(`/api/datasets/${datasetId}/download`);
  };

  // Dashboard API methods
  const getDashboardStats = async () => {
    return apiCall('/api/dashboard/stats');
  };

  const getRecentActivity = async () => {
    return apiCall('/api/dashboard/activity');
  };

  const value = {
    isLoading,
    error,
    setError,

    // Training methods
    startTraining,
    stopTraining,
    getTrainingStatus,
    getTrainingProgress,

    // Model methods
    getModels,
    deleteModel,
    deployModel,
    downloadModel,

    // Chat methods
    sendMessage,
    getChatHistory,
    clearChatHistory,

    // Dataset methods
    getDatasets,
    uploadDataset,
    deleteDataset,
    downloadDataset,

    // Dashboard methods
    getDashboardStats,
    getRecentActivity,
  };

  return (
    <ApiContext.Provider value={value}>
      {children}
    </ApiContext.Provider>
  );
};