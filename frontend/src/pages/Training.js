import React, { useState, useEffect } from 'react';
import { PlayIcon, StopIcon, CogIcon, ChartBarIcon } from '../components/Icons';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Grid,
  Chip,
  Divider,
} from '@mui/material';

const Training = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [config, setConfig] = useState({
    modelSize: 'small',
    epochs: 10,
    batchSize: 32,
    learningRate: 0.0003,
    dataset: 'wikitext'
  });
  const [viewDetailsDialog, setViewDetailsDialog] = useState(false);
  const [resumeDialog, setResumeDialog] = useState(false);

  // Poll for training progress
  useEffect(() => {
    let interval;
    if (isTraining) {
      interval = setInterval(async () => {
        try {
          const response = await fetch('http://localhost:8000/training/progress');
          if (response.ok) {
            const progress = await response.json();
            setTrainingProgress(progress);

            if (progress.status === 'completed' || progress.status === 'failed') {
              setIsTraining(false);
            }
          }
        } catch (error) {
          console.error('Failed to fetch training progress:', error);
        }
      }, 2000); // Poll every 2 seconds
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isTraining]);

  const startTraining = async () => {
    try {
      setIsTraining(true);
      // In a real implementation, you'd call the training API here
      console.log('Starting training with config:', config);
    } catch (error) {
      console.error('Failed to start training:', error);
      setIsTraining(false);
    }
  };

  const stopTraining = async () => {
    try {
      setIsTraining(false);
      setTrainingProgress(null);
      // In a real implementation, you'd call the stop training API here
      console.log('Stopping training');
    } catch (error) {
      console.error('Failed to stop training:', error);
    }
  };

  const formatTime = (seconds) => {
    if (!seconds) return '0s';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
  };

  const handleResumeTraining = async () => {
    try {
      setIsTraining(true);
      setResumeDialog(false);
      // Call resume training API
      const response = await fetch('http://localhost:8000/training/resume', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        throw new Error('Failed to resume training');
      }

      console.log('Training resumed successfully');
    } catch (error) {
      console.error('Failed to resume training:', error);
      setIsTraining(false);
      alert('Failed to resume training. Please check console for details.');
    }
  };

  const handleViewDetails = () => {
    setViewDetailsDialog(true);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-oxford-charcoal">Training</h1>
          <p className="text-oxford-gray mt-2">Train and fine-tune your MiniGPT models</p>
        </div>
        <div className="flex items-center space-x-3">
          {!isTraining ? (
            <button
              onClick={startTraining}
              className="btn-primary flex items-center space-x-2"
            >
              <PlayIcon className="w-5 h-5" />
              <span>Start Training</span>
            </button>
          ) : (
            <button
              onClick={stopTraining}
              className="bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-8 transition duration-200 ease-in-out rounded-lg flex items-center space-x-2"
            >
              <StopIcon className="w-5 h-5" />
              <span>Stop Training</span>
            </button>
          )}
        </div>
      </div>

      {/* Training Progress */}
      {isTraining && trainingProgress && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center space-x-3">
              <ChartBarIcon className="w-6 h-6 text-oxford-blue" />
              <h2 className="text-xl font-semibold text-oxford-charcoal">Training Progress</h2>
              <span className="badge badge-blue">{trainingProgress.status}</span>
            </div>
          </div>
          <div className="card-body">
            <div className="space-y-6">
              {/* Progress Bar */}
              <div>
                <div className="flex justify-between text-sm text-gray-600 mb-2">
                  <span>Overall Progress</span>
                  <span>{trainingProgress.progress_percentage?.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-oxford-blue h-3 rounded-full transition-all duration-300"
                    style={{ width: `${trainingProgress.progress_percentage || 0}%` }}
                  ></div>
                </div>
              </div>

              {/* Training Stats Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-oxford-gray-warm p-4 rounded-lg">
                  <div className="text-sm text-gray-600">Epoch</div>
                  <div className="text-2xl font-bold text-oxford-charcoal">
                    {trainingProgress.current_epoch}/{trainingProgress.total_epochs}
                  </div>
                </div>
                <div className="bg-oxford-gray-warm p-4 rounded-lg">
                  <div className="text-sm text-gray-600">Steps</div>
                  <div className="text-2xl font-bold text-oxford-charcoal">
                    {trainingProgress.current_step?.toLocaleString()}/{trainingProgress.total_steps?.toLocaleString()}
                  </div>
                </div>
                <div className="bg-oxford-gray-warm p-4 rounded-lg">
                  <div className="text-sm text-gray-600">Train Loss</div>
                  <div className="text-2xl font-bold text-oxford-charcoal">
                    {trainingProgress.train_loss?.toFixed(4) || 'N/A'}
                  </div>
                </div>
                <div className="bg-oxford-gray-warm p-4 rounded-lg">
                  <div className="text-sm text-gray-600">Val Loss</div>
                  <div className="text-2xl font-bold text-oxford-charcoal">
                    {trainingProgress.val_loss?.toFixed(4) || 'N/A'}
                  </div>
                </div>
              </div>

              {/* Additional Info */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-gray-200">
                <div>
                  <div className="text-sm text-gray-600">Model Name</div>
                  <div className="font-semibold text-oxford-charcoal">{trainingProgress.model_name}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Learning Rate</div>
                  <div className="font-semibold text-oxford-charcoal">{trainingProgress.learning_rate?.toExponential(2)}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Time Remaining</div>
                  <div className="font-semibold text-oxford-charcoal">
                    {formatTime(trainingProgress.estimated_time_remaining)}
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 pt-4 border-t border-gray-200">
                <button
                  onClick={() => setResumeDialog(true)}
                  className="btn-primary flex items-center space-x-2"
                  disabled={trainingProgress.status === 'running'}
                >
                  <PlayIcon className="w-4 h-4" />
                  <span>Resume Training</span>
                </button>
                <button
                  onClick={handleViewDetails}
                  className="btn-secondary flex items-center space-x-2"
                >
                  <ChartBarIcon className="w-4 h-4" />
                  <span>View Details</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Training Configuration */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center space-x-3">
            <CogIcon className="w-6 h-6 text-oxford-blue" />
            <h2 className="text-xl font-semibold text-oxford-charcoal">Training Configuration</h2>
          </div>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Model Size */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Size
              </label>
              <select
                value={config.modelSize}
                onChange={(e) => setConfig({...config, modelSize: e.target.value})}
                className="input-field"
                disabled={isTraining}
              >
                <option value="small">Small (4 layers, 128 dim)</option>
                <option value="medium">Medium (6 layers, 256 dim)</option>
                <option value="large">Large (12 layers, 512 dim)</option>
              </select>
            </div>

            {/* Dataset */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Dataset
              </label>
              <select
                value={config.dataset}
                onChange={(e) => setConfig({...config, dataset: e.target.value})}
                className="input-field"
                disabled={isTraining}
              >
                <option value="wikitext">WikiText-2</option>
                <option value="custom">Custom Dataset</option>
              </select>
            </div>

            {/* Epochs */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Epochs
              </label>
              <input
                type="number"
                value={config.epochs}
                onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                className="input-field"
                min="1"
                max="100"
                disabled={isTraining}
              />
            </div>

            {/* Batch Size */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Batch Size
              </label>
              <input
                type="number"
                value={config.batchSize}
                onChange={(e) => setConfig({...config, batchSize: parseInt(e.target.value)})}
                className="input-field"
                min="1"
                max="128"
                disabled={isTraining}
              />
            </div>

            {/* Learning Rate */}
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Learning Rate
              </label>
              <input
                type="number"
                value={config.learningRate}
                onChange={(e) => setConfig({...config, learningRate: parseFloat(e.target.value)})}
                className="input-field"
                step="0.0001"
                min="0.0001"
                max="0.01"
                disabled={isTraining}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Training History */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center space-x-3">
            <ChartBarIcon className="w-6 h-6 text-oxford-blue" />
            <h2 className="text-xl font-semibold text-oxford-charcoal">Recent Training Jobs</h2>
          </div>
        </div>
        <div className="card-body">
          <div className="space-y-3">
            <div className="flex items-center justify-between p-4 bg-oxford-gray-warm rounded-lg">
              <div className="flex items-center space-x-4">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <div>
                  <div className="font-semibold text-oxford-charcoal">miniGPT-v2</div>
                  <div className="text-sm text-gray-600">Completed • Small model • 95.2% accuracy</div>
                </div>
              </div>
              <div className="text-sm text-gray-500">2 hours ago</div>
            </div>

            <div className="flex items-center justify-between p-4 bg-oxford-gray-warm rounded-lg">
              <div className="flex items-center space-x-4">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <div>
                  <div className="font-semibold text-oxford-charcoal">miniGPT-v1</div>
                  <div className="text-sm text-gray-600">Completed • Small model • 92.8% accuracy</div>
                </div>
              </div>
              <div className="text-sm text-gray-500">1 day ago</div>
            </div>
          </div>
        </div>
      </div>

      {/* Resume Training Dialog */}
      <Dialog open={resumeDialog} onClose={() => setResumeDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Resume Training</DialogTitle>
        <DialogContent>
          <Typography mb={2}>
            Do you want to resume training from the last checkpoint?
          </Typography>
          <Box bgcolor="grey.100" p={2} borderRadius={1}>
            <Typography variant="body2" fontWeight="bold" mb={1}>Current Configuration:</Typography>
            <Typography variant="body2">Model Size: {config.modelSize}</Typography>
            <Typography variant="body2">Epochs: {config.epochs}</Typography>
            <Typography variant="body2">Batch Size: {config.batchSize}</Typography>
            <Typography variant="body2">Learning Rate: {config.learningRate}</Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResumeDialog(false)}>Cancel</Button>
          <Button onClick={handleResumeTraining} variant="contained">Resume Training</Button>
        </DialogActions>
      </Dialog>

      {/* View Details Dialog */}
      <Dialog open={viewDetailsDialog} onClose={() => setViewDetailsDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Training Details</DialogTitle>
        <DialogContent>
          {trainingProgress ? (
            <Box>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" fontWeight="bold" mb={2}>Training Progress</Typography>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Status:</Typography>
                    <Chip label={trainingProgress.status} color="primary" size="small" />
                  </Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Current Epoch:</Typography>
                    <Typography variant="body1">{trainingProgress.current_epoch} / {trainingProgress.total_epochs}</Typography>
                  </Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Current Step:</Typography>
                    <Typography variant="body1">{trainingProgress.current_step?.toLocaleString()} / {trainingProgress.total_steps?.toLocaleString()}</Typography>
                  </Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Progress:</Typography>
                    <Typography variant="body1">{trainingProgress.progress_percentage?.toFixed(1)}%</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" fontWeight="bold" mb={2}>Performance Metrics</Typography>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Training Loss:</Typography>
                    <Typography variant="body1">{trainingProgress.train_loss?.toFixed(4) || 'N/A'}</Typography>
                  </Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Validation Loss:</Typography>
                    <Typography variant="body1">{trainingProgress.val_loss?.toFixed(4) || 'N/A'}</Typography>
                  </Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Learning Rate:</Typography>
                    <Typography variant="body1">{trainingProgress.learning_rate?.toExponential(2) || 'N/A'}</Typography>
                  </Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Estimated Time Remaining:</Typography>
                    <Typography variant="body1">{formatTime(trainingProgress.estimated_time_remaining)}</Typography>
                  </Box>
                </Grid>
              </Grid>
              <Divider sx={{ my: 2 }} />
              <Typography variant="h6" fontWeight="bold" mb={2}>Model Information</Typography>
              <Box mb={2}>
                <Typography variant="body2" color="textSecondary">Model Name:</Typography>
                <Typography variant="body1">{trainingProgress.model_name || 'MiniGPT'}</Typography>
              </Box>
            </Box>
          ) : (
            <Typography>No training data available.</Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewDetailsDialog(false)} variant="contained">Close</Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

export default Training;