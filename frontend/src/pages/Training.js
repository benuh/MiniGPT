import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  Button,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  LinearProgress,
  Alert,
  Divider,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  Settings as ConfigIcon,
  Timeline as ProgressIcon,
} from '@mui/icons-material';

function Training() {
  const [trainingConfig, setTrainingConfig] = useState({
    modelName: 'MiniGPT-v3',
    dataset: 'stories.txt',
    epochs: 50,
    batchSize: 32,
    learningRate: 0.001,
    optimizer: 'adam',
    saveCheckpoints: true,
    useGPU: true,
  });

  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);

  const handleConfigChange = (field, value) => {
    setTrainingConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleStartTraining = () => {
    setIsTraining(true);
    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          return 100;
        }
        return prev + 2;
      });
    }, 1000);
  };

  const handleStopTraining = () => {
    setIsTraining(false);
    setTrainingProgress(0);
  };

  const datasets = ['stories.txt', 'shakespeare.txt', 'wiki_sample.txt', 'custom_data.txt'];
  const optimizers = ['adam', 'sgd', 'adamw', 'rmsprop'];

  return (
    <Box>
      <Typography variant="h4" fontWeight="bold" mb={3}>
        Model Training
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={3}>
                <ConfigIcon color="primary" />
                <Typography variant="h6" fontWeight="bold">
                  Training Configuration
                </Typography>
              </Box>

              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Model Name"
                    value={trainingConfig.modelName}
                    onChange={(e) => handleConfigChange('modelName', e.target.value)}
                    variant="outlined"
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Dataset</InputLabel>
                    <Select
                      value={trainingConfig.dataset}
                      label="Dataset"
                      onChange={(e) => handleConfigChange('dataset', e.target.value)}
                    >
                      {datasets.map((dataset) => (
                        <MenuItem key={dataset} value={dataset}>
                          {dataset}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Epochs"
                    type="number"
                    value={trainingConfig.epochs}
                    onChange={(e) => handleConfigChange('epochs', parseInt(e.target.value))}
                    variant="outlined"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Batch Size"
                    type="number"
                    value={trainingConfig.batchSize}
                    onChange={(e) => handleConfigChange('batchSize', parseInt(e.target.value))}
                    variant="outlined"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography gutterBottom>Learning Rate: {trainingConfig.learningRate}</Typography>
                  <Slider
                    value={trainingConfig.learningRate}
                    onChange={(e, value) => handleConfigChange('learningRate', value)}
                    min={0.0001}
                    max={0.01}
                    step={0.0001}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => value.toFixed(4)}
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Optimizer</InputLabel>
                    <Select
                      value={trainingConfig.optimizer}
                      label="Optimizer"
                      onChange={(e) => handleConfigChange('optimizer', e.target.value)}
                    >
                      {optimizers.map((optimizer) => (
                        <MenuItem key={optimizer} value={optimizer}>
                          {optimizer.toUpperCase()}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={trainingConfig.saveCheckpoints}
                        onChange={(e) => handleConfigChange('saveCheckpoints', e.target.checked)}
                      />
                    }
                    label="Save Checkpoints"
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={trainingConfig.useGPU}
                        onChange={(e) => handleConfigChange('useGPU', e.target.checked)}
                      />
                    }
                    label="Use GPU"
                  />
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              <Box display="flex" gap={2}>
                {!isTraining ? (
                  <Button
                    variant="contained"
                    startIcon={<StartIcon />}
                    onClick={handleStartTraining}
                    size="large"
                  >
                    Start Training
                  </Button>
                ) : (
                  <>
                    <Button
                      variant="outlined"
                      startIcon={<PauseIcon />}
                      size="large"
                    >
                      Pause
                    </Button>
                    <Button
                      variant="outlined"
                      color="error"
                      startIcon={<StopIcon />}
                      onClick={handleStopTraining}
                      size="large"
                    >
                      Stop
                    </Button>
                  </>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={3}>
                <ProgressIcon color="primary" />
                <Typography variant="h6" fontWeight="bold">
                  Training Progress
                </Typography>
              </Box>

              {isTraining && (
                <Alert severity="info" sx={{ mb: 3 }}>
                  Training in progress... This may take several hours to complete.
                </Alert>
              )}

              <Box mb={3}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">Overall Progress</Typography>
                  <Typography variant="body2" color="textSecondary">
                    {trainingProgress.toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={trainingProgress}
                  sx={{ height: 10, borderRadius: 5 }}
                />
              </Box>

              <Grid container spacing={2} mb={3}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">Current Epoch</Typography>
                  <Typography variant="h6" fontWeight="bold">
                    {Math.floor((trainingProgress / 100) * trainingConfig.epochs)} / {trainingConfig.epochs}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">Training Loss</Typography>
                  <Typography variant="h6" fontWeight="bold">
                    {isTraining ? (4.5 - (trainingProgress / 100) * 2).toFixed(3) : '---'}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">Learning Rate</Typography>
                  <Typography variant="h6" fontWeight="bold">
                    {trainingConfig.learningRate}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">Time Elapsed</Typography>
                  <Typography variant="h6" fontWeight="bold">
                    {isTraining ? `${Math.floor(trainingProgress / 10)}m ${Math.floor((trainingProgress % 10) * 6)}s` : '---'}
                  </Typography>
                </Grid>
              </Grid>

              <Box>
                <Typography variant="body2" color="textSecondary" mb={1}>Status</Typography>
                <Chip
                  label={isTraining ? 'Training' : 'Ready'}
                  color={isTraining ? 'primary' : 'default'}
                  variant={isTraining ? 'filled' : 'outlined'}
                />
              </Box>
            </CardContent>
          </Card>

          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" mb={2}>
                Recent Training Jobs
              </Typography>
              <Box>
                {[
                  { name: 'MiniGPT-v2', status: 'Completed', accuracy: '94.2%', date: '2 hours ago' },
                  { name: 'MiniGPT-v1', status: 'Completed', accuracy: '91.8%', date: '1 day ago' },
                  { name: 'MiniGPT-base', status: 'Failed', accuracy: '---', date: '2 days ago' },
                ].map((job, index) => (
                  <Box key={index} mb={2} pb={2} borderBottom={index < 2 ? 1 : 0} borderColor="divider">
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="body2" fontWeight="bold">{job.name}</Typography>
                      <Chip
                        label={job.status}
                        size="small"
                        color={job.status === 'Completed' ? 'success' : job.status === 'Failed' ? 'error' : 'default'}
                      />
                    </Box>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="caption" color="textSecondary">
                        Accuracy: {job.accuracy}
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        {job.date}
                      </Typography>
                    </Box>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Training;