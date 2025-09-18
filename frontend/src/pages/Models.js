import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  Chip,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  LinearProgress,
} from '@mui/material';
import {
  PlayArrow as DeployIcon,
  Stop as StopIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  Visibility as ViewIcon,
  Edit as EditIcon,
  Add as AddIcon,
} from '@mui/icons-material';

function Models() {
  const [models, setModels] = useState([
    {
      id: 1,
      name: 'MiniGPT-v2',
      version: '2.0.1',
      status: 'Active',
      accuracy: '94.2%',
      size: '125 MB',
      created: '2 hours ago',
      lastTrained: '2 hours ago',
      parameters: '50M',
    },
    {
      id: 2,
      name: 'MiniGPT-v1',
      version: '1.0.0',
      status: 'Inactive',
      accuracy: '91.8%',
      size: '110 MB',
      created: '1 day ago',
      lastTrained: '1 day ago',
      parameters: '45M',
    },
    {
      id: 3,
      name: 'MiniGPT-base',
      version: '0.9.0',
      status: 'Training',
      accuracy: '89.1%',
      size: '95 MB',
      created: '2 days ago',
      lastTrained: '2 days ago',
      parameters: '40M',
    },
  ]);

  const [openDialog, setOpenDialog] = useState(false);
  const [dialogType, setDialogType] = useState('');
  const [selectedModel, setSelectedModel] = useState(null);

  const getStatusColor = (status) => {
    switch (status) {
      case 'Active': return 'success';
      case 'Training': return 'warning';
      case 'Inactive': return 'default';
      default: return 'default';
    }
  };

  const handleAction = (action, model) => {
    setSelectedModel(model);
    setDialogType(action);
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setSelectedModel(null);
    setDialogType('');
  };

  const handleModelAction = () => {
    // Simulate action
    if (dialogType === 'deploy' && selectedModel) {
      setModels(prev => prev.map(model =>
        model.id === selectedModel.id
          ? { ...model, status: 'Active' }
          : { ...model, status: 'Inactive' }
      ));
    } else if (dialogType === 'delete' && selectedModel) {
      setModels(prev => prev.filter(model => model.id !== selectedModel.id));
    }
    handleCloseDialog();
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" fontWeight="bold">
          Model Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => handleAction('create', null)}
        >
          New Model
        </Button>
      </Box>

      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="body2">
                Total Models
              </Typography>
              <Typography variant="h4" fontWeight="bold">
                {models.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="body2">
                Active Models
              </Typography>
              <Typography variant="h4" fontWeight="bold">
                {models.filter(m => m.status === 'Active').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="body2">
                Training
              </Typography>
              <Typography variant="h4" fontWeight="bold">
                {models.filter(m => m.status === 'Training').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="body2">
                Best Accuracy
              </Typography>
              <Typography variant="h4" fontWeight="bold">
                94.2%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Card>
        <CardContent>
          <Typography variant="h6" fontWeight="bold" mb={2}>
            Models
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Name</strong></TableCell>
                  <TableCell><strong>Version</strong></TableCell>
                  <TableCell><strong>Status</strong></TableCell>
                  <TableCell><strong>Accuracy</strong></TableCell>
                  <TableCell><strong>Size</strong></TableCell>
                  <TableCell><strong>Parameters</strong></TableCell>
                  <TableCell><strong>Created</strong></TableCell>
                  <TableCell align="center"><strong>Actions</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {models.map((model) => (
                  <TableRow key={model.id} hover>
                    <TableCell>
                      <Typography fontWeight="bold">{model.name}</Typography>
                    </TableCell>
                    <TableCell>{model.version}</TableCell>
                    <TableCell>
                      <Chip
                        label={model.status}
                        color={getStatusColor(model.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{model.accuracy}</TableCell>
                    <TableCell>{model.size}</TableCell>
                    <TableCell>{model.parameters}</TableCell>
                    <TableCell>{model.created}</TableCell>
                    <TableCell align="center">
                      <Box display="flex" gap={1} justifyContent="center">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => handleAction('view', model)}
                        >
                          <ViewIcon fontSize="small" />
                        </IconButton>
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => handleAction('deploy', model)}
                          disabled={model.status === 'Training'}
                        >
                          {model.status === 'Active' ? <StopIcon fontSize="small" /> : <DeployIcon fontSize="small" />}
                        </IconButton>
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => handleAction('download', model)}
                        >
                          <DownloadIcon fontSize="small" />
                        </IconButton>
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleAction('delete', model)}
                          disabled={model.status === 'Active'}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>
          {dialogType === 'deploy' && `Deploy ${selectedModel?.name}`}
          {dialogType === 'delete' && `Delete ${selectedModel?.name}`}
          {dialogType === 'view' && `Model Details: ${selectedModel?.name}`}
          {dialogType === 'download' && `Download ${selectedModel?.name}`}
          {dialogType === 'create' && 'Create New Model'}
        </DialogTitle>
        <DialogContent>
          {dialogType === 'deploy' && (
            <Typography>
              Are you sure you want to deploy this model? This will make it the active model for chat and API requests.
            </Typography>
          )}
          {dialogType === 'delete' && (
            <Typography color="error">
              Are you sure you want to delete this model? This action cannot be undone.
            </Typography>
          )}
          {dialogType === 'view' && selectedModel && (
            <Box>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">Name:</Typography>
                  <Typography variant="body1" fontWeight="bold">{selectedModel.name}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">Version:</Typography>
                  <Typography variant="body1">{selectedModel.version}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">Status:</Typography>
                  <Chip label={selectedModel.status} color={getStatusColor(selectedModel.status)} size="small" />
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">Accuracy:</Typography>
                  <Typography variant="body1">{selectedModel.accuracy}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">Size:</Typography>
                  <Typography variant="body1">{selectedModel.size}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">Parameters:</Typography>
                  <Typography variant="body1">{selectedModel.parameters}</Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" color="textSecondary">Last Trained:</Typography>
                  <Typography variant="body1">{selectedModel.lastTrained}</Typography>
                </Grid>
              </Grid>
            </Box>
          )}
          {dialogType === 'download' && (
            <Box>
              <Typography mb={2}>
                Download {selectedModel?.name} model files
              </Typography>
              <LinearProgress variant="determinate" value={0} />
              <Typography variant="caption" color="textSecondary" mt={1}>
                Ready to download
              </Typography>
            </Box>
          )}
          {dialogType === 'create' && (
            <Box>
              <TextField
                fullWidth
                label="Model Name"
                margin="normal"
                placeholder="e.g., MiniGPT-v3"
              />
              <TextField
                fullWidth
                label="Base Model"
                margin="normal"
                select
                SelectProps={{ native: true }}
              >
                <option value="">Select base model</option>
                <option value="scratch">Train from scratch</option>
                <option value="minigpt-v2">Based on MiniGPT-v2</option>
                <option value="minigpt-v1">Based on MiniGPT-v1</option>
              </TextField>
              <TextField
                fullWidth
                label="Description"
                margin="normal"
                multiline
                rows={3}
                placeholder="Describe this model..."
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>
            Cancel
          </Button>
          <Button
            onClick={handleModelAction}
            variant="contained"
            color={dialogType === 'delete' ? 'error' : 'primary'}
          >
            {dialogType === 'deploy' && 'Deploy'}
            {dialogType === 'delete' && 'Delete'}
            {dialogType === 'download' && 'Download'}
            {dialogType === 'create' && 'Create'}
            {dialogType === 'view' && 'Close'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default Models;