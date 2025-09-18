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
  Alert,
  Divider,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
  Edit as EditIcon,
  Add as AddIcon,
  Storage as DataIcon,
  Assessment as AnalyticsIcon,
} from '@mui/icons-material';

function DataManagement() {
  const [datasets, setDatasets] = useState([
    {
      id: 1,
      name: 'stories.txt',
      type: 'Text',
      size: '2.3 MB',
      records: '1,247',
      status: 'Ready',
      uploaded: '2 hours ago',
      lastUsed: '2 hours ago',
    },
    {
      id: 2,
      name: 'shakespeare.txt',
      type: 'Text',
      size: '5.1 MB',
      records: '2,891',
      status: 'Ready',
      uploaded: '1 day ago',
      lastUsed: '1 day ago',
    },
    {
      id: 3,
      name: 'wiki_sample.txt',
      type: 'Text',
      size: '12.7 MB',
      records: '5,432',
      status: 'Processing',
      uploaded: '2 days ago',
      lastUsed: 'Never',
    },
  ]);

  const [openDialog, setOpenDialog] = useState(false);
  const [dialogType, setDialogType] = useState('');
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const getStatusColor = (status) => {
    switch (status) {
      case 'Ready': return 'success';
      case 'Processing': return 'warning';
      case 'Error': return 'error';
      default: return 'default';
    }
  };

  const handleAction = (action, dataset) => {
    setSelectedDataset(dataset);
    setDialogType(action);
    setOpenDialog(true);

    if (action === 'upload') {
      // Simulate upload progress
      setUploadProgress(0);
      const interval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            return 100;
          }
          return prev + 10;
        });
      }, 300);
    }
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setSelectedDataset(null);
    setDialogType('');
    setUploadProgress(0);
  };

  const handleDatasetAction = () => {
    if (dialogType === 'delete' && selectedDataset) {
      setDatasets(prev => prev.filter(dataset => dataset.id !== selectedDataset.id));
    } else if (dialogType === 'upload') {
      // Add new dataset
      const newDataset = {
        id: Date.now(),
        name: 'new_dataset.txt',
        type: 'Text',
        size: '1.5 MB',
        records: '892',
        status: 'Ready',
        uploaded: 'Just now',
        lastUsed: 'Never',
      };
      setDatasets(prev => [...prev, newDataset]);
    }
    handleCloseDialog();
  };

  const totalSize = datasets.reduce((acc, dataset) => {
    const size = parseFloat(dataset.size.split(' ')[0]);
    return acc + size;
  }, 0);

  const totalRecords = datasets.reduce((acc, dataset) => {
    const records = parseInt(dataset.records.replace(',', ''));
    return acc + records;
  }, 0);

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" fontWeight="bold">
          Data Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<UploadIcon />}
          onClick={() => handleAction('upload', null)}
        >
          Upload Dataset
        </Button>
      </Box>

      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Total Datasets
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {datasets.length}
                  </Typography>
                </Box>
                <DataIcon color="primary" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Total Size
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {totalSize.toFixed(1)} MB
                  </Typography>
                </Box>
                <AnalyticsIcon color="primary" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="body2">
                Total Records
              </Typography>
              <Typography variant="h4" fontWeight="bold">
                {totalRecords.toLocaleString()}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="body2">
                Ready for Training
              </Typography>
              <Typography variant="h4" fontWeight="bold">
                {datasets.filter(d => d.status === 'Ready').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Card mb={3}>
        <CardContent>
          <Typography variant="h6" fontWeight="bold" mb={2}>
            Upload Guidelines
          </Typography>
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>Supported formats:</strong> .txt, .json, .csv<br />
              <strong>Max file size:</strong> 100 MB<br />
              <strong>Recommended:</strong> UTF-8 encoded text files with clean, well-formatted content
            </Typography>
          </Alert>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h6" fontWeight="bold" mb={2}>
            Datasets
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Name</strong></TableCell>
                  <TableCell><strong>Type</strong></TableCell>
                  <TableCell><strong>Size</strong></TableCell>
                  <TableCell><strong>Records</strong></TableCell>
                  <TableCell><strong>Status</strong></TableCell>
                  <TableCell><strong>Uploaded</strong></TableCell>
                  <TableCell><strong>Last Used</strong></TableCell>
                  <TableCell align="center"><strong>Actions</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {datasets.map((dataset) => (
                  <TableRow key={dataset.id} hover>
                    <TableCell>
                      <Typography fontWeight="bold">{dataset.name}</Typography>
                    </TableCell>
                    <TableCell>{dataset.type}</TableCell>
                    <TableCell>{dataset.size}</TableCell>
                    <TableCell>{dataset.records}</TableCell>
                    <TableCell>
                      <Chip
                        label={dataset.status}
                        color={getStatusColor(dataset.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{dataset.uploaded}</TableCell>
                    <TableCell>{dataset.lastUsed}</TableCell>
                    <TableCell align="center">
                      <Box display="flex" gap={1} justifyContent="center">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => handleAction('view', dataset)}
                        >
                          <ViewIcon fontSize="small" />
                        </IconButton>
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => handleAction('download', dataset)}
                        >
                          <DownloadIcon fontSize="small" />
                        </IconButton>
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => handleAction('edit', dataset)}
                        >
                          <EditIcon fontSize="small" />
                        </IconButton>
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleAction('delete', dataset)}
                          disabled={dataset.status === 'Processing'}
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

      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="md" fullWidth>
        <DialogTitle>
          {dialogType === 'upload' && 'Upload Dataset'}
          {dialogType === 'delete' && `Delete ${selectedDataset?.name}`}
          {dialogType === 'view' && `Dataset Details: ${selectedDataset?.name}`}
          {dialogType === 'edit' && `Edit ${selectedDataset?.name}`}
          {dialogType === 'download' && `Download ${selectedDataset?.name}`}
        </DialogTitle>
        <DialogContent>
          {dialogType === 'upload' && (
            <Box>
              <Box
                sx={{
                  border: '2px dashed #ccc',
                  borderRadius: 2,
                  p: 4,
                  textAlign: 'center',
                  mb: 3,
                  cursor: 'pointer',
                  '&:hover': {
                    borderColor: 'primary.main',
                    bgcolor: 'primary.light',
                    opacity: 0.1,
                  },
                }}
              >
                <UploadIcon sx={{ fontSize: 48, color: 'grey.400', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  Drag and drop files here or click to browse
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Supported formats: .txt, .json, .csv (max 100MB)
                </Typography>
              </Box>

              {uploadProgress > 0 && (
                <Box mb={2}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="body2">Uploading...</Typography>
                    <Typography variant="body2">{uploadProgress}%</Typography>
                  </Box>
                  <LinearProgress variant="determinate" value={uploadProgress} />
                </Box>
              )}

              <TextField
                fullWidth
                label="Dataset Name"
                margin="normal"
                placeholder="e.g., my_custom_dataset"
              />
              <TextField
                fullWidth
                label="Description"
                margin="normal"
                multiline
                rows={3}
                placeholder="Describe this dataset..."
              />
            </Box>
          )}

          {dialogType === 'delete' && (
            <Typography color="error">
              Are you sure you want to delete this dataset? This action cannot be undone and will affect any models trained on this data.
            </Typography>
          )}

          {dialogType === 'view' && selectedDataset && (
            <Box>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" fontWeight="bold" mb={2}>Dataset Information</Typography>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Name:</Typography>
                    <Typography variant="body1" fontWeight="bold">{selectedDataset.name}</Typography>
                  </Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Type:</Typography>
                    <Typography variant="body1">{selectedDataset.type}</Typography>
                  </Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Size:</Typography>
                    <Typography variant="body1">{selectedDataset.size}</Typography>
                  </Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Records:</Typography>
                    <Typography variant="body1">{selectedDataset.records}</Typography>
                  </Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Status:</Typography>
                    <Chip label={selectedDataset.status} color={getStatusColor(selectedDataset.status)} size="small" />
                  </Box>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" fontWeight="bold" mb={2}>Usage Statistics</Typography>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Uploaded:</Typography>
                    <Typography variant="body1">{selectedDataset.uploaded}</Typography>
                  </Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Last Used:</Typography>
                    <Typography variant="body1">{selectedDataset.lastUsed}</Typography>
                  </Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">Models Trained:</Typography>
                    <Typography variant="body1">2</Typography>
                  </Box>
                </Grid>
              </Grid>
              <Divider sx={{ my: 2 }} />
              <Typography variant="h6" fontWeight="bold" mb={2}>Sample Content</Typography>
              <Box
                sx={{
                  bgcolor: 'grey.100',
                  p: 2,
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                  maxHeight: 200,
                  overflow: 'auto',
                }}
              >
                Once upon a time, in a land far away...<br />
                The quick brown fox jumps over the lazy dog.<br />
                Machine learning is transforming the world...<br />
                <Typography variant="caption" color="textSecondary" sx={{ fontStyle: 'italic' }}>
                  [Showing first 3 lines of {selectedDataset.records} total records]
                </Typography>
              </Box>
            </Box>
          )}

          {dialogType === 'edit' && selectedDataset && (
            <Box>
              <TextField
                fullWidth
                label="Dataset Name"
                margin="normal"
                defaultValue={selectedDataset.name}
              />
              <TextField
                fullWidth
                label="Description"
                margin="normal"
                multiline
                rows={3}
                placeholder="Update description..."
              />
            </Box>
          )}

          {dialogType === 'download' && (
            <Box>
              <Typography mb={2}>
                Download {selectedDataset?.name}
              </Typography>
              <LinearProgress variant="determinate" value={0} />
              <Typography variant="caption" color="textSecondary" mt={1}>
                Ready to download ({selectedDataset?.size})
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>
            Cancel
          </Button>
          <Button
            onClick={handleDatasetAction}
            variant="contained"
            color={dialogType === 'delete' ? 'error' : 'primary'}
          >
            {dialogType === 'upload' && 'Upload'}
            {dialogType === 'delete' && 'Delete'}
            {dialogType === 'edit' && 'Save Changes'}
            {dialogType === 'download' && 'Download'}
            {dialogType === 'view' && 'Close'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default DataManagement;