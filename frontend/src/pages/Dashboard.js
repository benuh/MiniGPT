import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Button,
  LinearProgress,
  IconButton,
} from '@mui/material';
import {
  SmartToy as ModelIcon,
  Timeline as TrainingIcon,
  Storage as DataIcon,
  Chat as ChatIcon,
  PlayArrow as PlayIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';

function Dashboard() {
  const stats = [
    { title: 'Active Models', value: '3', icon: <ModelIcon />, color: 'primary' },
    { title: 'Training Jobs', value: '1', icon: <TrainingIcon />, color: 'secondary' },
    { title: 'Datasets', value: '5', icon: <DataIcon />, color: 'success' },
    { title: 'Chat Sessions', value: '12', icon: <ChatIcon />, color: 'info' },
  ];

  const recentActivity = [
    { title: 'Model "MiniGPT-v2" training completed', time: '2 hours ago', type: 'success' },
    { title: 'New dataset "stories.txt" uploaded', time: '4 hours ago', type: 'info' },
    { title: 'Chat session started', time: '6 hours ago', type: 'default' },
    { title: 'Model evaluation completed', time: '1 day ago', type: 'success' },
  ];

  const getChipColor = (type) => {
    switch (type) {
      case 'success': return 'success';
      case 'info': return 'info';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" fontWeight="bold">
          Dashboard
        </Typography>
        <IconButton color="primary">
          <RefreshIcon />
        </IconButton>
      </Box>

      <Grid container spacing={3} mb={4}>
        {stats.map((stat, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography color="textSecondary" gutterBottom variant="body2">
                      {stat.title}
                    </Typography>
                    <Typography variant="h4" fontWeight="bold">
                      {stat.value}
                    </Typography>
                  </Box>
                  <Box color={`${stat.color}.main`}>
                    {stat.icon}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" mb={2}>
                Current Training Progress
              </Typography>
              <Box mb={2}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">MiniGPT-v3 Training</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Epoch 15/50 (68%)
                  </Typography>
                </Box>
                <LinearProgress variant="determinate" value={68} sx={{ height: 8, borderRadius: 4 }} />
              </Box>
              <Box display="flex" gap={2} mt={2}>
                <Button variant="contained" startIcon={<PlayIcon />} size="small">
                  Resume
                </Button>
                <Button variant="outlined" size="small">
                  View Details
                </Button>
              </Box>
            </CardContent>
          </Card>

          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" mb={2}>
                Quick Actions
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    variant="outlined"
                    fullWidth
                    startIcon={<TrainingIcon />}
                    sx={{ py: 2 }}
                  >
                    Start Training
                  </Button>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    variant="outlined"
                    fullWidth
                    startIcon={<ChatIcon />}
                    sx={{ py: 2 }}
                  >
                    New Chat
                  </Button>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    variant="outlined"
                    fullWidth
                    startIcon={<DataIcon />}
                    sx={{ py: 2 }}
                  >
                    Upload Data
                  </Button>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Button
                    variant="outlined"
                    fullWidth
                    startIcon={<ModelIcon />}
                    sx={{ py: 2 }}
                  >
                    Manage Models
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" fontWeight="bold" mb={2}>
                Recent Activity
              </Typography>
              <Box>
                {recentActivity.map((activity, index) => (
                  <Box key={index} mb={2} pb={2} borderBottom={index < recentActivity.length - 1 ? 1 : 0} borderColor="divider">
                    <Typography variant="body2" mb={1}>
                      {activity.title}
                    </Typography>
                    <Box display="flex" justifyContent="space-between" alignItems="center">
                      <Typography variant="caption" color="textSecondary">
                        {activity.time}
                      </Typography>
                      <Chip label={activity.type} size="small" color={getChipColor(activity.type)} />
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

export default Dashboard;