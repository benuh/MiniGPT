import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Alert,
  Chip,
} from '@mui/material';
import {
  Rocket as QuickStartIcon,
  School as LearnIcon,
  Speed as FastIcon,
  Security as SafeIcon,
} from '@mui/icons-material';
import ChatbotSetupGuide from '../components/ChatbotSetupGuide';

function GettingStarted() {
  const navigate = useNavigate();

  const handleNavigate = (path) => {
    navigate(path);
  };

  const features = [
    {
      icon: <FastIcon color="primary" />,
      title: 'Quick Setup',
      description: 'Get your chatbot running in under an hour with our guided workflow'
    },
    {
      icon: <LearnIcon color="secondary" />,
      title: 'Learn as You Go',
      description: 'Interactive help explains ML concepts while you build'
    },
    {
      icon: <SafeIcon color="success" />,
      title: 'Safe & Reliable',
      description: 'Automatic checkpoints and error handling protect your progress'
    }
  ];

  const quickActions = [
    {
      title: 'I have training data ready',
      description: 'Jump straight to uploading your text files',
      action: () => navigate('/data'),
      buttonText: 'Upload Data',
      color: 'primary'
    },
    {
      title: 'I want to try with sample data',
      description: 'Use our built-in datasets to test the system',
      action: () => navigate('/training'),
      buttonText: 'Start Training',
      color: 'secondary'
    },
    {
      title: 'I just want to see how it works',
      description: 'Explore the interface and learn about the process',
      action: () => navigate('/dashboard'),
      buttonText: 'Explore Dashboard',
      color: 'info'
    }
  ];

  return (
    <Box>
      {/* Header */}
      <Box textAlign="center" mb={4}>
        <Typography variant="h3" fontWeight="bold" mb={2}>
          Getting Started with MiniGPT
        </Typography>
        <Typography variant="h6" color="textSecondary" mb={3}>
          Build your own AI chatbot in 4 simple steps
        </Typography>

        <Grid container spacing={2} justifyContent="center" mb={3}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={4} key={index}>
              <Box display="flex" alignItems="center" gap={1}>
                {feature.icon}
                <Box textAlign="left">
                  <Typography variant="subtitle2" fontWeight="bold">
                    {feature.title}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    {feature.description}
                  </Typography>
                </Box>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Quick Actions */}
      <Alert severity="info" sx={{ mb: 4 }}>
        <Typography variant="subtitle2" fontWeight="bold" mb={1}>
          Choose Your Starting Point:
        </Typography>
        <Grid container spacing={2}>
          {quickActions.map((action, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="subtitle2" fontWeight="bold" mb={1}>
                    {action.title}
                  </Typography>
                  <Typography variant="body2" color="textSecondary" mb={2}>
                    {action.description}
                  </Typography>
                  <Button
                    variant="contained"
                    color={action.color}
                    onClick={action.action}
                    size="small"
                  >
                    {action.buttonText}
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Alert>

      {/* Main Flowchart Guide */}
      <ChatbotSetupGuide onNavigate={handleNavigate} />

      {/* Additional Resources */}
      <Box mt={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" fontWeight="bold" mb={2}>
              📚 Additional Resources
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" fontWeight="bold" mb={1}>
                  Sample Datasets
                </Typography>
                <Typography variant="body2" color="textSecondary" mb={2}>
                  Pre-loaded datasets you can use for training:
                </Typography>
                <Box display="flex" gap={1} flexWrap="wrap">
                  <Chip label="stories.txt" size="small" />
                  <Chip label="shakespeare.txt" size="small" />
                  <Chip label="wiki_sample.txt" size="small" />
                </Box>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" fontWeight="bold" mb={1}>
                  Need Help?
                </Typography>
                <Typography variant="body2" color="textSecondary" mb={2}>
                  Look for the help icons (?) throughout the interface for detailed explanations of ML concepts.
                </Typography>
                <Button variant="outlined" size="small" onClick={() => navigate('/training')}>
                  See Help Examples
                </Button>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
}

export default GettingStarted;