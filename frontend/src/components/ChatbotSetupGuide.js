import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Alert,
  Chip,
  IconButton,
  Collapse,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Storage as DataIcon,
  Settings as TrainIcon,
  Chat as ChatIcon,
  CheckCircle as CompleteIcon,
  ExpandMore as ExpandIcon,
  ExpandLess as CollapseIcon,
  Timeline as FlowIcon,
  Lightbulb as TipIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

const ChatbotSetupGuide = ({ onNavigate }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [expandedSteps, setExpandedSteps] = useState({});

  const toggleStepExpansion = (stepIndex) => {
    setExpandedSteps(prev => ({
      ...prev,
      [stepIndex]: !prev[stepIndex]
    }));
  };

  const steps = [
    {
      label: 'Prepare Your Data',
      icon: <DataIcon />,
      time: '5-10 minutes',
      difficulty: 'Easy',
      description: 'Upload and organize text data for training your chatbot',
      actions: [
        'Go to Data Management page',
        'Upload your text files (.txt, .json, .csv)',
        'Verify data quality and format',
        'Choose your primary dataset'
      ],
      tips: [
        'Start with 1-5MB of clean text for good results',
        'Remove special characters and formatting',
        'Use conversational data for better chat responses'
      ],
      warnings: [
        'Avoid copyrighted content',
        'Ensure data is clean and well-formatted'
      ],
      nextAction: () => onNavigate('/data'),
      buttonText: 'Upload Data',
      status: 'pending' // pending, in_progress, completed
    },
    {
      label: 'Configure Training',
      icon: <TrainIcon />,
      time: '2-3 minutes',
      difficulty: 'Medium',
      description: 'Set up training parameters for optimal learning',
      actions: [
        'Go to Training page',
        'Choose your uploaded dataset',
        'Set training parameters (start with defaults)',
        'Enable GPU if available',
        'Enable checkpoints for safety'
      ],
      tips: [
        'Start with 20-50 epochs for initial training',
        'Use batch size 32 for most GPUs',
        'Keep learning rate at 0.001 for beginners'
      ],
      warnings: [
        'Training can take 30 minutes to several hours',
        'Monitor GPU memory usage'
      ],
      nextAction: () => onNavigate('/training'),
      buttonText: 'Configure Training',
      status: 'pending'
    },
    {
      label: 'Train Your Model',
      icon: <StartIcon />,
      time: '30 minutes - 2 hours',
      difficulty: 'Easy',
      description: 'Start the training process and monitor progress',
      actions: [
        'Review your configuration',
        'Click "Start Training"',
        'Monitor training progress',
        'Wait for completion',
        'Check final accuracy metrics'
      ],
      tips: [
        'Training loss should decrease over time',
        'You can pause/resume training if needed',
        'Best model is automatically saved'
      ],
      warnings: [
        'Don\'t close the browser during training',
        'Ensure stable internet connection'
      ],
      nextAction: () => onNavigate('/training'),
      buttonText: 'Start Training',
      status: 'pending'
    },
    {
      label: 'Test Your Chatbot',
      icon: <ChatIcon />,
      time: '5-10 minutes',
      difficulty: 'Easy',
      description: 'Chat with your newly trained model and test responses',
      actions: [
        'Go to Chat page',
        'Select your trained model',
        'Adjust temperature for creativity level',
        'Start chatting and test responses',
        'Fine-tune settings as needed'
      ],
      tips: [
        'Lower temperature (0.3-0.5) for focused responses',
        'Higher temperature (0.8-1.2) for creative responses',
        'Test with various question types'
      ],
      warnings: [
        'First responses might need adjustment',
        'Model quality depends on training data'
      ],
      nextAction: () => onNavigate('/chat'),
      buttonText: 'Test Chatbot',
      status: 'pending'
    }
  ];

  const getDifficultyColor = (difficulty) => {
    switch (difficulty.toLowerCase()) {
      case 'easy': return 'success';
      case 'medium': return 'warning';
      case 'hard': return 'error';
      default: return 'default';
    }
  };

  const getTimeEstimate = () => {
    const totalTime = steps.reduce((acc, step) => {
      const time = step.time.split('-')[0];
      const minutes = parseInt(time.replace(/\D/g, ''));
      return acc + (isNaN(minutes) ? 30 : minutes);
    }, 0);

    if (totalTime >= 60) {
      return `${Math.floor(totalTime / 60)}h ${totalTime % 60}m`;
    }
    return `${totalTime}m`;
  };

  return (
    <Card sx={{ maxWidth: 800, mx: 'auto' }}>
      <CardContent>
        <Box display="flex" alignItems="center" gap={2} mb={3}>
          <FlowIcon color="primary" sx={{ fontSize: 32 }} />
          <Box>
            <Typography variant="h5" fontWeight="bold">
              Chatbot Setup Guide
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Complete workflow from data to working chatbot • Est. time: {getTimeEstimate()}
            </Typography>
          </Box>
        </Box>

        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            Follow these steps in order to create your custom chatbot. Each step builds on the previous one.
          </Typography>
        </Alert>

        <Stepper orientation="vertical" activeStep={activeStep}>
          {steps.map((step, index) => (
            <Step key={index}>
              <StepLabel
                icon={step.icon}
                sx={{
                  '& .MuiStepIcon-root': {
                    fontSize: '1.5rem',
                  }
                }}
              >
                <Box display="flex" alignItems="center" gap={2} flexWrap="wrap">
                  <Typography variant="h6" fontWeight="bold">
                    {step.label}
                  </Typography>
                  <Chip
                    label={step.difficulty}
                    size="small"
                    color={getDifficultyColor(step.difficulty)}
                  />
                  <Chip
                    label={step.time}
                    size="small"
                    variant="outlined"
                  />
                </Box>
              </StepLabel>

              <StepContent>
                <Box mb={2}>
                  <Typography variant="body1" paragraph>
                    {step.description}
                  </Typography>

                  {/* Action Steps */}
                  <Typography variant="subtitle2" fontWeight="bold" mb={1}>
                    📋 Steps to Complete:
                  </Typography>
                  <List dense sx={{ mb: 2 }}>
                    {step.actions.map((action, actionIndex) => (
                      <ListItem key={actionIndex} sx={{ py: 0.5 }}>
                        <ListItemIcon sx={{ minWidth: 32 }}>
                          <Typography variant="body2" color="primary" fontWeight="bold">
                            {actionIndex + 1}.
                          </Typography>
                        </ListItemIcon>
                        <ListItemText
                          primary={action}
                          primaryTypographyProps={{ variant: 'body2' }}
                        />
                      </ListItem>
                    ))}
                  </List>

                  {/* Tips Section */}
                  <Box mb={2}>
                    <Box
                      display="flex"
                      alignItems="center"
                      gap={1}
                      onClick={() => toggleStepExpansion(`${index}-tips`)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TipIcon color="warning" fontSize="small" />
                      <Typography variant="subtitle2" fontWeight="bold">
                        Pro Tips
                      </Typography>
                      <IconButton size="small">
                        {expandedSteps[`${index}-tips`] ? <CollapseIcon /> : <ExpandIcon />}
                      </IconButton>
                    </Box>
                    <Collapse in={expandedSteps[`${index}-tips`]}>
                      <List dense sx={{ pl: 2 }}>
                        {step.tips.map((tip, tipIndex) => (
                          <ListItem key={tipIndex} sx={{ py: 0.5 }}>
                            <ListItemIcon sx={{ minWidth: 24 }}>
                              <Box
                                sx={{
                                  width: 6,
                                  height: 6,
                                  borderRadius: '50%',
                                  bgcolor: 'warning.main'
                                }}
                              />
                            </ListItemIcon>
                            <ListItemText
                              primary={tip}
                              primaryTypographyProps={{ variant: 'body2' }}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Collapse>
                  </Box>

                  {/* Warnings Section */}
                  {step.warnings.length > 0 && (
                    <Box mb={2}>
                      <Box
                        display="flex"
                        alignItems="center"
                        gap={1}
                        onClick={() => toggleStepExpansion(`${index}-warnings`)}
                        sx={{ cursor: 'pointer' }}
                      >
                        <WarningIcon color="error" fontSize="small" />
                        <Typography variant="subtitle2" fontWeight="bold">
                          Important Notes
                        </Typography>
                        <IconButton size="small">
                          {expandedSteps[`${index}-warnings`] ? <CollapseIcon /> : <ExpandIcon />}
                        </IconButton>
                      </Box>
                      <Collapse in={expandedSteps[`${index}-warnings`]}>
                        <List dense sx={{ pl: 2 }}>
                          {step.warnings.map((warning, warningIndex) => (
                            <ListItem key={warningIndex} sx={{ py: 0.5 }}>
                              <ListItemIcon sx={{ minWidth: 24 }}>
                                <Box
                                  sx={{
                                    width: 6,
                                    height: 6,
                                    borderRadius: '50%',
                                    bgcolor: 'error.main'
                                  }}
                                />
                              </ListItemIcon>
                              <ListItemText
                                primary={warning}
                                primaryTypographyProps={{ variant: 'body2' }}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Collapse>
                    </Box>
                  )}

                  <Box display="flex" gap={2} mt={2}>
                    <Button
                      variant="contained"
                      onClick={step.nextAction}
                      startIcon={step.icon}
                    >
                      {step.buttonText}
                    </Button>

                    {index < steps.length - 1 && (
                      <Button
                        variant="outlined"
                        onClick={() => setActiveStep(index + 1)}
                      >
                        Next Step
                      </Button>
                    )}
                  </Box>
                </Box>
              </StepContent>
            </Step>
          ))}
        </Stepper>

        {activeStep === steps.length && (
          <Box textAlign="center" mt={3}>
            <CompleteIcon color="success" sx={{ fontSize: 48, mb: 2 }} />
            <Typography variant="h6" fontWeight="bold" mb={1}>
              🎉 Congratulations!
            </Typography>
            <Typography variant="body1" mb={2}>
              Your chatbot is ready! You can now chat with your custom AI model.
            </Typography>
            <Button
              variant="contained"
              onClick={() => onNavigate('/chat')}
              startIcon={<ChatIcon />}
            >
              Start Chatting
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ChatbotSetupGuide;