import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Avatar,
  Chip,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Divider,
} from '@mui/material';
import {
  Send as SendIcon,
  SmartToy as BotIcon,
  Person as UserIcon,
  Settings as SettingsIcon,
  Clear as ClearIcon,
  Download as ExportIcon,
} from '@mui/icons-material';

function Chat() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: 'Hello! I\'m MiniGPT. How can I help you today?',
      timestamp: new Date().toLocaleTimeString(),
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedModel, setSelectedModel] = useState('MiniGPT-v2');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(150);
  const [isGenerating, setIsGenerating] = useState(false);
  const messagesEndRef = useRef(null);

  const models = ['MiniGPT-v2', 'MiniGPT-v1', 'MiniGPT-base'];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isGenerating) return;

    const userMessage = {
      id: messages.length + 1,
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages(prev => [...prev, userMessage]);
    const messageToSend = inputMessage;
    setInputMessage('');
    setIsGenerating(true);

    try {
      // Call the real MiniGPT API
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageToSend,
          max_length: maxTokens,
          temperature: temperature,
          top_k: 50,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const botMessage = {
        id: messages.length + 2,
        type: 'bot',
        content: data.response,
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error calling API:', error);

      const errorMessage = {
        id: messages.length + 2,
        type: 'bot',
        content: `Sorry, I couldn't process your message. Error: ${error.message}. Make sure the backend is running at http://localhost:8000`,
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: 1,
        type: 'bot',
        content: 'Hello! I\'m MiniGPT. How can I help you today?',
        timestamp: new Date().toLocaleTimeString(),
      }
    ]);
  };

  const exportChat = () => {
    const chatHistory = messages.map(msg =>
      `[${msg.timestamp}] ${msg.type === 'user' ? 'You' : 'MiniGPT'}: ${msg.content}`
    ).join('\n');

    const blob = new Blob([chatHistory], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" fontWeight="bold">
          Chat with MiniGPT
        </Typography>
        <Box display="flex" gap={1}>
          <IconButton onClick={exportChat} color="primary">
            <ExportIcon />
          </IconButton>
          <IconButton onClick={clearChat} color="primary">
            <ClearIcon />
          </IconButton>
        </Box>
      </Box>

      <Box display="flex" gap={3} height="calc(100vh - 200px)">
        <Box flex={1}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', pb: 0 }}>
              <Box
                flex={1}
                sx={{
                  overflowY: 'auto',
                  mb: 2,
                  pr: 1,
                  '&::-webkit-scrollbar': {
                    width: '6px',
                  },
                  '&::-webkit-scrollbar-track': {
                    background: '#f1f1f1',
                    borderRadius: '3px',
                  },
                  '&::-webkit-scrollbar-thumb': {
                    background: '#c1c1c1',
                    borderRadius: '3px',
                  },
                }}
              >
                {messages.map((message) => (
                  <Box key={message.id} mb={2}>
                    <Box
                      display="flex"
                      justifyContent={message.type === 'user' ? 'flex-end' : 'flex-start'}
                      alignItems="flex-start"
                      gap={1}
                    >
                      {message.type === 'bot' && (
                        <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>
                          <BotIcon fontSize="small" />
                        </Avatar>
                      )}
                      <Box
                        sx={{
                          maxWidth: '70%',
                          bgcolor: message.type === 'user' ? 'primary.main' : 'grey.100',
                          color: message.type === 'user' ? 'white' : 'text.primary',
                          p: 2,
                          borderRadius: 2,
                          borderTopLeftRadius: message.type === 'bot' ? 0 : 2,
                          borderTopRightRadius: message.type === 'user' ? 0 : 2,
                        }}
                      >
                        <Typography variant="body1">{message.content}</Typography>
                        <Typography
                          variant="caption"
                          sx={{
                            mt: 1,
                            display: 'block',
                            opacity: 0.7,
                            fontSize: '0.75rem',
                          }}
                        >
                          {message.timestamp}
                        </Typography>
                      </Box>
                      {message.type === 'user' && (
                        <Avatar sx={{ bgcolor: 'grey.400', width: 32, height: 32 }}>
                          <UserIcon fontSize="small" />
                        </Avatar>
                      )}
                    </Box>
                  </Box>
                ))}
                {isGenerating && (
                  <Box display="flex" alignItems="center" gap={1} mb={2}>
                    <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>
                      <BotIcon fontSize="small" />
                    </Avatar>
                    <Box
                      sx={{
                        bgcolor: 'grey.100',
                        p: 2,
                        borderRadius: 2,
                        borderTopLeftRadius: 0,
                      }}
                    >
                      <Typography variant="body1" color="text.secondary">
                        Thinking...
                      </Typography>
                    </Box>
                  </Box>
                )}
                <div ref={messagesEndRef} />
              </Box>

              <Box display="flex" gap={1} pt={2} borderTop={1} borderColor="divider">
                <TextField
                  fullWidth
                  multiline
                  maxRows={3}
                  placeholder="Type your message..."
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={isGenerating}
                  variant="outlined"
                  size="small"
                />
                <Button
                  variant="contained"
                  onClick={handleSendMessage}
                  disabled={!inputMessage.trim() || isGenerating}
                  sx={{ minWidth: 'auto', px: 2 }}
                >
                  <SendIcon />
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Box>

        <Box width={300}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" gap={1} mb={3}>
                <SettingsIcon color="primary" />
                <Typography variant="h6" fontWeight="bold">
                  Chat Settings
                </Typography>
              </Box>

              <Box mb={3}>
                <FormControl fullWidth size="small">
                  <InputLabel>Model</InputLabel>
                  <Select
                    value={selectedModel}
                    label="Model"
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    {models.map((model) => (
                      <MenuItem key={model} value={model}>
                        {model}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>

              <Box mb={3}>
                <Typography gutterBottom variant="body2">
                  Temperature: {temperature}
                </Typography>
                <Slider
                  value={temperature}
                  onChange={(e, value) => setTemperature(value)}
                  min={0.1}
                  max={2.0}
                  step={0.1}
                  size="small"
                  valueLabelDisplay="auto"
                />
                <Typography variant="caption" color="textSecondary">
                  Controls randomness in responses
                </Typography>
              </Box>

              <Box mb={3}>
                <Typography gutterBottom variant="body2">
                  Max Tokens: {maxTokens}
                </Typography>
                <Slider
                  value={maxTokens}
                  onChange={(e, value) => setMaxTokens(value)}
                  min={50}
                  max={500}
                  step={10}
                  size="small"
                  valueLabelDisplay="auto"
                />
                <Typography variant="caption" color="textSecondary">
                  Maximum response length
                </Typography>
              </Box>

              <Divider sx={{ my: 2 }} />

              <Box>
                <Typography variant="body2" fontWeight="bold" mb={1}>
                  Current Session
                </Typography>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="caption" color="textSecondary">
                    Messages:
                  </Typography>
                  <Chip label={messages.length} size="small" />
                </Box>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="caption" color="textSecondary">
                    Model:
                  </Typography>
                  <Chip label={selectedModel} size="small" color="primary" />
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="caption" color="textSecondary">
                    Status:
                  </Typography>
                  <Chip
                    label={isGenerating ? 'Generating' : 'Ready'}
                    size="small"
                    color={isGenerating ? 'warning' : 'success'}
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Box>
    </Box>
  );
}

export default Chat;