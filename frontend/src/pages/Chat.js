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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Link,
} from '@mui/material';
import {
  Send as SendIcon,
  SmartToy as BotIcon,
  Person as UserIcon,
  Settings as SettingsIcon,
  Clear as ClearIcon,
  Download as ExportIcon,
  Key as KeyIcon,
} from '@mui/icons-material';
import MLTermDialog from '../components/MLTermDialog';
import MLTermTooltip from '../components/MLTermTooltip';

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
  const [selectedModel, setSelectedModel] = useState('local');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(150);
  const [isGenerating, setIsGenerating] = useState(false);
  const [helpDialog, setHelpDialog] = useState({ open: false, termKey: null });
  const [availableModels, setAvailableModels] = useState([]);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [authDialog, setAuthDialog] = useState({ open: false, provider: null });
  const [authStatus, setAuthStatus] = useState({});
  const [apiTokens, setApiTokens] = useState({
    huggingface: '',
    openai: '',
    anthropic: ''
  });
  const messagesEndRef = useRef(null);

  const handleLearnMore = (termKey) => {
    setHelpDialog({ open: true, termKey });
  };

  const handleCloseHelp = () => {
    setHelpDialog({ open: false, termKey: null });
  };

  const handleTermClick = (termKey) => {
    setHelpDialog({ open: true, termKey });
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setIsLoadingModels(true);
        const response = await fetch('http://localhost:8000/model/list');
        if (response.ok) {
          const data = await response.json();

          // Combine local and remote models for the dropdown
          const allModels = [];

          // Add local models
          if (data.local_models && data.local_models.length > 0) {
            allModels.push({
              key: 'local',
              name: 'Local Model',
              type: 'local',
              description: `${data.local_models.length} local model(s) available`
            });
          }

          // Add remote models
          if (data.remote_models && data.remote_models.length > 0) {
            data.remote_models.forEach(model => {
              allModels.push({
                key: model.key,
                name: model.name || model.key,
                type: 'remote',
                description: model.description || '',
                provider: model.provider || '',
                cost: model.cost || ''
              });
            });
          }

          setAvailableModels(allModels);

          // Set default model
          if (allModels.length > 0) {
            // Prefer remote models if no local models available
            const hasLocal = allModels.some(m => m.type === 'local');
            if (!hasLocal && allModels.length > 0) {
              // Default to first free remote model
              const freeRemote = allModels.find(m => m.cost && m.cost.includes('Free'));
              if (freeRemote) {
                setSelectedModel(freeRemote.key);
              } else if (allModels[0]) {
                setSelectedModel(allModels[0].key);
              }
            }
          }
        } else {
          console.error('Failed to fetch models:', response.statusText);
        }
      } catch (error) {
        console.error('Error fetching models:', error);
        // Fallback to default remote model
        setAvailableModels([
          {
            key: 'hf:gpt2',
            name: 'GPT-2 (HuggingFace)',
            type: 'remote',
            description: 'Free GPT-2 model from HuggingFace',
            provider: 'HuggingFace',
            cost: 'Free'
          }
        ]);
        setSelectedModel('hf:gpt2');
      } finally {
        setIsLoadingModels(false);
      }
    };

    fetchModels();
  }, []);

  // Fetch authentication status
  useEffect(() => {
    const fetchAuthStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/remote/auth-status');
        if (response.ok) {
          const status = await response.json();
          setAuthStatus(status);
        }
      } catch (error) {
        console.error('Error fetching auth status:', error);
      }
    };

    fetchAuthStatus();
  }, []);

  const handleSetApiToken = async (provider, token) => {
    try {
      const response = await fetch('http://localhost:8000/remote/set-token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ provider, token }),
      });

      if (response.ok) {
        // Update auth status
        const statusResponse = await fetch('http://localhost:8000/remote/auth-status');
        if (statusResponse.ok) {
          const status = await statusResponse.json();
          setAuthStatus(status);
        }

        // Clear model cache to refetch with new auth
        setAvailableModels([]);
        setIsLoadingModels(true);

        // Refetch models
        const modelsResponse = await fetch('http://localhost:8000/model/list');
        if (modelsResponse.ok) {
          const data = await modelsResponse.json();
          // [Same model processing logic as in the other useEffect]
          const allModels = [];
          if (data.local_models && data.local_models.length > 0) {
            allModels.push({
              key: 'local',
              name: 'Local Model',
              type: 'local',
              description: `${data.local_models.length} local model(s) available`
            });
          }
          if (data.remote_models && data.remote_models.length > 0) {
            data.remote_models.forEach(model => {
              allModels.push({
                key: model.key,
                name: model.name || model.key,
                type: 'remote',
                description: model.description || '',
                provider: model.provider || '',
                cost: model.cost || ''
              });
            });
          }
          setAvailableModels(allModels);
        }
        setIsLoadingModels(false);

        setAuthDialog({ open: false, provider: null });
        setApiTokens(prev => ({ ...prev, [provider]: '' }));

        return true;
      } else {
        throw new Error('Failed to set token');
      }
    } catch (error) {
      console.error('Error setting API token:', error);
      return false;
    }
  };

  const handleOpenAuthDialog = (provider) => {
    setAuthDialog({ open: true, provider });
  };

  const handleCloseAuthDialog = () => {
    setAuthDialog({ open: false, provider: null });
    setApiTokens(prev => ({ ...prev, [authDialog.provider]: '' }));
  };

  const handleSaveToken = async () => {
    const { provider } = authDialog;
    const token = apiTokens[provider];

    if (token.trim()) {
      const success = await handleSetApiToken(provider, token.trim());
      if (success) {
        // Success handled in handleSetApiToken
      } else {
        alert('Failed to set API token. Please try again.');
      }
    }
  };

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
      console.log('🔵 Sending chat request:', {
        message: messageToSend,
        model: selectedModel,
        max_length: maxTokens,
        temperature: temperature,
        top_k: 50,
      });

      // Call the real MiniGPT API
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageToSend,
          model: selectedModel,
          max_length: maxTokens,
          temperature: temperature,
          top_k: 50,
        }),
      });

      console.log('🔵 Response status:', response.status, response.statusText);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('❌ API Error:', {
          status: response.status,
          statusText: response.statusText,
          detail: errorData.detail || 'Unknown error'
        });

        throw new Error(`API Error (${response.status}): ${errorData.detail || response.statusText}`);
      }

      const data = await response.json();
      console.log('✅ API Response:', data);

      const botMessage = {
        id: messages.length + 2,
        type: 'bot',
        content: data.response,
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('❌ Frontend Error:', error);

      let errorMessage = '';
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        errorMessage = `🔌 Connection Error: Cannot reach backend at http://localhost:8000\n\nDebugging steps:\n1. Is the backend running? Check the backend terminal\n2. Is it on port 8000? Look for "Uvicorn running on http://0.0.0.0:8000"\n3. Try opening http://localhost:8000 in your browser`;
      } else if (error.message.includes('503')) {
        const modelType = selectedModel === 'local' ? 'local' : 'remote';
        if (modelType === 'local') {
          errorMessage = `🤖 Local Model Not Ready: ${error.message}\n\nOptions:\n1. Train a local model: cd backend && python -m minigpt.train\n2. Use a remote model instead (select from dropdown above)`;
        } else {
          errorMessage = `🌐 Remote Model Error: ${error.message}\n\nTrying alternative:\n1. Check your internet connection\n2. Try a different remote model\n3. For premium models (OpenAI, Claude), ensure API keys are set`;
        }
      } else if (error.message.includes('500')) {
        errorMessage = `💥 Server Error: ${error.message}\n\nCheck the backend terminal for detailed error logs and stack trace.`;
      } else {
        errorMessage = `❌ Unexpected Error: ${error.message}\n\nFull error: ${JSON.stringify(error, Object.getOwnPropertyNames(error), 2)}`;
      }

      const errorBotMessage = {
        id: messages.length + 2,
        type: 'bot',
        content: errorMessage,
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages(prev => [...prev, errorBotMessage]);
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
          <IconButton onClick={() => handleOpenAuthDialog('huggingface')} color="primary" title="API Keys">
            <KeyIcon />
          </IconButton>
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
                    disabled={isLoadingModels}
                  >
                    {isLoadingModels ? (
                      <MenuItem value="">Loading models...</MenuItem>
                    ) : availableModels.length === 0 ? (
                      <MenuItem value="">No models available</MenuItem>
                    ) : (
                      availableModels.map((model) => (
                        <MenuItem key={model.key} value={model.key}>
                          <Box>
                            <Typography variant="body2">
                              {model.name}
                            </Typography>
                            {model.type === 'remote' && (
                              <Typography variant="caption" color="textSecondary">
                                {model.provider} • {model.cost}
                              </Typography>
                            )}
                            {model.description && (
                              <Typography variant="caption" color="textSecondary" display="block">
                                {model.description}
                              </Typography>
                            )}
                          </Box>
                        </MenuItem>
                      ))
                    )}
                  </Select>
                </FormControl>
                {selectedModel && selectedModel !== 'local' && (
                  <Typography variant="caption" color="primary" mt={1} display="block">
                    🌐 Using remote model: {availableModels.find(m => m.key === selectedModel)?.name || selectedModel}
                  </Typography>
                )}
              </Box>

              <Box mb={3}>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Typography gutterBottom variant="body2">
                    Temperature: {temperature}
                  </Typography>
                  <MLTermTooltip
                    termKey="temperature"
                    onLearnMore={handleLearnMore}
                    size="small"
                  />
                </Box>
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
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Typography gutterBottom variant="body2">
                    Max Tokens: {maxTokens}
                  </Typography>
                  <MLTermTooltip
                    termKey="max_tokens"
                    onLearnMore={handleLearnMore}
                    size="small"
                  />
                </Box>
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
                  <Chip
                    label={availableModels.find(m => m.key === selectedModel)?.name || selectedModel}
                    size="small"
                    color={selectedModel === 'local' ? 'default' : 'primary'}
                    icon={selectedModel === 'local' ? undefined : <span>🌐</span>}
                  />
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

      {/* ML Terms Help Dialog */}
      <MLTermDialog
        termKey={helpDialog.termKey}
        open={helpDialog.open}
        onClose={handleCloseHelp}
        onTermClick={handleTermClick}
      />

      {/* API Authentication Dialog */}
      <Dialog open={authDialog.open} onClose={handleCloseAuthDialog} maxWidth="sm" fullWidth>
        <DialogTitle>
          API Keys & Authentication
        </DialogTitle>
        <DialogContent>
          <Box mb={2}>
            <Alert severity="info">
              Set API keys to access premium models and improve rate limits for free models.
            </Alert>
          </Box>

          {Object.entries(authStatus).map(([provider, status]) => (
            <Box key={provider} mb={3}>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Typography variant="h6" sx={{ textTransform: 'capitalize' }}>
                  {provider}
                </Typography>
                <Chip
                  label={status.authenticated ? 'Connected' : 'Not Connected'}
                  color={status.authenticated ? 'success' : 'default'}
                  size="small"
                />
              </Box>

              <Typography variant="body2" color="textSecondary" mb={2}>
                {status.description}
              </Typography>

              {!status.authenticated && (
                <Box>
                  <TextField
                    fullWidth
                    type="password"
                    placeholder={`Enter ${provider} API key`}
                    value={apiTokens[provider] || ''}
                    onChange={(e) => setApiTokens(prev => ({ ...prev, [provider]: e.target.value }))}
                    size="small"
                    sx={{ mb: 1 }}
                  />
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => {
                      setAuthDialog({ open: true, provider });
                    }}
                  >
                    Set {provider} API Key
                  </Button>
                </Box>
              )}

              {provider === 'huggingface' && (
                <Typography variant="caption" color="textSecondary" display="block" mt={1}>
                  Get free token at: <Link href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</Link>
                </Typography>
              )}
              {provider === 'openai' && (
                <Typography variant="caption" color="textSecondary" display="block" mt={1}>
                  Get API key at: <Link href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com/api-keys</Link>
                </Typography>
              )}
              {provider === 'anthropic' && (
                <Typography variant="caption" color="textSecondary" display="block" mt={1}>
                  Get API key at: <Link href="https://console.anthropic.com/" target="_blank">console.anthropic.com</Link>
                </Typography>
              )}
            </Box>
          ))}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseAuthDialog}>Close</Button>
          <Button
            onClick={handleSaveToken}
            variant="contained"
            disabled={!apiTokens[authDialog.provider]?.trim()}
          >
            Save API Key
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default Chat;