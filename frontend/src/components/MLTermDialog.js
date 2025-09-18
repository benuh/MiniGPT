import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  IconButton,
} from '@mui/material';
import {
  Help as HelpIcon,
  LightbulbOutlined as TipIcon,
  LinkOutlined as ExampleIcon,
  Close as CloseIcon,
} from '@mui/icons-material';
import { mlTerminology, getRelatedTerms } from '../data/mlTerminology';

const MLTermDialog = ({ termKey, open, onClose, onTermClick }) => {
  const term = mlTerminology[termKey];
  const relatedTerms = getRelatedTerms(termKey);

  if (!term) return null;

  const handleRelatedTermClick = (relatedTermKey) => {
    onTermClick(relatedTermKey);
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { borderRadius: 3 }
      }}
    >
      <DialogTitle sx={{ pb: 1 }}>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box display="flex" alignItems="center" gap={1}>
            <HelpIcon color="primary" />
            <Typography variant="h5" fontWeight="bold">
              {term.title}
            </Typography>
          </Box>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent>
        {/* Quick Definition */}
        <Box
          mb={3}
          p={2}
          bgcolor="primary.light"
          borderRadius={2}
          sx={{
            bgcolor: 'rgba(25, 118, 210, 0.08)',
            border: '1px solid',
            borderColor: 'primary.light'
          }}
        >
          <Typography variant="h6" fontWeight="bold" color="primary.main" mb={1}>
            Quick Definition
          </Typography>
          <Typography variant="body1" color="text.primary" sx={{ fontWeight: 500 }}>
            {term.definition}
          </Typography>
        </Box>

        {/* Detailed Explanation */}
        <Box mb={3}>
          <Typography variant="h6" fontWeight="bold" mb={2}>
            Detailed Explanation
          </Typography>
          <Typography variant="body1" paragraph>
            {term.detailed}
          </Typography>
        </Box>

        {/* Examples */}
        {term.examples && (
          <Box mb={3}>
            <Typography variant="h6" fontWeight="bold" mb={2} display="flex" alignItems="center" gap={1}>
              <ExampleIcon color="secondary" />
              Examples
            </Typography>
            <List dense>
              {term.examples.map((example, index) => (
                <ListItem key={index} sx={{ pl: 0 }}>
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    <Box
                      sx={{
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        bgcolor: 'secondary.main'
                      }}
                    />
                  </ListItemIcon>
                  <ListItemText primary={example} />
                </ListItem>
              ))}
            </List>
          </Box>
        )}

        {/* Tips */}
        {term.tips && (
          <Box mb={3}>
            <Typography variant="h6" fontWeight="bold" mb={2} display="flex" alignItems="center" gap={1}>
              <TipIcon color="warning" />
              Pro Tips
            </Typography>
            <List dense>
              {term.tips.map((tip, index) => (
                <ListItem key={index} sx={{ pl: 0 }}>
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    <TipIcon color="warning" sx={{ fontSize: 16 }} />
                  </ListItemIcon>
                  <ListItemText primary={tip} />
                </ListItem>
              ))}
            </List>
          </Box>
        )}

        {/* Related Terms */}
        {relatedTerms.length > 0 && (
          <Box>
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6" fontWeight="bold" mb={2}>
              Related Terms
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1}>
              {relatedTerms.map((relatedTerm) => (
                <Chip
                  key={relatedTerm.key}
                  label={relatedTerm.title}
                  onClick={() => handleRelatedTermClick(relatedTerm.key)}
                  color="primary"
                  variant="outlined"
                  sx={{
                    cursor: 'pointer',
                    '&:hover': {
                      bgcolor: 'primary.light',
                      color: 'primary.dark'
                    }
                  }}
                />
              ))}
            </Box>
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 3 }}>
        <Button onClick={onClose} variant="contained" size="large">
          Got it!
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default MLTermDialog;