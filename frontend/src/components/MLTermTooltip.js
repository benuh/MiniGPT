import React from 'react';
import {
  IconButton,
  Tooltip,
  Typography,
  Box,
} from '@mui/material';
import {
  HelpOutline as HelpIcon,
} from '@mui/icons-material';
import { mlTerminology } from '../data/mlTerminology';

const MLTermTooltip = ({ termKey, onLearnMore, size = 'small' }) => {
  const term = mlTerminology[termKey];

  if (!term) return null;

  const tooltipContent = (
    <Box sx={{ maxWidth: 300 }}>
      <Typography variant="subtitle2" fontWeight="bold" mb={1}>
        {term.title}
      </Typography>
      <Typography variant="body2" mb={1}>
        {term.definition}
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic' }}>
        Click for detailed explanation
      </Typography>
    </Box>
  );

  return (
    <Tooltip
      title={tooltipContent}
      arrow
      placement="top"
      componentsProps={{
        tooltip: {
          sx: {
            bgcolor: 'background.paper',
            color: 'text.primary',
            border: 1,
            borderColor: 'divider',
            boxShadow: 3,
            maxWidth: 320,
          }
        },
        arrow: {
          sx: {
            color: 'background.paper',
            '&::before': {
              border: 1,
              borderColor: 'divider',
            }
          }
        }
      }}
    >
      <IconButton
        size={size}
        onClick={() => onLearnMore(termKey)}
        sx={{
          color: 'primary.main',
          '&:hover': {
            bgcolor: 'primary.light',
            color: 'primary.dark'
          }
        }}
      >
        <HelpIcon fontSize={size === 'small' ? 'small' : 'medium'} />
      </IconButton>
    </Tooltip>
  );
};

export default MLTermTooltip;