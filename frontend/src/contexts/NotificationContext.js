import React, { createContext, useContext, useState } from 'react';
import { Snackbar, Alert } from '@mui/material';

const NotificationContext = createContext();

export const useNotification = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};

export const NotificationProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([]);

  const showNotification = (message, severity = 'info', duration = 4000) => {
    const id = Date.now();
    const notification = {
      id,
      message,
      severity,
      duration,
      open: true,
    };

    setNotifications(prev => [...prev, notification]);

    // Auto-hide notification after duration
    if (duration > 0) {
      setTimeout(() => {
        hideNotification(id);
      }, duration);
    }

    return id;
  };

  const hideNotification = (id) => {
    setNotifications(prev =>
      prev.map(notification =>
        notification.id === id
          ? { ...notification, open: false }
          : notification
      )
    );

    // Remove notification after animation
    setTimeout(() => {
      setNotifications(prev =>
        prev.filter(notification => notification.id !== id)
      );
    }, 300);
  };

  const showSuccess = (message, duration) => {
    return showNotification(message, 'success', duration);
  };

  const showError = (message, duration) => {
    return showNotification(message, 'error', duration);
  };

  const showWarning = (message, duration) => {
    return showNotification(message, 'warning', duration);
  };

  const showInfo = (message, duration) => {
    return showNotification(message, 'info', duration);
  };

  const clearAll = () => {
    setNotifications([]);
  };

  const value = {
    showNotification,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    hideNotification,
    clearAll,
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}

      {/* Render notifications */}
      {notifications.map((notification) => (
        <Snackbar
          key={notification.id}
          open={notification.open}
          autoHideDuration={notification.duration}
          onClose={() => hideNotification(notification.id)}
          anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
          sx={{ mt: 8 }}
        >
          <Alert
            onClose={() => hideNotification(notification.id)}
            severity={notification.severity}
            variant="filled"
            sx={{
              minWidth: '300px',
              boxShadow: 3,
            }}
          >
            {notification.message}
          </Alert>
        </Snackbar>
      ))}
    </NotificationContext.Provider>
  );
};