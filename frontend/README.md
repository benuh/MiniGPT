# MiniGPT Frontend

A modern React-based user interface for the MiniGPT AI training platform built with Material-UI.

## Features

- **Dashboard**: Overview of training progress, models, and system stats
- **Training Interface**: Configure and monitor model training with real-time progress
- **Chat Interface**: Interactive chat with trained models
- **Model Management**: Deploy, download, and manage trained models
- **Data Management**: Upload, organize, and manage training datasets
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Technology Stack

- **React 18** - Modern React with hooks and functional components
- **Material-UI (MUI) 5** - Comprehensive React component library
- **React Router 6** - Client-side routing
- **Socket.IO** - Real-time communication with backend
- **Axios** - HTTP client for API calls

## Quick Start

### Prerequisites

- Node.js 16 or higher
- npm or yarn package manager

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Copy environment configuration:
   ```bash
   cp .env.example .env
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

5. Open your browser to [http://localhost:3000](http://localhost:3000)

## Available Scripts

- `npm start` or `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run test` - Run test suite
- `npm run lint` - Check code style
- `npm run lint:fix` - Fix linting issues
- `npm run serve` - Build and serve production build

## Project Structure

```
frontend/
├── public/          # Static files
├── src/
│   ├── components/  # Reusable React components
│   │   └── Layout.js
│   ├── contexts/    # React context providers
│   │   ├── ApiContext.js
│   │   └── NotificationContext.js
│   ├── pages/       # Page components
│   │   ├── Dashboard.js
│   │   ├── Training.js
│   │   ├── Chat.js
│   │   ├── Models.js
│   │   └── DataManagement.js
│   ├── App.js       # Main app component
│   └── index.js     # Entry point
├── package.json
└── README.md
```

## Configuration

The application can be configured via environment variables:

- `REACT_APP_API_URL` - Backend API URL (default: http://localhost:8000)
- `REACT_APP_WS_URL` - WebSocket URL for real-time updates
- `REACT_APP_ENV` - Environment mode
- `REACT_APP_DEBUG` - Enable debug logging

## Backend Integration

This frontend is designed to work with the MiniGPT FastAPI backend. Make sure the backend is running on the configured API URL before using the frontend.

### API Endpoints Expected

- `GET /api/dashboard/stats` - Dashboard statistics
- `POST /api/training/start` - Start model training
- `GET /api/training/progress` - Get training progress
- `GET /api/models` - List available models
- `POST /api/chat` - Send chat message
- `GET /api/datasets` - List datasets
- `POST /api/datasets/upload` - Upload dataset

## Development

### Code Style

The project uses ESLint for code quality. Run `npm run lint` to check for issues.

### Components

All components are built using Material-UI and follow React best practices:
- Functional components with hooks
- Context for state management
- Responsive design patterns

### Real-time Updates

The app supports real-time updates via WebSocket for:
- Training progress
- Chat responses
- System notifications

## Production Build

To create a production build:

```bash
npm run build
```

This creates an optimized build in the `build/` directory ready for deployment.

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

MIT License - see the main project LICENSE file for details.