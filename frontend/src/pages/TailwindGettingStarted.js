import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  RobotIcon,
  TrainingIcon,
  ChatIcon,
  DataIcon,
  ChevronRightIcon,
} from '../components/Icons';

const TailwindGettingStarted = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>,
      title: 'Quick Setup',
      description: 'Get your chatbot running in under an hour with our guided workflow'
    },
    {
      icon: <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse"></div>,
      title: 'Learn as You Go',
      description: 'Interactive help explains ML concepts while you build'
    },
    {
      icon: <div className="w-3 h-3 bg-purple-400 rounded-full animate-pulse"></div>,
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
      icon: <DataIcon className="w-6 h-6" />,
      color: 'blue'
    },
    {
      title: 'I want to try with sample data',
      description: 'Use our built-in datasets to test the system',
      action: () => navigate('/training'),
      buttonText: 'Start Training',
      icon: <TrainingIcon className="w-6 h-6" />,
      color: 'green'
    },
    {
      title: 'I just want to explore',
      description: 'Browse the interface and learn about the process',
      action: () => navigate('/dashboard'),
      buttonText: 'Explore Dashboard',
      icon: <RobotIcon className="w-6 h-6" />,
      color: 'purple'
    }
  ];

  const steps = [
    {
      number: '01',
      title: 'Prepare Your Data',
      description: 'Upload and organize text data for training your chatbot',
      time: '5-10 min',
      difficulty: 'Easy',
      color: 'blue'
    },
    {
      number: '02',
      title: 'Configure Training',
      description: 'Set up training parameters for optimal learning',
      time: '2-3 min',
      difficulty: 'Medium',
      color: 'green'
    },
    {
      number: '03',
      title: 'Train Your Model',
      description: 'Start the training process and monitor progress',
      time: '30 min - 2 hours',
      difficulty: 'Easy',
      color: 'purple'
    },
    {
      number: '04',
      title: 'Test Your Chatbot',
      description: 'Chat with your newly trained model and test responses',
      time: '5-10 min',
      difficulty: 'Easy',
      color: 'orange'
    }
  ];

  const getColorClasses = (color) => {
    const colors = {
      blue: 'bg-blue-500 text-white hover:bg-blue-600',
      green: 'bg-green-500 text-white hover:bg-green-600',
      purple: 'bg-purple-500 text-white hover:bg-purple-600',
      orange: 'bg-orange-500 text-white hover:bg-orange-600'
    };
    return colors[color] || colors.blue;
  };

  const getBadgeClasses = (difficulty) => {
    const badges = {
      'Easy': 'bg-green-100 text-green-800',
      'Medium': 'bg-yellow-100 text-yellow-800',
      'Hard': 'bg-red-100 text-red-800'
    };
    return badges[difficulty] || badges.Easy;
  };

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="text-center space-y-6">
        <div className="flex justify-center">
          <div className="relative">
            <div className="absolute inset-0 bg-blue-400 rounded-full blur-xl opacity-30 animate-pulse"></div>
            <RobotIcon className="relative w-20 h-20 text-blue-600" />
          </div>
        </div>

        <div className="space-y-4">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900">
            Getting Started with <span className="text-blue-600">MiniGPT</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Build your own AI chatbot in 4 simple steps. No prior ML experience required.
          </p>
        </div>

        {/* Feature highlights */}
        <div className="flex flex-wrap justify-center gap-6 mt-8">
          {features.map((feature, index) => (
            <div key={index} className="flex items-center space-x-3 bg-white rounded-lg px-4 py-2 shadow-sm border border-gray-200">
              {feature.icon}
              <div className="text-left">
                <p className="font-semibold text-gray-900 text-sm">{feature.title}</p>
                <p className="text-gray-500 text-xs">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-8">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Choose Your Starting Point</h2>
          <p className="text-gray-600">Pick the option that best describes your situation</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {quickActions.map((action, index) => (
            <div key={index} className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200">
              <div className="flex items-center space-x-3 mb-4">
                <div className={`p-2 rounded-lg ${getColorClasses(action.color).replace('hover:bg-', 'bg-').replace(' text-white', '').replace(' hover:', ' ')}`}>
                  {action.icon}
                </div>
                <h3 className="font-semibold text-gray-900">{action.title}</h3>
              </div>
              <p className="text-gray-600 text-sm mb-4">{action.description}</p>
              <button
                onClick={action.action}
                className={`w-full px-4 py-2 rounded-lg font-medium transition-colors duration-200 ${getColorClasses(action.color)}`}
              >
                {action.buttonText}
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Step-by-step guide */}
      <div>
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Complete Workflow</h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Follow these steps in order to create your custom chatbot. Each step builds on the previous one.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {steps.map((step, index) => (
            <div key={index} className="relative">
              {/* Connection line */}
              {index < steps.length - 1 && (
                <div className="hidden lg:block absolute top-8 left-full w-6 h-0.5 bg-gray-200 z-0"></div>
              )}

              <div className="card hover:shadow-xl transition-shadow duration-300 relative z-10">
                <div className="card-body space-y-4">
                  {/* Step number */}
                  <div className="flex items-center justify-between">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center text-white font-bold text-lg ${getColorClasses(step.color).replace('hover:bg-', 'bg-').replace(' text-white', '').replace(' hover:', ' ')}`}>
                      {step.number}
                    </div>
                    <div className="flex space-x-2">
                      <span className={`badge ${getBadgeClasses(step.difficulty)}`}>
                        {step.difficulty}
                      </span>
                    </div>
                  </div>

                  {/* Content */}
                  <div>
                    <h3 className="font-bold text-gray-900 mb-2">{step.title}</h3>
                    <p className="text-gray-600 text-sm mb-3">{step.description}</p>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-500">⏱️ {step.time}</span>
                      <ChevronRightIcon className="w-4 h-4 text-gray-400" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Start button */}
        <div className="text-center mt-12">
          <button
            onClick={() => navigate('/data')}
            className="btn-primary text-lg px-8 py-4 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
          >
            <span className="flex items-center space-x-2">
              <span>Start Building Your Chatbot</span>
              <ChevronRightIcon className="w-5 h-5" />
            </span>
          </button>
        </div>
      </div>

      {/* Resources section */}
      <div className="bg-gray-900 rounded-2xl p-8 text-white">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-xl font-bold mb-4">📚 Sample Datasets</h3>
            <p className="text-gray-300 mb-4">Pre-loaded datasets you can use for training:</p>
            <div className="flex flex-wrap gap-2">
              {['stories.txt', 'shakespeare.txt', 'wiki_sample.txt'].map((dataset) => (
                <span key={dataset} className="bg-gray-800 px-3 py-1 rounded-full text-sm">
                  {dataset}
                </span>
              ))}
            </div>
          </div>
          <div>
            <h3 className="text-xl font-bold mb-4">❓ Need Help?</h3>
            <p className="text-gray-300 mb-4">
              Look for the help icons (?) throughout the interface for detailed explanations of ML concepts.
            </p>
            <button
              onClick={() => navigate('/training')}
              className="bg-white text-gray-900 px-4 py-2 rounded-lg hover:bg-gray-100 transition-colors duration-200 font-medium"
            >
              See Help Examples
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TailwindGettingStarted;