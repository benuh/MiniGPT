import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  RobotIcon,
  DashboardIcon,
  TrainingIcon,
  ChatIcon,
  ModelsIcon,
  DataIcon,
  MenuIcon,
  CloseIcon,
} from './Icons';

const TailwindLayout = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const navigationItems = [
    { text: 'Getting Started', icon: RobotIcon, path: '/getting-started' },
    { text: 'Dashboard', icon: DashboardIcon, path: '/dashboard' },
    { text: 'Training', icon: TrainingIcon, path: '/training' },
    { text: 'Chat', icon: ChatIcon, path: '/chat' },
    { text: 'Models', icon: ModelsIcon, path: '/models' },
    { text: 'Data', icon: DataIcon, path: '/data' },
  ];

  const handleNavigation = (path) => {
    navigate(path);
    setSidebarOpen(false);
  };

  const isActivePage = (path) => {
    return location.pathname === path;
  };

  const currentPageTitle = navigationItems.find(item =>
    isActivePage(item.path)
  )?.text || 'MiniGPT';

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        {/* Sidebar header */}
        <div className="flex items-center justify-between h-16 px-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="flex-shrink-0">
              <RobotIcon className="w-8 h-8 text-oxford-blue" />
            </div>
            <div className="flex-shrink-0">
              <h1 className="text-xl font-bold text-oxford-charcoal">MiniGPT</h1>
              <p className="text-xs text-gray-500">AI Training Platform</p>
            </div>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden p-1 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100"
          >
            <CloseIcon className="w-6 h-6" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="mt-6 px-3">
          <div className="space-y-1">
            {navigationItems.map((item) => {
              const IconComponent = item.icon;
              const isActive = isActivePage(item.path);

              return (
                <button
                  key={item.text}
                  onClick={() => handleNavigation(item.path)}
                  className={`
                    group w-full flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors duration-200
                    ${isActive
                      ? 'bg-primary-50 text-oxford-blue border-r-2 border-oxford-blue'
                      : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                    }
                  `}
                >
                  <IconComponent
                    className={`
                      mr-3 w-5 h-5 transition-colors duration-200
                      ${isActive ? 'text-oxford-blue' : 'text-gray-400 group-hover:text-gray-500'}
                    `}
                  />
                  <span className={isActive ? 'font-semibold' : ''}>{item.text}</span>
                </button>
              );
            })}
          </div>
        </nav>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200">
          <div className="text-xs text-gray-500 text-center">
            <p>Built with React & Tailwind</p>
            <p className="mt-1">© 2024 MiniGPT</p>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top header */}
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="flex items-center justify-between h-16 px-6">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100"
              >
                <MenuIcon className="w-6 h-6" />
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">{currentPageTitle}</h1>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="hidden md:flex items-center space-x-2 text-sm text-gray-500">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span>AI Training Platform</span>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <RobotIcon className="w-6 h-6 text-oxford-blue" />
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-auto bg-gray-50">
          <div className="container mx-auto px-6 py-8 max-w-7xl">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

export default TailwindLayout;