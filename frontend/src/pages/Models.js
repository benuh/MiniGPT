import React, { useState } from 'react';
import {
  PlayIcon,
  StopIcon,
  CogIcon,
  ChartBarIcon,
  TrainingIcon,
  DataIcon
} from '../components/Icons';

const Models = () => {
  const [models, setModels] = useState([
    {
      id: 1,
      name: 'MiniGPT-v2',
      version: '2.0.1',
      status: 'Active',
      accuracy: '94.2%',
      size: '125 MB',
      created: '2 hours ago',
      lastTrained: '2 hours ago',
      parameters: '50M',
    },
    {
      id: 2,
      name: 'MiniGPT-v1',
      version: '1.0.0',
      status: 'Inactive',
      accuracy: '89.7%',
      size: '98 MB',
      created: '1 day ago',
      lastTrained: '1 day ago',
      parameters: '35M',
    },
  ]);

  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [newModel, setNewModel] = useState({
    name: '',
    baseModel: '',
    description: ''
  });

  const getStatusColor = (status) => {
    switch (status) {
      case 'Active': return 'bg-green-100 text-green-800';
      case 'Training': return 'bg-yellow-100 text-yellow-800';
      case 'Inactive': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const handleCreateModel = () => {
    if (newModel.name && newModel.baseModel) {
      const model = {
        id: models.length + 1,
        name: newModel.name,
        version: '1.0.0',
        status: 'Training',
        accuracy: 'Training...',
        size: 'N/A',
        created: 'Just now',
        lastTrained: 'Just now',
        parameters: 'N/A',
      };
      setModels([...models, model]);
      setNewModel({ name: '', baseModel: '', description: '' });
      setShowCreateDialog(false);
    }
  };

  const handleDeleteModel = () => {
    if (selectedModel) {
      setModels(models.filter(model => model.id !== selectedModel.id));
      setShowDeleteDialog(false);
      setSelectedModel(null);
    }
  };

  const handleDeployModel = (model) => {
    setModels(prev => prev.map(m =>
      m.id === model.id
        ? { ...m, status: 'Active' }
        : { ...m, status: 'Inactive' }
    ));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-oxford-charcoal">Models</h1>
          <p className="text-oxford-gray mt-2">Manage and deploy your trained MiniGPT models</p>
        </div>
        <button
          onClick={() => setShowCreateDialog(true)}
          className="btn-primary flex items-center space-x-2"
        >
          <TrainingIcon className="w-5 h-5" />
          <span>Create New Model</span>
        </button>
      </div>

      {/* Model Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card">
          <div className="card-body">
            <div className="flex items-center">
              <div className="p-3 bg-oxford-blue bg-opacity-10 rounded-lg">
                <ChartBarIcon className="w-6 h-6 text-oxford-blue" />
              </div>
              <div className="ml-4">
                <div className="text-2xl font-bold text-oxford-charcoal">{models.length}</div>
                <div className="text-sm text-gray-600">Total Models</div>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-body">
            <div className="flex items-center">
              <div className="p-3 bg-green-100 rounded-lg">
                <PlayIcon className="w-6 h-6 text-green-600" />
              </div>
              <div className="ml-4">
                <div className="text-2xl font-bold text-oxford-charcoal">
                  {models.filter(m => m.status === 'Active').length}
                </div>
                <div className="text-sm text-gray-600">Active Models</div>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-body">
            <div className="flex items-center">
              <div className="p-3 bg-yellow-100 rounded-lg">
                <TrainingIcon className="w-6 h-6 text-yellow-600" />
              </div>
              <div className="ml-4">
                <div className="text-2xl font-bold text-oxford-charcoal">
                  {models.filter(m => m.status === 'Training').length}
                </div>
                <div className="text-sm text-gray-600">Training</div>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-body">
            <div className="flex items-center">
              <div className="p-3 bg-oxford-gold bg-opacity-20 rounded-lg">
                <DataIcon className="w-6 h-6 text-oxford-gold" />
              </div>
              <div className="ml-4">
                <div className="text-2xl font-bold text-oxford-charcoal">
                  {models.reduce((total, model) => {
                    const size = parseInt(model.size);
                    return total + (isNaN(size) ? 0 : size);
                  }, 0)} MB
                </div>
                <div className="text-sm text-gray-600">Total Size</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Models Table */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center space-x-3">
            <ChartBarIcon className="w-6 h-6 text-oxford-blue" />
            <h2 className="text-xl font-semibold text-oxford-charcoal">Model Library</h2>
          </div>
        </div>
        <div className="card-body p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-oxford-gray-warm">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                    Model
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                    Performance
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                    Details
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {models.map((model) => (
                  <tr key={model.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div>
                        <div className="text-sm font-medium text-oxford-charcoal">{model.name}</div>
                        <div className="text-sm text-gray-500">v{model.version}</div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`badge ${getStatusColor(model.status)}`}>
                        {model.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-oxford-charcoal">{model.accuracy}</div>
                      <div className="text-sm text-gray-500">{model.parameters} params</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-oxford-charcoal">{model.size}</div>
                      <div className="text-sm text-gray-500">Created {model.created}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex items-center space-x-2">
                        {model.status !== 'Active' && (
                          <button
                            onClick={() => handleDeployModel(model)}
                            className="text-green-600 hover:text-green-900 p-1 rounded"
                            title="Deploy Model"
                          >
                            <PlayIcon className="w-4 h-4" />
                          </button>
                        )}
                        {model.status === 'Active' && (
                          <button
                            onClick={() => setModels(prev => prev.map(m =>
                              m.id === model.id ? { ...m, status: 'Inactive' } : m
                            ))}
                            className="text-red-600 hover:text-red-900 p-1 rounded"
                            title="Stop Model"
                          >
                            <StopIcon className="w-4 h-4" />
                          </button>
                        )}
                        <button
                          onClick={() => {
                            setSelectedModel(model);
                            setShowDeleteDialog(true);
                          }}
                          className="text-red-600 hover:text-red-900 p-1 rounded"
                          title="Delete Model"
                        >
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
                          </svg>
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Create Model Dialog */}
      {showCreateDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-oxford-charcoal">Create New Model</h3>
            </div>
            <div className="px-6 py-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Model Name
                </label>
                <input
                  type="text"
                  value={newModel.name}
                  onChange={(e) => setNewModel({...newModel, name: e.target.value})}
                  placeholder="e.g., MiniGPT-v3"
                  className="input-field"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Base Model
                </label>
                <select
                  value={newModel.baseModel}
                  onChange={(e) => setNewModel({...newModel, baseModel: e.target.value})}
                  className="input-field"
                >
                  <option value="">Select base model</option>
                  <option value="scratch">Train from scratch</option>
                  <option value="minigpt-v2">Based on MiniGPT-v2</option>
                  <option value="minigpt-v1">Based on MiniGPT-v1</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Description
                </label>
                <textarea
                  value={newModel.description}
                  onChange={(e) => setNewModel({...newModel, description: e.target.value})}
                  placeholder="Describe this model..."
                  rows={3}
                  className="input-field resize-none"
                />
              </div>
            </div>
            <div className="px-6 py-4 border-t border-gray-200 flex justify-end space-x-3">
              <button
                onClick={() => setShowCreateDialog(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateModel}
                disabled={!newModel.name || !newModel.baseModel}
                className={`${newModel.name && newModel.baseModel ? 'btn-primary' : 'bg-gray-300 text-gray-500 cursor-not-allowed py-3 px-8 rounded-lg'}`}
              >
                Create Model
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Dialog */}
      {showDeleteDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-oxford-charcoal">Delete Model</h3>
            </div>
            <div className="px-6 py-4">
              <p className="text-gray-600">
                Are you sure you want to delete <strong>{selectedModel?.name}</strong>?
                This action cannot be undone.
              </p>
            </div>
            <div className="px-6 py-4 border-t border-gray-200 flex justify-end space-x-3">
              <button
                onClick={() => setShowDeleteDialog(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteModel}
                className="bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-8 transition duration-200 ease-in-out rounded-lg"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Models;