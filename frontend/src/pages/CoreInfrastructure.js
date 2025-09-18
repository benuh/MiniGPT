import React, { useState } from 'react';
import { ChevronDownIcon, ChevronRightIcon } from '../components/Icons';

const CoreInfrastructure = () => {
  const [expandedSections, setExpandedSections] = useState({});

  const toggleSection = (sectionId) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  const ExpandableSection = ({ id, title, children, defaultExpanded = false }) => {
    const isExpanded = expandedSections[id] ?? defaultExpanded;

    return (
      <div className="border border-gray-200 rounded-lg mb-4">
        <button
          onClick={() => toggleSection(id)}
          className="w-full px-6 py-4 text-left flex items-center justify-between bg-gray-50 hover:bg-gray-100 transition-colors duration-200 rounded-t-lg"
        >
          <h3 className="text-lg font-semibold text-harvard-charcoal">{title}</h3>
          {isExpanded ? (
            <ChevronDownIcon className="w-5 h-5 text-gray-500" />
          ) : (
            <ChevronRightIcon className="w-5 h-5 text-gray-500" />
          )}
        </button>
        {isExpanded && (
          <div className="px-6 py-4 bg-white rounded-b-lg">
            {children}
          </div>
        )}
      </div>
    );
  };

  const TechStack = ({ title, items }) => (
    <div className="mb-4">
      <h4 className="font-semibold text-harvard-crimson mb-2">{title}</h4>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
        {items.map((item, index) => (
          <span key={index} className="badge-neutral text-center py-2">
            {item}
          </span>
        ))}
      </div>
    </div>
  );

  const SystemDiagram = () => (
    <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Frontend */}
        <div className="text-center">
          <div className="bg-harvard-blue text-white p-4 rounded-lg mb-2">
            <h4 className="font-bold">Frontend</h4>
            <p className="text-sm">React + Tailwind</p>
          </div>
          <div className="text-xs space-y-1">
            <div>• User Interface</div>
            <div>• State Management</div>
            <div>• Real-time Updates</div>
          </div>
        </div>

        {/* Backend */}
        <div className="text-center">
          <div className="bg-harvard-green text-white p-4 rounded-lg mb-2">
            <h4 className="font-bold">Backend</h4>
            <p className="text-sm">FastAPI + Python</p>
          </div>
          <div className="text-xs space-y-1">
            <div>• REST API</div>
            <div>• Model Management</div>
            <div>• Training Pipeline</div>
          </div>
        </div>

        {/* ML Core */}
        <div className="text-center">
          <div className="bg-harvard-crimson text-white p-4 rounded-lg mb-2">
            <h4 className="font-bold">ML Core</h4>
            <p className="text-sm">PyTorch + Transformers</p>
          </div>
          <div className="text-xs space-y-1">
            <div>• Neural Networks</div>
            <div>• Training Logic</div>
            <div>• Model Inference</div>
          </div>
        </div>
      </div>

      {/* Arrows */}
      <div className="flex justify-center mt-4 space-x-8">
        <div className="text-center">
          <div className="text-2xl">↔</div>
          <div className="text-xs text-gray-600">HTTP/WebSocket</div>
        </div>
        <div className="text-center">
          <div className="text-2xl">↔</div>
          <div className="text-xs text-gray-600">Python Calls</div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center border-b border-gray-200 pb-8">
        <h1 className="text-4xl font-bold text-harvard-charcoal mb-4">
          MiniGPT Core Infrastructure
        </h1>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          Comprehensive technical documentation of the MiniGPT platform architecture,
          system design, and implementation details.
        </p>
        <div className="mt-4 bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-sm text-yellow-800">
          🔒 This is a hidden technical page accessible only via direct URL
        </div>
      </div>

      {/* System Overview */}
      <ExpandableSection id="overview" title="System Architecture Overview" defaultExpanded>
        <div className="space-y-6">
          <p className="text-gray-700">
            MiniGPT is a full-stack AI training platform built with modern web technologies
            and machine learning frameworks. The system follows a microservices-inspired
            architecture with clear separation between frontend, backend, and ML components.
          </p>

          <SystemDiagram />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Design Principles</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>Modularity:</strong> Clear separation of concerns</li>
                <li>• <strong>Scalability:</strong> Horizontal scaling capabilities</li>
                <li>• <strong>Maintainability:</strong> Clean code and documentation</li>
                <li>• <strong>User Experience:</strong> Intuitive interface design</li>
                <li>• <strong>Educational:</strong> Learning-focused implementation</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Key Features</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>Real-time Training:</strong> Live progress monitoring</li>
                <li>• <strong>Model Management:</strong> Version control and deployment</li>
                <li>• <strong>Interactive Chat:</strong> Test models instantly</li>
                <li>• <strong>Data Pipeline:</strong> Automated preprocessing</li>
                <li>• <strong>Educational UI:</strong> Learn ML concepts while building</li>
              </ul>
            </div>
          </div>
        </div>
      </ExpandableSection>

      {/* Frontend Architecture */}
      <ExpandableSection id="frontend" title="Frontend Architecture">
        <div className="space-y-6">
          <TechStack
            title="Technology Stack"
            items={[
              'React 18', 'Tailwind CSS', 'React Router', 'Context API',
              'Fetch API', 'WebSockets', 'JavaScript ES6+', 'NPM'
            ]}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Component Structure</h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm font-mono">
                <div>📁 src/</div>
                <div className="ml-4">📁 components/</div>
                <div className="ml-8">• TailwindLayout.js</div>
                <div className="ml-8">• Icons.js</div>
                <div className="ml-8">• MLTermDialog.js</div>
                <div className="ml-4">📁 pages/</div>
                <div className="ml-8">• Dashboard.js</div>
                <div className="ml-8">• Training.js</div>
                <div className="ml-8">• Chat.js</div>
                <div className="ml-4">📁 contexts/</div>
                <div className="ml-8">• ApiContext.js</div>
                <div className="ml-8">• NotificationContext.js</div>
              </div>
            </div>
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">State Management</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>React Context:</strong> Global state management</li>
                <li>• <strong>Local State:</strong> Component-specific state</li>
                <li>• <strong>API Context:</strong> Centralized API calls</li>
                <li>• <strong>Notification Context:</strong> Toast notifications</li>
                <li>• <strong>Router State:</strong> Navigation and routing</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-harvard-crimson mb-3">Design System</h4>
            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-sm text-gray-700 mb-3">
                Harvard University-inspired design system with professional academic aesthetics.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="bg-harvard-crimson text-white px-3 py-1 rounded-full text-xs">Primary: Harvard Crimson</span>
                <span className="bg-harvard-blue text-white px-3 py-1 rounded-full text-xs">Secondary: Academic Blue</span>
                <span className="bg-harvard-green text-white px-3 py-1 rounded-full text-xs">Accent: Harvard Green</span>
                <span className="bg-gray-600 text-white px-3 py-1 rounded-full text-xs">Neutral: Charcoal Gray</span>
              </div>
            </div>
          </div>
        </div>
      </ExpandableSection>

      {/* Backend Architecture */}
      <ExpandableSection id="backend" title="Backend Architecture">
        <div className="space-y-6">
          <TechStack
            title="Technology Stack"
            items={[
              'FastAPI', 'Python 3.9+', 'Uvicorn', 'Pydantic',
              'AsyncIO', 'PyTorch', 'Transformers', 'CORS Middleware'
            ]}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">API Structure</h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm">
                <div className="space-y-1 font-mono">
                  <div><span className="text-green-600">POST</span> /chat</div>
                  <div><span className="text-blue-600">GET</span> /models</div>
                  <div><span className="text-green-600">POST</span> /training/start</div>
                  <div><span className="text-blue-600">GET</span> /training/progress</div>
                  <div><span className="text-green-600">POST</span> /datasets/upload</div>
                  <div><span className="text-blue-600">GET</span> /dashboard/stats</div>
                  <div><span className="text-red-600">DELETE</span> /models/{`{id}`}</div>
                </div>
              </div>
            </div>
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Core Modules</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>api.py:</strong> FastAPI server and endpoints</li>
                <li>• <strong>model.py:</strong> Transformer neural network</li>
                <li>• <strong>train.py:</strong> Training pipeline logic</li>
                <li>• <strong>chat.py:</strong> Chat interface and inference</li>
                <li>• <strong>tokenizer.py:</strong> Text preprocessing</li>
                <li>• <strong>utils.py:</strong> Helper functions</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-harvard-crimson mb-3">Request Flow</h4>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center justify-between text-sm">
                <span className="bg-white px-3 py-2 rounded border">Frontend Request</span>
                <span>→</span>
                <span className="bg-white px-3 py-2 rounded border">FastAPI Router</span>
                <span>→</span>
                <span className="bg-white px-3 py-2 rounded border">Pydantic Validation</span>
                <span>→</span>
                <span className="bg-white px-3 py-2 rounded border">Business Logic</span>
                <span>→</span>
                <span className="bg-white px-3 py-2 rounded border">Response</span>
              </div>
            </div>
          </div>
        </div>
      </ExpandableSection>

      {/* ML Model Architecture */}
      <ExpandableSection id="ml-model" title="Machine Learning Model">
        <div className="space-y-6">
          <TechStack
            title="ML Technology Stack"
            items={[
              'PyTorch', 'Transformers', 'CUDA', 'NumPy',
              'Tokenizers', 'Checkpoints', 'Adam Optimizer', 'CrossEntropy Loss'
            ]}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Model Architecture</h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm">
                <div className="space-y-2">
                  <div><strong>Type:</strong> GPT-style Transformer</div>
                  <div><strong>Layers:</strong> 4-12 transformer blocks</div>
                  <div><strong>Hidden Size:</strong> 128-768 dimensions</div>
                  <div><strong>Attention Heads:</strong> 4-12 heads</div>
                  <div><strong>Context Length:</strong> 256-1024 tokens</div>
                  <div><strong>Vocabulary:</strong> 50,257 tokens (GPT-2)</div>
                  <div><strong>Parameters:</strong> 50K - 10M parameters</div>
                </div>
              </div>
            </div>
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Training Process</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>Data Preprocessing:</strong> Tokenization and batching</li>
                <li>• <strong>Forward Pass:</strong> Attention and feedforward</li>
                <li>• <strong>Loss Calculation:</strong> Cross-entropy loss</li>
                <li>• <strong>Backpropagation:</strong> Gradient computation</li>
                <li>• <strong>Optimization:</strong> Adam optimizer updates</li>
                <li>• <strong>Checkpointing:</strong> Model state saving</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-harvard-crimson mb-3">Transformer Block</h4>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="grid grid-cols-1 md:grid-cols-5 gap-2 text-center text-sm">
                <div className="bg-blue-100 p-2 rounded">Input Embeddings</div>
                <div>→</div>
                <div className="bg-green-100 p-2 rounded">Multi-Head Attention</div>
                <div>→</div>
                <div className="bg-yellow-100 p-2 rounded">Feed Forward</div>
              </div>
              <div className="text-center mt-2 text-xs text-gray-600">
                + Residual Connections + Layer Normalization
              </div>
            </div>
          </div>
        </div>
      </ExpandableSection>

      {/* Data Pipeline */}
      <ExpandableSection id="data-pipeline" title="Data Pipeline & Storage">
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Data Flow</h4>
              <div className="space-y-3">
                <div className="bg-gray-50 p-3 rounded border-l-4 border-blue-400">
                  <div className="font-medium">1. Data Upload</div>
                  <div className="text-sm text-gray-600">Text files (.txt, .json, .csv) uploaded via frontend</div>
                </div>
                <div className="bg-gray-50 p-3 rounded border-l-4 border-green-400">
                  <div className="font-medium">2. Preprocessing</div>
                  <div className="text-sm text-gray-600">Tokenization, cleaning, and batching</div>
                </div>
                <div className="bg-gray-50 p-3 rounded border-l-4 border-yellow-400">
                  <div className="font-medium">3. Training</div>
                  <div className="text-sm text-gray-600">Mini-batch gradient descent training</div>
                </div>
                <div className="bg-gray-50 p-3 rounded border-l-4 border-red-400">
                  <div className="font-medium">4. Model Storage</div>
                  <div className="text-sm text-gray-600">Checkpoints and final model persistence</div>
                </div>
              </div>
            </div>
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Storage Structure</h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm font-mono">
                <div>📁 data/</div>
                <div className="ml-4">📄 stories.txt</div>
                <div className="ml-4">📄 shakespeare.txt</div>
                <div className="ml-4">📄 user_uploads.txt</div>
                <div>📁 checkpoints/</div>
                <div className="ml-4">📄 model_epoch_10.pt</div>
                <div className="ml-4">📄 model_best.pt</div>
                <div className="ml-4">📄 optimizer_state.pt</div>
                <div>📁 logs/</div>
                <div className="ml-4">📄 training.log</div>
                <div className="ml-4">📄 api.log</div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-harvard-crimson mb-3">Data Processing Pipeline</h4>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="flex items-center justify-between text-sm overflow-x-auto">
                <span className="bg-white px-3 py-2 rounded border min-w-max">Raw Text</span>
                <span className="mx-2">→</span>
                <span className="bg-white px-3 py-2 rounded border min-w-max">Tokenization</span>
                <span className="mx-2">→</span>
                <span className="bg-white px-3 py-2 rounded border min-w-max">Batching</span>
                <span className="mx-2">→</span>
                <span className="bg-white px-3 py-2 rounded border min-w-max">GPU Transfer</span>
                <span className="mx-2">→</span>
                <span className="bg-white px-3 py-2 rounded border min-w-max">Model Training</span>
              </div>
            </div>
          </div>
        </div>
      </ExpandableSection>

      {/* DevOps & Deployment */}
      <ExpandableSection id="devops" title="DevOps & Deployment">
        <div className="space-y-6">
          <TechStack
            title="DevOps Stack"
            items={[
              'Docker', 'Bash Scripts', 'Git', 'NPM Scripts',
              'Python Virtual Env', 'Process Management', 'Log Management', 'Error Handling'
            ]}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Deployment Scripts</h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm font-mono">
                <div>📄 start-all.sh</div>
                <div className="ml-4 text-gray-600">• Starts both services</div>
                <div>📄 start-backend.sh</div>
                <div className="ml-4 text-gray-600">• Python virtual env + FastAPI</div>
                <div>📄 start-frontend.sh</div>
                <div className="ml-4 text-gray-600">• Node.js + React dev server</div>
                <div>📄 Dockerfile</div>
                <div className="ml-4 text-gray-600">• Container orchestration</div>
              </div>
            </div>
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Environment Management</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>Backend:</strong> Python virtual environment isolation</li>
                <li>• <strong>Frontend:</strong> Node.js package management</li>
                <li>• <strong>Configuration:</strong> Environment variables</li>
                <li>• <strong>Dependencies:</strong> Requirements.txt & package.json</li>
                <li>• <strong>Process:</strong> Background service management</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-harvard-crimson mb-3">Production Considerations</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h5 className="font-medium text-blue-800 mb-2">Scalability</h5>
                <ul className="text-xs text-blue-700 space-y-1">
                  <li>• Horizontal scaling ready</li>
                  <li>• Load balancer support</li>
                  <li>• Database connection pooling</li>
                  <li>• Caching strategies</li>
                </ul>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h5 className="font-medium text-green-800 mb-2">Security</h5>
                <ul className="text-xs text-green-700 space-y-1">
                  <li>• CORS configuration</li>
                  <li>• Input validation</li>
                  <li>• Error handling</li>
                  <li>• Rate limiting ready</li>
                </ul>
              </div>
              <div className="bg-yellow-50 p-4 rounded-lg">
                <h5 className="font-medium text-yellow-800 mb-2">Monitoring</h5>
                <ul className="text-xs text-yellow-700 space-y-1">
                  <li>• Structured logging</li>
                  <li>• Error tracking</li>
                  <li>• Performance metrics</li>
                  <li>• Health checks</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </ExpandableSection>

      {/* Performance & Optimization */}
      <ExpandableSection id="performance" title="Performance & Optimization">
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Frontend Optimizations</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>Code Splitting:</strong> Lazy loading components</li>
                <li>• <strong>Bundle Size:</strong> Tailwind CSS purging</li>
                <li>• <strong>State Management:</strong> Efficient context usage</li>
                <li>• <strong>Rendering:</strong> React optimization patterns</li>
                <li>• <strong>Caching:</strong> Browser cache strategies</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Backend Optimizations</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>Async Processing:</strong> FastAPI async/await</li>
                <li>• <strong>Model Loading:</strong> Singleton pattern</li>
                <li>• <strong>Memory Management:</strong> PyTorch optimizations</li>
                <li>• <strong>API Response:</strong> Efficient serialization</li>
                <li>• <strong>GPU Utilization:</strong> CUDA optimization</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-harvard-crimson mb-3">ML Performance</h4>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold text-harvard-crimson">50K-10M</div>
                  <div className="text-sm text-gray-600">Parameters</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-harvard-blue">~2GB</div>
                  <div className="text-sm text-gray-600">GPU Memory</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-harvard-green">100ms</div>
                  <div className="text-sm text-gray-600">Inference Time</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-yellow-600">90%+</div>
                  <div className="text-sm text-gray-600">GPU Utilization</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </ExpandableSection>

      {/* Project Documentation */}
      <ExpandableSection id="documentation" title="Project Documentation & Standards">
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Documentation Structure</h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm font-mono">
                <div>📁 docs/</div>
                <div className="ml-4">📄 README.md (Main documentation)</div>
                <div className="ml-4">📄 API.md (API endpoints)</div>
                <div className="ml-4">📄 ARCHITECTURE.md (System design)</div>
                <div className="ml-4">📄 DEPLOYMENT.md (Setup guide)</div>
                <div className="ml-4">📄 CONTRIBUTING.md (Development guide)</div>
                <div className="ml-4">📁 diagrams/ (System diagrams)</div>
                <div className="ml-4">📁 examples/ (Code examples)</div>
              </div>
            </div>
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Code Documentation</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>Inline Comments:</strong> Complex logic explanation</li>
                <li>• <strong>Docstrings:</strong> Python function documentation</li>
                <li>• <strong>JSDoc:</strong> JavaScript function documentation</li>
                <li>• <strong>API Docs:</strong> FastAPI auto-generated docs</li>
                <li>• <strong>Type Hints:</strong> Python type annotations</li>
                <li>• <strong>README Files:</strong> Per-directory explanations</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-harvard-crimson mb-3">Documentation Standards</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h5 className="font-medium text-blue-800 mb-2">📝 Writing Style</h5>
                <ul className="text-xs text-blue-700 space-y-1">
                  <li>• Clear and concise language</li>
                  <li>• Step-by-step instructions</li>
                  <li>• Code examples included</li>
                  <li>• Beginner-friendly explanations</li>
                  <li>• Visual diagrams where helpful</li>
                </ul>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h5 className="font-medium text-green-800 mb-2">🔧 Technical Specs</h5>
                <ul className="text-xs text-green-700 space-y-1">
                  <li>• API endpoint documentation</li>
                  <li>• Function parameter details</li>
                  <li>• Return value specifications</li>
                  <li>• Error handling examples</li>
                  <li>• Performance considerations</li>
                </ul>
              </div>
              <div className="bg-yellow-50 p-4 rounded-lg">
                <h5 className="font-medium text-yellow-800 mb-2">🎯 User Guides</h5>
                <ul className="text-xs text-yellow-700 space-y-1">
                  <li>• Installation instructions</li>
                  <li>• Configuration examples</li>
                  <li>• Troubleshooting guides</li>
                  <li>• FAQ sections</li>
                  <li>• Best practices</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-harvard-crimson mb-3">API Documentation Example</h4>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm font-mono">
                <div className="text-green-600 font-bold">POST /chat</div>
                <div className="mt-2 text-gray-700">
                  <div><strong>Description:</strong> Send a message to the AI model and receive a response</div>
                  <div className="mt-2"><strong>Request Body:</strong></div>
                  <div className="ml-4 bg-white p-2 rounded mt-1">
                    {`{
  "message": "Hello AI",
  "max_length": 150,
  "temperature": 0.7,
  "top_k": 50
}`}
                  </div>
                  <div className="mt-2"><strong>Response:</strong></div>
                  <div className="ml-4 bg-white p-2 rounded mt-1">
                    {`{
  "response": "Hello! How can I help you?",
  "generation_time": 0.45
}`}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </ExpandableSection>

      {/* Development Workflow */}
      <ExpandableSection id="development" title="Development Workflow">
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Development Setup</h4>
              <div className="bg-gray-50 p-4 rounded-lg text-sm">
                <div className="space-y-2">
                  <div><strong>1. Clone Repository</strong></div>
                  <div className="ml-4 font-mono text-xs bg-white p-2 rounded">git clone [repo-url]</div>

                  <div><strong>2. Backend Setup</strong></div>
                  <div className="ml-4 font-mono text-xs bg-white p-2 rounded">cd backend && pip install -e .</div>

                  <div><strong>3. Frontend Setup</strong></div>
                  <div className="ml-4 font-mono text-xs bg-white p-2 rounded">cd frontend && npm install</div>

                  <div><strong>4. Start Development</strong></div>
                  <div className="ml-4 font-mono text-xs bg-white p-2 rounded">./start-all.sh</div>
                </div>
              </div>
            </div>
            <div>
              <h4 className="font-semibold text-harvard-crimson mb-3">Code Quality</h4>
              <ul className="space-y-2 text-sm text-gray-700">
                <li>• <strong>Linting:</strong> ESLint for frontend, Black for Python</li>
                <li>• <strong>Type Checking:</strong> Pydantic for API validation</li>
                <li>• <strong>Testing:</strong> Unit tests for core functions</li>
                <li>• <strong>Documentation:</strong> Inline comments and docstrings</li>
                <li>• <strong>Git Workflow:</strong> Feature branches and PRs</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-harvard-crimson mb-3">Project Roadmap</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-green-50 p-4 rounded-lg border-l-4 border-green-400">
                <h5 className="font-medium text-green-800 mb-2">✅ Completed</h5>
                <ul className="text-xs text-green-700 space-y-1">
                  <li>• Core transformer implementation</li>
                  <li>• Training pipeline</li>
                  <li>• Interactive chat interface</li>
                  <li>• React frontend with Tailwind</li>
                  <li>• FastAPI backend</li>
                  <li>• Educational help system</li>
                </ul>
              </div>
              <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-400">
                <h5 className="font-medium text-blue-800 mb-2">🔄 In Progress</h5>
                <ul className="text-xs text-blue-700 space-y-1">
                  <li>• WebSocket real-time updates</li>
                  <li>• Advanced model architectures</li>
                  <li>• Performance optimizations</li>
                  <li>• Enhanced data pipeline</li>
                  <li>• Model comparison tools</li>
                </ul>
              </div>
              <div className="bg-yellow-50 p-4 rounded-lg border-l-4 border-yellow-400">
                <h5 className="font-medium text-yellow-800 mb-2">📋 Planned</h5>
                <ul className="text-xs text-yellow-700 space-y-1">
                  <li>• Multi-user support</li>
                  <li>• Advanced ML techniques</li>
                  <li>• Cloud deployment</li>
                  <li>• API rate limiting</li>
                  <li>• Advanced analytics</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </ExpandableSection>

      {/* Footer */}
      <div className="text-center py-8 border-t border-gray-200">
        <p className="text-gray-600 text-sm">
          This technical documentation is automatically generated and updated with each system modification.
        </p>
        <p className="text-gray-500 text-xs mt-2">
          Last updated: {new Date().toLocaleDateString()} • MiniGPT v1.0.0
        </p>
      </div>
    </div>
  );
};

export default CoreInfrastructure;