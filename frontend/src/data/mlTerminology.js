// ML Terminology Database for Educational Tooltips
export const mlTerminology = {
  epochs: {
    title: "Epochs",
    definition: "One complete pass through the entire training dataset.",
    detailed: "During each epoch, the model sees every example in your training data exactly once. More epochs generally lead to better learning, but too many can cause overfitting.",
    examples: [
      "5 epochs = model sees all data 5 times",
      "Typical range: 10-100 epochs",
      "Watch for loss plateauing to know when to stop"
    ],
    tips: [
      "Start with fewer epochs (10-20) to test quickly",
      "Monitor validation loss to prevent overfitting",
      "Use early stopping to automatically halt training"
    ],
    relatedTerms: ["batch_size", "learning_rate", "overfitting"]
  },

  batch_size: {
    title: "Batch Size",
    definition: "Number of training examples processed together before updating model weights.",
    detailed: "Instead of updating the model after each example, we group examples into batches. Larger batches provide more stable gradients but require more memory.",
    examples: [
      "Batch size 32 = process 32 examples, then update",
      "Common sizes: 16, 32, 64, 128",
      "Larger batches = more memory needed"
    ],
    tips: [
      "Start with 32 - good balance of speed and stability",
      "Reduce if you get out-of-memory errors",
      "Power of 2 sizes (16, 32, 64) work well with GPUs"
    ],
    relatedTerms: ["epochs", "learning_rate", "gradient_descent"]
  },

  learning_rate: {
    title: "Learning Rate",
    definition: "Controls how big steps the model takes when learning from mistakes.",
    detailed: "Like the speed of learning. Too high and the model might overshoot the optimal solution. Too low and it learns very slowly or gets stuck.",
    examples: [
      "0.001 = small, careful steps (common default)",
      "0.01 = bigger steps, faster learning",
      "0.0001 = very conservative, slow learning"
    ],
    tips: [
      "Start with 0.001 - works for most cases",
      "If loss jumps around wildly, reduce learning rate",
      "If learning is too slow, try increasing slightly"
    ],
    relatedTerms: ["optimizer", "gradient_descent", "loss_function"]
  },

  optimizer: {
    title: "Optimizer",
    definition: "The algorithm that adjusts model weights based on the calculated errors.",
    detailed: "Different optimizers have different strategies for updating the model. Each has advantages for different types of problems.",
    examples: [
      "Adam: Adaptive, works well for most problems",
      "SGD: Simple, reliable, good for fine-tuning",
      "AdamW: Adam with better weight decay"
    ],
    tips: [
      "Adam is a safe default choice",
      "SGD can work better for some vision tasks",
      "Try AdamW if you have overfitting issues"
    ],
    relatedTerms: ["learning_rate", "gradient_descent", "momentum"]
  },

  temperature: {
    title: "Temperature",
    definition: "Controls randomness in text generation. Higher = more creative, lower = more focused.",
    detailed: "Temperature affects how the model chooses its next word. It's like controlling creativity vs. consistency in the model's responses.",
    examples: [
      "0.1 = Very focused, predictable responses",
      "0.7 = Balanced creativity and coherence",
      "1.5 = Very creative, potentially chaotic"
    ],
    tips: [
      "0.7 is good for most conversations",
      "Use 0.1-0.3 for factual, precise answers",
      "Use 1.0+ for creative writing or brainstorming"
    ],
    relatedTerms: ["top_k", "sampling", "generation"]
  },

  max_tokens: {
    title: "Max Tokens",
    definition: "Maximum number of words/pieces the model will generate in its response.",
    detailed: "Tokens are pieces of words (usually words or parts of words). This limits how long the response can be.",
    examples: [
      "50 tokens ≈ 1-2 sentences",
      "150 tokens ≈ 1 paragraph",
      "500 tokens ≈ several paragraphs"
    ],
    tips: [
      "Start with 100-150 for chat responses",
      "Increase for longer, detailed answers",
      "Lower values generate responses faster"
    ],
    relatedTerms: ["temperature", "context_length", "tokenization"]
  },

  loss_function: {
    title: "Loss Function",
    definition: "Measures how wrong the model's predictions are. Training tries to minimize this.",
    detailed: "The loss function calculates the difference between what the model predicted and the correct answer. Lower loss = better performance.",
    examples: [
      "Cross-entropy loss for text prediction",
      "Mean squared error for regression",
      "Loss starts high, should decrease over time"
    ],
    tips: [
      "Watch loss decrease during training",
      "If loss stops improving, training might be done",
      "Sudden loss spikes might indicate learning rate too high"
    ],
    relatedTerms: ["gradient_descent", "backpropagation", "optimization"]
  },

  overfitting: {
    title: "Overfitting",
    definition: "When the model memorizes training data instead of learning general patterns.",
    detailed: "Like studying for a test by memorizing specific questions instead of understanding concepts. The model does great on training data but poorly on new data.",
    examples: [
      "Training accuracy: 99%, Test accuracy: 60% = overfitting",
      "Model gives perfect answers to training examples",
      "But struggles with slightly different questions"
    ],
    tips: [
      "Use validation data to detect overfitting",
      "Stop training when validation loss starts increasing",
      "Add regularization or reduce model complexity"
    ],
    relatedTerms: ["validation", "regularization", "generalization"]
  },

  gradient_descent: {
    title: "Gradient Descent",
    definition: "The fundamental algorithm that teaches the model by showing it its mistakes.",
    detailed: "Like rolling a ball downhill to find the bottom of a valley. The algorithm calculates which direction reduces error the most and moves the model parameters in that direction.",
    examples: [
      "Calculate error on current prediction",
      "Determine which weights caused the error",
      "Adjust weights to reduce future errors"
    ],
    tips: [
      "This happens automatically during training",
      "Learning rate controls the step size",
      "Batch size affects how smooth the descent is"
    ],
    relatedTerms: ["learning_rate", "optimizer", "backpropagation"]
  },

  checkpoints: {
    title: "Checkpoints",
    definition: "Saved snapshots of the model during training, like save points in a video game.",
    detailed: "Regular saves of the model's state so you can resume training if it's interrupted, or go back to a better version if training goes wrong.",
    examples: [
      "Save every 100 steps or every epoch",
      "Keep best performing checkpoint",
      "Resume training from last checkpoint"
    ],
    tips: [
      "Always enable checkpoints for long training",
      "Save to different files to avoid corruption",
      "Keep the best checkpoint even if training continues"
    ],
    relatedTerms: ["model_saving", "training_resume", "best_model"]
  },

  gpu_training: {
    title: "GPU Training",
    definition: "Using graphics cards to speed up model training by 10-100x compared to CPU.",
    detailed: "GPUs have thousands of small cores perfect for the parallel calculations needed in neural networks. Much faster than CPU for training.",
    examples: [
      "CPU training: hours to days",
      "GPU training: minutes to hours",
      "NVIDIA GPUs work best with PyTorch"
    ],
    tips: [
      "Enable if you have a compatible GPU",
      "Monitor GPU memory usage",
      "Reduce batch size if you get out-of-memory errors"
    ],
    relatedTerms: ["batch_size", "memory_usage", "cuda"]
  }
};

// Helper function to get related terms
export const getRelatedTerms = (termKey) => {
  const term = mlTerminology[termKey];
  if (!term?.relatedTerms) return [];

  return term.relatedTerms.map(relatedKey => ({
    key: relatedKey,
    title: mlTerminology[relatedKey]?.title || relatedKey
  }));
};

// Helper function to search terms
export const searchTerms = (query) => {
  const lowercaseQuery = query.toLowerCase();
  return Object.entries(mlTerminology).filter(([key, term]) =>
    key.toLowerCase().includes(lowercaseQuery) ||
    term.title.toLowerCase().includes(lowercaseQuery) ||
    term.definition.toLowerCase().includes(lowercaseQuery)
  );
};