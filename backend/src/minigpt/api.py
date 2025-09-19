"""
FastAPI web service for MiniGPT model inference
Provides REST API endpoints for text generation and model management
"""

import os
import json
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import torch

from .model import MiniGPT
from .tokenizer import get_tokenizer
from .chat import ChatBot
from .utils import get_device, load_checkpoint, get_checkpoints_dir, find_best_checkpoint
from .evaluate import ModelEvaluator
from .pretrained import PretrainedModelImporter, list_available_models
from .remote import RemoteModelManager, list_remote_models


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_length: int = Field(50, ge=1, le=200, description="Maximum tokens to generate")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")


class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    full_text: str
    generation_time: float
    tokens_generated: int


class ChatMessage(BaseModel):
    message: str
    model: str = Field("local", description="Model to use: 'local' or remote model key like 'hf:gpt2'")
    max_length: int = Field(50, ge=1, le=200)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=1, le=100)


class ChatResponse(BaseModel):
    response: str
    generation_time: float


class ModelInfo(BaseModel):
    model_name: str
    parameters: int
    vocab_size: int
    context_length: int
    device: str
    status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    memory_usage: Optional[Dict[str, float]] = None


class TrainingProgressResponse(BaseModel):
    status: str
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    train_loss: float
    val_loss: float
    learning_rate: float
    estimated_time_remaining: float
    model_name: str
    progress_percentage: float


# Global instances
model_manager = None
pretrained_importer = None
remote_manager = None


class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.chatbot = None
        self.device = get_device()
        self.model_info = {}

    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Load model from checkpoint"""
        try:
            logger.info(f"Loading model from {model_path}")

            # Load checkpoint
            checkpoint = load_checkpoint(model_path, self.device)
            config = checkpoint.get('config', {})
            model_config = config.get('model', {})

            # Initialize model
            self.model = MiniGPT(**model_config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Initialize tokenizer
            self.tokenizer = get_tokenizer("gpt2")

            # Initialize chatbot
            self.chatbot = ChatBot(model_path)

            # Store model info
            self.model_info = {
                'model_path': model_path,
                'parameters': self.model.count_parameters(),
                'vocab_size': self.model.vocab_size,
                'context_length': self.model.block_size,
                'device': str(self.device),
                'config': config
            }

            logger.info(f"Model loaded successfully: {self.model_info['parameters']:,} parameters")
            return self.model_info

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    def is_ready(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None and self.tokenizer is not None

    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 0.8, top_k: int = 50) -> Dict[str, Any]:
        """Generate text using the loaded model"""
        if not self.is_ready():
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            import time
            start_time = time.time()

            # Generate text
            full_text = self.chatbot.generate_text(prompt, max_length, temperature, top_k)
            generated_text = full_text[len(prompt):].strip()

            end_time = time.time()
            generation_time = end_time - start_time

            # Count tokens (approximate)
            tokens_generated = len(generated_text.split())

            return {
                'prompt': prompt,
                'generated_text': generated_text,
                'full_text': full_text,
                'generation_time': generation_time,
                'tokens_generated': tokens_generated
            }

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="MiniGPT API",
    description="REST API for MiniGPT text generation model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager and pre-trained importer
@app.on_event("startup")
async def startup_event():
    global model_manager, pretrained_importer, remote_manager
    model_manager = ModelManager()
    pretrained_importer = PretrainedModelImporter()
    remote_manager = RemoteModelManager()

    # Try to load the best model automatically
    try:
        best_checkpoint = find_best_checkpoint()
        if best_checkpoint:
            model_manager.load_model(str(best_checkpoint))
            logger.info(f"Auto-loaded model: {best_checkpoint}")
        else:
            logger.info("No trained models found. Train a model first with 'python autoTest.py'")
    except Exception as e:
        logger.warning(f"Failed to auto-load model: {str(e)}")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    memory_usage = None

    if torch.cuda.is_available():
        memory_usage = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "cached_mb": torch.cuda.memory_reserved() / 1024 / 1024
        }

    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_ready() if model_manager else False,
        device=str(model_manager.device) if model_manager else "unknown",
        memory_usage=memory_usage
    )


# Model management endpoints
@app.post("/model/load")
async def load_model(model_name: str):
    """Load a specific model checkpoint"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    checkpoints_dir = get_checkpoints_dir()
    model_path = checkpoints_dir / model_name

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    info = model_manager.load_model(str(model_path))
    return {"message": f"Model {model_name} loaded successfully", "info": info}


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the currently loaded model"""
    if not model_manager or not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="No model loaded")

    info = model_manager.model_info
    return ModelInfo(
        model_name=Path(info['model_path']).name,
        parameters=info['parameters'],
        vocab_size=info['vocab_size'],
        context_length=info['context_length'],
        device=info['device'],
        status="ready"
    )


@app.get("/model/list")
async def list_models():
    """List available model checkpoints and remote models"""
    # Local models
    checkpoints_dir = get_checkpoints_dir()
    local_models = []
    for checkpoint in checkpoints_dir.glob("*.pt"):
        local_models.append({
            "key": "local",
            "name": checkpoint.name,
            "path": str(checkpoint),
            "size_mb": checkpoint.stat().st_size / (1024 * 1024),
            "modified": checkpoint.stat().st_mtime,
            "is_best": checkpoint.name == "best_model.pt",
            "type": "local"
        })

    # Remote models
    remote_models = []
    if remote_manager:
        remote_model_info = list_remote_models()
        for key, info in remote_model_info.items():
            remote_models.append({
                "key": key,
                "name": info.get("name", key),
                "description": info.get("description", ""),
                "provider": info.get("provider", "unknown"),
                "cost": info.get("cost", "unknown"),
                "type": "remote"
            })

    return {
        "local_models": sorted(local_models, key=lambda x: x["modified"], reverse=True),
        "remote_models": remote_models
    }


# Text generation endpoints
@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text from a prompt"""
    if not model_manager or not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="No model loaded")

    result = model_manager.generate_text(
        request.prompt,
        request.max_length,
        request.temperature,
        request.top_k
    )

    return GenerationResponse(**result)


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Single-turn chat with the model"""
    logger.info(f"🔵 Chat request received: '{message.message}' (model={message.model}, temp={message.temperature}, max_len={message.max_length})")

    try:
        import time
        start_time = time.time()

        # Check if using remote model
        if message.model != "local":
            logger.info(f"🌐 Using remote model: {message.model}")
            if not remote_manager:
                raise HTTPException(status_code=503, detail="Remote model system not initialized")

            response_text = remote_manager.generate_text(
                message.model,
                message.message,
                max_tokens=message.max_length,
                temperature=message.temperature
            )
        else:
            # Use local model
            logger.info("🤖 Using local model")
            if not model_manager or not model_manager.is_ready():
                logger.error("❌ No local model loaded or model not ready")
                raise HTTPException(status_code=503, detail="No local model loaded. Please train a model first or use a remote model.")

            logger.info(f"🤖 Generating response...")
            result = model_manager.generate_text(
                message.message,
                message.max_length,
                message.temperature,
                message.top_k
            )
            response_text = result["generated_text"]

        generation_time = time.time() - start_time

        logger.info(f"✅ Response generated in {generation_time:.2f}s: '{response_text[:50]}{'...' if len(response_text) > 50 else ''}'")

        return ChatResponse(
            response=response_text,
            generation_time=generation_time
        )

    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.exception("Full error traceback:")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/generate/stream")
async def stream_generate(
    prompt: str,
    max_length: int = 50,
    temperature: float = 0.8,
    top_k: int = 50
):
    """Stream text generation (simplified version)"""
    if not model_manager or not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="No model loaded")

    async def generate_stream():
        # This is a simplified streaming implementation
        # In a real implementation, you'd modify the model to yield tokens
        result = model_manager.generate_text(prompt, max_length, temperature, top_k)

        # Simulate streaming by yielding words
        words = result["generated_text"].split()
        for word in words:
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.1)  # Simulate generation delay

        yield f"data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/plain")


# Utility endpoints
@app.get("/training/progress", response_model=TrainingProgressResponse)
async def get_training_progress():
    """Get current training progress"""
    progress_file = "training_progress.json"

    if not Path(progress_file).exists():
        raise HTTPException(status_code=404, detail="No training in progress")

    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)

        # Calculate progress percentage
        if progress_data["total_steps"] > 0:
            progress_percentage = (progress_data["current_step"] / progress_data["total_steps"]) * 100
        else:
            progress_percentage = 0.0

        return TrainingProgressResponse(
            **progress_data,
            progress_percentage=progress_percentage
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read training progress: {str(e)}")


# Pre-trained model endpoints
@app.get("/pretrained/available")
async def list_available_pretrained_models():
    """List all available pre-trained models for import"""
    return {"models": list_available_models()}


@app.get("/pretrained/imported")
async def list_imported_pretrained_models():
    """List all imported pre-trained models"""
    if not pretrained_importer:
        raise HTTPException(status_code=503, detail="Pre-trained model system not initialized")

    return {"models": pretrained_importer.list_imported_models()}


@app.post("/pretrained/import/{model_key}")
async def import_pretrained_model_endpoint(model_key: str, background_tasks: BackgroundTasks):
    """Import a pre-trained model"""
    if not pretrained_importer:
        raise HTTPException(status_code=503, detail="Pre-trained model system not initialized")

    if model_key not in list_available_models():
        available = list(list_available_models().keys())
        raise HTTPException(status_code=404, detail=f"Model {model_key} not available. Available: {available}")

    def import_model():
        try:
            path = pretrained_importer.import_model(model_key)
            logger.info(f"Successfully imported {model_key} to {path}")
        except Exception as e:
            logger.error(f"Failed to import {model_key}: {e}")

    background_tasks.add_task(import_model)
    return {"message": f"Started importing {model_key}", "model_key": model_key}


@app.delete("/pretrained/remove/{model_key}")
async def remove_pretrained_model(model_key: str):
    """Remove an imported pre-trained model"""
    if not pretrained_importer:
        raise HTTPException(status_code=503, detail="Pre-trained model system not initialized")

    success = pretrained_importer.remove_model(model_key)
    if success:
        return {"message": f"Model {model_key} removed successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")


# Remote API settings endpoints
@app.post("/remote/set-token")
async def set_api_token(provider: str, token: str):
    """Set API token for remote provider"""
    import os

    token_env_vars = {
        "huggingface": "HUGGINGFACE_API_TOKEN",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }

    if provider not in token_env_vars:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    env_var = token_env_vars[provider]
    os.environ[env_var] = token

    # Clear cached clients to pick up new token
    if remote_manager:
        remote_manager.clients.clear()

    return {"message": f"Token set for {provider}", "provider": provider}


@app.get("/remote/auth-status")
async def get_auth_status():
    """Check authentication status for remote providers"""
    import os

    status = {
        "huggingface": {
            "authenticated": bool(os.getenv("HUGGINGFACE_API_TOKEN")),
            "required": False,
            "description": "Optional - improves rate limits and access to some models"
        },
        "openai": {
            "authenticated": bool(os.getenv("OPENAI_API_KEY")),
            "required": True,
            "description": "Required for OpenAI models (GPT-3.5, GPT-4)"
        },
        "anthropic": {
            "authenticated": bool(os.getenv("ANTHROPIC_API_KEY")),
            "required": True,
            "description": "Required for Claude models"
        }
    }

    return status


@app.get("/pretrained/info/{model_key}")
async def get_pretrained_model_info(model_key: str):
    """Get information about a pre-trained model"""
    if not pretrained_importer:
        raise HTTPException(status_code=503, detail="Pre-trained model system not initialized")

    try:
        return pretrained_importer.get_model_info(model_key)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "MiniGPT API",
        "version": "1.0.0",
        "description": "REST API for MiniGPT text generation",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "chat": "/chat",
            "model_info": "/model/info",
            "model_list": "/model/list",
            "training_progress": "/training/progress",
            "pretrained_available": "/pretrained/available",
            "pretrained_imported": "/pretrained/imported",
            "remote_auth_status": "/remote/auth-status",
            "remote_set_token": "/remote/set-token",
            "docs": "/docs"
        }
    }


def create_app():
    """Factory function to create the FastAPI app"""
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Run the FastAPI server"""
    uvicorn.run(
        "minigpt.api:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        access_log=True
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniGPT API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--model", type=str, help="Model checkpoint to load on startup")

    args = parser.parse_args()

    if args.model:
        # Pre-load model
        manager = ModelManager()
        manager.load_model(args.model)
        model_manager = manager

    run_server(args.host, args.port, args.workers)