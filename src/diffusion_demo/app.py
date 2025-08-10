"""
Main FastAPI application for DiffuChatGPT
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import config
from .utils.logger import app_logger
from .models.base import ModelManager
from .models.openai_model import OpenAIModel
from .models.local_model import AutoregressiveModel
from .models.llada_model import LLaDAModel
from .api.routes import router
from . import __version__


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Validate configuration
    config.validate()
    
    # Create FastAPI app
    app = FastAPI(
        title="DiffuChatGPT",
        description="Multi-model chat interface with OpenAI, Autoregressive Models, and LLaDA",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="src/diffusion_demo/static"), name="static")
    
    # Include API routes
    app.include_router(router)
    
    return app


def initialize_models():
    """Initialize all model instances"""
    app_logger.info("🚀 Initializing DiffuChatGPT models...")
    
    model_manager = ModelManager()
    
    # Initialize OpenAI model
    try:
        openai_model = OpenAIModel()
        model_manager.register_model("openai", openai_model)
        app_logger.info("✅ OpenAI model initialized")
    except Exception as e:
        app_logger.error(f"❌ Failed to initialize OpenAI model: {e}")
    
    # Initialize Autoregressive model
    try:
        autoregressive_model = AutoregressiveModel()
        model_manager.register_model("autoregressive", autoregressive_model)
        app_logger.info("✅ Autoregressive model initialized")
    except Exception as e:
        app_logger.error(f"❌ Failed to initialize Autoregressive model: {e}")
    
    # Initialize LLaDA model (temporarily disabled for testing)
    # try:
    #     llada_model = LLaDAModel()
    #     model_manager.register_model("llada", llada_model)
    #     app_logger.info("✅ LLaDA model initialized")
    # except Exception as e:
    #     app_logger.error(f"❌ Failed to initialize LLaDA model: {e}")
    
    app_logger.info("🎯 Model initialization completed")
    
    return model_manager


def main():
    """Main application entry point"""
    app_logger.info(f"🚀 Starting DiffuChatGPT v{__version__}")
    
    # Create app
    app = create_app()
    
    # Initialize models and store in app state
    model_manager = initialize_models()
    app.state.model_manager = model_manager
    
    # Start server
    if config.server.reload:
        # Use import string for reload mode
        uvicorn.run(
            "diffusion_demo.app:app",
            host=config.server.host,
            port=config.server.port,
            reload=True,
            log_level=config.logging.log_level.lower()
        )
    else:
        # Use direct app object for non-reload mode
        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.port,
            reload=False,
            log_level=config.logging.log_level.lower()
        )


# Create global app instance for uvicorn import
app = create_app()

# Initialize models for the global app instance
try:
    model_manager = initialize_models()
    app.state.model_manager = model_manager
except Exception as e:
    app_logger.error(f"❌ Failed to initialize models for global app: {e}")

if __name__ == "__main__":
    main()
