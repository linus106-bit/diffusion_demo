# DiffuChatGPT - Multi-Model Chat Interface

A modern, multi-model chat interface supporting OpenAI GPT, Local Transformers models, and LLaDA diffusion models.

## 🚀 Features

- **🌐 OpenAI Integration**: Chat with GPT-3.5, GPT-4, and other OpenAI models
- **🏠 Autoregressive Models**: Run SmolLM2 and other autoregressive Transformers models
- **🧠 LLaDA**: Advanced diffusion-based text generation
- **🎭 Remask Feature**: Interactive text editing with mask tokens
- **📊 Comprehensive Logging**: Detailed request/response logging
- **⚙️ Configurable**: Environment-based configuration
- **🔧 Modular Architecture**: Clean, maintainable code structure

## 📁 Project Structure

```
diffusion_demo/
├── src/
│   └── diffusion_demo/
│       ├── __init__.py              # Package initialization
│       ├── config.py                # Configuration management
│       ├── app.py                   # Main FastAPI application
│       ├── api/                     # API layer
│       │   ├── models.py            # Pydantic schemas
│       │   └── routes.py            # API routes
│       ├── models/                  # Model implementations
│       │   ├── base.py              # Base model interface
│       │   ├── openai_model.py      # OpenAI model
│       │   ├── local_model.py       # Autoregressive model
│       │   └── llada_model.py       # LLaDA model
│       ├── utils/                   # Utilities
│       │   └── logger.py            # Logging utilities
│       ├── templates/               # HTML templates
│       │   ├── base.html            # Base template
│       │   ├── openai_chat.html     # OpenAI chat interface
│       │   ├── autoregressive_chat.html      # Autoregressive model interface
│       │   └── llada_chat.html      # LLaDA interface
│       └── static/                  # Static assets
│           ├── css/                 # Stylesheets
│           └── js/                  # JavaScript
├── main.py                          # Entry point
├── setup.py                         # Package setup
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Quick Start

1. **Clone and install**:
```bash
git clone <repository-url>
cd diffusion_demo
pip install -e .
```

2. **Set environment variables**:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export SERVER_PORT=8080
```

3. **Run the application**:
```bash
python main.py
```

### Environment Configuration

Create a `.env` file or set environment variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_DEFAULT_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.7

# Autoregressive Model Configuration
AUTOREGRESSIVE_MODEL_NAME=HuggingFaceTB/SmolLM2-135M-Instruct
AUTOREGRESSIVE_MAX_TOKENS=1000
AUTOREGRESSIVE_TEMPERATURE=0.7
AUTOREGRESSIVE_DEVICE=auto

# LLaDA Configuration
LLADA_MODEL_NAME=GSAI-ML/LLaDA-8B-Instruct
LLADA_MAX_TOKENS=128
LLADA_TEMPERATURE=0.0
LLADA_STEPS=128
LLADA_BLOCK_LENGTH=32
LLADA_CFG_SCALE=0.0
LLADA_REMASKING=low_confidence

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
SERVER_DEBUG=false
SERVER_RELOAD=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/diffusion_demo.log
```

## 🎮 Usage

### Web Interface

1. **OpenAI Chat**: Visit `http://localhost:8080/`
2. **Autoregressive Models**: Visit `http://localhost:8080/autoregressive`
3. **LLaDA**: Visit `http://localhost:8080/llada`

### API Endpoints

- `GET /` - OpenAI chat interface
- `GET /autoregressive` - Autoregressive model interface
- `GET /llada` - LLaDA interface
- `POST /chat` - OpenAI chat API
- `POST /chat/autoregressive` - Autoregressive model chat API
- `POST /chat/llada` - LLaDA chat API
- `GET /models` - List available OpenAI models
- `GET /health` - Health check
- `GET /docs` - API documentation

### Remask Feature

1. **Select text** in any chat message
2. **Right-click** to open context menu
3. **Choose "🎭 Remask"** option
4. **Set mask count** (number of `<|mask|>` tokens)
5. **Wait for regeneration** with blinking mask tokens

## 🔧 Development

### Project Structure Benefits

#### **1. Modular Architecture**
- **Separation of concerns**: API, models, and UI are separate
- **Easy testing**: Each component can be tested independently
- **Scalable**: Easy to add new models or features

#### **2. Configuration Management**
- **Environment-based**: All settings via environment variables
- **Type-safe**: Dataclass-based configuration
- **Validated**: Configuration validation on startup

#### **3. Model Abstraction**
- **Unified interface**: All models implement `BaseModel`
- **Plugin architecture**: Easy to add new model types
- **Error handling**: Graceful fallbacks and error reporting

#### **4. Logging System**
- **Structured logging**: JSON and human-readable formats
- **Rotating files**: Automatic log rotation
- **Message tracking**: Detailed chat message logging

### Adding New Models

1. **Create model class**:
```python
from .base import BaseModel

class MyModel(BaseModel):
    def load(self) -> bool:
        # Load your model
        pass
    
    def generate_response(self, messages, **kwargs) -> str:
        # Generate response
        pass
```

2. **Register in app**:
```python
my_model = MyModel()
model_manager.register_model("my_model", my_model)
```

3. **Add API route**:
```python
@router.post("/chat/my-model")
async def chat_my_model(chat_request: ChatRequest):
    # Handle requests
    pass
```

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8080/health
```

### Model Information
```bash
curl http://localhost:8080/model-info/openai
curl http://localhost:8080/model-info/autoregressive
curl http://localhost:8080/model-info/llada
```

### Logs
- **Application logs**: `logs/diffusion_demo.log`
- **Chat messages**: `logs/chat_messages_YYYYMMDD.log`

## 🚀 Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -e .

EXPOSE 8080
CMD ["python", "main.py"]
```

### Production Settings

```bash
export SERVER_DEBUG=false
export SERVER_RELOAD=false
export LOG_LEVEL=WARNING
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆚 Migration from Old Version

### Key Changes

1. **Package Structure**: Moved to `src/` layout
2. **Configuration**: Centralized config management
3. **Models**: Abstracted model interface
4. **Logging**: Enhanced logging system
5. **API**: Cleaner route organization

### Migration Steps

1. **Backup** your current setup
2. **Install** the new version
3. **Update** environment variables
4. **Test** all functionality
5. **Deploy** when ready

The refactored version maintains full backward compatibility while providing a much cleaner, more maintainable codebase! 🎉
