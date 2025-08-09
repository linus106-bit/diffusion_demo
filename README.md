# AI Chat Demo - OpenAI GPT Powered

A ChatGPT-like web interface built with FastAPI that connects directly to OpenAI's GPT API.

## Features

- 🎨 Modern ChatGPT-like UI design
- 💬 Real-time chat interface with FastAPI backend
- 🔄 Model selection dropdown (GPT-3.5, GPT-4, etc.)
- 📱 Responsive design for mobile and desktop
- ⚡ Fast async responses with OpenAI GPT API
- 🔌 Direct OpenAI API integration
- 📚 Automatic API documentation with FastAPI

## Prerequisites

1. **OpenAI API Key**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Python 3.8+**: Make sure you have Python installed

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

You need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Environment Configuration (Optional)

Create a `.env` file in the project root (optional):

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Or set environment variables directly:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Application

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

The application will be available at `http://localhost:8080`

### 5. API Documentation

FastAPI automatically generates interactive API documentation:
- **Swagger UI**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`

## Configuration

### Default Settings

- **OpenAI API**: Uses official OpenAI API endpoints
- **FastAPI Port**: `8080`
- **Default Model**: `gpt-3.5-turbo`
- **Max Tokens**: `1000`
- **Temperature**: `0.7`

### Customization

You can modify the following in `app.py`:

- Model parameters (temperature, max_tokens, etc.)
- Available models in the dropdown
- UI styling and behavior

## API Endpoints

- `GET /`: Main chat interface
- `POST /chat`: Send messages to OpenAI GPT (with Pydantic validation)
- `POST /chat/stream`: Streaming chat endpoint for real-time responses
- `GET /models`: Get available models from OpenAI API
- `GET /docs`: Interactive API documentation (Swagger UI)
- `GET /redoc`: Alternative API documentation (ReDoc)

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your OpenAI API key is set correctly
2. **Model Not Found**: Verify the model name is available in your OpenAI account
3. **Rate Limits**: OpenAI has rate limits; upgrade your plan if needed
4. **Port Conflicts**: Change the FastAPI port if 8080 is already in use

### Debugging

Enable FastAPI development mode with auto-reload:
```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload --log-level debug
```

## File Structure

```
diffusion_demo/
├── app.py                 # FastAPI backend server
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── templates/
    └── index.html        # Chat interface template
```

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests!
