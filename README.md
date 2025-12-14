# Confession to Wisdom - Voice to Wisdom Assistant

A Streamlit application that converts voice input (confessions/stories) to text, retrieves relevant Chinese proverbs, and generates explanations with text-to-speech output.

## Features

- 🎙️ **Audio Input**: Record voice through browser microphone
- 🗣️ **Speech-to-Text**: Transcribe audio using Thai Whisper models
- 🔍 **Proverb Retrieval**: Search relevant proverbs from ChromaDB vector store
- 🧠 **LLM Integration**: Generate structured responses (Thai proverb + explanation) using Azure OpenAI
- 🔊 **Text-to-Speech**: Convert explanations to speech using Microsoft Edge TTS

## System Requirements

### Tested Environment
- **OS**: Linux (WSL2 on Windows)
- **GPU**: NVIDIA CUDA (optional, CPU supported)
- **Python**: 3.8+
- **RAM**: 8GB+ recommended
- **Disk Space**: 5GB+ (for model caching)

### Hardware Notes
- GPU is used for Whisper and LLM inference (if available)
- ChromaDB uses CPU to avoid WSL2 double-free crashes
- First run downloads ~3-5GB of models
- CPU-only mode supported but slower

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Azure OpenAI API key (for LLM)
- Microphone access in browser (for audio recording)
- Git (for cloning and pushing to GitHub)

## Installation

### 1. Clone or navigate to project directory
```bash
cd /path/to/fusion/extra
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
```

### 3. Activate virtual environment

**On Linux/macOS:**
```bash
source venv/bin/activate
```

**On Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**On Windows (CMD):**
```bash
venv\Scripts\activate.bat
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the project directory with the following:

```env
GPT5_API_KEY=your_azure_openai_api_key
```

Or set the environment variable directly:
```bash
export GPT5_API_KEY="your_azure_openai_api_key"
```

## Running the Application

```bash
streamlit run confession2wisdom.py
```

The application will:
1. Start a local server (typically http://localhost:8501)
2. Open in your default browser
3. Display the Streamlit UI with audio input controls

## Usage Guide

### Step 1: Load Models
1. Go to the sidebar (⚙️ **Settings**)
2. Select your preferred **Speech-to-Text Model**:
   - Thai Whisper large (recommended)
   - OpenAI Whisper Small
   - OpenAI Whisper Base
3. Click **🔄 โหลดโมเดล** button to load models

### Step 2: Record Audio
1. In the **🎙️ Recording Audio** section, click the microphone icon
2. Speak into your microphone (allow browser access when prompted)
3. Click the recording icon again to stop

### Step 3: Transcribe
1. Click **⚡ ถอดเสียงเป็นข้อความ** to convert audio to text
2. Review and edit the transcribed text in the text area if needed

### Step 4: Retrieve Proverb
1. Click **🔍 ขอสุภาษิต** to get relevant proverb
2. The app will display:
   - Thai proverb (✨)
   - Detailed explanation
3. Click **🔊 อ่านออกมา** to hear the explanation (requires TTS)

## Troubleshooting

### "Missing GPT5_API_KEY in environment variables"
- Ensure your `.env` file exists in the project directory
- Verify the API key is correct
- Try restarting the application

### Audio input not working
- Check browser microphone permissions
- Ensure you're using HTTPS or localhost
- Try a different browser (Chrome/Edge recommended)

### Model loading takes too long
- This is normal for the first load (can take 5-10 minutes for large models)
- Models are cached after first load
- Consider using smaller models (OpenAI Whisper Base) if you have limited resources

### Memory/CUDA issues
- The app defaults to CPU for ChromaDB to avoid WSL2 conflicts
- GPU is used for Whisper and LLM if available
- Reduce batch size or use smaller models if running out of memory

## Project Structure

```
.
├── confession2wisdom.py       # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .env                      # Environment variables (create this)
├── Chinese_proverbs.csv      # Proverb database
├── chroma_db/               # ChromaDB vector store (auto-created)
└── README_STREAMLIT.md      # Additional Streamlit documentation
```

## Models Used

- **Speech-to-Text**: biodatlab/whisper-th-medium-combined (Thai Whisper)
- **Embeddings**: BAAI/bge-m3
- **LLM**: Azure OpenAI (gpt-4.1 or gpt-5-mini)
- **Text-to-Speech**: Microsoft Edge TTS (Neural Voice Thai)

## Dependencies

See `requirements.txt` for complete list. Main packages:
- `streamlit`: Web framework
- `torch`: Deep learning framework
- `transformers`: Hugging Face models
- `langchain`: LLM framework
- `chromadb`: Vector database
- `openai`: Azure OpenAI client
- `edge-tts`: Text-to-speech
- `python-dotenv`: Environment variable management

## Performance Notes

- First load may take 5-15 minutes (model downloading and caching)
- Subsequent loads are faster (models cached)
- Recommended: 8GB+ RAM
- Optional: GPU support (CUDA) for faster processing

## License

This project uses open-source models and libraries. Check individual licenses for details.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review application error messages
3. Check console logs in terminal

---

**Last Updated**: December 2024
