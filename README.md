# TalentMatch PoC

AI-powered talent matching system using LangGraph for intelligent resume processing and job matching with confidence scoring and automated status routing.

## Features

- **Resume Processing**: Supports both PDF and text file uploads with intelligent text extraction
- **Smart Information Extraction**: Uses LLM to extract structured candidate data (skills, experience, education)
- **Intelligent Job Matching**: Vector similarity matching with confidence scoring
- **Automated Status Routing**: 
  - 🟢 **Auto Matched** (≥0.9 confidence): Automatic approval for strong matches
  - 🟡 **Recruiter Review** (0.6-0.89 confidence): Human review required for partial matches
  - 🔴 **Rejected** (<0.6 confidence): Automatic rejection for poor matches
- **Robust Error Handling**: JSON parsing with fallback mechanisms for LLM responses
- **FastAPI Backend**: RESTful API with comprehensive error handling
- **Streamlit UI**: User-friendly interface for resume upload and results visualization

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd poc-talent-match
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Running the Application

1. Start the FastAPI backend:
```bash
uvicorn main:app --reload
```

2. Start the Streamlit UI (in a new terminal):
```bash
streamlit run streamlit_app.py
```

## Project Structure

```
poc-talent-match/
├── api/
│   └── endpoints.py      # FastAPI endpoints with robust error handling
├── graph/
│   └── workflow.py      # LangGraph workflow definition
├── agents/
│   ├── ingest.py        # Resume ingestion agent (PDF/text support)
│   ├── extract.py       # Information extraction agent with JSON parsing
│   ├── classify.py      # Skills classification agent (optional LLM)
│   ├── match.py         # Job matching agent with confidence scoring
│   └── qa.py            # Q&A evaluation agent (optional LLM)
├── models/
│   ├── __init__.py      # Data models and state definitions
│   ├── base.py          # Core data models (Skill, CandidateProfile)
│   ├── job.py           # Job posting models
│   ├── matching.py      # Match result models
│   └── state.py         # Graph state management
├── data/
│   ├── job_catalog.json # Job database with varied skill requirements
│   └── resume_sample.txt # Sample resume for testing
├── utils.py             # JSON parsing utilities with fallback handling
├── main.py              # FastAPI application
├── streamlit_app.py     # Streamlit UI
└── requirements.txt     # Project dependencies
```

## Workflow

1. **Resume Ingestion**: Processes uploaded PDF or text files with intelligent text extraction
2. **Information Extraction**: Uses LLM to extract structured candidate data with robust JSON parsing
3. **Skills Classification**: Categorizes and standardizes skills (optional LLM enhancement)
4. **Job Matching**: Vector similarity matching with confidence scoring and status routing
5. **Q&A Evaluation**: Performs detailed candidate assessment (optional LLM review)

## Job Catalog

The system includes a comprehensive job catalog (`data/job_catalog.json`) with diverse roles:
- **ML/AI Roles**: Senior ML Engineer, AI Systems Engineer, Data Scientist
- **Backend Roles**: Principal Backend Engineer, Senior Full Stack Engineer
- **Specialized Roles**: Lead Frontend Engineer, Senior Java Developer, C++ Systems Engineer
- **DevOps Roles**: Lead DevOps Engineer

Each job includes required/preferred skills, experience requirements, and detailed descriptions for realistic matching scenarios.

## API Endpoints

- `POST /api/v1/match/resume`: Upload and process a resume (PDF or text)
  - Returns: List of job matches with confidence scores and status
  - Status codes: Auto Matched, Recruiter Review, or Rejected
- `GET /docs`: Interactive API documentation (Swagger UI)
- `GET /`: Health check endpoint

## Technical Details

### Dependencies
- **Python 3.11+**
- **LangGraph**: Workflow orchestration and state management
- **FastAPI**: High-performance API framework with automatic documentation
- **Streamlit**: Interactive web UI for resume upload and results
- **OpenAI**: GPT-4 and text-embedding-3-small for language processing
- **FAISS**: Vector similarity search for job matching
- **PyPDF**: PDF text extraction
- **Pydantic**: Data validation and serialization

### Key Improvements
- **Robust JSON Parsing**: Handles LLM responses with markdown wrappers and malformed JSON
- **Flexible File Support**: Processes both PDF and text files
- **Confidence Scoring**: Intelligent matching with automated status routing
- **Error Resilience**: Comprehensive error handling with fallback mechanisms
- **Dynamic Job Loading**: Jobs loaded from JSON catalog for easy updates

### Testing
Test the system with the included sample resume:
```bash
curl -X POST -F "resume=@data/resume_sample.txt" http://localhost:8000/api/v1/match/resume
```

## License

[Add License Information]
