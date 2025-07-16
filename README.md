# TalentMatch PoC

AI-powered talent matching system using LangGraph for intelligent resume processing and job matching.

## Features

- Resume ingestion and parsing
- Skill extraction and classification
- Intelligent job matching
- Q&A-based candidate evaluation
- FastAPI backend with Streamlit UI

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
│   └── endpoints.py      # FastAPI endpoints
├── graph/
│   └── workflow.py      # LangGraph workflow definition
├── agents/
│   ├── ingest.py        # Resume ingestion agent
│   ├── extract.py       # Information extraction agent
│   ├── classify.py      # Skills classification agent
│   ├── match.py         # Job matching agent
│   └── qa.py            # Q&A evaluation agent
├── models/
│   └── __init__.py      # Data models and state definitions
├── main.py              # FastAPI application
├── streamlit_app.py     # Streamlit UI
└── requirements.txt     # Project dependencies
```

## Workflow

1. **Resume Ingestion**: Processes uploaded resume documents
2. **Information Extraction**: Extracts structured data from resumes
3. **Skills Classification**: Categorizes and standardizes skills
4. **Job Matching**: Matches candidates with job requirements
5. **Q&A Evaluation**: Performs detailed candidate assessment

## API Endpoints

- `POST /upload-resume`: Upload and process a resume
- `POST /match-jobs`: Match resume against job catalog
- `GET /candidates/{id}`: Get candidate details
- `GET /matches/{id}`: Get match results

## Development

- Python 3.11+
- Uses LangGraph for workflow orchestration
- FastAPI for backend API
- Streamlit for user interface
- OpenAI for language processing

## License

[Add License Information]
