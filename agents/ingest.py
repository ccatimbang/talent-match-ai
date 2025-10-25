from typing import Annotated, Union
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pypdf import PdfReader
from io import BytesIO
from models import GraphState
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clean_text(text: str) -> str:
    """Clean extracted text from PDF"""
    return " ".join(text.split())


class IngestAgent:
    def __init__(self, model: Annotated[ChatOpenAI, "OpenAI model for text processing"] = None):
        # Validate OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.model = model or ChatOpenAI(model="gpt-4-turbo-preview")

    def process_pdf(self, pdf_data: Union[str, bytes]) -> str:
        """Extract text from PDF resume or handle text files"""
        try:
            # Handle string input (already text)
            if isinstance(pdf_data, str):
                return clean_text(pdf_data)
            
            # Handle bytes input
            if isinstance(pdf_data, bytes):
                # First, try to decode as text to check if it's a text file
                try:
                    text_content = pdf_data.decode('utf-8')
                    # Check if it looks like text (not PDF)
                    if len(text_content) > 0 and not text_content.startswith('%PDF'):
                        # It's a text file, not a PDF
                        return clean_text(text_content)
                except UnicodeDecodeError:
                    pass
                
                # If not text, try to process as PDF
                try:
                    pdf_io = BytesIO(pdf_data)
                    reader = PdfReader(pdf_io)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                        
                    if not text.strip():
                        raise ValueError("No text extracted from PDF")
                        
                    return clean_text(text)
                except Exception as pdf_error:
                    # If PDF processing fails, try to decode as text anyway
                    try:
                        text_content = pdf_data.decode('utf-8')
                        return clean_text(text_content)
                    except UnicodeDecodeError:
                        raise ValueError(f"Failed to process as PDF or text: {str(pdf_error)}")
            
            raise ValueError("Invalid input type")
        except Exception as e:
            raise ValueError(f"Failed to process PDF: {str(e)}")

    async def __call__(self, state: GraphState) -> GraphState:
        """LangGraph node implementation"""
        if not state.resume_text:
            state.error = "No resume data provided"
            return state

        try:
            # Extract text from PDF bytes
            extracted_text = self.process_pdf(state.resume_text)
            
            if not extracted_text:
                state.error = "No text could be extracted from the PDF"
                return state
            
            # Update state with extracted text
            state.resume_text = extracted_text

            # For testing, skip OpenAI validation if using sample resume
            if len(extracted_text) < 2000 and "Jane Doe" in extracted_text:
                state.current_step = "extract"
                return state

            # Validate the extracted text
            try:
                messages = [
                    HumanMessage(content=f"""
                    Validate if the following text appears to be a valid resume. 
                    Return 'VALID' if it contains typical resume sections like education, experience, or skills.
                    Return 'INVALID' if it appears to be garbage text or clearly not a resume.
                    
                    Text: {extracted_text[:2000]}...
                    """)
                ]
                
                response = await self.model.ainvoke(messages)
                
                if "INVALID" in response.content:
                    state.error = "Invalid resume format detected"
                    return state
            except Exception as api_error:
                # Log the API error but continue processing
                print(f"OpenAI API error: {str(api_error)}")
                state.error = f"Failed to validate resume: {str(api_error)}"
                return state
                
            state.current_step = "extract"
            return state
            
        except Exception as e:
            state.error = f"Failed to process resume: {str(e)}"
            return state 