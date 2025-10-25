import json
import re
from typing import Any, Dict, Optional


def parse_llm_json_response(content: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse JSON response from LLM, handling markdown code blocks and other formatting issues.
    
    Args:
        content: Raw response content from LLM
        fallback: Fallback data to return if parsing fails
        
    Returns:
        Parsed JSON data as dictionary
        
    Raises:
        ValueError: If JSON parsing fails and no fallback is provided
    """
    if not content or not content.strip():
        if fallback is not None:
            return fallback
        raise ValueError("Empty response content")
    
    # Clean the content
    cleaned_content = content.strip()
    
    # Remove markdown code block wrappers
    if cleaned_content.startswith("```json"):
        # Extract content between ```json and ```
        match = re.search(r'```json\s*(.*?)\s*```', cleaned_content, re.DOTALL)
        if match:
            cleaned_content = match.group(1).strip()
    elif cleaned_content.startswith("```"):
        # Extract content between ``` and ```
        match = re.search(r'```\s*(.*?)\s*```', cleaned_content, re.DOTALL)
        if match:
            cleaned_content = match.group(1).strip()
    
    # Remove any remaining markdown artifacts
    cleaned_content = re.sub(r'^```.*$', '', cleaned_content, flags=re.MULTILINE)
    cleaned_content = cleaned_content.strip()
    
    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Cleaned content: {cleaned_content[:200]}...")
        
        if fallback is not None:
            print("Using fallback data")
            return fallback
        
        raise ValueError(f"Failed to parse JSON response: {e}")


def safe_parse_llm_json(content: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Safely parse LLM JSON response with comprehensive error handling.
    
    Args:
        content: Raw response content from LLM
        fallback: Fallback data to return if parsing fails
        
    Returns:
        Parsed JSON data as dictionary, or fallback if parsing fails
    """
    try:
        return parse_llm_json_response(content, fallback)
    except Exception as e:
        print(f"Failed to parse LLM response: {e}")
        if fallback is not None:
            return fallback
        # Return a minimal valid response
        return {"error": f"Failed to parse response: {str(e)}"}
