import streamlit as st
import requests
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="TalentMatch",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 TalentMatch")
st.write("Upload a resume to find matching job opportunities")

# File upload
uploaded_file = st.file_uploader("Choose a resume PDF", type="pdf")

if uploaded_file:
    # Show spinner during processing
    with st.spinner("Analyzing resume..."):
        try:
            # Prepare file for upload
            files = {"resume": ("resume.pdf", uploaded_file, "application/pdf")}
            
            # Call API
            response = requests.post(
                "http://localhost:8000/api/v1/match/resume",
                files=files
            )
            response.raise_for_status()
            matches = response.json()
            
            # Display results
            st.subheader("🎉 Matching Results")
            
            for match in matches:
                with st.expander(f"📋 {match['matched_job']['title']} (Score: {match['confidence_score']:.2f})"):
                    # Job details
                    st.write("### Job Details")
                    st.write(f"**Description:** {match['matched_job']['description']}")
                    st.write("**Required Skills:**")
                    for skill in match['matched_job']['required_skills']:
                        st.write(f"- {skill['name']} ({skill['level']})")
                    
                    if match['matched_job'].get('preferred_skills'):
                        st.write("**Preferred Skills:**")
                        for skill in match['matched_job']['preferred_skills']:
                            st.write(f"- {skill['name']} ({skill['level']})")
                    
                    # Match analysis
                    st.write("### Match Analysis")
                    st.write(match['reasoning'])
                    
                    # Status
                    status_color = {
                        "auto_matched": "🟢",
                        "recruiter_review": "🟡",
                        "rejected": "🔴"
                    }
                    st.write(f"**Status:** {status_color.get(match['status'], '⚪')} {match['status']}")
                    
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {str(e)}")
            
# Add sidebar with info
with st.sidebar:
    st.header("About")
    st.write("""
    TalentMatch uses AI to analyze resumes and find the best matching jobs based on:
    - Skills and experience
    - Career trajectory
    - Role requirements
    
    The system provides confidence scores and detailed reasoning for each match.
    """)
    
    st.header("How it Works")
    st.write("""
    1. Upload your resume (PDF format)
    2. AI extracts key information
    3. Skills are classified and enriched
    4. Vector similarity finds matching jobs
    5. LLM performs detailed match analysis
    6. Results are routed based on confidence
    """) 