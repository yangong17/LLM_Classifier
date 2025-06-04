# CSVision

A Flask web application for CSV data analysis and processing, developed as my final project for STATS418 - "Tools in Data Science", UCLA MASDS Spring 2025

**Check out a live demo here:**
https://stats418-csvision-228019890474.us-central1.run.app

**Note:** The default API key is linked to a demo account with limited credits (~$3 remaining as of 6/2/2025)

To use your own OpenAI API key:

* Enter it directly in the dashboard's API Key field, or
* Modify the .env file in the project root


## Overview
CSVision is designed to help data scientists and analysts convert unstructured text data (like job descriptions, news articles, or personal reflections) into structured, numerical data suitable for statistical analysis. It provides an end-to-end pipeline for text processing, classification, and analysis.

## Data Processing Pipeline
1. **Input**: CSV file containing unstructured text data
2. **Text Processing**:
   - **Summarization**: Condense long text into key points
   - **Classification**: Convert text into categorical/numerical ratings
3. **Analysis**: Statistical analysis of the processed data

Example Pipeline:
```
Raw Text Data → Summarized Text → Numerical Classifications → Statistical Analysis
(e.g., Job Description → Key Requirements → Skills Ratings (1-5) → Correlation Analysis)
```

## Features
- **Summarization Mode**: Generate concise summaries of text data using OpenAI's GPT models
- **Classification Mode**: Classify text entries with customizable categories and scoring
- **Analysis Mode**: Perform statistical analysis including:
  - Pearson correlations
  - Full correlation matrices
  - Distribution analysis

## Use Cases
- **Job Market Analysis**: Convert job descriptions into standardized skill ratings
- **Content Analysis**: Transform articles/posts into categorical data
- **Feedback Analysis**: Convert open-ended responses into quantifiable metrics
- **Research Coding**: Transform qualitative data into quantitative variables

## Tech Stack
- Backend: Flask (Python)
- Frontend: HTML/CSS/JavaScript
- AI Integration: OpenAI API
- Data Processing: Pandas, NumPy

## Setup
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
MODEL=gpt-4.1
```
4. Run the application:
```bash
python app.py
```

## Usage
1. Upload your CSV file containing unstructured text
2. Select the text column to process
3. Choose processing mode:
   - **Summarization**: Generate structured summaries of text
   - **Classification**: Convert summaries into numerical ratings (1-5) across categories
   - **Analysis**: Analyze relationships between classified variables

## Example Workflow
1. Upload CSV with job descriptions
2. Summarize descriptions into key requirements
3. Classify requirements into skill categories (e.g., Technical, Soft Skills, Experience)
4. Generate numerical ratings for each category
5. Analyze correlations between different skills/requirements

## Note
This application requires an active OpenAI API key with access to GPT models. Billing is based on token usage.
