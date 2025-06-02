# CSVision

A Flask web application for CSV data analysis and processing, developed as my final project for STATS418 - "Tools in Data Science", UCLA MASDS Spring 2025

## Features
- **Summarization Mode**: Generate concise summaries of text data using OpenAI's GPT models
- **Classification Mode**: Classify text entries with customizable categories and scoring
- **Analysis Mode**: Perform statistical analysis including:
  - Pearson correlations
  - Full correlation matrices
  - Distribution analysis

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
1. Upload your CSV file
2. Select the column(s) to process
3. Choose processing mode:
   - Summarization: Generate text summaries
   - Classification: Categorize entries
   - Analysis: Statistical analysis of numerical columns

## Note
This application requires an active OpenAI API key with access to GPT models. Billing is based on token usage.
