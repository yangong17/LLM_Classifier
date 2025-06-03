import os
import openai
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, session, send_file, jsonify, Response
from werkzeug.utils import secure_filename
import traceback
from dotenv import load_dotenv
from collections import OrderedDict
import json
from datetime import datetime, timedelta
import requests
import logging

# === Load Environment Variables ===
load_dotenv()

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Constants ===
MODEL_PRICING = {
    'gpt-4': {
        'display_name': 'GPT-4 ($0.03/$0.06 per 1K tokens)',
        'input': 30.00,  # Price per 1M tokens
        'cached_input': 7.50,
        'output': 60.00
    },
    'gpt-4-turbo-preview': {
        'display_name': 'GPT-4 Turbo ($0.01/$0.03 per 1K tokens)',
        'input': 10.00,
        'cached_input': 2.50,
        'output': 30.00
    },
    'gpt-4-1106-preview': {
        'display_name': 'GPT-4.1-nano ($0.01/$0.02 per 1K tokens)',
        'input': 10.00,
        'cached_input': 2.50,
        'output': 20.00
    },
    'gpt-3.5-turbo': {
        'display_name': 'GPT-3.5 Turbo ($0.0005/$0.0015 per 1K tokens)',
        'input': 0.50,
        'cached_input': 0.10,
        'output': 1.50
    }
}

# === Sample Prompts ===
SAMPLE_PROMPTS = {
    'summarization': OrderedDict([
        ('default', {
            'name': 'Default Summarization',
            'template': '''Please provide a concise summary of the following text, highlighting the key points and main ideas:

{csv column input}'''
        }),
        ('trip_report', {
            'name': 'Trip Report Analysis',
            'template': '''Act as my research assistant in psychedelic medicine. I will provide you with a first-person psychedelic trip report. Your task is to summarize it using the structure below:

Trip Report:
{csv column input}

End of trip report. As my research assistant, here is the format of the summary I want you to write for me:

Structure:
1. A brief overview of the experience in bullet points
2. Descriptions of the following details, each with:
    - A description bullet point, about 3-5 sentences each.
    - A quote bullet point pulled directly from the report for support

The details are: 

Experience Level
- How experienced were they with psychedelics? Have they had previous trips?
- Quote: "..."

Control/Environmental Safety Level
- How much control or environmental safety did the user have during the experience?
- Quote: "..."

Contextual Understanding
- How much guiding mental, spiritual, or philosophical framework did the user have going into the trip?
- Quote: "..."

Intention
- Did they have a specific reason or goal? Was preparation involved?
- Quote: "..."

Outcome
- Was the experience positive, negative, or mixed?
- Quote: "..."

Integration Practice
- Efforts to integrate afterward?
- Quote: "..." or "No integration practices were mentioned."

Important Notes:
- Use bullet points for both the description and its corresponding quote.
- Accuracy is HIGHLY IMPORTANT - include only what is explicitly stated.'''
        }),
        ('bullet_points', {
            'name': 'Bullet Point Summary',
            'template': '''Please summarize the following text into clear, concise bullet points:

{csv column input}

Please format your response as:
• Key point 1
• Key point 2
• Key point 3'''
        }),
        ('executive', {
            'name': 'Executive Summary',
            'template': '''Please provide an executive summary of the following text, including:
1. Main objective/purpose
2. Key findings/points
3. Conclusions/recommendations

Text:
{csv column input}'''
        })
    ]),
    'classification': OrderedDict([
        ('default', {
            'name': 'Default Classification',
            'template': '''Your task is to analyze and classify the following text across five key dimensions. Use the scoring guidelines below to assign scores from 1 to 5 for each category.

Scoring Guidelines:

Complexity (1-5)
How technically or conceptually complex is the content?

1 - Very simple; basic concepts that require no specialized knowledge
3 - Moderate complexity; requires some domain knowledge or analytical thinking
5 - Highly complex; requires deep expertise or handles multiple intricate concepts

Clarity (1-5)
How clear and well-written is the text?

1 - Unclear or confusing; poor structure, grammar issues, or hard to follow
3 - Moderately clear; some organization but could be more precise or better structured
5 - Exceptionally clear; well-organized, precise language, easy to understand

Relevance (1-5)
How relevant is the content to its intended purpose/topic?

1 - Not relevant; off-topic or tangential to the main subject
3 - Moderately relevant; addresses the topic but includes unnecessary information
5 - Highly relevant; focused and directly addresses the core topic

Actionability (1-5)
How easily can the information be acted upon?

1 - Not actionable; purely theoretical or lacks practical application
3 - Somewhat actionable; provides some guidance but requires additional information
5 - Highly actionable; clear, specific steps or recommendations that can be implemented

Impact (1-5)
What is the potential impact if the information is acted upon?

1 - Minimal impact; changes would be superficial or insignificant
3 - Moderate impact; would lead to notable but not transformative changes
5 - High impact; could lead to significant or transformative changes

Text to analyze:
{csv column input}

Instructions:
Rate each category using the rubric above. Use only explicitly stated information and avoid assumptions. If information is ambiguous or missing, default to the more conservative (lower) rating.

Respond in the following exact format (labels plus number, one per line, no extra text):

Example:
Complexity: 4
Clarity: 3
Relevance: 5
Actionability: 4
Impact: 3'''
        }),
        ('trip_report', {
            'name': 'Trip Report Classification',
            'template': '''You are analyzing a summarized psychedelic trip report, which contains the following sections: Experience Level, Control/Environmental Safety Level, Contextual Understanding, Intention, Integration Practice, and Outcome.

Your task is to assign numeric scores to each of these categories based on the content of the summary. Use the anchor descriptions below as a rubric to standardize your ratings, especially focusing on scores 1 (low), 3 (moderate), and 5 (high).

Scoring Guidelines:
Experience (1–5)
How experienced is the author with psychedelics?

1 – First-time or very inexperienced; explicitly states it's their first or second time.
3 – Moderate experience; has used psychedelics a few times, with some familiarity.
5 – Highly experienced; repeated use over time, demonstrates fluency in terminology or practices.

Control / Environmental Safety (1–5)
How much control or safety did they have in their setting?

1 – Chaotic or unsafe setting; peer pressure, no preparation, unexpected events.
3 – Mixed environment; some planning but minor disruptions or lack of supervision.
5 – Fully intentional and safe; planned set and setting, possibly guided, supportive surroundings.

Contextual Understanding (1–5)
How strong was their guiding framework (philosophy, spirituality, therapeutic lens)?

1 – No context; took the substance without a clear reason or framework.
3 – Some awareness; references general curiosity or healing goals without depth.
5 – Clear and deep framework; describes specific beliefs, spiritual practices, or therapeutic models.

Intention (1–5)
How clearly did the author state their purpose for the trip?

1 – No clear intention; used impulsively or recreationally without stated reason.
3 – Vague or mixed intention; curiosity or general self-exploration.
5 – Strong, well-defined intention; healing trauma, spiritual growth, therapy, etc.

Integration Practice (0 or 1)
Did the author take steps to reflect on or apply their experience afterward?

0 – No mention of integration; no signs of journaling, reflection, or life changes.
1 – Integration effort described; mentions discussing with others, journaling, lifestyle changes, or applying lessons learned.

Outcome (1–5)
How positive or negative was the experience overall?

1 – Overwhelmingly negative or traumatic; described as distressing, disorienting, or harmful.
3 – Mixed or neutral; some insight or pleasant moments but also challenges or confusion.
5 – Deeply positive and transformative; described as healing, life-changing, or profoundly meaningful.

Trip Summary to analyze:
{csv column input}

Instructions:
Please rate each category using the rubric above. Use only explicitly stated information in the summary and avoid making assumptions. If information is ambiguous or missing, default to the most conservative (lower) rating.

Respond in the following exact format (labels plus number, one per line, no extra text):

Example:
Experience: 4  
Control: 3  
Context: 4  
Intention: 5  
Integration: 0  
Outcome: 4'''
        }),
        ('sentiment', {
            'name': 'Sentiment Analysis',
            'template': '''Your task is to analyze the sentiment of the following text across four dimensions. Use the scoring guidelines below to assign appropriate scores.

Scoring Guidelines:

Positivity (1-5)
How positive or negative is the overall tone and content?

1 - Very negative; predominantly critical, pessimistic, or expressing distress
3 - Neutral; balanced mix of positive and negative elements
5 - Very positive; predominantly optimistic, enthusiastic, or expressing satisfaction

Intensity (1-5)
How strong or intense are the expressed emotions and opinions?

1 - Very mild; minimal emotional expression or passion
3 - Moderate; clear but controlled emotional expression
5 - Very intense; strong emotional language or passionate expression

Objectivity (1-5)
How objective or subjective is the content?

1 - Very subjective; primarily personal opinions and feelings
3 - Mixed; combines factual information with personal views
5 - Very objective; focuses on facts and evidence with minimal bias

Confidence (1-5)
How confident or certain is the tone of the content?

1 - Very uncertain; lots of hedging language or expressed doubt
3 - Moderate confidence; some qualifications but generally assured
5 - Very confident; strong assertions and definitive statements

Text to analyze:
{csv column input}

Instructions:
Rate each category using the rubric above. Use only explicitly stated information and avoid assumptions. If information is ambiguous or missing, default to the more conservative (lower) rating.

Respond in the following exact format (labels plus number, one per line, no extra text):

Example:
Positivity: 4
Intensity: 3
Objectivity: 5
Confidence: 4'''
        }),
        ('topic', {
            'name': 'Topic Classification',
            'template': '''Your task is to classify the relevance of the following text to different topic areas. Use the scoring guidelines below to assign a relevance score from 0 to 5 for each category.

Scoring Guidelines:

For each category, rate the relevance as follows:
0 - Not relevant at all; no connection to the topic
1 - Minimal relevance; brief or indirect mentions
2 - Slight relevance; some related concepts but not central
3 - Moderate relevance; clear connection but not primary focus
4 - High relevance; significant focus on the topic
5 - Primary topic; central focus of the text

Topic Categories:

Technology
- Hardware, software, digital systems, innovation
- AI, programming, cybersecurity, digital transformation
- Technical infrastructure, emerging tech trends

Business
- Corporate operations, management, strategy
- Marketing, sales, finance, entrepreneurship
- Industry analysis, market trends

Science
- Research, experiments, scientific methods
- Natural sciences, physics, biology, chemistry
- Scientific discoveries, academic findings

Politics
- Government, policy, legislation
- Political movements, elections, democracy
- International relations, public policy

Entertainment
- Media, arts, culture, recreation
- Movies, music, games, sports
- Celebrity news, lifestyle content

Health
- Medical information, wellness
- Healthcare systems, treatments
- Mental health, nutrition, fitness

Education
- Learning, teaching, academic content
- Educational systems, methods
- Training, skill development

Other
- Topics not fitting above categories
- Miscellaneous or unique subjects
- Cross-disciplinary content

Text to analyze:
{csv column input}

Instructions:
Rate each category using the rubric above. Use only explicitly stated information and avoid assumptions. If information is ambiguous or missing, default to the more conservative (lower) rating.

Respond in the following exact format (labels plus number, one per line, no extra text):

Example:
Technology: 5
Business: 3
Science: 4
Politics: 0
Entertainment: 1
Health: 2
Education: 3
Other: 1'''
        })
    ])
}

# === Flask Setup ===
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for session management, not authentication
UPLOAD_FOLDER = '/tmp/uploads'  # Change to /tmp for Cloud Run
OUTPUT_FOLDER = '/tmp/outputs'  # Change to /tmp for Cloud Run

# Ensure directories exist and are writable
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    logger.info(f"Created directories: {UPLOAD_FOLDER} and {OUTPUT_FOLDER}")
except Exception as e:
    logger.error(f"Error creating directories: {str(e)}")

def get_billing_info():
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {"error": "No API key found"}

        # Get current date for billing query
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # Get last 90 days
        
        # Format dates for API
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Make request to usage endpoint instead
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.get(
            f"https://api.openai.com/v1/dashboard/billing/usage?start_date={start_date_str}&end_date={end_date_str}",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            total_usage = data.get("total_usage", 0) / 100  # Convert from cents to dollars
            return {
                "total_balance": f"${total_usage:.2f}",
                "status": "success"
            }
        else:
            return {"error": f"API returned status code {response.status_code}"}
            
    except Exception as e:
        print(f"Error fetching billing info: {str(e)}")
        return {"error": str(e)}

@app.route('/get_billing_info', methods=['GET'])
def billing_info():
    return jsonify(get_billing_info())

def is_api_key_valid():
    api_key = os.getenv('OPENAI_API_KEY')
    logger.info(f"Checking API key validity. Key present: {bool(api_key)}")
    if not api_key or not api_key.strip():
        logger.error("No API key found in environment")
        return False
    try:
        logger.info(f"Testing OpenAI connection with API key: {mask_api_key(api_key)}")
        client = openai.OpenAI(api_key=api_key)
        logger.info("Created OpenAI client, attempting to list models...")
        client.models.list()
        logger.info("OpenAI connection test successful")
        return True
    except openai.APIConnectionError as e:
        logger.error(f"Connection Error details: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return "connection_error"
    except Exception as e:
        logger.error(f"API Error details: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return False

def mask_api_key(api_key):
    if not api_key or len(api_key) < 6:
        return "••••••"
    return f"{api_key[:2]}••••{api_key[-4:]}"

def calculate_stats(df, column, selected_model=None, template=None):
    if column not in df.columns:
        return None
    
    # Calculate character counts
    char_counts = df[column].astype(str).str.len()
    
    stats = {
        'row_count': len(df),
        'mean_chars': int(char_counts.mean()),
        'median_chars': int(char_counts.median()),
        'total_chars': int(char_counts.sum()),
        'costs': {}
    }
    
    # Estimate tokens (rough estimate: 4 chars per token)
    prompt_template_chars = len(template) if template else 200
    input_tokens_per_row = (char_counts + prompt_template_chars) / 4
    output_tokens_per_row = (char_counts * 1.5) / 4  # Assuming output is ~1.5x input length
    
    total_input_tokens = input_tokens_per_row.sum()
    total_output_tokens = output_tokens_per_row.sum()
    
    # Convert tokens to millions for pricing calculation
    total_input_tokens_millions = total_input_tokens / 1_000_000
    total_output_tokens_millions = total_output_tokens / 1_000_000
    
    # Calculate costs only for the selected model if specified
    models_to_calculate = [selected_model] if selected_model else MODEL_PRICING.keys()
    
    for model in models_to_calculate:
        if model in MODEL_PRICING:
            prices = MODEL_PRICING[model]
            stats['costs'][model] = {
                'display_name': prices['display_name'],
                'input_price': prices['input'],
                'cached_input_price': prices['cached_input'],
                'output_price': prices['output'],
                'estimated_input_tokens': int(total_input_tokens),
                'estimated_output_tokens': int(total_output_tokens),
                'estimated_input_cost': round(total_input_tokens_millions * prices['input'], 4),
                'estimated_cached_input_cost': round(total_input_tokens_millions * prices['cached_input'], 4),
                'estimated_output_cost': round(total_output_tokens_millions * prices['output'], 4),
                'estimated_total_cost': round(total_input_tokens_millions * prices['input'] + total_output_tokens_millions * prices['output'], 4),
                'estimated_total_cost_cached': round(total_input_tokens_millions * prices['cached_input'] + total_output_tokens_millions * prices['output'], 4)
            }
    
    return stats

def get_preview_data():
    if 'csv_path' not in session or 'selected_column' not in session:
        return None
    
    try:
        df = pd.read_csv(session['csv_path'])
        column = session['selected_column']
        
        if column not in df.columns:
            return None
            
        # Get first 5 entries and convert to string
        preview_data = df[column].head(5).apply(str).tolist()
        return preview_data
    except Exception as e:
        print(f"Error getting preview data: {str(e)}")
        return None

def get_prompt_preview():
    if 'csv_path' not in session or 'selected_column' not in session:
        return None
        
    try:
        df = pd.read_csv(session['csv_path'])
        column = session['selected_column']
        
        if column not in df.columns:
            return None
            
        # Get first entry and convert to string
        sample_text = str(df[column].iloc[0])
        
        # Get current template
        template = request.form.get('custom_prompt', SAMPLE_PROMPTS['summarization']['default']['template'])
        
        # Replace placeholder with sample text
        preview = template.replace('{csv column input}', sample_text)
        return preview
    except Exception as e:
        print(f"Error generating prompt preview: {str(e)}")
        return None

def parse_classification_result(result):
    """Parse classification result into a dictionary of category:score pairs"""
    try:
        categories = {}
        lines = result.strip().split('\n')
        
        # Skip lines until we find actual category:score pairs
        start_idx = 0
        for i, line in enumerate(lines):
            if ':' in line and any(c.isdigit() for c in line):
                start_idx = i
                break
        
        # Process only the relevant lines
        for line in lines[start_idx:]:
            if ':' in line:
                category, score = line.split(':', 1)
                category = category.strip().lower()
                # Remove any parenthetical descriptions
                category = category.split('(')[0].strip()
                try:
                    # Extract just the number from the score
                    score_str = ''.join(c for c in score if c.isdigit())
                    score = int(score_str)
                    if 0 <= score <= 5:  # Validate score range
                        categories[category] = score
                except ValueError:
                    continue  # Skip invalid scores instead of failing
        return categories if categories else None
    except Exception as e:
        print(f"Error parsing classification result: {str(e)}")
        return None

# === Routes ===
@app.route('/', methods=['GET', 'POST'])
def index():
    # Step 1: File Upload
    if 'csv_path' not in session:
        return render_template('dashboard.html',
                            step='upload',
                            csv_uploaded=False,
                            models=MODEL_PRICING,
                            sample_prompts=SAMPLE_PROMPTS)
    
    # Load columns from CSV
    try:
        df = pd.read_csv(session['csv_path'])
        columns = df.columns.tolist()
        session['columns'] = columns  # Store columns in session
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return f"Error loading CSV: {str(e)}"
    
    # Step 2: Column Selection
    if 'selected_column' not in session:
        return render_template('dashboard.html',
                            step='select_column',
                            csv_uploaded=True,
                            columns=columns,
                            models=MODEL_PRICING,
                            sample_prompts=SAMPLE_PROMPTS)
    
    # Step 3: API Key (if needed) and Main Dashboard
    api_key_status = is_api_key_valid()
    
    if api_key_status == "connection_error":
        return render_template('dashboard.html',
                            step='dashboard',
                            needs_api_key=True,
                            connection_error=True,
                            error_message="❌ Connection error. Please check your internet connection and try again.",
                            csv_uploaded=True,
                            api_key=mask_api_key(os.getenv('OPENAI_API_KEY')),
                            model=os.getenv('MODEL', 'gpt-4'),
                            models=MODEL_PRICING,
                            columns=columns,
                            selected_column=session.get('selected_column'))
    
    # Calculate stats if we have both file and column and not in analysis mode
    stats = None
    if 'csv_path' in session and 'selected_column' in session and session.get('mode') != 'analyze':
        try:
            selected_model = session.get('model', 'gpt-4')
            stats = calculate_stats(df, session['selected_column'], selected_model)
        except Exception as e:
            print(f"Error calculating stats: {str(e)}")
    
    return render_template('dashboard.html',
                        step='dashboard',
                        needs_api_key=not api_key_status,
                        csv_uploaded=True,
                        api_key=mask_api_key(os.getenv('OPENAI_API_KEY')),
                        model=os.getenv('MODEL', 'gpt-4'),
                        models=MODEL_PRICING,
                        columns=columns,
                        selected_column=session.get('selected_column'),
                        preview_data=get_preview_data() if session.get('mode') != 'analyze' else None,
                        stats=stats,
                        sample_prompts=SAMPLE_PROMPTS,
                        prompt_template=request.form.get('custom_prompt', SAMPLE_PROMPTS['summarization']['default']['template']) if session.get('mode') != 'analyze' else None,
                        prompt_preview=get_prompt_preview() if session.get('mode') != 'analyze' else None,
                        mode=session.get('mode', 'summarize'))

@app.route('/upload_file', methods=['POST'])
def upload_file():
    logger.info("Starting file upload")
    if 'csv_file' not in request.files:
        logger.error("No file part in request")
        return redirect(url_for('index'))
    
    file = request.files['csv_file']
    if file:
        try:
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            logger.info(f"Saving file to: {path}")
            file.save(path)
            session['csv_path'] = path
            
            # Load columns
            try:
                df = pd.read_csv(path)
                session['columns'] = df.columns.tolist()
                logger.info(f"Successfully loaded CSV with columns: {df.columns.tolist()}")
            except Exception as e:
                logger.error(f"Error loading CSV: {str(e)}")
                return f"Error loading CSV: {str(e)}"
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return f"Error saving file: {str(e)}"
    
    return redirect(url_for('index'))

@app.route('/update_column', methods=['POST'])
def update_column():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
            
        column = data.get('column')
        if not column:
            return jsonify({'status': 'error', 'message': 'No column specified'}), 400
            
        # Validate that the column exists in the CSV
        if 'csv_path' not in session:
            return jsonify({'status': 'error', 'message': 'No CSV file loaded'}), 400
            
        # Get columns from session or load from CSV
        columns = session.get('columns', [])
        if not columns:
            try:
                df = pd.read_csv(session['csv_path'])
                columns = df.columns.tolist()
                session['columns'] = columns
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Error reading CSV: {str(e)}'}), 500
            
        if column not in columns:
            return jsonify({'status': 'error', 'message': f'Column "{column}" not found in CSV'}), 400
            
        session['selected_column'] = column
        return jsonify({'status': 'success', 'message': f'Successfully updated column to {column}'})
        
    except Exception as e:
        print(f"Error in update_column: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/update_api_key', methods=['POST'])
def update_api_key():
    data = request.get_json()
    api_key = data.get('api_key', '').strip()
    
    if not api_key:
        return jsonify({'status': 'error', 'message': 'No API key provided'}), 400

    try:
        # Sanitize the API key - remove any non-ASCII characters and whitespace
        api_key = ''.join(char for char in api_key if 32 <= ord(char) <= 126)
        api_key = api_key.strip()

        if not api_key:
            return jsonify({
                'status': 'error',
                'message': 'API key contains invalid characters. Please ensure you\'ve copied it correctly from OpenAI.'
            }), 400
        
        # Test the API key
        try:
            client = openai.OpenAI(api_key=api_key)
            client.models.list()
        except openai.AuthenticationError:
            return jsonify({
                'status': 'error',
                'message': 'Invalid API key. Please check your API key and try again.'
            }), 400
        except openai.APIConnectionError as e:
            return jsonify({
                'status': 'error',
                'message': 'Could not connect to OpenAI. Please check your internet connection.'
            }), 400
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error validating API key: {str(e)}'
            }), 400
        
        try:
            # Save the key
            if not os.path.exists('.env'):
                with open('.env', 'w', encoding='utf-8') as f:
                    f.write(f'OPENAI_API_KEY={api_key}\nMODEL=gpt-4')
            else:
                with open('.env', 'r', encoding='utf-8') as f:
                    env_lines = f.readlines()
                
                with open('.env', 'w', encoding='utf-8') as f:
                    key_written = False
                    for line in env_lines:
                        if line.startswith('OPENAI_API_KEY='):
                            f.write(f'OPENAI_API_KEY={api_key}\n')
                            key_written = True
                        else:
                            f.write(line)
                    if not key_written:
                        f.write(f'\nOPENAI_API_KEY={api_key}')
            
            # Reload environment
            load_dotenv()
            
            # Verify the key was saved and loaded correctly
            if not is_api_key_valid():
                raise Exception("API key was not saved correctly")
                
            return jsonify({'status': 'success'})
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'API key is valid but could not be saved: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing API key: {str(e)}'
        }), 400

@app.route('/update_model', methods=['POST'])
def update_model():
    data = request.get_json()
    model = data.get('model')
    if model in MODEL_PRICING:
        # Update .env file
        with open('.env', 'r') as f:
            env_lines = f.readlines()
        
        with open('.env', 'w') as f:
            for line in env_lines:
                if line.startswith('MODEL='):
                    f.write(f'MODEL={model}\n')
                else:
                    f.write(line)
                    
        # Recalculate stats with new model
        if 'csv_path' in session and 'selected_column' in session:
            try:
                df = pd.read_csv(session['csv_path'])
                stats = calculate_stats(df, session['selected_column'], model)
                return jsonify({
                    'status': 'success',
                    'stats': stats
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                })
    
    return jsonify({'status': 'error', 'message': 'Invalid model'})

@app.route('/update_mode', methods=['POST'])
def update_mode():
    data = request.get_json()
    mode = data.get('mode')
    if mode in ['summarize', 'classify', 'analyze']:
        session['mode'] = mode
        # Update template based on mode
        if mode == 'classify':
            session['template'] = SAMPLE_PROMPTS['classification']['default']['template']
        elif mode == 'analyze':
            # No template needed for analysis mode
            session['template'] = None
        else:
            session['template'] = SAMPLE_PROMPTS['summarization']['default']['template']
    return jsonify({'status': 'success'})

@app.route('/update_cost_stats', methods=['POST'])
def update_cost_stats():
    data = request.get_json()
    template = data.get('template')
    
    if not template:
        return jsonify({
            'status': 'error',
            'message': 'No template provided'
        })
    
    try:
        df = pd.read_csv(session['csv_path'])
        column = session['selected_column']
        selected_model = session.get('model', 'gpt-4')
        
        stats = calculate_stats(df, column, selected_model, template)
        return jsonify({
            'status': 'success',
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/process', methods=['GET', 'POST'])
def process():
    logger.info("Starting process route")
    if not is_api_key_valid():
        logger.error("API key validation failed")
        return "Please set your OpenAI API key first", 400
        
    try:
        logger.info("Loading CSV file")
        df = pd.read_csv(session['csv_path'])
        col = session['selected_column']
        mode = request.args.get('mode', 'summarize')
        logger.info(f"Processing mode: {mode}")
        
        # Handle both GET and POST methods
        if request.method == 'POST':
            prompt_template = request.form['custom_prompt']
        else:
            prompt_template = request.args.get('custom_prompt')
        
        if not prompt_template:
            logger.error("No prompt template provided")
            return "No prompt template provided", 400
        
        api_key = os.getenv('OPENAI_API_KEY')
        model = os.getenv('MODEL', 'gpt-3.5-turbo')
        
        logger.info(f"Initializing OpenAI client with model: {model}")
        logger.info(f"API key present: {'Yes' if api_key else 'No'}")
        
        # Create a client using the new OpenAI format
        try:
            client = openai.OpenAI(api_key=api_key)
            # Test the connection
            client.models.list()
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            return f"Error initializing OpenAI client: {str(e)}", 500

        # Add environment variable logging
        logger.info("Current environment variables:")
        for key in ['OPENAI_API_KEY', 'MODEL', 'PORT', 'FLASK_APP', 'FLASK_ENV']:
            value = os.getenv(key)
            if key == 'OPENAI_API_KEY' and value:
                logger.info(f"{key}: {mask_api_key(value)}")
            else:
                logger.info(f"{key}: {value}")

        results = []
        parsed_results = []  # For classification mode
        total_rows = len(df[col])
        output_filename = f'{mode}_output.csv'
        
        def generate_response(prompt):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error in LLM call: {str(e)}")
                return f"LLM ERROR: {str(e)}"

        # Create SSE response
        def generate():
            for i, row in enumerate(df[col]):
                try:
                    print(f"→ Processing row {i + 1} of {total_rows}")
                    prompt = prompt_template.replace('{csv column input}', str(row))
                    
                    # Get LLM response
                    result = generate_response(prompt)
                    results.append(result)
                    
                    # For classification mode, validate and parse the result
                    if mode == 'classify':
                        parsed = parse_classification_result(result)
                        if parsed is None:
                            error_msg = "Invalid classification format. Expected format: Category: Score"
                            yield f"data: {json.dumps({'error': error_msg})}\n\n"
                            parsed_results.append({})  # Add empty dict to maintain row alignment
                        else:
                            parsed_results.append(parsed)
                    
                    # Send progress and result
                    progress = {
                        'current': i + 1,
                        'total': total_rows,
                        'result': result,
                        'status': 'processing'
                    }
                    yield f"data: {json.dumps(progress)}\n\n"
                    
                except Exception as e:
                    print(f"❌ Error on row {i+1}:", e)
                    traceback.print_exc()
                    error_msg = str(e)
                    results.append(f"ERROR: {error_msg}")
                    if mode == 'classify':
                        parsed_results.append({})
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"

            # Add results to dataframe
            if mode == 'classify':
                # Get all unique categories
                categories = set()
                for result in parsed_results:
                    categories.update(result.keys())
                
                # Create columns for each category
                for category in categories:
                    column_name = category.title()  # Capitalize first letter
                    df[column_name] = [result.get(category, None) for result in parsed_results]
                
                # Add raw output column
                df['Raw_Output'] = results
            else:
                df['Summary'] = results

            # Save file
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            out_path = os.path.join(OUTPUT_FOLDER, output_filename)
            df.to_csv(out_path, index=False)
            
            # Send completion message with download path
            completion_data = {
                'status': 'complete',
                'file': f'/download/{output_filename}',
                'mode': mode
            }
            
            yield f"data: {json.dumps(completion_data)}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        traceback.print_exc()
        return f"Error during processing: {str(e)}", 400

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join(OUTPUT_FOLDER, filename),
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
    except Exception as e:
        return f"Error downloading file: {str(e)}", 404

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    try:
        data = request.get_json()
        predictors = data.get('predictors', [])
        outcome = data.get('outcome')
        
        if not predictors or not outcome:
            return jsonify({
                'status': 'error',
                'message': 'Please select both predictor and outcome variables'
            }), 400
        
        # Read the CSV file
        df = pd.read_csv(session['csv_path'])
        
        # Calculate correlations
        variables = predictors + [outcome]
        correlation_matrix = df[variables].corr()
        
        # Format correlations as a dictionary
        correlations = {}
        for v1 in variables:
            for v2 in variables:
                correlations[f"{v1}-{v2}"] = correlation_matrix.loc[v1, v2]
        
        # Calculate distributions for predictor variables
        distributions = {}
        scatter_data = {}
        for var in predictors:
            values = df[var].dropna().tolist()
            distributions[var] = {
                'values': values,
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            
            # Get data for scatter plots (only where both x and y are not null)
            mask = df[[var, outcome]].notna().all(axis=1)
            scatter_data[var] = {
                'x': df[var][mask].tolist(),
                'y': df[outcome][mask].tolist()
            }
        
        return jsonify({
            'status': 'success',
            'correlations': correlations,
            'distributions': distributions,
            'scatter_data': scatter_data
        })
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/update_prompt_preview', methods=['POST'])
def update_prompt_preview():
    try:
        data = request.get_json()
        template = data.get('template')
        
        if not template:
            return jsonify({'status': 'error', 'message': 'No template provided'}), 400
            
        # Get current CSV data
        if 'csv_path' not in session or 'selected_column' not in session:
            return jsonify({'status': 'error', 'message': 'No CSV data available'}), 400
            
        df = pd.read_csv(session['csv_path'])
        column = session['selected_column']
        
        if column not in df.columns:
            return jsonify({'status': 'error', 'message': 'Selected column not found'}), 400
            
        # Get first entry
        csv_data = str(df[column].iloc[0])
        
        # Replace placeholder with sample text
        preview = template.replace('{csv column input}', csv_data)
        
        # Wrap the CSV data in a span for styling
        preview_html = preview.replace(csv_data, f'<span class="csv-data">{csv_data}</span>')
        
        return jsonify({
            'status': 'success',
            'preview': preview_html
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
