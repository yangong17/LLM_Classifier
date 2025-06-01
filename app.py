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

# === Load Environment Variables ===
load_dotenv()

# === Constants ===
MODEL_PRICING = {
    'gpt-4.1': {
        'display_name': 'GPT-4.1 ($2.00/$8.00 per 1M tokens)',
        'input': 2.00,  # Price per 1M tokens
        'cached_input': 0.50,
        'output': 8.00
    },
    'gpt-4.1-mini': {
        'display_name': 'GPT-4.1 Mini ($0.40/$1.60 per 1M tokens)',
        'input': 0.40,
        'cached_input': 0.10,
        'output': 1.60
    },
    'gpt-4.1-nano': {
        'display_name': 'GPT-4.1 Nano ($0.10/$0.40 per 1M tokens)',
        'input': 0.10,
        'cached_input': 0.025,
        'output': 0.40
    },
    'gpt-4o': {
        'display_name': 'GPT-4O ($2.50/$10.00 per 1M tokens)',
        'input': 2.50,
        'cached_input': 1.25,
        'output': 10.00
    },
    'gpt-3.5-turbo': {
        'display_name': 'GPT-3.5 Turbo ($0.50/$1.50 per 1M tokens)',
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
            'template': '''Please classify the following text into these categories and assign a score from 1-5 for each (1=lowest, 5=highest). Return only labels and scores, one per line:

Complexity: (technical/conceptual complexity)
Clarity: (how clear and well-written)
Relevance: (topic relevance)
Actionability: (can be acted upon)
Impact: (potential impact if acted upon)

Text:
{csv column input}'''
        }),
        ('trip_report', {
            'name': 'Trip Report Classification',
            'template': '''You are analyzing a summarized psychedelic trip report. It contains sections such as Experience Level, Control/Environmental Safety Level, Contextual Understanding, Intention, Integration Practice, and Outcome.

Assign a score to each of the following categories:

Experience: 1–5  
Control: 1–5  
Context: 1–5  
Intention: 1–5  
Integration: 1 if integration efforts were described, 0 otherwise  
Outcome: 1–5  (1 = distressing/negative, 5 = positive/meaningful)

Respond in the following exact format (labels plus number, one per line, no extra text):

Example:
Experience: 4  
Control: 3  
Context: 4  
Intention: 5  
Integration: 0  
Outcome: 4

Trip Summary:
{csv column input}'''
        }),
        ('sentiment', {
            'name': 'Sentiment Analysis',
            'template': '''Analyze the sentiment of the following text and provide scores from 1-5 for each aspect. Return only labels and scores, one per line:

Positivity: (1=very negative, 5=very positive)
Intensity: (1=mild, 5=strong)
Objectivity: (1=very subjective, 5=very objective)
Confidence: (1=low confidence, 5=high confidence)

Text:
{csv column input}'''
        }),
        ('topic', {
            'name': 'Topic Classification',
            'template': '''Classify the following text into these topic categories. For each category, assign a relevance score from 0-5 (0=not relevant, 5=highly relevant). Return only labels and scores, one per line:

Technology: 
Business: 
Science: 
Politics: 
Entertainment: 
Health: 
Education: 
Other: 

Text:
{csv column input}'''
        })
    ])
}

# === Flask Setup ===
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for session management, not authentication
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def is_api_key_valid():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or not api_key.strip():
        return False
    try:
        openai.api_key = api_key
        client = openai.OpenAI(api_key=api_key)
        client.models.list()  # Test the connection
        return True
    except openai.APIConnectionError as e:
        print(f"Connection Error: {str(e)}")
        return "connection_error"
    except Exception as e:
        print(f"API Error: {str(e)}")
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
                            model=os.getenv('MODEL', 'gpt-4.1'),
                            models=MODEL_PRICING,
                            columns=columns,
                            selected_column=session.get('selected_column'))
    
    # Calculate stats if we have both file and column and not in analysis mode
    stats = None
    if 'csv_path' in session and 'selected_column' in session and session.get('mode') != 'analyze':
        try:
            selected_model = session.get('model', 'gpt-4.1')
            stats = calculate_stats(df, session['selected_column'], selected_model)
        except Exception as e:
            print(f"Error calculating stats: {str(e)}")
    
    return render_template('dashboard.html',
                        step='dashboard',
                        needs_api_key=not api_key_status,
                        csv_uploaded=True,
                        api_key=mask_api_key(os.getenv('OPENAI_API_KEY')),
                        model=os.getenv('MODEL', 'gpt-4.1'),
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
    if 'csv_file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['csv_file']
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        session['csv_path'] = path
        
        # Load columns
        try:
            df = pd.read_csv(path)
            session['columns'] = df.columns.tolist()
        except Exception as e:
            return f"Error loading CSV: {str(e)}"
        
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
        # Test the API key
        openai.api_key = api_key
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        
        try:
            # Save the key
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write(f'OPENAI_API_KEY={api_key}\nMODEL=gpt-4.1')
            else:
                with open('.env', 'r') as f:
                    env_lines = f.readlines()
                
                with open('.env', 'w') as f:
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
            'message': f'Invalid API key: {str(e)}'
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
        selected_model = session.get('model', 'gpt-4.1')
        
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
    if not is_api_key_valid():
        return "Please set your OpenAI API key first", 400
        
    try:
        df = pd.read_csv(session['csv_path'])
        col = session['selected_column']
        mode = request.args.get('mode', 'summarize')  # Get mode from request parameters
        
        # Handle both GET and POST methods
        if request.method == 'POST':
            prompt_template = request.form['custom_prompt']
        else:
            prompt_template = request.args.get('custom_prompt')
        
        if not prompt_template:
            return "No prompt template provided", 400
        
        openai.api_key = os.getenv('OPENAI_API_KEY')
        model = os.getenv('MODEL', 'gpt-4.1')
        
        # Create a client using the new OpenAI format
        client = openai.OpenAI(api_key=openai.api_key)

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
            
            # Send completion message with download path and additional info
            completion_data = {
                'status': 'complete',
                'file': f'/download/{output_filename}',
                'mode': mode,
                'output_path': out_path,  # Include the output path in response
                'offer_classification': mode == 'summarize'  # Only offer classification after summarization
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
        for var in predictors:
            values = df[var].dropna().tolist()
            distributions[var] = {
                'values': values,
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return jsonify({
            'status': 'success',
            'correlations': correlations,
            'distributions': distributions
        })
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
