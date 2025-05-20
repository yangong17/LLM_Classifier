import os
import openai
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, session, send_file, jsonify
from werkzeug.utils import secure_filename
import traceback
from dotenv import load_dotenv

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
    'gpt-4.5-preview': {
        'display_name': 'GPT-4.5 Preview ($75.00/$150.00 per 1M tokens)',
        'input': 75.00,
        'cached_input': 37.50,
        'output': 150.00
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
    'default': {
        'name': 'Default Template',
        'template': '{csv column input}'
    },
    'job_description': {
        'name': 'Job Description Analysis',
        'template': '''Analyze the following job description and categorize it into these key areas:
1. Required Technical Skills
2. Required Soft Skills
3. Experience Level
4. Company Benefits
5. Primary Responsibilities

Job Description:
{csv column input}'''
    },
    'summarize': {
        'name': 'Text Summarization',
        'template': '''Please provide a concise summary of the following text, highlighting the key points and main ideas:

{csv column input}'''
    },
    'classify': {
        'name': 'Text Classification',
        'template': '''Please classify the following text into appropriate categories based on its content and themes:

{csv column input}'''
    }
}

# === Flask Setup ===
app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
    prompt_template_chars = len(template) if template else 200  # Use actual template length if provided
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
            input_cost = total_input_tokens_millions * prices['input']
            cached_input_cost = total_input_tokens_millions * prices['cached_input']
            output_cost = total_output_tokens_millions * prices['output']
            total_cost = input_cost + output_cost
            total_cost_cached = cached_input_cost + output_cost
            
            stats['costs'][model] = {
                'display_name': prices['display_name'],
                'input_price': prices['input'],
                'cached_input_price': prices['cached_input'],
                'output_price': prices['output'],
                'estimated_input_tokens': int(total_input_tokens),
                'estimated_output_tokens': int(total_output_tokens),
                'estimated_input_cost': round(input_cost, 4),
                'estimated_cached_input_cost': round(cached_input_cost, 4),
                'estimated_output_cost': round(output_cost, 4),
                'estimated_total_cost': round(total_cost, 4),
                'estimated_total_cost_cached': round(total_cost_cached, 4)
            }
    
    return stats

# === Initialize from Environment ===
def init_env():
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write('OPENAI_API_KEY=\nMODEL=gpt-4.1')

init_env()

def is_api_key_valid():
    api_key = os.getenv('OPENAI_API_KEY')
    return api_key and api_key.strip() and api_key != 'your_api_key_here'

def mask_api_key(api_key):
    if not api_key or len(api_key) < 6:
        return "••••••"
    return f"{api_key[:2]}••••{api_key[-4:]}"

# === Main Route ===
@app.route('/', methods=['GET', 'POST'])
def index():
    # Step 1: File Upload
    if 'csv_path' not in session:
        return render_template('dashboard.html',
                            step='upload',
                            csv_uploaded=False,
                            models=MODEL_PRICING,
                            sample_prompts=SAMPLE_PROMPTS)
    
    # Step 2: Column Selection
    if 'selected_column' not in session:
        try:
            df = pd.read_csv(session['csv_path'])
            columns = df.columns.tolist()
            return render_template('dashboard.html',
                                step='select_column',
                                csv_uploaded=True,
                                columns=columns,
                                models=MODEL_PRICING,
                                sample_prompts=SAMPLE_PROMPTS)
        except Exception as e:
            return f"Error loading CSV: {str(e)}"
    
    # Step 3: API Key (if needed) and Main Dashboard
    api_key_valid = is_api_key_valid()
    
    # Calculate stats if we have both file and column
    stats = None
    if 'csv_path' in session and 'selected_column' in session:
        try:
            df = pd.read_csv(session['csv_path'])
            selected_model = session.get('model', 'gpt-4.1')  # Default to gpt-4.1
            stats = calculate_stats(df, session['selected_column'], selected_model)
        except Exception as e:
            print(f"Error calculating stats: {str(e)}")
    
    return render_template('dashboard.html',
                        step='dashboard',
                        needs_api_key=not api_key_valid,
                        csv_uploaded=True,
                        api_key=mask_api_key(os.getenv('OPENAI_API_KEY')),
                        model=os.getenv('MODEL', 'gpt-4.1'),
                        models=MODEL_PRICING,
                        columns=session.get('columns', []),
                        selected_column=session.get('selected_column'),
                        preview_data=get_preview_data(),
                        stats=stats,
                        sample_prompts=SAMPLE_PROMPTS,
                        prompt_template=request.form.get('custom_prompt', SAMPLE_PROMPTS['default']['template']),
                        prompt_preview=get_prompt_preview())

# === File Upload Handler ===
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

# === Settings Update Routes ===
@app.route('/update_column', methods=['POST'])
def update_column():
    data = request.get_json()
    column = data.get('column')
    if column:
        session['selected_column'] = column
    return jsonify({'status': 'success'})

@app.route('/update_api_key', methods=['POST'])
def update_api_key():
    data = request.get_json()
    api_key = data.get('api_key')
    if api_key:
        try:
            # Test the API key using the models endpoint (free)
            openai.api_key = api_key
            openai.Model.list()
            
            # If the API call was successful, save the key
            with open('.env', 'r') as f:
                env_lines = f.readlines()
            
            with open('.env', 'w') as f:
                for line in env_lines:
                    if line.startswith('OPENAI_API_KEY='):
                        f.write(f'OPENAI_API_KEY={api_key}\n')
                    else:
                        f.write(line)
            load_dotenv()
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
    return jsonify({'status': 'error', 'message': 'No API key provided'}), 400

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
    if mode in ['summarize', 'classify']:
        session['mode'] = mode
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

# === Helper Functions ===
def get_preview_data():
    if 'csv_path' not in session or 'selected_column' not in session:
        return None
    
    try:
        df = pd.read_csv(session['csv_path'])
        column = session['selected_column']
        
        if column not in df.columns:
            return None
            
        # Get first 5 entries
        preview_data = df[column].head(5).tolist()
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
            
        # Get first entry
        sample_text = df[column].iloc[0]
        
        # Get current template
        template = request.form.get('custom_prompt', SAMPLE_PROMPTS['default']['template'])
        
        # Replace placeholder with sample text
        preview = template.replace('{csv column input}', str(sample_text))
        return preview
    except Exception as e:
        print(f"Error generating prompt preview: {str(e)}")
        return None

# === Process Route ===
@app.route('/process', methods=['POST'])
def process():
    if not is_api_key_valid():
        return "Please set your OpenAI API key first", 400
        
    try:
        df = pd.read_csv(session['csv_path'])
        col = session['selected_column']
        mode = session.get('mode', 'summarize')
        prompt_template = request.form['custom_prompt']
        
        openai.api_key = os.getenv('OPENAI_API_KEY')
        model = os.getenv('MODEL', 'gpt-4.1')

        results = []
        for i, row in enumerate(df[col]):
            try:
                print(f"→ Processing row {i + 1}")
                prompt = prompt_template.replace('{csv column input}', str(row))
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                )
                result = response.choices[0].message.content.strip()
                results.append(result)
            except Exception as e:
                print(f"❌ LLM error on row {i+1}:", e)
                traceback.print_exc()
                results.append("LLM ERROR")

        # Add results to dataframe
        result_column = 'Summary' if mode == 'summarize' else 'Classification'
        df[result_column] = results

        # Save and send file
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        out_path = os.path.join(OUTPUT_FOLDER, f'{mode}_output.csv')
        df.to_csv(out_path, index=False)
        return send_file(out_path, as_attachment=True)

    except Exception as e:
        traceback.print_exc()
        return f"Error during processing: {str(e)}"

@app.route('/get_prompt_preview', methods=['POST'])
def get_prompt_preview_route():
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
        
        if column not in df.columns:
            return jsonify({
                'status': 'error',
                'message': 'Invalid column'
            })
            
        # Get first entry
        sample_text = df[column].iloc[0]
        
        # Replace placeholder with sample text
        preview = template.replace('{csv column input}', str(sample_text))
        
        return jsonify({
            'status': 'success',
            'preview': preview
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
