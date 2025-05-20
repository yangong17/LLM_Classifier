import os
import openai
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, session, send_file
from werkzeug.utils import secure_filename
import traceback

# === Flask Setup ===
app = Flask(__name__)
app.secret_key = 'supersecret'  # Change this in production
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Upload CSV ===
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['csv_file']
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            session['csv_path'] = path
            return redirect(url_for('main'))
    return '''
    <h2>Upload CSV File</h2>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="csv_file" required><br><br>
      <input type="submit" value="Upload">
    </form>
    '''

# === Main Screen: API Key, Column Selection, Mode Selection ===
@app.route('/main', methods=['GET', 'POST'])
def main():
    if 'csv_path' not in session:
        return redirect(url_for('index'))

    if 'api_key' not in session or 'model' not in session:
        if request.method == 'POST':
            session['api_key'] = request.form['api_key']
            session['model'] = request.form['model']
            return redirect(url_for('main'))
        return '''
        <h2>Enter OpenAI Credentials</h2>
        <form method="post">
            API Key: <input type="text" name="api_key" required><br>
            Model: <input type="text" name="model" value="gpt-4"><br><br>
            <input type="submit" value="Continue">
        </form>
        '''

    try:
        df = pd.read_csv(session['csv_path'])
        session['columns'] = df.columns.tolist()
    except Exception as e:
        return f"<pre>❌ Failed to load CSV: {e}</pre>"

    if request.method == 'POST' and 'column' in request.form:
        session['selected_column'] = request.form['column']
        return redirect(url_for(request.form['mode']))  # summarize or classify

    column_options = ''.join([f'<option value="{col}">{col}</option>' for col in session['columns']])
    return f'''
    <h2>Select Column and Mode</h2>
    <form method="post">
        <label>Column:</label>
        <select name="column">{column_options}</select><br><br>
        <button name="mode" value="summarize">Summarize</button>
        <button name="mode" value="classify">Classify</button>
    </form>
    '''

# === Summarization Form and Processing ===
@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        try:
            df = pd.read_csv(session['csv_path'])
            col = session['selected_column']
            prompt_template = request.form['custom_prompt']
            openai.api_key = session['api_key']
            model = session['model']

            if col not in df.columns:
                return f"<pre>❌ Column '{col}' not found in CSV.</pre>"

            summaries = []
            for i, row in enumerate(df[col]):
                try:
                    print(f"→ Processing row {i + 1}")
                    prompt = prompt_template.replace('{csv column input}', str(row))
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                    )
                    summary = response.choices[0].message.content.strip()
                    summaries.append(summary)
                except Exception as e:
                    print(f"❌ LLM error on row {i+1}:", e)
                    traceback.print_exc()
                    summaries.append("LLM ERROR")

            df['Summary'] = summaries

            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            out_path = os.path.join(OUTPUT_FOLDER, 'summary_output.csv')
            df.to_csv(out_path, index=False)
            print("✅ Summary CSV saved to", out_path)
            return send_file(out_path, as_attachment=True)

        except Exception as e:
            traceback.print_exc()
            return f"<pre>❌ Error during summarization: {e}</pre>"

    # Render summarization prompt UI
    return '''
    <h2>Enter Prompt Template</h2>
    <form method="post">
        <textarea name="custom_prompt" rows="20" cols="100" required>{csv column input}</textarea><br><br>
        <input type="submit" value="Run Summarization">
    </form>
    <p>Use <code>{csv column input}</code> where each row's text should go.</p>
    '''

# === Placeholder Classify Route ===
@app.route('/classify')
def classify():
    return "<h2>Classification mode coming soon.</h2>"

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
