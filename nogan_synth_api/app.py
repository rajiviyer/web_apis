from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from nogan_synthesizer import NoGANSynth
from nogan_synthesizer.preprocessing import wrap_category_columns, unwrap_category_columns
from genai_evaluation import multivariate_ecdf, ks_statistic
import pandas as pd
import numpy as np
import os
import random
import time
import re

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # Get user inputs from the form
        file = request.files['file']
        file_name = file.filename
        file_type = request.form['file_type']
        delimiter = request.form['delimiter']

        # Save the uploaded file
        file.save(os.path.join('data', file_name))

        # Redirect to the second page for column selection and row generation
        return redirect(url_for('generate', file_name = file_name, file_type=file_type, delimiter=delimiter))

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        try:
            file_name = request.args.get('file_name')
            file_type = request.args.get('file_type')
            delimiter = request.args.get('delimiter')

            # Load data into a pandas DataFrame based on file type
            if file_type == 'csv':
                df = pd.read_csv(os.path.join('data', file_name), delimiter=delimiter)
            elif file_type == 'text':
                df = pd.read_csv(os.path.join('data', file_name), delimiter=delimiter)
            elif file_type == 'excel':
                df = pd.read_excel(os.path.join('data', file_name))            

            # Get column names for category selection
            columns = df.columns.tolist()
            
            if re.search(r'[^a-zA-Z0-9_]', "".join(columns)):
                raise ValueError("Special Characters or Space present in Column Names. Please remove them & try again")

            return render_template('generate.html', columns=columns)
        
        except Exception as e:
            error_message = f'Error: {str(e)}'
            return render_template('results.html', 
                                   success_message=None,
                                   error_message=error_message)                

    elif request.method == 'POST':
        try:
            category_columns = request.form.getlist('category_columns')
            num_rows = int(request.form['num_rows'])

            # Load the previously uploaded file
            file_name = request.args.get('file_name')
            delimeter = request.args.get('delimiter')
            
            train_data = pd.read_csv(os.path.join('data', file_name),
                               delimiter=delimeter)
            train_data = train_data.dropna()
            print(f"Datafile Shape: {train_data.shape}")
            print(f"Number of rows: {num_rows}")
                        
            if category_columns:
                wrapped_data, idx_to_key, _ = \
                        wrap_category_columns(train_data,
                                              category_columns)
                train_data = wrapped_data
            
            # Train NoGAN
            seed = np.random.randint(low=1, high=9999999, size=1)
            bins = [100] * len(train_data.columns)
                            
            nogan = NoGANSynth(train_data,
                               random_seed=seed)
            
            nogan.fit(bins = bins)
            
            synth_data = nogan.generate_synthetic_data(num_rows)
            
            # Calculate ECDFs & KS Stats 
            _, ecdf_train, ecdf_nogan_synth = \
                        multivariate_ecdf(train_data, 
                                        synth_data, 
                                        n_nodes = 5000,
                                        verbose = False,
                                        random_seed=seed)
                        
            ks_stat = ks_statistic(ecdf_train, ecdf_nogan_synth)            

            if category_columns:
                generated_data = unwrap_category_columns(data=synth_data,
                                                 idx_to_key=idx_to_key, cat_cols=category_columns)
            else:
                generated_data = synth_data                

            # Generate a unique CSV file name
            timestamp = int(time.time())
            print(f"Filname type: {type(file_name)}")
            csv_filename = f"result_{file_name.split('.')[0]}_{timestamp}.csv"
            generated_data.to_csv(os.path.join('data', csv_filename), index=False)

            success_message = f'Synthetic Data file {csv_filename} generated successfully. Download it <a href="/download/{csv_filename}">here</a>.\n KS Statistic is: {ks_stat:0.4f}'

            # Redirect to the results page and pass the success message
            return render_template('results.html', 
                                   success_message=success_message, error_message=None)

        except Exception as e:
            error_message = f'Error: {str(e)}'
            return render_template('results.html', 
                                   success_message=None,
                                   error_message=error_message)

@app.route('/download/<filename>')
def download(filename):
    # Serve the file for download from the 'data' directory
    return send_from_directory('data', filename, as_attachment=True)            

if __name__ == '__main__':
    app.run(debug=True)
