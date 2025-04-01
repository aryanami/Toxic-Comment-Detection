from flask import Flask, request, render_template
import requests

app = Flask(__name__)

# Replace with your actual API Gateway endpoint
API_ENDPOINT = "https://pcj494kz5m.execute-api.us-east-2.amazonaws.com/prod/detect_toxic_detection"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        try:
            response = requests.get(f"{API_ENDPOINT}?text={text}")
            if response.status_code == 200:
                result = response.json()
                # Ensure the result has the expected structure
                if 'probability' not in result:
                    result['probability'] = 0.0
                if 'is_toxic' not in result:
                    result['is_toxic'] = False
            else:
                result = {
                    'error': f"API request failed with status {response.status_code}",
                    'probability': 0.0,
                    'is_toxic': False
                }
        except requests.exceptions.RequestException as e:
            result = {
                'error': str(e),
                'probability': 0.0,
                'is_toxic': False
            }
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)