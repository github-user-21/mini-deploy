
import sys
import os
import logging
from flask import Flask, jsonify

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Final  # import your main function

app = Flask(__name__)
app.config['DEBUG'] = True  # Enable debug mode

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/data', methods=['GET'])
def get_data():
    try:
        x, y = Final.main()  # Call the function to get x and y
        return jsonify({"Reco": x, "Future": y}), 200
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
