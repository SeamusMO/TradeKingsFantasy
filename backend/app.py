from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import main as main 
import player as player 

app = Flask(__name__)
CORS(app)  # Allow the browser extension to talk to this server

@app.route('/process-trade', methods=['POST'])
def process_trade():
    try:
        data = request.json
        image_data = data['image']
        coords = data['coords']

        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        img_bytes = base64.b64decode(image_data)
        full_img = Image.open(BytesIO(img_bytes))

        # If the user selected an area only, crop to that selection first (frontend sends selection)
        crop_area = (
            coords['x'],
            coords['y'],
            coords['x'] + coords['width'],
            coords['y'] + coords['height']
        )
        selected_img = full_img.crop(crop_area)

        result = main.perform_full_parse_from_image(selected_img)

        return jsonify({"status": "success", "data": result})
    except Exception as e:
        print("Error in process_trade:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("Server running on http://localhost:5000")
    app.run(port=5000)