import os
import base64
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from google import genai
from google.genai import types
import fal_client

load_dotenv()

app = Flask(__name__, static_folder="static")

google_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/generate-image", methods=["POST"])
def generate_image():
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        response = google_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            ),
        )

        if not response.candidates or not response.candidates[0].content.parts:
            return jsonify({"error": "No image generated — prompt may have been blocked by safety filters"}), 400

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                return jsonify({
                    "image_base64": image_b64,
                    "mime_type": part.inline_data.mime_type or "image/png",
                })

        return jsonify({"error": "Response contained no image data"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-3d", methods=["POST"])
def generate_3d():
    data = request.json
    image_b64 = data.get("image_base64", "")

    if not image_b64:
        return jsonify({"error": "Image data is required"}), 400

    try:
        image_bytes = base64.b64decode(image_b64)

        # Save to temp file and upload to FAL CDN
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_bytes)
            temp_path = f.name

        image_url = fal_client.upload_file(temp_path)
        os.unlink(temp_path)

        # Generate 3D with Trellis
        result = fal_client.subscribe(
            "fal-ai/trellis",
            arguments={"image_url": image_url},
        )

        return jsonify({
            "model_url": result["model_mesh"]["url"],
            "timings": result.get("timings", {}),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/simulation")
def simulation():
    return send_from_directory("static", "simulation.html")


if __name__ == "__main__":
    app.run(debug=True, port=5050)
