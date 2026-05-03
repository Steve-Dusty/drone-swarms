import os
import base64
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from google import genai
from google.genai import types
import fal_client
import anthropic

load_dotenv()

app = Flask(__name__, static_folder="static")

# Initialize google_client only if API key is available
google_client = None
if "GOOGLE_API_KEY" in os.environ:
    google_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize Anthropic client
anthropic_client = None
if "ANTHROPIC_API_KEY" in os.environ:
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/generate-image", methods=["POST"])
def generate_image():
    if not google_client:
        return jsonify({"error": "Google API key not configured"}), 503

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


@app.route("/api/generate-killchain", methods=["POST"])
def generate_killchain():
    if not anthropic_client:
        return jsonify({"error": "Anthropic API key not configured"}), 503

    data = request.json
    objective = data.get("objective", "")

    if not objective:
        return jsonify({"error": "Objective is required"}), 400

    try:
        # Call Claude API to analyze objective and generate kill chain
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a military mission planning AI. Given a mission objective, generate an appropriate kill chain sequence.

Available kill chain nodes:
- mission-trigger: Operator confirms target coordinates and rules of engagement
- isr-phase: Scout drones survey area of operations, build terrain picture, identify threats
- swarm-assignment: Attack drones assigned sectors based on terrain masking and battery
- engagement-authorization: Human operator approves engagement (REQUIRED BY LAW)
- execute: Swarm executes with autonomous deconfliction
- battle-damage-assessment: Post-strike evaluation of target destruction
- report-summary: Mission summary and ROE compliance

Mission Objective: {objective}

Return ONLY a JSON array of node types in execution order. The engagement-authorization node is MANDATORY for any lethal operation.

Example response: ["mission-trigger", "isr-phase", "swarm-assignment", "engagement-authorization", "execute", "battle-damage-assessment", "report-summary"]

Respond with JSON only, no explanation."""
                }
            ]
        )

        # Parse response
        response_text = message.content[0].text.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```" in response_text:
            import re
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

        import json
        nodes = json.loads(response_text)

        # Validate nodes
        valid_nodes = ['mission-trigger', 'isr-phase', 'swarm-assignment', 'engagement-authorization', 'execute', 'battle-damage-assessment', 'report-summary']
        nodes = [n for n in nodes if n in valid_nodes]

        # Ensure engagement-authorization is included for lethal ops
        if 'execute' in nodes and 'engagement-authorization' not in nodes:
            # Insert before execute
            exec_index = nodes.index('execute')
            nodes.insert(exec_index, 'engagement-authorization')

        return jsonify({"nodes": nodes})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
