import os
import base64
import tempfile
import json
from datetime import datetime, timezone
from urllib import request as urlrequest
from urllib import error as urlerror
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

openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def foundry_settings():
    return {
        "url": os.environ.get("FOUNDRY_URL", "").rstrip("/"),
        "ontology_rid": os.environ.get("FOUNDRY_ONTOLOGY_RID", os.environ.get("ONTOLOGY_RID", "")),
        "token": os.environ.get("FOUNDRY_TOKEN", ""),
        "actions": {
            "createMission": os.environ.get("FOUNDRY_ACTION_CREATE_MISSION", "create-example-cask-gps-position"),
            "updateDroneStatus": os.environ.get("FOUNDRY_ACTION_UPDATE_DRONE_STATUS", "create-example-cask-gps-position"),
            "updateDroneTelemetry": os.environ.get("FOUNDRY_ACTION_UPDATE_DRONE_TELEMETRY", "create-example-cask-gps-position"),
            "createKillChainEvent": os.environ.get("FOUNDRY_ACTION_CREATE_KILLCHAIN_EVENT", "create-example-cask-gps-position"),
        }
    }


def foundry_ready():
    settings = foundry_settings()
    return bool(settings["url"] and settings["ontology_rid"] and settings["token"])


def apply_foundry_action(action_key, parameters):
    settings = foundry_settings()

    if not foundry_ready():
        raise RuntimeError("Foundry env vars missing. Set FOUNDRY_URL, FOUNDRY_ONTOLOGY_RID, and FOUNDRY_TOKEN.")

    action_name = settings["actions"].get(action_key, action_key)
    endpoint = f'{settings["url"]}/api/v2/ontologies/{settings["ontology_rid"]}/actions/{action_name}/apply'
    payload = json.dumps({"parameters": parameters}).encode("utf-8")
    req = urlrequest.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f'Bearer {settings["token"]}',
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=15) as response:
            body = response.read().decode("utf-8") or "{}"
            return json.loads(body)
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Foundry action {action_name} failed ({exc.code}): {body}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"Foundry request failed: {exc.reason}") from exc


def apply_foundry_action_batch(action_key, parameter_sets):
    settings = foundry_settings()

    if not foundry_ready():
        raise RuntimeError("Foundry env vars missing. Set FOUNDRY_URL, FOUNDRY_ONTOLOGY_RID, and FOUNDRY_TOKEN.")

    action_name = settings["actions"].get(action_key, action_key)
    endpoint = f'{settings["url"]}/api/v2/ontologies/{settings["ontology_rid"]}/actions/{action_name}/applyBatch'
    payload = json.dumps({
        "requests": [{"parameters": params} for params in parameter_sets]
    }).encode("utf-8")
    req = urlrequest.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f'Bearer {settings["token"]}',
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=20) as response:
            body = response.read().decode("utf-8") or "{}"
            return json.loads(body)
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Foundry batch action {action_name} failed ({exc.code}): {body}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"Foundry batch request failed: {exc.reason}") from exc


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def sim_to_geo(x, z):
    base_lat = 34.6500
    base_lon = 43.9000
    return {
        "lat": base_lat + (z / 1000.0),
        "lon": base_lon + (x / 1000.0),
    }


def create_position_event(device_id, name, x=0, z=0, altitude=0, speed=0, course=0):
    geo = sim_to_geo(x, z)
    return apply_foundry_action("createKillChainEvent", {
        "deviceId": device_id,
        "name": name,
        "latitude": geo["lat"],
        "longitude": geo["lon"],
        "altitudeM": altitude,
        "speedKnots": speed,
        "courseDeg": course,
        "fixQuality": 5,
        "numSatellites": 12,
        "timestamp": iso_now(),
    })


def parse_json_object(text):
    text = text.strip()
    if "```" in text:
        import re
        match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
    return json.loads(text)


def generate_mission_plan_with_openai(objective):
    if not openai_api_key:
        raise RuntimeError("OpenAI API key not configured")

    endpoint = "https://api.openai.com/v1/chat/completions"
    prompt = f"""You are a mission planner for a terrain-aware drone swarm simulation.

Given a mission objective, return a JSON object with this exact shape:
{{
  "nodes": ["mission-trigger", "isr-phase", "swarm-assignment", "engagement-authorization", "execute", "battle-damage-assessment", "report-summary"],
  "plan": {{
    "target": {{
      "x": number,
      "z": number,
      "label": string
    }},
    "swarm_mode": "smart" | "dumb",
    "environment": "alpine" | "desert" | "arctic" | "volcanic" | "tropical",
    "formation": "swarm" | "delta" | "perimeter" | "grid" | "helix",
    "requires_approval": boolean,
    "reasoning": string
  }}
}}

Rules:
- x and z are simulation map coordinates between -220 and 220.
- If the objective references west, target x should be negative.
- If the objective references east, target x should be positive.
- If the objective references north, target z should be negative.
- If the objective references south, target z should be positive.
- If the objective mentions avoiding SAMs, terrain masking, valleys, or cover, choose swarm_mode "smart".
- If the objective is lethal, include engagement-authorization before execute.
- Return JSON only.

Mission objective: {objective}
"""

    payload = {
        "model": openai_model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You produce strict JSON for a drone mission planner."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    req = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            raw = json.loads(response.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            return parse_json_object(content)
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI mission planner failed ({exc.code}): {body}") from exc
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OpenAI returned an invalid mission plan: {exc}") from exc


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


@app.route("/mapbox-simulation")
def mapbox_simulation():
    return send_from_directory("static", "mapbox-simulation.html")


@app.route("/api/foundry/health")
def foundry_health():
    settings = foundry_settings()
    return jsonify({
        "configured": foundry_ready(),
        "url": settings["url"],
        "ontology_rid": settings["ontology_rid"],
        "actions": settings["actions"],
    })


@app.route("/api/foundry/mission", methods=["POST"])
def create_foundry_mission():
    data = request.json or {}
    mission_id = data.get("id", "mission")
    objective = data.get("objective", "Terrain-aware strike mission")
    summary = f"MISSION START | {mission_id} | {objective[:60]}"

    try:
        result = create_position_event(
            device_id=f"event-{mission_id}",
            name=summary,
            altitude=200,
            speed=1,
            course=0,
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/foundry/drone-lost", methods=["POST"])
def foundry_drone_lost():
    data = request.json or {}
    drone_id = data.get("drone_id", "unknown")
    mission_id = data.get("mission_id", "mission")

    try:
        result = create_position_event(
            device_id=f"loss-{drone_id}",
            name=f"DRONE LOST | {mission_id} | {drone_id}",
            altitude=25,
            speed=0,
            course=270,
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/foundry/telemetry", methods=["POST"])
def foundry_telemetry():
    data = request.json or {}
    drones = data.get("drones", [])
    if not drones:
        return jsonify({"ok": True, "skipped": True, "count": 0})

    try:
        results = []
        for drone in drones[:20]:
            geo = sim_to_geo(drone.get("position_x", 0), drone.get("position_z", 0))
            params = {
                "deviceId": drone.get("drone_id"),
                "name": f"{drone.get('formation', 'Swarm')} {drone.get('drone_id')}",
                "latitude": geo["lat"],
                "longitude": geo["lon"],
                "altitudeM": max(0, float(drone.get("position_y", 0))),
                "speedKnots": max(30.0, float(drone.get("battery_pct", 0))),
                "courseDeg": 90.0,
                "fixQuality": 5,
                "numSatellites": 12,
                "timestamp": iso_now(),
            }
            results.append(apply_foundry_action("updateDroneTelemetry", params))
        return jsonify({"ok": True, "count": len(results), "result": results[-1] if results else {}})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/foundry/killchain-event", methods=["POST"])
def foundry_killchain_event():
    data = request.json or {}
    mission_id = data.get("mission_id", "mission")
    node_name = data.get("node_name", "Event")
    action_taken = data.get("action_taken", "APPROVED")
    confidence_score = data.get("confidence_score", 1.0)

    try:
        result = create_position_event(
            device_id=f"event-{mission_id}",
            name=f"{node_name} | {action_taken} | conf {confidence_score}",
            altitude=150,
            speed=2,
            course=45,
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/generate-killchain", methods=["POST"])
def generate_killchain():
    data = request.json
    objective = data.get("objective", "")

    if not objective:
        return jsonify({"error": "Objective is required"}), 400

    try:
        if openai_api_key:
            result = generate_mission_plan_with_openai(objective)
            nodes = result.get("nodes", [])
            plan = result.get("plan", {})
        elif anthropic_client:
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
"""
                    }
                ]
            )
            response_text = message.content[0].text.strip()
            if "```" in response_text:
                import re
                json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
            nodes = json.loads(response_text)
            plan = {}
        else:
            return jsonify({"error": "No mission-planning model configured"}), 503

        # Validate nodes
        valid_nodes = ['mission-trigger', 'isr-phase', 'swarm-assignment', 'engagement-authorization', 'execute', 'battle-damage-assessment', 'report-summary']
        nodes = [n for n in nodes if n in valid_nodes]

        # Ensure engagement-authorization is included for lethal ops
        if 'execute' in nodes and 'engagement-authorization' not in nodes:
            # Insert before execute
            exec_index = nodes.index('execute')
            nodes.insert(exec_index, 'engagement-authorization')

        return jsonify({"nodes": nodes, "plan": plan})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
