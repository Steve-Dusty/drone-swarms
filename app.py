import os
import base64
import tempfile
import json
from datetime import datetime, timezone
from urllib import request as urlrequest
from urllib import error as urlerror
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from dotenv import load_dotenv
from google import genai
from google.genai import types
import fal_client
from openai import OpenAI

load_dotenv()

app = Flask(__name__, static_folder="static")

google_client = None
if os.environ.get("GOOGLE_API_KEY"):
    google_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# ═══════════════════════════════════════
#  PALANTIR FOUNDRY INTEGRATION
# ═══════════════════════════════════════
def foundry_settings():
    return {
        "url": os.environ.get("FOUNDRY_URL", "").rstrip("/"),
        "ontology_rid": os.environ.get("FOUNDRY_ONTOLOGY_RID", ""),
        "token": os.environ.get("FOUNDRY_TOKEN", ""),
        "actions": {
            "createMission": os.environ.get("FOUNDRY_ACTION_CREATE_MISSION", "create-example-cask-gps-position"),
            "updateDroneTelemetry": os.environ.get("FOUNDRY_ACTION_UPDATE_DRONE_TELEMETRY", "create-example-cask-gps-position"),
            "createKillChainEvent": os.environ.get("FOUNDRY_ACTION_CREATE_KILLCHAIN_EVENT", "create-example-cask-gps-position"),
        },
    }


def foundry_ready():
    s = foundry_settings()
    return bool(s["url"] and s["ontology_rid"] and s["token"])


def apply_foundry_action(action_key, parameters):
    s = foundry_settings()
    if not foundry_ready():
        raise RuntimeError("Foundry env vars missing")
    action_name = s["actions"].get(action_key, action_key)
    endpoint = f'{s["url"]}/api/v2/ontologies/{s["ontology_rid"]}/actions/{action_name}/apply'
    payload = json.dumps({"parameters": parameters}).encode("utf-8")
    req = urlrequest.Request(endpoint, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f'Bearer {s["token"]}',
    }, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urlerror.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Foundry {action_name} failed ({e.code}): {body}")
    except urlerror.URLError as e:
        raise RuntimeError(f"Foundry request failed: {e.reason}")


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def sim_to_geo(x, z):
    return {"lat": 34.65 + z / 1000.0, "lon": 43.90 + x / 1000.0}


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


@app.route("/api/drone-comms", methods=["POST"])
def drone_comms():
    data = request.json
    blue = data.get("blue", [])
    red = data.get("red", [])
    blue_hp = data.get("blueHP", 100)
    red_hp = data.get("redHP", 100)

    state_lines = []
    for i, d in enumerate(blue):
        state_lines.append(
            f"BLUE-{i+1} role={d['role']} pos=({d['x']:.0f},{d['z']:.0f}) alt={d['y']:.0f}m alive={d['alive']}"
        )
    for i, d in enumerate(red):
        state_lines.append(
            f"RED-{i+1} pos=({d['x']:.0f},{d['z']:.0f}) alt={d['y']:.0f}m alive={d['alive']}"
        )
    state_lines.append(f"BLUE_OUTPOST_HP={blue_hp:.0f}% RED_OUTPOST_HP={red_hp:.0f}%")
    state_str = "\n".join(state_lines)

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the internal reasoning engine of each drone in a coordinated swarm. "
                        "Output each drone's FULL THOUGHT PROCESS — like reading an AI agent's internal monologue.\n\n"
                        "Each drone should output 2-4 sentences covering:\n"
                        "1. SITUATIONAL AWARENESS: What do I see? Where are enemies? What are my teammates doing?\n"
                        "2. THREAT ASSESSMENT: What's dangerous? What's the priority?\n"
                        "3. COORDINATION LOGIC: How am I working with the other BLUE drones? Who needs help?\n"
                        "4. DECISION: What am I doing and WHY?\n\n"
                        "Be SPECIFIC with drone IDs, positions, distances, HP percentages.\n\n"
                        "Example output for one drone:\n"
                        '"SCAN: RED-1 at (80,0) is 65m east, RED-2 at (70,10) is 48m northeast. '
                        "BLUE-3 is engaging RED-2 solo — they're outnumbered. Outpost HP at 45%, need to press. "
                        "DECISION: Breaking from current vector to support BLUE-3's engagement on RED-2, "
                        'approaching from south to create crossfire angle."\n\n'
                        "Reply ONLY as a JSON array of strings, one per blue drone, in order."
                    ),
                },
                {"role": "user", "content": state_str},
            ],
            temperature=0.8,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        # Find the JSON array in the response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            comms = json.loads(raw[start:end])
            return jsonify(comms)
        # If no array found, split by newlines
        comms = [line.strip().strip('"').strip("'") for line in raw.split("\n") if line.strip()]
        return jsonify(comms[:len(blue)])
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        sys.stdout.flush()
        # Fallback: generate detailed local reasoning
        fallback = []
        alive_red = [r for r in red if r.get("alive")]
        for i, d in enumerate(blue):
            if not d.get("alive"):
                fallback.append("SYSTEMS OFFLINE. Hull integrity compromised. Initiating emergency restart sequence. Awaiting clearance to rejoin formation.")
            elif d.get("role") == "attack":
                nearest = min((((d['x']-r['x'])**2 + (d['z']-r['z'])**2)**0.5 for r in alive_red), default=999)
                fallback.append(
                    f"SCAN: {len(alive_red)} hostiles detected. Nearest threat {nearest:.0f}m away. "
                    f"Enemy outpost at {red_hp:.0f}% integrity. "
                    f"DECISION: {'Nearest hostile in range — engaging before pushing target.' if nearest < 40 else 'Clear path to objective — accelerating attack vector on outpost.'}"
                )
            else:
                threats_near = sum(1 for r in alive_red if ((d['x']-r['x'])**2 + (d['z']-r['z'])**2)**0.5 < 60)
                fallback.append(
                    f"SCAN: Monitoring perimeter. {threats_near} threats within defense zone. "
                    f"Friendly outpost at {blue_hp:.0f}% integrity. "
                    f"DECISION: {'Threats approaching — intercepting nearest hostile to protect base.' if threats_near > 0 else 'Sector clear — maintaining patrol orbit and scanning for incursions.'}"
                )
        return jsonify(fallback), 200


@app.route("/api/battle-stream", methods=["POST"])
def battle_stream():
    data = request.json
    blue = data.get("blue", [])
    red = data.get("red", [])
    blue_hp = data.get("blueHP", 100)
    red_hp = data.get("redHP", 100)

    state_lines = []
    for i, d in enumerate(blue):
        state_lines.append(f"BLUE-{i+1} role={d['role']} pos=({d['x']:.0f},{d['z']:.0f}) alt={d['y']:.0f}m alive={d['alive']}")
    for i, d in enumerate(red):
        state_lines.append(f"RED-{i+1} pos=({d['x']:.0f},{d['z']:.0f}) alt={d['y']:.0f}m alive={d['alive']}")
    state_lines.append(f"BLUE_OUTPOST_HP={blue_hp:.0f}% RED_OUTPOST_HP={red_hp:.0f}%")

    def generate():
        try:
            stream = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a tactical AI battle narrator providing REAL-TIME analysis of a drone swarm battle. "
                            "Write a LONG, detailed, streaming tactical analysis. Think like a military commander watching drone feeds.\n\n"
                            "Cover ALL of these in detail:\n"
                            "1. SITUATION OVERVIEW — what's happening right now across the entire battlefield\n"
                            "2. BLUE TEAM ANALYSIS — what each blue drone is doing, their coordination, formation assessment\n"
                            "3. RED TEAM THREATS — enemy positions, movement patterns, danger assessment\n"
                            "4. COORDINATION GRAPH — which drones are working together, communication links, who's supporting whom\n"
                            "5. TACTICAL ASSESSMENT — is the current strategy working? what should change?\n"
                            "6. PREDICTED OUTCOMES — what will happen in the next 10 seconds based on current trajectories\n"
                            "7. DECISION TREE — for each blue drone: IF [condition] THEN [action] ELSE [alternative]\n\n"
                            "Use specific positions, distances, drone IDs. Be verbose. Write at least 400 words. "
                            "Use headers like [SITUATION] [COORDINATION] [THREATS] etc. "
                            "This is meant to show the AI's full thinking process — don't hold back."
                        ),
                    },
                    {"role": "user", "content": "\n".join(state_lines)},
                ],
                temperature=0.85,
                max_tokens=1200,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'t': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'t': f'[ANALYSIS ERROR: {str(e)}]'})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/foundry/health")
def foundry_health():
    s = foundry_settings()
    return jsonify({"configured": foundry_ready(), "url": s["url"], "ontology_rid": s["ontology_rid"]})


@app.route("/api/foundry/mission", methods=["POST"])
def create_foundry_mission():
    data = request.json or {}
    mission_id = data.get("id", "mission")
    objective = data.get("objective", "Strike mission")
    try:
        geo = sim_to_geo(0, 0)
        result = apply_foundry_action("createKillChainEvent", {
            "deviceId": f"event-{mission_id}", "name": f"MISSION START | {mission_id} | {objective[:60]}",
            "latitude": geo["lat"], "longitude": geo["lon"], "altitudeM": 200,
            "speedKnots": 1, "courseDeg": 0, "fixQuality": 5, "numSatellites": 12, "timestamp": iso_now(),
        })
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/foundry/telemetry", methods=["POST"])
def foundry_telemetry():
    data = request.json or {}
    drones = data.get("drones", [])
    if not drones:
        return jsonify({"ok": True, "count": 0})
    try:
        results = []
        for d in drones[:20]:
            geo = sim_to_geo(d.get("x", 0), d.get("z", 0))
            results.append(apply_foundry_action("updateDroneTelemetry", {
                "deviceId": d.get("id", "drone"), "name": f"Drone {d.get('id', '?')}",
                "latitude": geo["lat"], "longitude": geo["lon"],
                "altitudeM": max(0, float(d.get("y", 0))), "speedKnots": 30.0,
                "courseDeg": 90.0, "fixQuality": 5, "numSatellites": 12, "timestamp": iso_now(),
            }))
        return jsonify({"ok": True, "count": len(results)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/foundry/killchain-event", methods=["POST"])
def foundry_killchain_event():
    data = request.json or {}
    mission_id = data.get("mission_id", "mission")
    node_name = data.get("node_name", "Event")
    action_taken = data.get("action_taken", "APPROVED")
    try:
        geo = sim_to_geo(0, 0)
        result = apply_foundry_action("createKillChainEvent", {
            "deviceId": f"event-{mission_id}", "name": f"{node_name} | {action_taken}",
            "latitude": geo["lat"], "longitude": geo["lon"], "altitudeM": 150,
            "speedKnots": 2, "courseDeg": 45, "fixQuality": 5, "numSatellites": 12, "timestamp": iso_now(),
        })
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
        resp = openai_client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You produce strict JSON for a drone mission planner."},
                {"role": "user", "content": f"""Given a mission objective, return JSON:
{{"nodes": ["mission-trigger","isr-phase","swarm-assignment","engagement-authorization","execute","battle-damage-assessment","report-summary"],
"plan": {{"target":{{"x":number,"z":number,"label":string}},"swarm_mode":"smart"|"dumb","formation":"swarm"|"delta"|"perimeter","reasoning":string}}}}
Rules: x,z between -220 and 220. If lethal, include engagement-authorization before execute. Return JSON only.
Mission: {objective}"""},
            ],
            max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        result = json.loads(raw)
        nodes = result.get("nodes", [])
        valid = ['mission-trigger','isr-phase','swarm-assignment','engagement-authorization','execute','battle-damage-assessment','report-summary']
        nodes = [n for n in nodes if n in valid]
        if 'execute' in nodes and 'engagement-authorization' not in nodes:
            nodes.insert(nodes.index('execute'), 'engagement-authorization')
        return jsonify({"nodes": nodes, "plan": result.get("plan", {})})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/flowchart")
def flowchart():
    return send_from_directory("static", "flowchart.html")


@app.route("/api/config")
def config():
    return jsonify({
        "mapbox_token": os.environ.get("MAPBOX_TOKEN", ""),
        "google_maps_key": os.environ.get("GOOGLE_MAPS_API_KEY", ""),
    })


@app.route("/api/splats")
def list_splats():
    splat_dir = os.path.join("static", "splats")
    if not os.path.isdir(splat_dir):
        return jsonify([])
    files = [f for f in os.listdir(splat_dir) if f.endswith(('.splat', '.ply', '.ksplat', '.spz'))]
    files.sort()
    return jsonify(files)


@app.route("/simulation")
def simulation():
    return send_from_directory("static", "simulation.html")


@app.route("/training")
def training():
    return send_from_directory("static", "training.html")


@app.route("/splat-test")
def splat_test():
    return send_from_directory("static", "splat-test.html")


if __name__ == "__main__":
    app.run(debug=True, port=5050)
