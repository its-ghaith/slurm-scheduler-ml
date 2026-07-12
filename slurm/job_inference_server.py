import base64
import cgi
import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from ultralytics import YOLO


LOGGER = logging.getLogger("job_inference_server")
MODEL_CACHE: dict[str, YOLO] = {}
REGISTRY_ROOT = Path(os.environ.get("JOB_MODEL_REGISTRY_DIR", "/workspace-cache/model_registry"))
HOST = os.environ.get("INFERENCE_HOST", "0.0.0.0")
PORT = int(os.environ.get("INFERENCE_PORT", "7860"))


def _json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, body: str, status: int = 200, content_type: str = "text/html; charset=utf-8"):
    payload = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(payload)


def _registry_dir(job_id: str) -> Path:
    return REGISTRY_ROOT / f"job_{job_id}"


def _metadata_for_job(job_id: str) -> dict | None:
    metadata_path = _registry_dir(job_id) / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _available_jobs() -> list[dict]:
    jobs: list[dict] = []
    if not REGISTRY_ROOT.exists():
        return jobs
    for item in sorted(REGISTRY_ROOT.glob("job_*")):
        metadata = {}
        metadata_path = item / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        jobs.append(
            {
                "job_id": item.name.removeprefix("job_"),
                "model_path": str(item / "best.pt"),
                "has_model": (item / "best.pt").exists(),
                "metadata": metadata,
            }
        )
    return jobs


def _load_model(job_id: str) -> YOLO:
    if job_id in MODEL_CACHE:
        return MODEL_CACHE[job_id]

    model_path = _registry_dir(job_id) / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Kein exportiertes Modell fuer Job {job_id} gefunden unter {model_path}.")

    model = YOLO(str(model_path))
    MODEL_CACHE[job_id] = model
    return model


def _predict(job_id: str, image_bytes: bytes) -> dict:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Bild konnte nicht gelesen werden.")

    model = _load_model(job_id)
    results = model.predict(source=image, verbose=False, save=False, conf=0.25, max_det=500)
    result = results[0]
    boxes = result.boxes
    count = int(len(boxes)) if boxes is not None else 0
    confidences = []
    if boxes is not None and boxes.conf is not None:
        confidences = [round(float(x), 4) for x in boxes.conf.detach().cpu().tolist()]

    plotted = result.plot()
    ok, encoded = cv2.imencode(".png", plotted)
    if not ok:
        raise RuntimeError("Annotiertes Bild konnte nicht kodiert werden.")

    metadata = _metadata_for_job(job_id) or {}
    return {
        "job_id": job_id,
        "prediction_count": count,
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "confidences": confidences,
        "metadata": metadata,
        "annotated_image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def _ui_html(job_id: str | None) -> str:
    preselected = job_id or ""
    return f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SLURM Job Inference</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --card: #fffdf7;
      --ink: #1f2a2a;
      --accent: #0f766e;
      --accent-2: #d97706;
      --line: #d8cfbf;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Trebuchet MS", serif;
      background: linear-gradient(135deg, #f8f4ea, #efe5d2 55%, #f5f1e8);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 20px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 20px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 28px rgba(31, 42, 42, 0.08);
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: 28px;
    }}
    .muted {{
      color: #5d685f;
      font-size: 14px;
      line-height: 1.5;
    }}
    label {{
      display: block;
      font-weight: 700;
      margin: 14px 0 8px;
    }}
    select, input[type=file], button {{
      width: 100%;
      box-sizing: border-box;
      border-radius: 12px;
      border: 1px solid #b9c4bb;
      padding: 12px 14px;
      font-size: 15px;
      background: white;
    }}
    button {{
      margin-top: 16px;
      border: 0;
      background: linear-gradient(135deg, var(--accent), #155e75);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{
      filter: brightness(1.05);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fffaf0;
    }}
    .stat .k {{
      font-size: 12px;
      color: #667066;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .stat .v {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .imgbox {{
      border: 1px dashed #b8aa8a;
      border-radius: 18px;
      min-height: 420px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      background: repeating-linear-gradient(
        45deg,
        #faf7ef,
        #faf7ef 12px,
        #f5efdf 12px,
        #f5efdf 24px
      );
    }}
    .imgbox img {{
      width: 100%;
      height: auto;
      display: none;
    }}
    .meta {{
      margin-top: 16px;
      font-size: 14px;
      line-height: 1.6;
      white-space: pre-wrap;
      background: #f7f5ee;
      border-radius: 14px;
      padding: 12px;
      border: 1px solid var(--line);
    }}
    .status {{
      margin-top: 12px;
      color: var(--accent-2);
      min-height: 20px;
      font-weight: 700;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .stats {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Job Inference Playground</h1>
    <div class="grid">
      <div class="card">
        <div class="muted">Waehle einen trainierten SLURM-Job, lade ein Bild hoch und lasse genau das Modell dieses Jobs auf dem Bild inferieren.</div>
        <label for="jobId">Job</label>
        <select id="jobId"></select>
        <label for="imageFile">Bild</label>
        <input id="imageFile" type="file" accept="image/*" />
        <button id="runBtn">Vorhersage starten</button>
        <div class="status" id="status"></div>
      </div>
      <div class="card">
        <div class="stats">
          <div class="stat"><div class="k">Job</div><div class="v" id="jobStat">-</div></div>
          <div class="stat"><div class="k">Autos erkannt</div><div class="v" id="countStat">-</div></div>
          <div class="stat"><div class="k">Ø Confidence</div><div class="v" id="confStat">-</div></div>
        </div>
        <div class="imgbox"><img id="resultImage" alt="Inference Ergebnis" /><div id="placeholder">Noch keine Vorhersage.</div></div>
        <div class="meta" id="metaBox">Noch keine Metadaten.</div>
      </div>
    </div>
  </div>
  <script>
    const preselectedJob = {json.dumps(preselected)};
    const statusEl = document.getElementById('status');
    const jobEl = document.getElementById('jobId');
    const imgEl = document.getElementById('resultImage');
    const placeholderEl = document.getElementById('placeholder');
    const countStat = document.getElementById('countStat');
    const confStat = document.getElementById('confStat');
    const jobStat = document.getElementById('jobStat');
    const metaBox = document.getElementById('metaBox');

    async function loadJobs() {{
      const res = await fetch('/jobs');
      const jobs = await res.json();
      jobEl.innerHTML = '';
      jobs.forEach((job) => {{
        const opt = document.createElement('option');
        opt.value = job.job_id;
        const modelVersion = job.metadata && job.metadata.model_version ? ` - ${job.metadata.model_version}` : '';
        opt.textContent = `Job ${job.job_id}${modelVersion}`;
        if (job.job_id === preselectedJob) opt.selected = true;
        jobEl.appendChild(opt);
      }});
      if (!jobEl.value && jobEl.options.length > 0) {{
        jobEl.options[0].selected = true;
      }}
      if (jobEl.options.length === 0) {{
        statusEl.textContent = 'Noch keine exportierten Modelle vorhanden.';
      }}
    }}

    async function runPrediction() {{
      const file = document.getElementById('imageFile').files[0];
      const jobId = jobEl.value;
      if (!jobId) {{
        statusEl.textContent = 'Bitte zuerst einen Job waehlen.';
        return;
      }}
      if (!file) {{
        statusEl.textContent = 'Bitte zuerst ein Bild auswaehlen.';
        return;
      }}
      statusEl.textContent = 'Vorhersage laeuft ...';
      const formData = new FormData();
      formData.append('image', file);
      const res = await fetch(`/predict?job_id=${{encodeURIComponent(jobId)}}`, {{
        method: 'POST',
        body: formData
      }});
      const payload = await res.json();
      if (!res.ok) {{
        statusEl.textContent = payload.error || 'Vorhersage fehlgeschlagen.';
        return;
      }}
      imgEl.src = `data:image/png;base64,${{payload.annotated_image_base64}}`;
      imgEl.style.display = 'block';
      placeholderEl.style.display = 'none';
      countStat.textContent = payload.prediction_count;
      confStat.textContent = payload.avg_confidence == null ? '-' : payload.avg_confidence;
      jobStat.textContent = payload.job_id;
      metaBox.textContent = JSON.stringify(payload.metadata, null, 2);
      statusEl.textContent = 'Vorhersage abgeschlossen.';
    }}

    document.getElementById('runBtn').addEventListener('click', runPrediction);
    loadJobs().catch((err) => {{
      statusEl.textContent = 'Jobs konnten nicht geladen werden: ' + err;
    }});
  </script>
</body>
</html>"""


class InferenceHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            _json_response(self, {"status": "ok", "jobs": len(_available_jobs())})
            return
        if parsed.path == "/jobs":
            _json_response(self, _available_jobs())
            return
        if parsed.path == "/ui" or parsed.path == "/":
            query = parse_qs(parsed.query)
            job_id = query.get("job_id", [None])[0]
            _text_response(self, _ui_html(job_id))
            return
        _json_response(self, {"error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            _json_response(self, {"error": "Not found"}, status=404)
            return

        query = parse_qs(parsed.query)
        job_id = query.get("job_id", [None])[0]
        if not job_id:
            _json_response(self, {"error": "job_id fehlt."}, status=400)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        file_item = form["image"] if "image" in form else None
        if file_item is None or not getattr(file_item, "file", None):
            _json_response(self, {"error": "Bilddatei fehlt."}, status=400)
            return

        try:
            payload = _predict(job_id, file_item.file.read())
            _json_response(self, payload)
        except FileNotFoundError as exc:
            _json_response(self, {"error": str(exc)}, status=404)
        except Exception as exc:
            LOGGER.exception("Vorhersage fehlgeschlagen")
            _json_response(self, {"error": str(exc)}, status=500)

    def log_message(self, fmt: str, *args):
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), InferenceHandler)
    LOGGER.info("Inference server listening on %s:%s", HOST, PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
