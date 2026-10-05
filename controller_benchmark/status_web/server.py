from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("BENCHMARK_ROOT", "/workspace-cache/controller-benchmarks"))
PRETRAINING_ROOT = Path(
    os.environ.get("PRETRAINING_STATUS_ROOT", "/workspace-cache/controller-pretraining-status")
)
DEFAULT_RUN_ID = os.environ.get("RUN_ID", "")
HOST = os.environ.get("STATUS_HOST", "0.0.0.0")
PORT = int(os.environ.get("STATUS_PORT", "8088"))
OFFLINE_REPLAY_ROOT = Path(
    os.environ.get("OFFLINE_REPLAY_ROOT", str(ROOT / "offline-replays"))
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def latest_run_id() -> str:
    if DEFAULT_RUN_ID:
        return DEFAULT_RUN_ID
    candidates = [p for p in ROOT.iterdir() if p.is_dir() and (p / "status.json").exists()]
    if not candidates:
        return ""
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1].name


def fmt_elapsed(seconds: Any) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "n/a"
    hours, rem = divmod(max(total, 0), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def fmt_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100.0:.2f} %".replace(".", ",")
    except (TypeError, ValueError):
        return "n/a"


def fmt_number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}".replace(".", ",")
    except (TypeError, ValueError):
        return str(value)


def fmt_wh(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f} Wh".replace(".", ",")
    except (TypeError, ValueError):
        return "n/a"


def fmt_pp(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100.0:.2f} pp".replace(".", ",")
    except (TypeError, ValueError):
        return "n/a"


def last_epoch(run_id: str, job_id: str | None) -> dict[str, Any] | None:
    if not job_id:
        return None
    path = ROOT / run_id / "metrics" / f"epoch_timeline_job_{job_id}.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return None
    latest: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_kind = event.get("event")
            if event_kind == "end" or (
                event_kind is None and event.get("epoch") is not None
            ):
                latest = event
    return latest


def _best_quality(summary: dict[str, Any]) -> float | None:
    quality_metric = str(summary.get("quality_metric") or "quality_score")
    for value in (
        summary.get("best_quality_score"),
        summary.get(f"best_{quality_metric}"),
        summary.get("final_best_map50_95"),
    ):
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    observed: list[float] = []
    for row in summary.get("epochs") or []:
        value = row.get("best_quality_score")
        if value is None:
            value = row.get(quality_metric)
        if value is None:
            value = row.get("quality_score")
        try:
            if value is not None:
                observed.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(observed) if observed else None


def _metric_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _controller_by_id(shadow: dict[str, Any], controller_id: str) -> dict[str, Any] | None:
    for controller in shadow.get("controllers") or []:
        if controller.get("controller_id") == controller_id:
            return controller
    return None


def _total_controller_energy(controller: dict[str, Any] | None) -> float | None:
    if not controller:
        return None
    value = controller.get("counterfactual_total_energy_wh")
    if value is None:
        training = controller.get("training_energy_to_stop_wh")
        overhead = controller.get("controller_overhead_energy_wh") or 0.0
        if training is None:
            return None
        value = float(training) + float(overhead)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def live_shadow_results_text(run_id: str, rows: list[dict[str, Any]]) -> str:
    result_rows: list[dict[str, Any]] = []
    metrics = ROOT / run_id / "metrics"
    for row in rows:
        if row.get("state") != "COMPLETED":
            continue
        job_id = str(row.get("job_id") or "")
        case_id = str(row.get("case_id") or "")
        if not job_id or not case_id:
            continue
        epoch_summary = _metric_json(metrics / f"epoch_summary_job_{job_id}.json")
        shadow = _metric_json(metrics / f"shadow_summary_job_{job_id}.json")
        if not epoch_summary or not shadow:
            continue
        standard = _controller_by_id(shadow, "standard-es-10")
        controller = _controller_by_id(shadow, "rapec-g-v4")
        if not standard or not controller:
            continue

        full_quality = _best_quality(epoch_summary)
        full_energy = None
        try:
            full_energy = float(epoch_summary.get("total_gpu_energy_kwh") or 0.0) * 1000.0
        except (TypeError, ValueError):
            full_energy = None
        if not full_energy:
            try:
                full_energy = float(shadow.get("full100_epoch_training_energy_wh") or 0.0)
            except (TypeError, ValueError):
                full_energy = None
        es_quality = standard.get("best_quality_at_stop")
        controller_quality = controller.get("best_quality_at_stop")
        es_energy = _total_controller_energy(standard)
        controller_energy = _total_controller_energy(controller)
        loss_full = (
            float(full_quality) - float(controller_quality)
            if full_quality is not None and controller_quality is not None
            else None
        )
        loss_es = (
            float(es_quality) - float(controller_quality)
            if es_quality is not None and controller_quality is not None
            else None
        )
        saving_full = (
            (float(full_energy) - float(controller_energy)) / float(full_energy)
            if full_energy and controller_energy is not None
            else None
        )
        saving_es = (
            (float(es_energy) - float(controller_energy)) / float(es_energy)
            if es_energy and controller_energy is not None
            else None
        )
        result_rows.append(
            {
                "Dataset": case_id,
                "Full100 Job": job_id,
                "ES Virtual Job": f"{job_id}-es",
                "V4 Virtual Job": f"{job_id}-v4",
                "Qualität Full": fmt_percent(full_quality),
                "Qualität ES": fmt_percent(es_quality),
                "Qualität Controller": fmt_percent(controller_quality),
                "Verlust vs Full": fmt_pp(loss_full),
                "Verlust vs ES": fmt_pp(loss_es),
                "Energie Full100": fmt_wh(full_energy),
                "Energie ES": fmt_wh(es_energy),
                "Energie Controller": fmt_wh(controller_energy),
                "Stop ES": str(standard.get("stop_epoch") or "n/a"),
                "Stop Controller": str(controller.get("stop_epoch") or "n/a"),
                "Ersparnis vs Full": fmt_percent(saving_full),
                "Ersparnis vs ES": fmt_percent(saving_es),
            }
        )
    if not result_rows:
        return "Live-Shadow Ergebnis-Tabelle: noch kein abgeschlossenes Dataset mit Shadow-Metriken vorhanden."

    columns = [
        "Dataset",
        "Full100 Job",
        "ES Virtual Job",
        "V4 Virtual Job",
        "Qualität Full",
        "Qualität ES",
        "Qualität Controller",
        "Verlust vs Full",
        "Verlust vs ES",
        "Energie Full100",
        "Energie ES",
        "Energie Controller",
        "Stop ES",
        "Stop Controller",
        "Ersparnis vs Full",
        "Ersparnis vs ES",
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in result_rows))
        for column in columns
    }
    lines = [
        "Live-Shadow Ergebnis-Tabelle: 1 realer Full100-Job + 2 virtuelle Jobs je Dataset",
        f"Representierte Jobs: {len(result_rows) * 3} ({len(result_rows)} real, {len(result_rows)} ES virtuell, {len(result_rows)} RAPEC-G v4 virtuell)",
        " ".join(f"{column:<{widths[column]}}" for column in columns),
        " ".join("-" * widths[column] for column in columns),
    ]
    for result in result_rows:
        lines.append(" ".join(f"{result[column]:<{widths[column]}}" for column in columns))
    return "\n".join(lines)


def offline_replay_results_text(run_id: str) -> str:
    candidates = [
        ROOT / run_id / "rapec-g-v5-48-offline-replay.txt",
        OFFLINE_REPLAY_ROOT / "rapec-g-v5-48-offline-replay.txt",
    ]
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path.read_text(encoding="utf-8-sig").rstrip()
        except OSError:
            continue
    return "Offline-Replay V4/V5: noch kein Bericht vorhanden."


def latest_pretraining_log() -> Path | None:
    if not PRETRAINING_ROOT.exists():
        return None
    candidates = [
        path
        for pattern in (
            "current.out",
            "hard-current.out",
            "expensive-current.out",
            "hardest-current.out",
            "pretrain-cross-domain-*.out",
        )
        for path in PRETRAINING_ROOT.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1]


def pretraining_status_text() -> str:
    path = latest_pretraining_log()
    if path is None:
        return "Pretraining : no active or mirrored pretraining log found."

    latest: dict[str, Any] | None = None
    completed_sources: set[str] = set()
    source_cases: set[str] = set()
    tail: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            tail.append(line)
            tail = tail[-8:]
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_case = str(row.get("source_case") or "")
            if source_case:
                source_cases.add(source_case)
            latest = row
            try:
                checkpoint = int(row.get("checkpoint"))
                total = int(row.get("pretraining_checkpoints"))
            except (TypeError, ValueError):
                continue
            if source_case and checkpoint >= total:
                completed_sources.add(source_case)

    lines = [
        "Pretraining Source Models",
        f"Log file              : {path.name}",
        f"Known source cases    : {len(source_cases)}",
        f"Completed source cases: {len(completed_sources)}",
    ]
    if latest:
        source_case = latest.get("source_case", "n/a")
        checkpoint = latest.get("checkpoint", "n/a")
        total = latest.get("pretraining_checkpoints", "n/a")
        metric = latest.get("raw_quality_metric") or "quality_score"
        raw_quality = latest.get("raw_quality_value", latest.get("quality_score"))
        quality = latest.get("quality_score")
        raw_text = fmt_percent(raw_quality) if latest.get("quality_transform") == "identity" else fmt_number(raw_quality)
        lines.extend(
            [
                f"Current source case  : {source_case}",
                f"Current checkpoint   : {checkpoint}/{total}",
                f"Raw quality ({metric}) : {raw_text}",
                f"Quality score        : {fmt_percent(quality)}",
                f"Train loss           : {fmt_number(latest.get('train_loss'))}",
                f"Learning rate        : {fmt_number(latest.get('learning_rate'))}",
            ]
        )
    lines.extend(["", "Recent log lines:", *tail])
    return "\n".join(lines)


def text_status(run_id: str) -> str:
    status_path = ROOT / run_id / "status.json"
    if not run_id or not status_path.exists():
        return "No benchmark status found.\n\n" + pretraining_status_text() + "\n"

    status = load_json(status_path)
    job_id = status.get("current_job_id")
    epoch = last_epoch(run_id, str(job_id) if job_id else None)
    quality = None
    map50 = None
    last_epoch_number: Any = "n/a"
    if epoch:
        quality = epoch.get("quality_score", epoch.get("map50_95"))
        map50 = epoch.get("map50")
        last_epoch_number = epoch.get("epoch", "n/a")

    rows = status.get("runs", []) or []
    sequence_w = max([len("sequence")] + [len(str(r.get("sequence", ""))) for r in rows])
    case_w = max([len("case_id")] + [len(str(r.get("case_id", ""))) for r in rows])
    strategy_w = max([len("strategy")] + [len(str(r.get("strategy", ""))) for r in rows])
    job_w = max([len("job_id")] + [len(str(r.get("job_id", ""))) for r in rows])
    state_w = max([len("state")] + [len(str(r.get("state", ""))) for r in rows])

    table_header = (
        f"{'sequence':>{sequence_w}} {'case_id':<{case_w}} "
        f"{'strategy':<{strategy_w}} {'job_id':>{job_w}} {'state':<{state_w}}"
    )
    table_rule = (
        f"{'-' * sequence_w:>{sequence_w}} {'-' * case_w:<{case_w}} "
        f"{'-' * strategy_w:<{strategy_w}} {'-' * job_w:>{job_w}} {'-' * state_w:<{state_w}}"
    )
    table_lines = [table_header, table_rule]
    for row in rows:
        table_lines.append(
            f"{str(row.get('sequence', '')):>{sequence_w}} "
            f"{str(row.get('case_id', '')):<{case_w}} "
            f"{str(row.get('strategy', '')):<{strategy_w}} "
            f"{str(row.get('job_id', '')):>{job_w}} "
            f"{str(row.get('state', '')):<{state_w}}"
        )

    error = status.get("error") or ""
    current_seconds = status.get("current_job_running_seconds")
    if current_seconds is None and rows:
        current = next((r for r in rows if str(r.get("job_id")) == str(job_id)), None)
        submitted_at = current.get("submitted_at") if current else None
        if submitted_at:
            try:
                submitted = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
                current_seconds = (datetime.now(timezone.utc) - submitted).total_seconds()
            except ValueError:
                current_seconds = None

    benchmark_text = "\n".join(
        [
            f"RunId       : {status.get('benchmark_run_id', run_id)}",
            f"State       : {status.get('state', '')}",
            f"Progress    : {status.get('completed_runs', 0)}/{status.get('total_runs', 0)}",
            f"CurrentJob  : {status.get('current_job_id') or ''}",
            f"CurrentCase : {status.get('current_case_id') or ''}",
            f"Updated     : {status.get('updated_at') or ''}",
            f"Error       : {error}",
            "",
            "",
            f"Laufzeit               : {fmt_elapsed(current_seconds)}",
            f"Aktuelle letzte Epoche : {last_epoch_number}",
            f"Quality / mAP50-95     : {fmt_percent(quality)}",
            f"mAP50                  : {fmt_percent(map50)}",
            "",
            "",
            *table_lines,
            "",
            live_shadow_results_text(run_id, rows),
            "",
            offline_replay_results_text(run_id),
            "",
        ]
    )
    return benchmark_text + "\n" + pretraining_status_text() + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        run_id = self.path.split("runId=", 1)[1].split("&", 1)[0] if "runId=" in self.path else latest_run_id()
        body = text_status(run_id)
        if self.path.startswith("/api/status"):
            payload = {"runId": run_id, "text": body}
            data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.startswith("/status.txt"):
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        escaped = html.escape(body)
        page = f"""<!doctype html>
<html lang=\"de\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Controller Benchmark Status</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; background: #111827; color: #e5e7eb; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
    main {{ padding: 18px; max-width: 1800px; margin: 0 auto; }}
    h1 {{ font-size: 18px; margin: 0 0 14px; color: #93c5fd; }}
    pre {{ white-space: pre; overflow-x: auto; background: #020617; border: 1px solid #334155; border-radius: 14px; padding: 16px; line-height: 1.42; font-size: 13px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 10px; }}
    .hint {{ color: #94a3b8; font-size: 12px; }}
    button {{ cursor: pointer; border: 1px solid #475569; border-radius: 999px; padding: 6px 12px; background: #0f766e; color: #f8fafc; font: inherit; font-size: 12px; }}
    button[aria-pressed=\"false\"] {{ background: #334155; color: #cbd5e1; }}
    #refresh-status {{ min-width: 112px; color: #94a3b8; font-size: 12px; }}
  </style>
</head>
<body>
<main>
  <h1>Controller Benchmark Status</h1>
  <div class=\"toolbar\">
    <button id=\"refresh-toggle\" type=\"button\" aria-pressed=\"true\">Auto-refresh: ON</button>
    <span id=\"refresh-status\">Refresh in 15 s</span>
    <span class=\"hint\">Textansicht: <a style=\"color:#93c5fd\" href=\"/status.txt?runId={html.escape(run_id)}\">/status.txt</a></span>
  </div>
  <pre>{escaped}</pre>
</main>
<script>
  (() => {{
    const storageKey = "controller-benchmark-auto-refresh";
    const refreshSeconds = 15;
    const button = document.getElementById("refresh-toggle");
    const status = document.getElementById("refresh-status");
    let enabled = true;
    let remaining = refreshSeconds;

    try {{
      enabled = localStorage.getItem(storageKey) !== "off";
    }} catch (_error) {{
      enabled = true;
    }}

    const render = () => {{
      button.textContent = `Auto-refresh: ${{enabled ? "ON" : "OFF"}}`;
      button.setAttribute("aria-pressed", String(enabled));
      status.textContent = enabled ? `Refresh in ${{remaining}} s` : "Refresh paused";
    }};

    button.addEventListener("click", () => {{
      enabled = !enabled;
      remaining = refreshSeconds;
      try {{
        localStorage.setItem(storageKey, enabled ? "on" : "off");
      }} catch (_error) {{
        // Auto-refresh still works for this tab if browser storage is unavailable.
      }}
      render();
    }});

    window.setInterval(() => {{
      if (!enabled) return;
      remaining -= 1;
      if (remaining <= 0) {{
        window.location.reload();
        return;
      }}
      render();
    }}, 1000);

    render();
  }})();
</script>
</body>
</html>"""
        data = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"controller benchmark status web listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()
