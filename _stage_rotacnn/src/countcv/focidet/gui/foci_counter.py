import json
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
from plyer import filechooser

from countcv.focidet.main import main

# Page Config (MUST be first Streamlit command)
st.set_page_config(page_title="FOCI Counter", layout="centered")

# Constants
DEFAULT_IMG_DIR = str((Path(".") / "data" / "uc_cells" / "images" / "test").absolute())
MODEL_MAP = {
	"Both": "both",
	"YOLO": "yolo",
	"Densitymap": "density",
}
SETTINGS_FILE = Path("src") / "countcv" / "focidet" / "gui" / "foci_counter_settings.json"
PERSISTENT_KEYS = ["img_dir", "output_dir", "model_label", "format", "write_images"]


# Logging
class StreamlitLogHandler(logging.Handler):
	def __init__(self, placeholder):
		super().__init__()
		self.placeholder = placeholder

	def emit(self, record):
		msg = self.format(record)
		st.session_state.log_output += msg + "\n"
		self.placeholder.code(st.session_state.log_output)


def setup_logger(placeholder, filename: Path):
	logger = logging.getLogger("countcv")
	logger.setLevel(logging.INFO)
	logger.propagate = False
	if not logger.handlers:
		if filename:
			handler_file = logging.FileHandler(filename)
			handler_file.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
			logger.addHandler(handler_file)
		handler = StreamlitLogHandler(placeholder)
		handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
		logger.addHandler(handler)
	return logger


def clear_logger():
	logging.getLogger("countcv").handlers.clear()


# Persistence helpers
def load_settings() -> dict:
	"""Load settings from JSON file, return empty dict if missing or corrupt."""
	if SETTINGS_FILE.exists():
		try:
			return json.loads(SETTINGS_FILE.read_text())
		except (json.JSONDecodeError, OSError):
			return {}
	return {}


def save_settings():
	"""Write the persistent session state keys to the JSON file."""
	data = {k: st.session_state[k] for k in PERSISTENT_KEYS if k in st.session_state}
	SETTINGS_FILE.write_text(json.dumps(data, indent=2))


# Callbacks
def browse_folder(state_key):
	result = filechooser.choose_dir()
	if result:
		st.session_state[state_key] = result[0]
		save_settings()


def start_run():
	st.session_state.running = True
	st.session_state.stop_flag = False
	st.session_state.run_status = None
	st.session_state.log_output = ""
	st.session_state.error_message = ""


def request_stop():
	st.session_state.stop_flag = True


def on_change_save(key):
	"""Generic on_change callback that persists settings."""
	save_settings()


# UI Helpers
def directory_row(label, key, help: str | None = None):
	col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
	col1.text_input(label, key=key, help=help, on_change=save_settings)
	col2.button("Browse", key=f"{key}_browse", on_click=browse_folder, args=(key,))


def init_logo():
	st.logo("src/countcv/focidet/gui/static/logo/uba_ki_logo.svg", size="large")


# Session State Initialization
def init_session_state():
	saved = load_settings()
	defaults = {
		"running": False,
		"stop_flag": False,
		"run_status": None,
		"log_output": "",
		"output_dir": "",
		"img_dir": DEFAULT_IMG_DIR,
		"processed_count": 0,
		"error_message": "",
		# Persistent UI defaults — overridden by saved settings below
		"model_label": "Both",
		"format": "german",
		"write_images": False,
	}
	for k, v in defaults.items():
		if k not in st.session_state:
			# Use saved value if available, otherwise fall back to default
			st.session_state[k] = saved.get(k, v)


# Start
init_session_state()
init_logo()
# UI
st.title("FOCI Counter")

directory_row("Image Directory", "img_dir", help="Directory containing the images to be processed.")
directory_row(
	"Output Directory",
	"output_dir",
	"Directory where you want to see the results file. A new sub-directory will be created there with a timestamp",
)

model_label = st.radio(
	"Models",
	list(MODEL_MAP.keys()),
	horizontal=True,
	help="Select the model to use for counting. Details can be found in the wiki.",
	on_change=save_settings,
	key="model_label",
)
model_arg = MODEL_MAP[model_label]

format = st.radio(
	"Excel Format",
	["german", "english"],
	horizontal=True,
	help=(
		"Switch between German and English formats. German format uses comma as decimal separator, "
		"English format uses dot."
	),
	on_change=save_settings,
	key="format",
)

write_images = st.toggle(
	"Write result images",
	key="write_images",
	help=(
		"Toggle to write result images to disk. This is helpful for debugging and visual inspection but takes "
		"additional time and disk space."
	),
	on_change=save_settings,
)

# Run / Stop button
if st.session_state.running:
	st.button("Stop", type="secondary", on_click=request_stop)
else:
	st.button("Run", type="primary", on_click=start_run)

# Progress Section
progress_placeholder = st.empty()
if st.session_state.run_status == "done":
	progress_placeholder.progress(
		1.0,
		text=f"✅ Done! Processed {st.session_state.processed_count} images",
	)
elif st.session_state.run_status == "stopped":
	progress_placeholder.progress(0, text="⏹ Stopped by user.")
elif st.session_state.run_status == "error":
	progress_placeholder.progress(
		0,
		text=f"❌ Error: {st.session_state.error_message}",
	)
elif st.session_state.running:
	progress_bar = progress_placeholder.progress(0, text="Starting…")
else:
	progress_placeholder.empty()

# Log Output
st.subheader("Log")
log_placeholder = st.empty()
log_placeholder.code(st.session_state.log_output)


# Execution Block
def update_progress(current, total):
	if st.session_state.stop_flag:
		raise InterruptedError("Stopped by user.")
	progress_bar.progress(
		current / total,
		text=f"Processing {current}/{total}…",
	)


out_path = Path(st.session_state.output_dir) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

if st.session_state.running:
	out_path.mkdir(parents=True, exist_ok=True)
	logger = setup_logger(log_placeholder, out_path / "foci_counter_log.log")
	try:
		counts = main(
			out_path,
			st.session_state.img_dir,
			"checkpoints",
			write_images,
			model_arg,
			progress_callback=update_progress,
			format=format,
		)
		st.session_state.run_status = "done"
		st.session_state.processed_count = len(counts)
	except InterruptedError:
		st.session_state.run_status = "stopped"
		st.session_state.log_output += "Run has been stopped.\n"
	except Exception as e:
		logger.error(str(e), exc_info=True)
		st.session_state.run_status = "error"
		st.session_state.error_message = str(e)
	finally:
		st.session_state.running = False
		clear_logger()
		st.rerun()
