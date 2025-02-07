import hashlib
import logging
import threading
import time
import trace
import traceback
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import List, Optional

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename

from srt.config import settings
from srt.lib import (
    CSVProcessConfig,
    OutputFormat,
    SourceMapping,
    SRTProcessConfig,
    clean_filename,
    infer_field_mapping,
    process_csv,
    process_srt,
    read_csv,
)
from srt.schema import ConversionProgress, ConversionStatus

ROOT = Path(__file__).parent.parent.absolute()

app = Flask(__name__, template_folder="srt/templates", root_path=str(ROOT))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1000 * 1000

limiter = Limiter(
    app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"]
)

# logging with filename, function, linenumber etc
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class ProcessingTracker:
    """Tracks the progress and state of a processing job"""

    progress_queue: List[ConversionProgress] = field(default_factory=list)
    error: Optional[str] = None
    output_filename: Optional[str] = None
    is_complete: bool = False
    _thread: Optional[threading.Thread] = None

    def log_progress(self, message: str):
        """Add a progress message to the queue"""
        progress = ConversionProgress(message=message, status=ConversionStatus.RUNNING)
        self.progress_queue.append(progress)

    def set_error(self, error: str):
        """Set error state"""
        self.is_complete = True
        progress = ConversionProgress(message=str(error), status=ConversionStatus.ERROR)
        self.progress_queue.append(progress)

    def set_complete(self, output_filename: str):
        """Set completion state"""
        self.output_filename = output_filename
        self.is_complete = True
        progress = ConversionProgress(
            message=f"Processing complete: {output_filename}",
            status=ConversionStatus.DONE,
            payload=output_filename,
        )
        self.progress_queue.append(progress)

    def start_processing(self, target, *args, **kwargs):
        """Start processing in a thread"""
        self._thread = threading.Thread(target=target, args=args, kwargs=kwargs)
        self._thread.start()

    def generate_events(self):
        """Generate SSE events for progress updates"""
        while not self.is_complete or self.progress_queue:
            while self.progress_queue:
                progress = self.progress_queue.pop(0)
                yield f"data: {progress.model_dump_json()}\n\n"
            time.sleep(0.1)

    def stream_response(self):
        """Create a streaming response for progress updates"""
        return Response(
            stream_with_context(self.generate_events()), mimetype="text/event-stream"
        )


def json_error(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return decorated_function


@app.route("/")
def index():
    return render_template("upload_csv.html", active_tab="csv")


@app.route("/srt")
def srt_upload():
    return render_template("upload_srt.html", active_tab="srt")


@app.route("/csv")
def csv_upload():
    return render_template("upload_csv.html", active_tab="csv")


@app.route("/upload/srt", methods=["POST"])
@limiter.limit("10 per minute")
@json_error
def upload_srt():
    """Handle SRT file upload and generate deck"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filter_file = request.files.get("filter_file")
    output_format = request.form.get("format", "apkg")

    if not file or not file.filename:
        return jsonify({"error": "Invalid file"}), 400

    ignore_words = set()
    if filter_file and filter_file.filename:
        ignore_words = {
            line.strip()
            for line in filter_file.read().decode("utf-8").splitlines()
            if line.strip()
        }

    logging.info("Filtering %d records.", len(ignore_words))

    file_content = file.read()
    upload_dir = settings.upload_dir
    upload_dir.mkdir(exist_ok=True)

    filename = secure_filename(file.filename)
    filename = f"{filename}_{str(hash(file_content))}.srt"

    srt_path = upload_dir / filename
    srt_path = srt_path.absolute()
    srt_path.write_bytes(file_content)
    tracker = ProcessingTracker()

    include_audio = bool(request.form.get("include_audio"))

    def _process_srt():
        try:
            output_dir = settings.output_dir
            output_dir.mkdir(exist_ok=True, parents=True)

            clean_name = clean_filename(filename)
            ext = "apkg" if output_format == "apkg" else "pdf"
            output_path = output_dir / f"{clean_name}.{ext}"

            config = SRTProcessConfig(
                srt_path=srt_path,
                output_path=output_path,
                output_format=OutputFormat(output_format),
                include_audio=include_audio,
                deck_name=clean_name,
                ignore_words=ignore_words,
                progress_logger=tracker.log_progress,
            )

            process_srt(config)
            tracker.set_complete(output_path.name)
        except Exception as e:
            stack = traceback.format_exc()
            tracker.set_error(stack)
        finally:
            srt_path.unlink(missing_ok=True)

    tracker.start_processing(_process_srt)
    return tracker.stream_response()


@app.route("/csv/analyze", methods=["POST"])
@json_error
def analyze_csv():
    """Analyze CSV structure and suggest field mappings"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Invalid file"}), 400

    preview_data = file.stream.read().decode("utf-8")
    file.stream.seek(0)  # Reset stream position

    separator, df = read_csv(preview_data)
    suggestions = infer_field_mapping(df)

    df = df.dropna(axis="columns")

    return jsonify(
        {
            "headers": df.columns.tolist(),
            "preview_rows": df.head(5).to_dict(orient="records"),
            "separator": separator,
            "suggestions": suggestions,
        }
    )


@app.route("/upload/csv", methods=["POST"])
@json_error
def upload_csv():
    """Handle CSV file upload with field mapping"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filter_file = request.files.get("filter_file")
    if not file or not file.filename:
        return jsonify({"error": "Invalid file"}), 400

    ignore_words = set()
    if filter_file and filter_file.filename:
        ignore_words = {
            line.strip()
            for line in filter_file.read().decode("utf-8").splitlines()
            if line.strip()
        }

    logging.info("Filtering %d records.", len(ignore_words))

    # Get field mappings from form
    mapping = SourceMapping(
        term=request.form["term_field"],
        reading=request.form["reading_field"],
        meaning=request.form["meaning_field"],
        context_native=request.form.get("context_native_field"),
        context_en=request.form.get("context_en_field"),
    )

    output_format = request.form.get("format", "apkg")
    separator = request.form.get("separator", ",")

    # Read CSV data into DataFrame
    file_content = file.read().decode("utf-8")
    separator, df = read_csv(file_content)

    # Process CSV with config
    output_dir = settings.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    clean_name = clean_filename(secure_filename(file.filename))
    nonce = hashlib.md5(file_content.encode("utf-8")).hexdigest()
    output_path = output_dir / f"{clean_name}-{nonce}.{output_format}"

    tracker = ProcessingTracker()
    config = CSVProcessConfig(
        df=df,
        output_path=output_path,
        output_format=OutputFormat(output_format),
        field_mapping=mapping,
        include_audio=bool(request.form.get("include_audio")),
        deck_name=clean_name,
        ignore_words=ignore_words,
        progress_logger=tracker.log_progress,
    )

    def _process_csv():
        try:
            process_csv(config)
            tracker.set_complete(output_path.name)
        except Exception as e:
            tracker.set_error(traceback.format_exc())

    tracker.start_processing(_process_csv)
    return tracker.stream_response()


@app.route("/download/<filename>")
def download(filename):
    """Download generated deck file"""
    output_dir = settings.output_dir
    file_path = output_dir / secure_filename(filename)
    logging.info("Downloading file: %s", file_path)

    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path, as_attachment=True, download_name=filename)


def main():
    app.run(debug=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
