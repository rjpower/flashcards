import logging
from functools import wraps
from pathlib import Path

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
    return render_template("upload_srt.html", active_tab="srt")


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

    def generate():
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
                include_audio=bool(request.form.get("include_audio")),
                deck_name=clean_name,
                ignore_words=ignore_words,
            )

            for progress in process_srt(config):
                yield f"data: {progress.model_dump_json()}\n\n"

        finally:
            srt_path.unlink(missing_ok=True)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


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
        context_jp=request.form.get("context_jp_field"),
        context_en=request.form.get("context_en_field"),
        level=request.form.get("level_field"),
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
    output_path = output_dir / f"{clean_name}.{output_format}"

    config = CSVProcessConfig(
        df=df,
        output_path=output_path,
        output_format=OutputFormat(output_format),
        field_mapping=mapping,
        include_audio=bool(request.form.get("include_audio")),
        deck_name=clean_name,
        ignore_words=ignore_words,
    )

    def generate():
        for progress in process_csv(config):
            yield f"data: {progress.model_dump_json()}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


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
    app.run(debug=True, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
