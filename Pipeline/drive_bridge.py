"""
drive_bridge.py — Google Drive bridge for local Streamlit ↔ Colab communication.

Environment auto-detection:
  - Colab: uses google.colab.auth (no credentials file needed)
  - Local: uses OAuth 2.0 with credentials.json + token.json (stored repo-adjacent)

All Drive folder paths imported from config.py — no hardcoded values.
"""

from __future__ import annotations

import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Environment detection
# ═══════════════════════════════════════════════════════════════════

def _is_colab() -> bool:
    try:
        import google.colab
        return True
    except ImportError:
        return False


if _is_colab():
    from google.colab import auth as _colab_auth
    from google.colab import drive as _colab_drive
    from googleapiclient.discovery import build as _build_service
    from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
    _COLAB = True
else:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build as _build_service
    from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
    _COLAB = False

_SCOPES = ["https://www.googleapis.com/auth/drive"]
_SERVICE: Any = None
_FOLDER_CACHE: dict[str, str] = {}  # logical path → Drive folder ID


def _is_ssl_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in ("ssl", "record layer", "broken pipe",
                                 "connection reset", "eof occurred", "connection aborted"))


# ═══════════════════════════════════════════════════════════════════
# Authentication
# ═══════════════════════════════════════════════════════════════════

def authenticate() -> Any:
    """Authenticate and return a Google Drive service object. Idempotent."""
    global _SERVICE
    if _SERVICE is not None:
        return _SERVICE

    if _COLAB:
        _colab_auth.authenticate_user()
        _colab_drive.mount("/content/drive")
        _SERVICE = _build_service("drive", "v3")
        logger.info("Drive bridge: authenticated via Colab")
        return _SERVICE

    # ── Local OAuth 2.0 flow ──
    # Search for credentials.json: repo root, Politakis/, or cwd
    creds: Credentials | None = None
    creds_path: Path | None = None

    for candidate in (
        Path(__file__).parent,                     # Politakis/
        Path(__file__).parent / "credentials.json",
        Path("credentials.json"),
        Path("Politakis") / "credentials.json",
    ):
        if candidate.is_file():
            creds_path = candidate
            break
        elif (candidate / "credentials.json").is_file():
            creds_path = candidate / "credentials.json"
            break

    if creds_path is None and Path("credentials.json").is_file():
        creds_path = Path("credentials.json")
    if creds_path is None and (Path(__file__).parent / "credentials.json").is_file():
        creds_path = Path(__file__).parent / "credentials.json"

    if creds_path is None:
        raise FileNotFoundError(
            "credentials.json not found. "
            "Place it in the repo root, Politakis/, or alongside drive_bridge.py."
        )

    # Store token.json next to credentials.json to avoid leaking into repo root
    token_path = creds_path.parent / "token.json"

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)
        logger.info("Drive bridge: loaded cached token from %s", token_path)
        # Hard-return guard: if the cached token is valid, never fall through
        # to the browser OAuth flow. This prevents a new tab from opening on
        # every Streamlit rerun cycle.
        if creds and creds.valid:
            _SERVICE = _build_service("drive", "v3", credentials=creds)
            return _SERVICE

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            logger.info("Drive bridge: refreshed expired token")
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(creds_path), _SCOPES
            )
            creds = flow.run_local_server(port=0)
            logger.info("Drive bridge: completed OAuth flow")

        token_path.write_text(creds.to_json())
        logger.info("Drive bridge: token saved to %s", token_path)

    _SERVICE = _build_service("drive", "v3", credentials=creds)
    return _SERVICE


# ═══════════════════════════════════════════════════════════════════
# Folder helpers
# ═══════════════════════════════════════════════════════════════════

def get_or_create_folder(name: str, parent_id: str | None = None) -> str:
    """Return the Drive folder ID for *name*, creating it if it doesn't exist.
    Supports nested paths like 'ece22073/input' — traverses level by level.
    Retries once after resetting the service on transient SSL/network errors.
    """
    global _SERVICE
    for attempt in range(2):
        try:
            service = authenticate()
            parts = [p for p in name.split("/") if p]
            cur_parent = parent_id

            for part in parts:
                query = (
                    f"mimeType='application/vnd.google-apps.folder' "
                    f"and name='{part}' and trashed=false"
                )
                if cur_parent:
                    query += f" and '{cur_parent}' in parents"

                results = (
                    service.files()
                    .list(q=query, fields="files(id, name, createdTime)",
                          orderBy="createdTime", pageSize=5)
                    .execute()
                )
                items = results.get("files", [])

                if items:
                    cur_parent = items[0]["id"]
                else:
                    folder_meta = {
                        "name": part,
                        "mimeType": "application/vnd.google-apps.folder",
                    }
                    if cur_parent:
                        folder_meta["parents"] = [cur_parent]
                    folder = service.files().create(body=folder_meta, fields="id").execute()
                    cur_parent = folder["id"]
                    logger.info("Drive: created folder '%s' (id: %s)", part, cur_parent)

            return cur_parent

        except Exception as exc:
            if attempt == 0 and _is_ssl_error(exc):
                logger.warning("SSL/network error in get_or_create_folder, resetting service: %s", exc)
                _SERVICE = None
                continue
            raise


def _resolve_folder_id(folder_name: str) -> str:
    """Get folder ID for a logical path like 'ece22073/input'. Caches results."""
    if folder_name not in _FOLDER_CACHE:
        _FOLDER_CACHE[folder_name] = get_or_create_folder(folder_name)
    return _FOLDER_CACHE[folder_name]


# ═══════════════════════════════════════════════════════════════════
# File operations
# ═══════════════════════════════════════════════════════════════════

def upload_file(
    local_path: str | Path,
    drive_folder: str,
    *,
    filename: str | None = None,
) -> str:
    """Upload a file to Drive, overwriting if it already exists. Returns the Drive file ID."""
    service = authenticate()
    local_path = Path(local_path)
    target_name = filename or local_path.name
    folder_id = _resolve_folder_id(drive_folder)
    target_name_escaped = target_name.replace("'", "\\'")
    query = f"'{folder_id}' in parents and name='{target_name_escaped}' and trashed=false"
    existing = service.files().list(q=query, fields="files(id)", pageSize=1).execute().get("files", [])

    media = MediaFileUpload(
        str(local_path), mimetype="application/octet-stream", resumable=True
    )

    if existing:
        file_id = existing[0]["id"]
        f = service.files().update(fileId=file_id, media_body=media, fields="id").execute()
        logger.info("Drive: updated '%s' → %s/%s (id: %s)", local_path.name, drive_folder, target_name, file_id)
        return file_id
    else:
        file_meta = {"name": target_name, "parents": [folder_id]}
        f = service.files().create(body=file_meta, media_body=media, fields="id").execute()
        logger.info("Drive: uploaded '%s' → %s/%s (id: %s)", local_path.name, drive_folder, target_name, f["id"])
        return f["id"]


def upload_bytes(
    data: bytes,
    drive_folder: str,
    filename: str,
    *,
    mimetype: str = "application/octet-stream",
) -> str:
    """Upload raw bytes to Drive, overwriting if the file already exists. Returns file ID."""
    service = authenticate()
    folder_id = _resolve_folder_id(drive_folder)

    filename_escaped = filename.replace("'", "\\'")
    query = f"'{folder_id}' in parents and name='{filename_escaped}' and trashed=false"
    existing = service.files().list(q=query, fields="files(id)", pageSize=1).execute().get("files", [])

    from googleapiclient.http import MediaIoBaseUpload
    fh = io.BytesIO(data)
    media = MediaIoBaseUpload(fh, mimetype=mimetype, resumable=True)

    if existing:
        file_id = existing[0]["id"]
        f = service.files().update(fileId=file_id, media_body=media, fields="id").execute()
        logger.info("Drive: updated %d bytes → %s/%s (id: %s)", len(data), drive_folder, filename, file_id)
        return file_id
    else:
        file_meta = {"name": filename, "parents": [folder_id]}
        f = service.files().create(body=file_meta, media_body=media, fields="id").execute()
        logger.info("Drive: uploaded %d bytes → %s/%s (id: %s)", len(data), drive_folder, filename, f["id"])
        return f["id"]


def list_files(drive_folder: str) -> list[dict]:
    """List all non-trashed files in a Drive folder. Returns list of {id, name, createdTime}.
    Retries once after resetting the service on transient SSL/network errors.
    """
    global _SERVICE
    for attempt in range(2):
        try:
            service = authenticate()
            folder_id = _resolve_folder_id(drive_folder)
            results = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed=false",
                    fields="files(id, name, createdTime, size)",
                    pageSize=100,
                )
                .execute()
            )
            return results.get("files", [])
        except Exception as exc:
            if attempt == 0 and _is_ssl_error(exc):
                logger.warning("SSL/network error in list_files, resetting service: %s", exc)
                _SERVICE = None
                continue
            raise


def download_file(file_id: str, local_path: str | Path) -> None:
    """Download a file from Drive by ID."""
    service = authenticate()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    request = service.files().get_media(fileId=file_id)
    with open(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    logger.info("Drive: downloaded file (id: %s) → %s", file_id, local_path)


def read_json(file_id: str) -> dict:
    """Read a JSON file from Drive by ID and return parsed dict."""
    service = authenticate()
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return json.loads(fh.read().decode("utf-8"))


def read_bytes(file_id: str) -> bytes:
    """Download a file from Drive by ID and return raw bytes."""
    service = authenticate()
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()


def write_json(
    data: dict,
    drive_folder: str,
    filename: str,
) -> str:
    """Serialize a dict to JSON and upload to Drive. Returns file ID."""
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    return upload_bytes(
        payload.encode("utf-8"),
        drive_folder,
        filename,
        mimetype="application/json",
    )


def delete_file(file_id: str) -> None:
    """Trash a file on Drive by ID."""
    service = authenticate()
    service.files().delete(fileId=file_id).execute()
    logger.info("Drive: deleted file (id: %s)", file_id)


def move_file(file_id: str, target_folder: str) -> str:
    """Move a file to a different Drive folder. Returns new parent folder ID."""
    service = authenticate()
    # Get current parents to remove them
    f = service.files().get(fileId=file_id, fields="parents").execute()
    prev_parents = ",".join(f.get("parents", []))
    target_id = _resolve_folder_id(target_folder)
    service.files().update(
        fileId=file_id,
        addParents=target_id,
        removeParents=prev_parents,
        fields="id, parents",
    ).execute()
    logger.info("Drive: moved file (id: %s) → %s", file_id, target_folder)
    return target_id


# ═══════════════════════════════════════════════════════════════════
# High-level job helpers
# ═══════════════════════════════════════════════════════════════════

def init_drive_structure() -> None:
    """Idempotently create the full Drive folder tree."""
    for folder in (
        config.DRIVE_INPUT,
        config.DRIVE_INPUT_JOBS,
        config.DRIVE_INPUT_PROCESSED,
        config.DRIVE_OUTPUT,
        config.DRIVE_OUTPUT_PODCASTS,
        config.DRIVE_MODELS_CACHE,
    ):
        get_or_create_folder(folder)
    logger.info("Drive: folder structure initialized")


def write_status(job_id: str, status: config.StatusDict) -> str:
    """Write a status.json for a job into output/{job_id}/."""
    job_output = f"{config.DRIVE_OUTPUT}/{job_id}"
    get_or_create_folder(job_output)
    return write_json(status, job_output, "status.json")


def read_status(job_id: str) -> config.StatusDict | None:
    """Read status.json for a job. Returns None if not found."""
    try:
        files = list_files(f"{config.DRIVE_OUTPUT}/{job_id}")
        # Sort files by createdTime descending to get the newest status.json
        try:
            files_sorted = sorted(files, key=lambda x: x.get("createdTime", ""), reverse=True)
        except Exception:
            files_sorted = files
        for f in files_sorted:
            if f["name"] == "status.json":
                return read_json(f["id"])
    except Exception as e:
        logger.warning("read_status(%s) failed: %s", job_id, e, exc_info=True)
    return None


def find_new_input_files() -> list[dict]:
    """List files in the input/ folder that haven't been processed yet."""
    return list_files(config.DRIVE_INPUT)


def find_new_podcast_jobs() -> list[dict]:
    """List JSON files in input/podcast_jobs/."""
    files = list_files(config.DRIVE_INPUT_JOBS)
    return [f for f in files if f["name"].endswith(".json")]


def archive_input_file(file_id: str) -> None:
    """Move a processed input file to the processed/ archive."""
    move_file(file_id, config.DRIVE_INPUT_PROCESSED)


def generate_job_id() -> str:
    """Generate a unique job ID."""
    return str(uuid.uuid4())[:8]


def list_job_history() -> list[dict]:
    """Return all ASR job records from the output folder, newest first.
    Each entry: {job_id, stage, updated_at, error, drive_created, filename}.
    Reads status.json + meta.json per job folder.
    Skips system folders (e.g. 'podcasts') by requiring 8-char hex names.

    Uses direct folder IDs from the top-level listing to avoid O(N) path
    resolution calls — reduces API calls from ~5N to ~2N+1.
    """
    import re
    _JOB_RE = re.compile(r"^[0-9a-f]{8}$")

    try:
        # Resolve output folder once (cached after first call)
        output_folder_id = _resolve_folder_id(config.DRIVE_OUTPUT)
        service = authenticate()
        top_resp = (
            service.files()
            .list(
                q=f"'{output_folder_id}' in parents and trashed=false",
                fields="files(id, name, createdTime, mimeType)",
                pageSize=100,
            )
            .execute()
        )
        top_items = top_resp.get("files", [])
    except Exception as exc:
        logger.warning("list_job_history: %s", exc)
        return []

    results = []
    for item in top_items:
        name = item.get("name", "")
        if not _JOB_RE.match(name):
            continue
        try:
            # Use the folder ID directly — no path resolution needed
            job_folder_id = item["id"]
            # Cache this folder ID so other callers don't re-resolve it
            _FOLDER_CACHE[f"{config.DRIVE_OUTPUT}/{name}"] = job_folder_id

            job_resp = (
                service.files()
                .list(
                    q=f"'{job_folder_id}' in parents and trashed=false",
                    fields="files(id, name, createdTime)",
                    pageSize=50,
                )
                .execute()
            )
            job_files = job_resp.get("files", [])

            try:
                job_files_sorted = sorted(job_files, key=lambda x: x.get("createdTime", ""), reverse=True)
            except Exception:
                job_files_sorted = job_files

            status_id = meta_id = None
            for jf in job_files_sorted:
                if jf["name"] == "status.json" and not status_id:
                    status_id = jf["id"]
                elif jf["name"] == "meta.json" and not meta_id:
                    meta_id = jf["id"]

            if not status_id:
                continue

            status = read_json(status_id)
            meta   = read_json(meta_id) if meta_id else {}

            results.append({
                "job_id":        name,
                "stage":         status.get("stage", "unknown"),
                "updated_at":    status.get("updated_at", ""),
                "error":         status.get("error"),
                "drive_created": item.get("createdTime", ""),
                "filename":      meta.get("filename", ""),
            })
        except Exception:
            continue

    results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return results
