"""
PC Use MCP Server (portable)
Exposes the host machine's capabilities to Eva via MCP HTTP transport.
The same server runs on any host (Windows desktop, Ubuntu server, ...);
clients connect over HTTP. See NOTES.txt for the security/hardening plan.
"""

import sys
import os
import socket
import io
import base64
import glob
import json
import subprocess
import threading
import time
import uuid
import requests
import uvicorn
from datetime import datetime
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
load_dotenv()

# Ensure stdout handles unicode on Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from mcp.server.fastmcp import FastMCP

host = os.getenv("MCP_HOST", "0.0.0.0")
port = int(os.getenv("MCP_PORT", "8001"))
CAMERA_SEND_COUNT = int(os.getenv("CAMERA_SEND_COUNT", "3"))
lmstudio_url = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1/")
vision_model = os.getenv("VISION_MODEL", "qwen/qwen3.5-9b")

# --- Layer 0: network + auth hardening (all backward-compatible) -------------
# MCP_AUTH_TOKEN: if set, every request must send `Authorization: Bearer <token>`.
#   If unset, auth is disabled (legacy behavior) and a loud warning is printed.
# MCP_ALLOWED_ORIGINS: comma-separated CORS allowlist. Defaults to "*" (legacy).
# Recommended production posture: set MCP_HOST=127.0.0.1 (or the tailnet IP),
#   set MCP_AUTH_TOKEN, and set MCP_ALLOWED_ORIGINS to the client origin(s).
AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("MCP_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
# Paths that must stay reachable without a token (health/liveness probes only).
AUTH_EXEMPT_PATHS = tuple(
    p.strip() for p in os.getenv("MCP_AUTH_EXEMPT_PATHS", "/health").split(",") if p.strip()
)

# --- Layer 1: least-privilege guards (all backward-compatible) ---------------
# FILE_ALLOWED_ROOTS: os.pathsep-separated absolute dirs the file tools may touch.
#   Empty (default) = no confinement (legacy: any path the process can reach).
# BLOCK_PRIVATE_IPS: "1" makes web_fetch/http_request refuse private/loopback/
#   link-local targets (SSRF guard). Default "0" (legacy). HTTP_ALLOWED_HOSTS is
#   an explicit exception list (e.g. the LM Studio host) checked before blocking.
FILE_ALLOWED_ROOTS = [
    os.path.realpath(p.strip())
    for p in os.getenv("FILE_ALLOWED_ROOTS", "").split(os.pathsep)
    if p.strip()
]
BLOCK_PRIVATE_IPS = os.getenv("BLOCK_PRIVATE_IPS", "0").strip() in ("1", "true", "True")
HTTP_ALLOWED_HOSTS = {
    h.strip().lower()
    for h in os.getenv("HTTP_ALLOWED_HOSTS", "").split(",")
    if h.strip()
}


def _check_path(path: str):
    """Return (canonical_path, None) if allowed, or (None, error_str) if denied.

    Canonicalizes with realpath (resolves symlinks and ..) then requires the
    result to be inside one of FILE_ALLOWED_ROOTS. No-op when the allowlist is
    empty (legacy behavior)."""
    canonical = os.path.realpath(path)
    if not FILE_ALLOWED_ROOTS:
        return canonical, None
    for root in FILE_ALLOWED_ROOTS:
        if canonical == root or canonical.startswith(root + os.sep):
            return canonical, None
    return None, (
        f"Permission denied: {path!r} is outside the allowed roots. "
        f"Allowed: {FILE_ALLOWED_ROOTS}. "
        f"(An operator can grant this path — see the permission broker.)"
    )


def _check_url(url: str):
    """Return None if the URL may be fetched, or an error string if blocked.

    When BLOCK_PRIVATE_IPS is on, resolves the host and rejects loopback,
    private, link-local, and reserved ranges unless the host is explicitly in
    HTTP_ALLOWED_HOSTS. No-op when BLOCK_PRIVATE_IPS is off (legacy)."""
    if not BLOCK_PRIVATE_IPS:
        return None
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Blocked: only http/https URLs allowed, got scheme {parsed.scheme!r}."
    hostname = parsed.hostname or ""
    if hostname.lower() in HTTP_ALLOWED_HOSTS:
        return None
    try:
        infos = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except Exception as e:
        return f"Blocked: could not resolve host {hostname!r}: {e}"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return (
                f"Blocked: {hostname} resolves to non-public address {ip}. "
                f"Add it to HTTP_ALLOWED_HOSTS to allow."
            )
    return None


mcp = FastMCP(os.getenv("MCP_SERVER_NAME", os.getenv("COMPUTERNAME") or socket.gethostname()), host=host, port=port)

# ---------------------------------------------------------------------------
# Camera — always-on background capture
# ---------------------------------------------------------------------------

CAPTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'captures'))
CAPTURE_INTERVAL = int(os.getenv("CAPTURE_INTERVAL_SECONDS", "30"))
MAX_CAPTURES = int(os.getenv("imageStorageMax", "10"))

os.makedirs(CAPTURES_DIR, exist_ok=True)

def _background_capture_loop():
    import cv2
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[Background capture] Could not open webcam — background capture disabled.")
        return
    print(f"[Background capture] Started. Interval: {CAPTURE_INTERVAL}s, keeping last {MAX_CAPTURES} frames.")
    while True:
        ret, frame = cam.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            filepath = os.path.join(CAPTURES_DIR, f"cap_{timestamp}.jpg")
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            # Prune oldest files beyond MAX_CAPTURES
            files = sorted(glob.glob(os.path.join(CAPTURES_DIR, "cap_*.jpg")))
            while len(files) > MAX_CAPTURES:
                os.remove(files.pop(0))
        time.sleep(CAPTURE_INTERVAL)

_capture_thread = threading.Thread(target=_background_capture_loop, daemon=True)
_capture_thread.start()


def _load_captures() -> list[tuple[datetime, str]]:
    """Return sorted list of (datetime, filepath) for all captures on disk."""
    files = sorted(glob.glob(os.path.join(CAPTURES_DIR, "cap_*.jpg")))
    result = []
    for f in files:
        ts_raw = os.path.basename(f)[4:-4]  # cap_20260319T123456.jpg → 20260319T123456
        try:
            dt = datetime.strptime(ts_raw, "%Y%m%dT%H%M%S")
            result.append((dt, f))
        except ValueError:
            pass
    return result

def _ensure_vision_model_loaded():
    """Load the vision model in LM Studio if it isn't already running."""
    base = lmstudio_url.rstrip("/").rsplit("/v1", 1)[0]
    try:
        resp = requests.get(lmstudio_url.rstrip("/") + "/models", timeout=10)
        loaded_ids = [m.get("id", "") for m in resp.json().get("data", [])]
        if vision_model in loaded_ids:
            return
    except Exception as e:
        print(f"[Vision] Could not check loaded models: {e}")
        return
    print(f"[Vision] Loading model {vision_model}...")
    try:
        requests.post(f"{base}/api/v0/models/load", json={"model": vision_model}, timeout=300)
        print(f"[Vision] Model load request sent.")
    except Exception as e:
        print(f"[Vision] Failed to load model: {e}")


def _describe_capture(filepath: str, dt: datetime) -> dict:
    """Send a capture to the vision model and return its description."""
    import cv2
    img = cv2.imread(filepath)
    if img is not None:
        img = cv2.resize(img, (640, 480))
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    else:
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = requests.post(
            lmstudio_url.rstrip("/") + "/chat/completions",
            json={
                "model": vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe in detail what you see in this image."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}"
                                }
                            }
                        ]
                    }
                ]
            },
            timeout=180
        )
        response.raise_for_status()
        description = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        description = f"[Vision model error: {e}]"
    return {"timestamp": dt.isoformat(), "description": description}


@mcp.tool()
def get_images(after: str = "", before: str = "", count: int = 3) -> str:
    """
    Retrieve webcam captures, describe each one using the vision model, and return the descriptions.
    Optionally filter by time range using ISO timestamps (e.g. '2026-03-19T17:00:00').
    If no captures fall within the requested range, returns descriptions of the most recent images instead
    and reports the available time range so you can adjust your next request.

    after:  return only captures after this ISO timestamp (inclusive). Leave blank for no lower bound.
    before: return only captures before this ISO timestamp (inclusive). Leave blank for no upper bound.
    count:  max number of images to describe (default 3).
    """
    count = max(1, min(count, CAMERA_SEND_COUNT))
    _ensure_vision_model_loaded()
    captures = _load_captures()

    if not captures:
        return json.dumps({"error": "No captured images available yet — background capture may still be initializing."})

    oldest_ts = captures[0][0].isoformat()
    newest_ts = captures[-1][0].isoformat()
    available_range = {"oldest": oldest_ts, "newest": newest_ts}

    filtered = captures
    if after:
        try:
            after_dt = datetime.fromisoformat(after)
            filtered = [(dt, f) for dt, f in filtered if dt >= after_dt]
        except ValueError:
            return json.dumps({"error": f"Invalid 'after' timestamp: {after!r}. Use ISO format e.g. '2026-03-19T17:00:00'."})
    if before:
        try:
            before_dt = datetime.fromisoformat(before)
            filtered = [(dt, f) for dt, f in filtered if dt <= before_dt]
        except ValueError:
            return json.dumps({"error": f"Invalid 'before' timestamp: {before!r}. Use ISO format e.g. '2026-03-19T17:00:00'."})

    selected = filtered[-count:] if filtered else captures[-count:]
    note = None if filtered else f"No captures found in the requested range. Returning the {len(selected)} most recent instead."

    descriptions = [_describe_capture(f, dt) for dt, f in selected]

    result = {"available_range": available_range, "observations": descriptions}
    if note:
        result["note"] = note
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

@mcp.tool()
def run_command(command: str) -> str:
    """
    Run a shell command on Fate's Windows desktop and return its output.
    Useful for checking system state, listing files, or running scripts.
    Timeout is 30 seconds. Avoid long-running or interactive commands.
    """
    try:
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] run_command: {command}")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode != 0:
            return f"Exit code {result.returncode}:\n{result.stderr}"
        return result.stdout or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Exception: {e}"


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------

JOBS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mcp_jobs'))
os.makedirs(JOBS_DIR, exist_ok=True)

@mcp.tool()
def run_command_bg(command: str, job_id: str = "") -> str:
    """
    Run a long-running shell command in the background on Fate's desktop.
    Returns immediately with a job_id. Output is written to a file.
    Use read_job_output(job_id) in a later MCP call to retrieve results.
    """
    if not job_id:
        job_id = str(uuid.uuid4())[:8]
    output_file = os.path.join(JOBS_DIR, f"{job_id}.txt")

    def _run():
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"[STARTED] {datetime.now().isoformat()}\n[CMD] {command}\n")
            f.flush()
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True,
                text=True, timeout=300, encoding='utf-8', errors='replace'
            )
            with open(output_file, 'a', encoding='utf-8') as f:
                if result.stdout:
                    f.write(f"[STDOUT]\n{result.stdout}\n")
                if result.stderr:
                    f.write(f"[STDERR]\n{result.stderr}\n")
                f.write(f"[EXIT] {result.returncode}\n[DONE] {datetime.now().isoformat()}\n")
        except Exception as e:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"[ERROR] {e}\n")

    threading.Thread(target=_run, daemon=True).start()
    print(f"[run_command_bg] job_id={job_id} cmd={command}")
    return json.dumps({"job_id": job_id, "output_file": output_file, "status": "started"})


@mcp.tool()
def read_job_output(job_id: str) -> str:
    """
    Read the output of a background job started with run_command_bg.
    Returns the current contents of the job output file.
    Check for [DONE] or [ERROR] at the end to know if it has finished.
    """
    output_file = os.path.join(JOBS_DIR, f"{job_id}.txt")
    try:
        with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        if "[DONE]" in content or "[ERROR]" in content:
            try:
                os.remove(output_file)
            except Exception:
                pass
        return content
    except FileNotFoundError:
        return f"Job '{job_id}' not found. Available jobs: {os.listdir(JOBS_DIR)}"
    except Exception as e:
        return f"Error reading job output: {e}"


# ---------------------------------------------------------------------------
# File access
# ---------------------------------------------------------------------------

@mcp.tool()
def read_file(path: str) -> str:
    """
    Read the contents of a file on Fate's Windows desktop.
    path should be an absolute Windows path, e.g. 'C:\\Users\\fhard\\notes.txt'.
    Returns the file contents as text, or an error message if the file cannot be read.
    """
    path, err = _check_path(path)
    if err:
        return err
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except PermissionError:
        return f"Permission denied: {path}"
    except FileNotFoundError:
        return f"File not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def write_file(path: str, content: str, append: bool = False) -> str:
    """
    Write or append text content to a file on Fate's Windows desktop.
    path: absolute Windows path, e.g. 'C:\\Users\\fhard\\notes.txt'
    content: text to write
    append: if True, appends to existing file; if False (default), overwrites
    Returns 'OK' on success or an error message.
    """
    path, err = _check_path(path)
    if err:
        return err
    try:
        mode = 'a' if append else 'w'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)
        return "OK"
    except PermissionError:
        return f"Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@mcp.tool()
def list_dir(path: str) -> str:
    """
    List the contents of a directory on Fate's Windows desktop.
    path: absolute Windows path, e.g. 'C:\\Users\\fhard\\Documents'
    Returns a JSON array of entries with name, type (file/dir), and size.
    """
    path, err = _check_path(path)
    if err:
        return err
    try:
        entries = []
        for entry in os.scandir(path):
            try:
                entries.append({
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else None
                })
            except Exception:
                entries.append({"name": entry.name, "type": "unknown", "size": None})
        return json.dumps(sorted(entries, key=lambda e: (e["type"] == "file", e["name"])))
    except PermissionError:
        return f"Permission denied: {path}"
    except FileNotFoundError:
        return f"Directory not found: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


# ---------------------------------------------------------------------------
# Compute & networking
# ---------------------------------------------------------------------------

@mcp.tool()
def python_exec(code: str) -> str:
    """
    Execute Python code on Fate's Windows desktop and return printed output or a 'result' variable.
    Useful for math, data processing, and simple computations.
    For API calls use http_request instead.
    """
    import io as _io
    print(f"[python_exec] {code[:80]}")
    stdout_capture = _io.StringIO()
    local_ns = {}
    old_stdout = sys.stdout
    try:
        sys.stdout = stdout_capture
        exec(code, {}, local_ns)
        sys.stdout = old_stdout
        output = stdout_capture.getvalue().strip()
        if output:
            return output
        if "result" in local_ns:
            return str(local_ns["result"])
        return "OK (no output)"
    except Exception as e:
        sys.stdout = old_stdout
        return f"Exception: {e}"


@mcp.tool()
def web_fetch(url: str) -> str:
    """
    Fetch the text content of any public URL from Fate's desktop.
    Useful for reading documentation, API references, or web pages.
    """
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        print(f"[web_fetch] {url}")
        return resp.text
    except Exception as e:
        return f"Exception: {e}"


@mcp.tool()
def http_request(method: str, url: str, headers: dict = {}, body: dict = {}) -> str:
    """
    Make an HTTP request from Fate's desktop and return the response.
    method: 'GET', 'POST', 'PATCH', 'DELETE', 'PUT'
    url: full URL including query params for GET requests
    headers: dict of request headers, e.g. {"Authorization": "Bearer key"}
    body: dict to send as JSON body (for POST/PATCH/PUT). Leave empty for GET/DELETE.
    Returns JSON with 'status', 'headers', and 'body' keys.
    """
    try:
        print(f"[http_request] {method} {url}")
        resp = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=body if body else None,
            timeout=30
        )
        try:
            resp_body = resp.json()
        except Exception:
            resp_body = resp.text
        return json.dumps({"status": resp.status_code, "headers": dict(resp.headers), "body": resp_body})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

@mcp.tool()
def get_datetime() -> str:
    """Returns the current date and time on Fate's desktop machine."""
    return datetime.now().isoformat()


@mcp.tool()
def get_location() -> str:
    """
    Returns EVA's current physical location by geolocating the server's public IP.
    Provides city, region, country, coordinates, timezone, ISP, and public IP address.
    Useful for location-aware tasks and answering 'where are you?' questions.
    No arguments needed — automatically detects the server's outbound IP.
    """
    import socket
    try:
        resp = requests.get("http://ip-api.com/json", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            return json.dumps({"error": "Geolocation lookup failed", "details": data})
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            local_ip = "unknown"
        return json.dumps({
            "public_ip": data.get("query"),
            "city": data.get("city"),
            "region": data.get("regionName"),
            "country": data.get("country"),
            "country_code": data.get("countryCode"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),
            "timezone": data.get("timezone"),
            "isp": data.get("isp"),
            "hostname": hostname,
            "local_ip": local_ip,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Obsidian vault — registered only when VAULT_ROOT points at a vault on this
# machine, so an instance running without one simply does not expose the tools.
# ---------------------------------------------------------------------------

try:
    from vault import register_vault_tools
    register_vault_tools(mcp)
except Exception as _e:
    print(f"[vault] tools unavailable: {_e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

import hmac
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Require `Authorization: Bearer <MCP_AUTH_TOKEN>` on every request.

    No-op when MCP_AUTH_TOKEN is unset (legacy behavior). Uses a constant-time
    compare to avoid leaking the token via timing. CORS preflight (OPTIONS) and
    configured exempt paths (e.g. /health) are allowed through unauthenticated.
    """

    def __init__(self, app, token: str, exempt_paths: tuple = ()):
        super().__init__(app)
        self.token = token
        self.exempt_paths = exempt_paths

    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS" or request.url.path in self.exempt_paths:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        provided = auth[7:] if auth.lower().startswith("bearer ") else ""
        if not (provided and hmac.compare_digest(provided, self.token)):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


if __name__ == "__main__":
    print(f"PC Use MCP server starting on {host}:{port}")

    app = mcp.streamable_http_app()

    # Order matters: CORS added last runs first (outermost), so preflight and
    # CORS headers are handled before the auth check rejects anything.
    app.add_middleware(
        BearerAuthMiddleware,
        token=AUTH_TOKEN,
        exempt_paths=AUTH_EXEMPT_PATHS,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"],
    )

    if AUTH_TOKEN:
        print(f"[MCP] Bearer auth ENABLED (exempt: {', '.join(AUTH_EXEMPT_PATHS) or 'none'})")
    else:
        print("[MCP] ⚠ WARNING: MCP_AUTH_TOKEN not set — auth DISABLED. "
              "Anyone who can reach this port has full host control. "
              "Set MCP_AUTH_TOKEN and bind MCP_HOST to localhost/tailnet for production.")
    if ALLOWED_ORIGINS == ["*"]:
        print("[MCP] ⚠ WARNING: CORS allow_origins is '*'. Set MCP_ALLOWED_ORIGINS to lock it down.")
    if host == "0.0.0.0":
        print("[MCP] ⚠ WARNING: bound to 0.0.0.0 (all interfaces). "
              "Prefer MCP_HOST=127.0.0.1 or the tailnet IP.")

    while True:
        try:
            uvicorn.run(app, host=host, port=port)
        except Exception as e:
            print(f"[MCP] Server error ({e}), restarting in 1s...")
            time.sleep(1)
