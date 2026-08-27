"""Local static file server for the LQMS reference frontend, with small shims
for the handful of ASP.NET-only requests this page fires automatically on load:

  * PageMethod calls (BindAuditType, BindFindingType, BindBranchList,
    BindAuditRefCriteria, getCounts, BindCAPAGrid, CheckAdminUserStatus) --
    only exist on the real IIS/ASP.NET server. Python's stock http.server
    can't execute them at all (POST -> 501), and the page's own error handler
    pops a blocking alert("Error") on failure. Shimmed with an empty-but-valid
    ASP.NET AJAX response ({"d": "<NewDataSet></NewDataSet>"}), so those
    dropdowns/grid/counters simply come up empty instead of erroring.

  * WebResource.axd / ScriptResource.axd requests -- ASP.NET's embedded-script
    handler (validation/postback helper scripts compiled into the real
    server). Shimmed with an empty, valid application/javascript response so
    the <script> tag loads cleanly instead of 404ing; any function it would
    have defined is simply undefined if something later tries to call it,
    same as if the network request had failed outright.

It deliberately does NOT shim state-changing PageMethod calls (save/insert/
update/approve/send-mail/etc.) -- faking a "success" response for those would
silently lie about data being persisted. Everything else still 404s/501s
exactly as before; this is local dev ergonomics only, not a reimplementation
of the ASP.NET backend.

Font files (Font Awesome / Glyphicons .woff/.woff2/.ttf) are a separate,
genuine gap -- they're just missing from assets/fonts/ in this export -- and
are NOT shimmed here, since serving fake/empty bytes for a font request
trades a 404 for a "failed to decode font" error, which isn't actually
cleaner. See the project README for how to fill those in.

Usage:
    python3 dev_server.py [port]   # defaults to 5510
"""

import os
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

EMPTY_DATASET_RESPONSE = b'{"d": "<NewDataSet></NewDataSet>"}'
EMPTY_SCRIPT_RESPONSE = b"// stubbed by dev_server.py -- not available outside the real ASP.NET server\n"

# Read-only PageMethod calls this page fires on load, matched by exact path.
SHIMMED_PAGE_METHODS = (
    "CAPAMain.aspx/BindAuditType",
    "CAPAMain.aspx/BindFindingType",
    "CAPAMain.aspx/BindBranchList",
    "CAPAMain.aspx/getCounts",
    "CAPAMain.aspx/BindCAPAGrid",
    "CAPAMain.aspx/CheckAdminUserStatus",
    "CreateAuditSchedule.aspx/BindAuditRefCriteria",
)


class DevRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path.endswith(".axd"):
            self._send(200, "application/javascript; charset=utf-8", EMPTY_SCRIPT_RESPONSE)
            return
        super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length:
            self.rfile.read(content_length)  # drain the request body

        path = self.path.split("?", 1)[0].lstrip("/")
        if path in SHIMMED_PAGE_METHODS:
            self._send(200, "application/json; charset=utf-8", EMPTY_DATASET_RESPONSE)
            return

        self.send_error(501, "Unsupported method ('POST')")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quiet down the default per-request logging; keep it to non-2xx so the
        # console stays readable while developing.
        if not str(args[1]).startswith("2"):
            super().log_message(format, *args)


def start_backend_process():
    """Start the FastAPI backend server on port 8010 if not already running."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    backend_dir = os.path.join(project_root, "backend")
    venv_python = os.path.join(backend_dir, ".venv", "bin", "python")
    
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable
    cmd = [python_bin, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8010"]
    
    try:
        proc = subprocess.Popen(cmd, cwd=backend_dir)
        print(f"🚀 Started Backend API server on http://localhost:8010 (PID: {proc.pid})")
        return proc
    except Exception as exc:
        print(f"⚠️ Could not automatically start backend: {exc}")
        return None


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5510
    
    # Automatically launch backend alongside frontend
    backend_proc = start_backend_process()
    
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("", port), DevRequestHandler)
    print(f"✨ Serving frontend on http://localhost:{port}/index.html")
    print(f"🔑 GitHub OAuth entry: http://localhost:8010/api/auth/github/login")
    print(f"Shimming {len(SHIMMED_PAGE_METHODS)} read-only ASP.NET PageMethod calls (see dev_server.py).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dev server...")
    finally:
        if backend_proc:
            backend_proc.terminate()


if __name__ == "__main__":
    main()
