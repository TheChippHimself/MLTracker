import http.server
import socketserver
import threading
import json
import time
import os
import urllib.parse
from pathlib import Path

class SSEHandler(http.server.SimpleHTTPRequestHandler):
    latest_pose = None
    vrm_filename = None

    def __init__(self, *args, **kwargs):
        # Serve files from the static directory and current directory
        static_dir = Path(__file__).parent / "static"
        cwd = Path.cwd()
        super().__init__(*args, directory=str(cwd), **kwargs)

    def translate_path(self, path):
        # Intercept / to serve our index.html from static dir
        parsed_path = urllib.parse.urlparse(path).path
        if parsed_path == "/" or parsed_path == "/index.html":
            return str(Path(__file__).parent / "static" / "index.html")
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        if self.path == '/pose':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            if SSEHandler.latest_pose is not None:
                self.wfile.write(json.dumps(SSEHandler.latest_pose).encode())
            else:
                self.wfile.write(b"{}")
        elif self.path == '/config':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"vrm_file": SSEHandler.vrm_filename}).encode())
        else:
            super().do_GET()

def start_vrm_server(vrm_filename: str, port: int = 8080):
    SSEHandler.vrm_filename = vrm_filename
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("localhost", port), SSEHandler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    return httpd
