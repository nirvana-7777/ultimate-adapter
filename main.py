#!/usr/bin/env python3
"""
Ultimate Adapter - Flask wrapper for UltimateAdapter
"""

import logging
import os
import uuid
from datetime import datetime

from flask import Flask, Response, jsonify, make_response, request

from ultimate_adapter import create_adapter


# Configure comprehensive logging
def setup_logging():
    """Configure comprehensive logging"""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # File handler for all logs
    file_handler = logging.FileHandler(
        f"logs/ultimate_adapter_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Reduce verbosity for some libraries
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create adapter instance
adapter = create_adapter()

# Configuration
SERVER_PORT = int(os.environ.get("SERVER_PORT", 8080))
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"
# Support both env var names for backward compatibility
MPEGTS_PROXY_URL = os.environ.get(
    "MPEGTS_PROXY_URL", os.environ.get("HLS_CONVERTER_URL", "http://mpegts-proxy:8000")
)


@app.before_request
def assign_request_id():
    """Assign a unique ID to each request for tracing"""
    request.request_id = str(uuid.uuid4())[:8]
    logger.debug(
        f"Assigned Request ID: {request.request_id} for {request.method} {request.path}"
    )


@app.before_request
def log_request_info():
    """Log information about each request"""
    if request.path != "/favicon.ico":  # Skip favicon requests
        request_id = getattr(request, "request_id", "unknown")
        logger.debug(
            f"Request {request_id}: {request.method} {request.path} - "
            f"Content-Type: {request.content_type}, "
            f"Content-Length: {request.content_length}"
        )


@app.after_request
def add_request_id_header(response):
    """Add request ID to response headers"""
    if hasattr(request, "request_id"):
        response.headers["X-Request-ID"] = request.request_id
    return response


@app.route("/player_api.php", methods=["GET", "POST"])
def player_api():
    """Main API endpoint"""
    request_id = getattr(request, "request_id", "unknown")

    # Get parameters
    if request.method == "GET":
        params = request.args.to_dict()
    else:
        params = request.form.to_dict()

    username = params.get("username")
    action = params.get("action", "")

    # Log full request details
    client_ip = request.remote_addr
    #    user_agent = request.user_agent.string[:100] if request.user_agent else "Unknown"

    logger.info(
        f"API Request {request_id} - IP: {client_ip}, "
        f"Action: '{action}', User: {username}, Method: {request.method}"
    )

    # Log full parameters (excluding sensitive data)
    safe_params = params.copy()
    if "password" in safe_params:
        safe_params["password"] = "***REDACTED***"
    logger.debug(f"Request {request_id} parameters: {safe_params}")

    # Handle request via adapter
    response_data = adapter.handle_api_request(action, params)

    # Log response status
    if isinstance(response_data, dict) and "error" in response_data:
        logger.warning(
            f"API Error {request_id} for action '{action}': {response_data['error']}"
        )
        status_code = 401 if "Invalid credentials" in response_data["error"] else 400
        return jsonify(response_data), status_code

    # Log successful response summary
    if isinstance(response_data, list):
        logger.info(
            f"API Success {request_id} - Action: '{action}', returned {len(response_data)} items"
        )
    else:
        logger.info(
            f"API Success {request_id} - Action: '{action}', returned dict response"
        )

    return jsonify(response_data)


@app.route("/live/<username>/<password>/<int:stream_id>")
@app.route("/live/<username>/<password>/<int:stream_id>.<ext>")
def live_stream(username, password, stream_id, ext=None):
    """Live stream endpoint with MPEG-TS proxy conversion"""
    request_id = getattr(request, "request_id", "unknown")

    # Log the request
    logger.info(
        f"Live stream request {request_id}: user={username}, "
        f"stream={stream_id}, ext={ext}"
    )

    # Get stream URL from adapter
    dash_url = adapter.get_stream_url(username, password, stream_id)

    if not dash_url:
        logger.warning(
            f"Stream not found {request_id}: user={username}, stream={stream_id}"
        )
        return "Unauthorized or stream not found", 404

    # If client expects TS/M3U8, use MPEG-TS proxy
    if ext and ext in ["ts", "m3u8"]:
        # Check if MPEGTS proxy is configured (supports both env var names)
        mpegts_proxy_url = os.environ.get(
            "MPEGTS_PROXY_URL", os.environ.get("HLS_CONVERTER_URL")
        )

        if mpegts_proxy_url:
            try:
                # Check if dash_url is a pipe:// FFmpeg command
                if dash_url.startswith("pipe://"):
                    # Extract the actual URL from the FFmpeg command
                    import re

                    # Look for -i "URL" or -i URL pattern
                    match = re.search(r'-i\s+"([^"]+)"', dash_url) or re.search(
                        r"-i\s+(\S+)", dash_url
                    )
                    if match:
                        actual_url = match.group(1)
                        logger.info(f"Extracted URL from pipe command: {actual_url}")
                        dash_url = actual_url
                    else:
                        logger.error(
                            f"Could not extract URL from pipe command: {dash_url[:100]}"
                        )
                        return "Invalid stream URL format", 500

                # Get channel name for metadata
                channel = adapter.get_channel_by_id(stream_id)
                stream_name = channel.name if channel else f"Stream {stream_id}"

                # Build MPEG-TS proxy URL with stream parameter
                from urllib.parse import urlencode

                params = {"url": dash_url, "name": stream_name}
                mpegts_url = f"{mpegts_proxy_url}/stream?{urlencode(params)}"

                logger.info(
                    f"Proxying DASH to MPEG-TS for {request_id}: stream={stream_id}"
                )

                # Redirect to MPEG-TS proxy
                return Response(
                    "Redirecting to MPEG-TS stream...",
                    status=302,
                    headers={"Location": mpegts_url},
                )

            except Exception as e:
                logger.error(f"MPEG-TS proxy error for {request_id}: {e}")
                # Fall through to DASH redirect
        else:
            logger.warning(
                f"No MPEG-TS proxy configured for {request_id}, "
                f"falling back to DASH"
            )

    # Redirect to original DASH stream
    logger.info(f"Redirecting to DASH stream {request_id}: url={dash_url}")

    return Response(
        "Redirecting to stream...", status=302, headers={"Location": dash_url}
    )


@app.route("/movie/<username>/<password>/<int:vod_id>.<ext>")
def vod_stream(username, password, vod_id, ext):
    """VOD stream endpoint"""
    request_id = getattr(request, "request_id", "unknown")
    logger.info(
        f"VOD stream request {request_id}: user={username}, vod={vod_id}, ext={ext}"
    )
    return "VOD not supported", 501


@app.route("/series/<username>/<password>/<int:series_id>.<ext>")
def series_stream(username, password, series_id, ext):
    """Series stream endpoint"""
    request_id = getattr(request, "request_id", "unknown")
    logger.info(
        f"Series stream request {request_id}: user={username}, series={series_id}, ext={ext}"
    )
    return "Series not supported", 501


@app.route("/xmltv.php")
def xmltv_epg():
    """XMLTV EPG endpoint"""
    request_id = getattr(request, "request_id", "unknown")
    username = request.args.get("username")
    password = request.args.get("password")

    logger.info(f"XMLTV request {request_id}: user={username}")

    # Generate XMLTV via adapter
    xmltv_data = adapter.generate_xmltv_epg(username, password)

    if not xmltv_data:
        logger.warning(f"XMLTV unauthorized {request_id}: user={username}")
        return "Unauthorized", 401

    response = make_response(xmltv_data)
    response.headers["Content-Type"] = "application/xml; charset=utf-8"
    return response


@app.route("/m3u/<username>/<password>")
def m3u_playlist(username, password):
    """M3U playlist endpoint"""
    request_id = getattr(request, "request_id", "unknown")
    logger.info(f"M3U request {request_id}: user={username}")

    # Generate M3U via adapter
    m3u_data = adapter.generate_m3u_playlist(username, password)

    if not m3u_data:
        logger.warning(f"M3U unauthorized {request_id}: user={username}")
        return "Unauthorized", 401

    response = make_response(m3u_data)
    response.headers["Content-Type"] = "audio/x-mpegurl; charset=utf-8"
    filename = f"{username}_playlist.m3u"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@app.route("/get.php")
def get_stream():
    """Alternative stream endpoint"""
    request_id = getattr(request, "request_id", "unknown")
    username = request.args.get("username")
    password = request.args.get("password")
    stream_id = request.args.get("stream_id")

    if not username or not password or not stream_id:
        logger.warning(
            f"Missing parameters {request_id}: username={username}, stream_id={stream_id}"
        )
        return "Missing parameters", 400

    try:
        stream_id_int = int(stream_id)
    except ValueError:
        logger.warning(f"Invalid stream ID {request_id}: {stream_id}")
        return "Invalid stream ID", 400

    logger.info(
        f"Get stream request {request_id}: user={username}, stream={stream_id_int}"
    )

    # Redirect to live stream endpoint
    return f"/live/{username}/{password}/{stream_id_int}", 302


# --- Admin/Utility Endpoints ---


@app.route("/admin/flush_cache")
def flush_cache():
    """Flush adapter cache (admin only)"""
    request_id = getattr(request, "request_id", "unknown")
    cache_type = request.args.get("type")
    logger.info(f"Flush cache request {request_id}: type={cache_type}")

    result = adapter.flush_cache(cache_type)
    return jsonify(result)


@app.route("/admin/stats")
def get_stats():
    """Get adapter statistics"""
    request_id = getattr(request, "request_id", "unknown")
    logger.info(f"Stats request {request_id}")

    stats = adapter.get_stats()
    return jsonify(stats)


@app.route("/admin/test_backend")
def test_backend():
    """Test connection to Ultimate Backend"""
    request_id = getattr(request, "request_id", "unknown")
    logger.info(f"Test backend request {request_id}")

    # Try to fetch providers
    providers = adapter._make_request("/api/providers")

    if providers:
        providers_list = providers.get("providers", [])
        logger.info(
            f"Backend test successful {request_id}: {len(providers_list)} providers found"
        )
        return jsonify(
            {
                "status": "connected",
                "providers_count": len(providers_list),
                "backend_url": adapter.ultimate_backend_url,
            }
        )
    else:
        logger.error(
            f"Backend test failed {request_id}: Cannot connect to Ultimate Backend"
        )
        return (
            jsonify(
                {
                    "status": "disconnected",
                    "backend_url": adapter.ultimate_backend_url,
                    "error": "Cannot connect to Ultimate Backend",
                }
            ),
            503,
        )


@app.route("/health")
def health():
    """Health check endpoint"""
    request_id = getattr(request, "request_id", "unknown")
    logger.debug(f"Health check request {request_id}")

    # Test backend connection
    providers = adapter._make_request("/api/providers")
    backend_ok = bool(providers)

    providers_data = adapter.cache["providers"]["data"]
    providers_count = len(providers_data) if providers_data else 0

    return jsonify(
        {
            "status": "healthy" if backend_ok else "degraded",
            "backend_connected": backend_ok,
            "adapter": {
                "channels_count": len(adapter.get_channels()),
                "providers_count": providers_count,
            },
        }
    )


@app.route("/")
def index():
    """Welcome page"""
    request_id = getattr(request, "request_id", "unknown")
    logger.info(f"Index page request {request_id}")

    stats = adapter.get_stats()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ultimate Adapter</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .info {{
                background: #f5f5f5;
                padding: 20px;
                border-radius: 5px;
            }}
            .endpoint {{
                background: #e9f7fe;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #2196F3;
            }}
            code {{ background: #eee; padding: 2px 5px; }}
            .request-id {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Ultimate Adapter</h1>
        <div class="request-id">Request ID: {request_id}</div>

        <div class="info">
            <h3>Backend: {adapter.ultimate_backend_url}</h3>
            <p><strong>Channels:</strong> {stats['channels_total']}</p>
            <p><strong>Providers:</strong> {stats['providers_count']}</p>
        </div>

        <h2>API Endpoints</h2>

        <div class="endpoint">
            <h3>Main API</h3>
            <p>
                <code>
                    GET /player_api.php?username=user&password=pass&
                    action=get_live_categories
                </code>
            </p>
            <p>Actions: get_live_categories, get_live_streams, get_short_epg</p>
        </div>

        <div class="endpoint">
            <h3>Streams</h3>
            <p><code>GET /live/username/password/stream_id</code></p>
        </div>

        <div class="endpoint">
            <h3>Playlists</h3>
            <p><code>GET /m3u/username/password</code> (M3U Playlist)</p>
            <p><code>GET /xmltv.php?username=user&password=pass</code> (XMLTV EPG)</p>
        </div>

        <div class="endpoint">
            <h3>Admin</h3>
            <p><code>GET /health</code> (Health check)</p>
            <p><code>GET /admin/stats</code> (Statistics)</p>
            <p><code>GET /admin/flush_cache?type=channels</code> (Clear cache)</p>
        </div>

        <p><a href="/health">Check Health</a> | <a href="/admin/stats">View Stats</a></p>
    </body>
    </html>
    """

    return html


# --- Error Handlers ---


@app.errorhandler(404)
def not_found(error):
    """Log and handle 404 errors"""
    request_id = getattr(request, "request_id", "unknown")
    logger.warning(
        f"404 Not Found {request_id} - Path: {request.path}, "
        f"Method: {request.method}, IP: {request.remote_addr}"
    )
    return (
        jsonify(
            {"error": "Not found", "endpoint": request.path, "request_id": request_id}
        ),
        404,
    )


@app.errorhandler(405)
def method_not_allowed(error):
    """Log and handle 405 errors"""
    request_id = getattr(request, "request_id", "unknown")
    logger.warning(
        f"405 Method Not Allowed {request_id} - Path: {request.path}, "
        f"Method: {request.method}, IP: {request.remote_addr}"
    )
    return jsonify({"error": "Method not allowed", "request_id": request_id}), 405


@app.errorhandler(500)
def internal_error(error):
    """Log and handle 500 errors"""
    request_id = getattr(request, "request_id", "unknown")
    logger.error(f"Internal server error {request_id}: {error}", exc_info=True)
    return jsonify({"error": "Internal server error", "request_id": request_id}), 500


if __name__ == "__main__":
    logger.info(f"Starting Ultimate Adapter on port {SERVER_PORT}")
    logger.info(f"Backend URL: {adapter.ultimate_backend_url}")
    default_creds = f"{adapter.default_username}/{adapter.default_password}"
    logger.info(f"Default credentials: {default_creds}")

    # Initial cache load
    logger.info("Loading initial data...")
    adapter._load_providers()
    adapter.get_channels()

    app.run(host="0.0.0.0", port=SERVER_PORT, debug=DEBUG_MODE)
