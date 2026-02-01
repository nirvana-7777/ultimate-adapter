#!/usr/bin/env python3
"""
Ultimate Xtream Adapter - Flask wrapper for UltimateAdapter
"""

import os
import logging
from flask import Flask, request, jsonify, Response, make_response
from ultimate_adapter import create_adapter, UltimateAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create adapter instance
adapter = create_adapter()

# Configuration
SERVER_PORT = int(os.environ.get("SERVER_PORT", 8080))
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"


@app.route('/player_api.php', methods=['GET', 'POST'])
def player_api():
    """Main Xtream API endpoint"""
    # Get parameters
    if request.method == 'GET':
        params = request.args.to_dict()
    else:
        params = request.form.to_dict()

    username = params.get('username')
    password = params.get('password')
    action = params.get('action', '')

    logger.info(f"Xtream API request: action={action}, user={username}")

    # Handle request via adapter
    response_data = adapter.handle_xtream_request(action, params)

    # Return appropriate response
    if isinstance(response_data, dict) and 'error' in response_data:
        return jsonify(response_data), 401 if 'Invalid credentials' in response_data['error'] else 400

    return jsonify(response_data)


@app.route('/live/<username>/<password>/<int:stream_id>')
def live_stream(username, password, stream_id):
    """Live stream endpoint"""
    logger.info(f"Live stream request: user={username}, stream={stream_id}")

    # Get stream URL from adapter
    stream_url = adapter.get_stream_url(username, password, stream_id)

    if not stream_url:
        return "Unauthorized or stream not found", 404

    # Redirect to actual stream
    return Response(
        f'Redirecting to stream...',
        status=302,
        headers={'Location': stream_url}
    )


@app.route('/movie/<username>/<password>/<int:vod_id>.<ext>')
def vod_stream(username, password, vod_id, ext):
    """VOD stream endpoint"""
    return "VOD not supported", 501


@app.route('/series/<username>/<password>/<int:series_id>.<ext>')
def series_stream(username, password, series_id, ext):
    """Series stream endpoint"""
    return "Series not supported", 501


@app.route('/xmltv.php')
def xmltv_epg():
    """XMLTV EPG endpoint"""
    username = request.args.get('username')
    password = request.args.get('password')

    logger.info(f"XMLTV request: user={username}")

    # Generate XMLTV via adapter
    xmltv_data = adapter.generate_xmltv_epg(username, password)

    if not xmltv_data:
        return "Unauthorized", 401

    response = make_response(xmltv_data)
    response.headers['Content-Type'] = 'application/xml; charset=utf-8'
    return response


@app.route('/m3u/<username>/<password>')
def m3u_playlist(username, password):
    """M3U playlist endpoint"""
    logger.info(f"M3U request: user={username}")

    # Generate M3U via adapter
    m3u_data = adapter.generate_m3u_playlist(username, password)

    if not m3u_data:
        return "Unauthorized", 401

    response = make_response(m3u_data)
    response.headers['Content-Type'] = 'audio/x-mpegurl; charset=utf-8'
    response.headers['Content-Disposition'] = f'attachment; filename="{username}_playlist.m3u"'
    return response


@app.route('/get.php')
def get_stream():
    """Alternative stream endpoint"""
    username = request.args.get('username')
    password = request.args.get('password')
    stream_id = request.args.get('stream_id')
    stream_type = request.args.get('type', 'm3u8')

    if not username or not password or not stream_id:
        return "Missing parameters", 400

    try:
        stream_id_int = int(stream_id)
    except ValueError:
        return "Invalid stream ID", 400

    # Redirect to live stream endpoint
    return f'/live/{username}/{password}/{stream_id_int}', 302


# --- Admin/Utility Endpoints ---

@app.route('/admin/flush_cache')
def flush_cache():
    """Flush adapter cache (admin only)"""
    cache_type = request.args.get('type')
    result = adapter.flush_cache(cache_type)
    return jsonify(result)


@app.route('/admin/stats')
def get_stats():
    """Get adapter statistics"""
    stats = adapter.get_stats()
    return jsonify(stats)


@app.route('/admin/test_backend')
def test_backend():
    """Test connection to Ultimate Backend"""
    # Try to fetch providers
    providers = adapter._make_request("/api/providers")

    if providers:
        return jsonify({
            "status": "connected",
            "providers_count": len(providers.get('providers', [])),
            "backend_url": adapter.ultimate_backend_url
        })
    else:
        return jsonify({
            "status": "disconnected",
            "backend_url": adapter.ultimate_backend_url,
            "error": "Cannot connect to Ultimate Backend"
        }), 503


@app.route('/health')
def health():
    """Health check endpoint"""
    # Test backend connection
    providers = adapter._make_request("/api/providers")
    backend_ok = bool(providers)

    return jsonify({
        "status": "healthy" if backend_ok else "degraded",
        "backend_connected": backend_ok,
        "adapter": {
            "channels_count": len(adapter.get_channels()),
            "providers_count": len(adapter.cache['providers']['data']) if adapter.cache['providers']['data'] else 0
        }
    })


@app.route('/')
def index():
    """Welcome page"""
    stats = adapter.get_stats()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ultimate Xtream Adapter</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .info {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
            .endpoint {{ background: #e9f7fe; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; }}
            code {{ background: #eee; padding: 2px 5px; }}
        </style>
    </head>
    <body>
        <h1>Ultimate Xtream Adapter</h1>

        <div class="info">
            <h3>Backend: {adapter.ultimate_backend_url}</h3>
            <p><strong>Channels:</strong> {stats['channels_total']}</p>
            <p><strong>Providers:</strong> {stats['providers_count']}</p>
        </div>

        <h2>Xtream API Endpoints</h2>

        <div class="endpoint">
            <h3>Main API</h3>
            <p><code>GET /player_api.php?username=user&password=pass&action=get_live_categories</code></p>
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
    return jsonify({"error": "Not found", "endpoint": request.path}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    logger.info(f"Starting Ultimate Xtream Adapter on port {SERVER_PORT}")
    logger.info(f"Backend URL: {adapter.ultimate_backend_url}")
    logger.info(f"Default credentials: {adapter.default_username}/{adapter.default_password}")

    # Initial cache load
    logger.info("Loading initial data...")
    adapter._load_providers()
    adapter.get_channels()

    app.run(
        host='0.0.0.0',
        port=SERVER_PORT,
        debug=DEBUG_MODE
    )