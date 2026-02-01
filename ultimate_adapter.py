#!/usr/bin/env python3
"""
Ultimate Adapter - Xtream API adapter for Ultimate Backend
Maps Xtream API calls to Ultimate Backend endpoints
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StreamType(Enum):
    LIVE = "live"
    VOD = "vod"
    SERIES = "series"


@dataclass
class User:
    username: str
    password: str
    exp_date: str = "1999999999"  # Unix timestamp
    max_connections: int = 1
    is_trial: int = 0
    active_cons: int = 0
    created_at: str = "1609459200"  # 2021-01-01
    status: str = "Active"


@dataclass
class Channel:
    stream_id: int
    num: int
    name: str
    stream_type: str = "live"
    stream_icon: str = ""
    category_id: str = "1"
    category_name: str = ""
    epg_channel_id: str = ""
    added: str = "1609459200"
    is_adult: int = 0
    custom_sid: str = ""
    tv_archive: int = 0
    direct_source: str = ""

    # Ultimate Backend specific
    provider_name: str = ""
    original_channel_id: str = ""


@dataclass
class Category:
    category_id: str
    category_name: str
    parent_id: int = 0


@dataclass
class EPGProgram:
    id: str
    epg_id: str
    title: str
    description: str = ""
    start: str = ""
    end: str = ""
    start_timestamp: str = ""
    stop_timestamp: str = ""
    channel_id: str = ""


class UltimateAdapter:
    """
    Adapter that translates Xtream API calls to Ultimate Backend API
    calls
    """

    def __init__(
            self,
            ultimate_backend_url: str,
            default_username: str = "user",
            default_password: str = "pass",
    ):

        self.ultimate_backend_url = ultimate_backend_url.rstrip("/")
        self.default_username = default_username
        self.default_password = default_password

        # Cache for performance
        self.cache = {
            "channels": {"data": None, "timestamp": 0},
            "categories": {"data": None, "timestamp": 0},
            "providers": {"data": None, "timestamp": 0},
            "epg": {"data": None, "timestamp": 0},
        }

        # Channel mapping
        self.channel_map = {}  # stream_id -> (provider, channel_id)
        self.reverse_channel_map = {}  # (provider, channel_id) -> stream_id

        # Initialize
        self._load_providers()

        init_msg = (
            f"Ultimate Adapter initialized for backend: " f"{ultimate_backend_url}"
        )
        logger.info(init_msg)

    def _make_request(
            self,
            endpoint: str,
            method: str = "GET",
            params: Optional[Dict] = None,
            data: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make HTTP request to Ultimate Backend"""
        url = f"{self.ultimate_backend_url}/{endpoint.lstrip('/')}"

        headers = {"User-Agent": "UltimateAdapter/1.0", "Accept": "application/json"}

        try:
            if method == "GET":
                response = requests.get(
                    url, params=params, headers=headers, timeout=(10, 60)
                )
            elif method == "POST":
                headers["Content-Type"] = "application/json"
                response = requests.post(
                    url, json=data, headers=headers, timeout=(10, 60)
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()

            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    # Some endpoints return plain text (M3U)
                    return {"_raw": response.text}
            else:
                logger.error(f"Request failed: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    def _load_providers(self, force_refresh: bool = False) -> bool:
        """Load providers from Ultimate Backend"""
        cache_key = "providers"

        # Check cache (5 minute TTL)
        if not force_refresh and self.cache[cache_key]["data"]:
            cache_age = time.time() - self.cache[cache_key]["timestamp"]
            if cache_age < 300:  # 5 minutes
                return True

        try:
            response = self._make_request("/api/providers")
            if not response:
                logger.error("Failed to fetch providers")
                return False

            self.cache[cache_key]["data"] = response.get("providers", [])
            self.cache[cache_key]["timestamp"] = time.time()

            providers_count = len(self.cache[cache_key]["data"])
            logger.info(f"Loaded {providers_count} providers")
            return True

        except Exception as e:
            logger.error(f"Error loading providers: {e}")
            return False

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user (simplified - in production use database)"""
        # For demo purposes, accept default credentials
        if username == self.default_username and password == self.default_password:
            return User(
                username=username,
                password=password,
                exp_date="1999999999",  # Never expires
                max_connections=1,
                is_trial=0,
                active_cons=0,
                status="Active",
            )
        return None

    def get_server_info(self) -> Dict:
        """Get server information for Xtream API"""
        return {
            "url": self.ultimate_backend_url,
            "port": "7777",
            "rtmp_port": "1935",
            "timezone": "UTC",
            "time_now": str(int(time.time())),
            "process": True,
        }

    def get_categories(
            self, stream_type: StreamType = StreamType.LIVE
    ) -> List[Category]:
        """Get categories (providers become categories)"""
        cache_key = "categories"

        # Check cache
        if self.cache[cache_key]["data"]:
            cache_age = time.time() - self.cache[cache_key]["timestamp"]
            if cache_age < 300:  # 5 minutes
                return self.cache[cache_key]["data"]

        if not self._load_providers():
            return []

        categories = []
        providers = self.cache["providers"]["data"]

        # Add "All" category
        categories.append(
            Category(category_id="0", category_name="All Channels", parent_id=0)
        )

        # Each provider becomes a category
        for idx, provider in enumerate(providers, 1):
            provider_label = provider.get(
                "label", provider.get("name", f"Provider {idx}")
            )
            categories.append(
                Category(
                    category_id=str(idx), category_name=provider_label, parent_id=0
                )
            )

        self.cache[cache_key]["data"] = categories
        self.cache[cache_key]["timestamp"] = time.time()

        return categories

    def get_channels(self, category_id: Optional[str] = None) -> List[Channel]:
        """Get channels, optionally filtered by category"""

        # If specific category requested, use lazy loading
        if category_id and category_id != "0":
            return self._get_channels_for_category(category_id)

        # Load all channels (for category "0" or no category)
        cache_key = "channels"

        # Check cache
        if self.cache[cache_key]["data"]:
            cache_age = time.time() - self.cache[cache_key]["timestamp"]
            if cache_age < 300:  # 5 minutes
                return self.cache[cache_key]["data"]

        return self._load_all_channels()

    def _get_channels_for_category(self, category_id: str) -> List[Channel]:
        """Load channels only for specific category (lazy loading)"""
        cache_key = f"channels_cat_{category_id}"

        # Check cache
        if cache_key not in self.cache:
            self.cache[cache_key] = {"data": None, "timestamp": 0}

        if self.cache[cache_key]["data"]:
            cache_age = time.time() - self.cache[cache_key]["timestamp"]
            if cache_age < 300:  # 5 minutes
                return self.cache[cache_key]["data"]

        if not self._load_providers():
            return []

        providers = self.cache["providers"]["data"]

        # Find the provider for this category
        try:
            provider_idx = int(category_id)
            if provider_idx < 1 or provider_idx > len(providers):
                logger.warning(f"Invalid category_id: {category_id}")
                return []
            provider = providers[provider_idx - 1]
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing category_id {category_id}: {e}")
            return []

        provider_name = provider.get("name")
        provider_label = provider.get("label", provider_name)

        logger.info(f"Loading channels for category {category_id} "
                    f"(provider: {provider_name})")

        channels = []
        stream_counter = 1

        try:
            endpoint = f"/api/providers/{provider_name}/channels"
            response = self._make_request(endpoint)

            if not response or "channels" not in response:
                logger.warning(f"No channels found for provider: "
                               f"{provider_name}")
                return []

            for channel_data in response["channels"]:
                # Backend uses capitalized fields: Name, Id, LogoUrl
                channel_id = channel_data.get("Id", "")
                channel_name = channel_data.get("Name", "Unknown")
                channel_logo = channel_data.get("LogoUrl", "")

                # Skip radio channels (optional)
                if channel_data.get("IsRadio", False):
                    continue

                # Create stream URL
                stream_url = (
                    f"{self.ultimate_backend_url}/api/providers/"
                    f"{provider_name}/channels/{channel_id}/"
                    f"stream/decrypted/index.mpd"
                )

                # Create channel object
                channel = Channel(
                    stream_id=stream_counter,
                    num=stream_counter,
                    name=channel_name,
                    stream_icon=channel_logo,
                    category_id=category_id,
                    category_name=provider_label,
                    epg_channel_id=channel_id,
                    direct_source=stream_url,
                    provider_name=provider_name,
                    original_channel_id=channel_id,
                )

                channels.append(channel)
                stream_counter += 1

        except Exception as e:
            logger.error(f"Error fetching channels for {provider_name}: {e}")
            return []

        # Cache the result
        self.cache[cache_key]["data"] = channels
        self.cache[cache_key]["timestamp"] = time.time()

        logger.info(f"Loaded {len(channels)} channels for category "
                    f"{category_id}")

        return channels

    def _load_all_channels(self) -> List[Channel]:
        """Load channels from all providers"""

    def _load_all_channels(self) -> List[Channel]:
        """Load channels from all providers"""
        cache_key = "channels"

        if not self._load_providers():
            return []

        all_channels = []
        stream_counter = 1

        providers = self.cache["providers"]["data"]

        logger.info(f"Loading channels from all {len(providers)} providers...")

        for provider_idx, provider in enumerate(providers, 1):
            provider_name = provider.get("name")
            provider_label = provider.get("label", provider_name)

            # Fetch channels for this provider
            try:
                endpoint = f"/api/providers/{provider_name}/channels"
                response = self._make_request(endpoint)
                if not response or "channels" not in response:
                    log_msg = f"No channels found for provider: " f"{provider_name}"
                    logger.warning(log_msg)
                    continue

                provider_channels = response["channels"]

                for channel_data in provider_channels:
                    # Backend uses capitalized fields: Name, Id, LogoUrl
                    channel_id = channel_data.get("Id", "")
                    channel_name = channel_data.get("Name", "Unknown")
                    channel_logo = channel_data.get("LogoUrl", "")

                    # Skip radio channels (optional)
                    if channel_data.get("IsRadio", False):
                        continue

                    # Create stream URL
                    stream_url = (
                        f"{self.ultimate_backend_url}/api/providers/"
                        f"{provider_name}/channels/{channel_id}/"
                        f"stream/decrypted/index.mpd"
                    )

                    # Create channel object
                    channel = Channel(
                        stream_id=stream_counter,
                        num=stream_counter,
                        name=channel_name,
                        stream_icon=channel_logo,
                        category_id=str(provider_idx),
                        category_name=provider_label,
                        epg_channel_id=channel_id,
                        direct_source=stream_url,
                        provider_name=provider_name,
                        original_channel_id=channel_id,
                    )

                    # Store mapping
                    self.channel_map[stream_counter] = (provider_name, channel_id)
                    self.reverse_channel_map[(provider_name, channel_id)] = (
                        stream_counter
                    )

                    all_channels.append(channel)
                    stream_counter += 1

            except Exception as e:
                log_msg = f"Error fetching channels for {provider_name}: {e}"
                logger.error(log_msg)
                continue

        # Cache results
        self.cache[cache_key]["data"] = all_channels
        self.cache[cache_key]["timestamp"] = time.time()

        logger.info(f"Loaded {len(all_channels)} channels")

        # Filter if category specified
        return self._filter_channels_by_category(all_channels, category_id)

    def _filter_channels_by_category(
            self, channels: List[Channel], category_id: Optional[str]
    ) -> List[Channel]:
        """Filter channels by category ID"""
        if not category_id or category_id == "0":
            return channels

        # Find category name from category ID
        categories = self.get_categories()
        category_name = None
        for cat in categories:
            if cat.category_id == category_id:
                category_name = cat.category_name
                break

        if not category_name:
            return channels

        # Filter channels by category name
        return [c for c in channels if c.category_name == category_name]

    def get_channel_by_id(self, stream_id: int) -> Optional[Channel]:
        """Get a specific channel by stream ID"""
        channels = self.get_channels()
        for channel in channels:
            if channel.stream_id == stream_id:
                return channel
        return None

    def get_epg(
            self, stream_id: Optional[int] = None, limit: int = 12
    ) -> List[EPGProgram]:
        """Get EPG data for a channel or all channels"""
        cache_key = "epg"

        # Check cache (1 hour TTL for EPG)
        if self.cache[cache_key]["data"]:
            cache_age = time.time() - self.cache[cache_key]["timestamp"]
            if cache_age < 3600:  # 1 hour
                epg_data = self.cache[cache_key]["data"]
                if stream_id:
                    filtered = [p for p in epg_data if p.channel_id == str(stream_id)]
                    return filtered[:limit]
                return epg_data[:limit]

        # Try to get EPG from Ultimate Backend
        try:
            # Check if Ultimate Backend has EPG endpoint
            response = self._make_request("/api/epg/xmltv-channels")
            if not response:
                # Return empty EPG
                self.cache[cache_key]["data"] = []
                self.cache[cache_key]["timestamp"] = time.time()
                return []

            # For now, create dummy EPG data
            # In a real implementation, you would parse XMLTV data
            epg_programs = []
            program_counter = 1

            for channel in self.get_channels():
                # Create some dummy programs
                for i in range(3):
                    start_time = datetime.now() + timedelta(hours=i)
                    end_time = start_time + timedelta(hours=1)

                    program = EPGProgram(
                        id=str(program_counter),
                        epg_id=str(program_counter),
                        title=f"Program {program_counter}",
                        description=f"Description for program " f"{program_counter}",
                        start=start_time.strftime("%H:%M"),
                        end=end_time.strftime("%H:%M"),
                        start_timestamp=str(int(start_time.timestamp())),
                        stop_timestamp=str(int(end_time.timestamp())),
                        channel_id=str(channel.stream_id),
                    )
                    epg_programs.append(program)
                    program_counter += 1

            self.cache[cache_key]["data"] = epg_programs
            self.cache[cache_key]["timestamp"] = time.time()

            if stream_id:
                filtered = [p for p in epg_programs if p.channel_id == str(stream_id)]
                return filtered[:limit]
            return epg_programs[:limit]

        except Exception as e:
            logger.error(f"Error fetching EPG: {e}")
            return []

    def generate_m3u_playlist(self, username: str, password: str) -> str:
        """Generate M3U playlist for user"""
        # Verify authentication
        user = self.authenticate(username, password)
        if not user:
            return ""

        channels = self.get_channels()

        m3u_content = "#EXTM3U\n"

        for channel in channels:
            # EXTINF line with metadata
            extinf = (
                f'#EXTINF:-1 tvg-id="{channel.epg_channel_id}" '
                f'tvg-logo="{channel.stream_icon}" '
                f'group-title="{channel.category_name}",{channel.name}'
            )
            m3u_content += extinf + "\n"

            # Add stream URL
            m3u_content += f"{channel.direct_source}\n"

        return m3u_content

    def generate_xmltv_epg(self, username: str, password: str) -> str:
        """Generate XMLTV EPG data"""
        # Verify authentication
        user = self.authenticate(username, password)
        if not user:
            return ""

        channels = self.get_channels()
        epg_data = self.get_epg()

        # Build XMLTV structure
        xmltv = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xmltv += (
            '<tv generator-info-name="Ultimate Adapter" '
            'source-info-name="Ultimate Backend">\n'
        )

        # Add channels
        for channel in channels:
            xmltv += f'  <channel id="{channel.epg_channel_id}">\n'
            xmltv += f"    <display-name>{channel.name}</display-name>\n"
            if channel.stream_icon:
                xmltv += f'    <icon src="{channel.stream_icon}"/>\n'
            xmltv += "  </channel>\n"

        # Add programs
        for program in epg_data:
            prog_start = f"{program.start_timestamp} +0000"
            prog_stop = f"{program.stop_timestamp} +0000"
            xmltv += (
                f'  <programme start="{prog_start}" '
                f'stop="{prog_stop}" '
                f'channel="{program.channel_id}">\n'
            )
            xmltv += f"    <title>{program.title}</title>\n"
            if program.description:
                xmltv += f"    <desc>{program.description}</desc>\n"
            xmltv += "  </programme>\n"

        xmltv += "</tv>"

        return xmltv

    def get_stream_url(
            self, username: str, password: str, stream_id: int
    ) -> Optional[str]:
        """Get actual stream URL for a channel"""
        # Verify authentication
        user = self.authenticate(username, password)
        if not user:
            return None

        # Get channel info
        channel = self.get_channel_by_id(stream_id)
        if not channel:
            return None

        return channel.direct_source

    def handle_xtream_request(self, action: str, params: Dict) -> Dict:
        """Handle Xtream API request and return appropriate response"""

        # Extract common parameters
        username = params.get("username")
        password = params.get("password")

        # Verify authentication for most actions
        if action not in ["", "test"]:
            user = self.authenticate(username, password)
            if not user:
                return {"error": "Invalid credentials"}

        # Handle different actions
        if action == "" or action == "test":
            # Return user info
            user = self.authenticate(username, password)
            if not user:
                return {"error": "Invalid credentials"}

            return {"user_info": asdict(user), "server_info": self.get_server_info()}

        elif action == "get_live_categories":
            categories = self.get_categories(StreamType.LIVE)
            return [asdict(cat) for cat in categories]

        elif action == "get_live_streams":
            category_id = params.get("category_id")
            channels = self.get_channels(category_id)
            return [asdict(channel) for channel in channels]

        elif action == "get_vod_categories":
            # VOD not currently supported
            return []

        elif action == "get_vod_streams":
            # VOD not currently supported
            return []

        elif action == "get_series_categories":
            # Series not currently supported
            return []

        elif action == "get_series":
            # Series not currently supported
            return []

        elif action == "get_short_epg":
            stream_id = params.get("stream_id")
            limit = int(params.get("limit", 12))

            if stream_id:
                try:
                    stream_id_int = int(stream_id)
                    epg = self.get_epg(stream_id_int, limit)
                except ValueError:
                    epg = []
            else:
                epg = self.get_epg(limit=limit)

            return [asdict(program) for program in epg]

        else:
            return {"error": f"Unknown action: {action}"}

    def flush_cache(self, cache_type: Optional[str] = None) -> Dict:
        """Flush cache"""
        if cache_type:
            if cache_type in self.cache:
                self.cache[cache_type] = {"data": None, "timestamp": 0}
                return {"message": f"Cache {cache_type} cleared"}
            else:
                return {"error": f"Unknown cache type: {cache_type}"}
        else:
            # Clear all caches
            for key in self.cache:
                self.cache[key] = {"data": None, "timestamp": 0}
            self.channel_map.clear()
            self.reverse_channel_map.clear()
            return {"message": "All caches cleared"}

    def get_stats(self) -> Dict:
        """Get adapter statistics"""
        channels = self.get_channels()

        providers_data = self.cache["providers"]["data"]
        providers_count = len(providers_data) if providers_data else 0

        return {
            "channels_total": len(channels),
            "providers_count": providers_count,
            "categories_count": len(self.get_categories()),
            "cache_status": {
                key: "valid" if self.cache[key]["data"] else "empty"
                for key in self.cache
            },
            "channel_map_size": len(self.channel_map),
            "backend_url": self.ultimate_backend_url,
        }


# Convenience functions for Flask app
def create_adapter() -> UltimateAdapter:
    """Factory function to create adapter instance"""
    backend_url = os.environ.get("ULTIMATE_BACKEND_URL", "http://ultimate-backend:7777")
    username = os.environ.get("DEFAULT_USERNAME", "user")
    password = os.environ.get("DEFAULT_PASSWORD", "pass")

    return UltimateAdapter(
        ultimate_backend_url=backend_url,
        default_username=username,
        default_password=password,
    )


# Test function
def test_adapter():
    """Test the adapter"""
    adapter = UltimateAdapter(
        ultimate_backend_url="http://localhost:7777",
        default_username="user",
        default_password="pass",
    )

    print("Testing Ultimate Adapter...")
    print("=" * 50)

    # Test authentication
    user = adapter.authenticate("user", "pass")
    print(f"Authentication: {'SUCCESS' if user else 'FAILED'}")

    # Test getting categories
    categories = adapter.get_categories()
    print(f"Categories: {len(categories)} found")
    for cat in categories[:3]:  # Show first 3
        print(f"  - {cat.category_name} (ID: {cat.category_id})")

    # Test getting channels
    channels = adapter.get_channels()
    print(f"Channels: {len(channels)} found")
    for ch in channels[:3]:  # Show first 3
        print(f"  - {ch.name} (ID: {ch.stream_id})")

    # Test Xtream request
    print("\nTesting Xtream API simulation...")
    response = adapter.handle_xtream_request(
        "get_live_categories", {"username": "user", "password": "pass"}
    )
    result = len(response) if isinstance(response, list) else "error"
    print(f"Categories response: {result}")

    # Test M3U generation
    m3u = adapter.generate_m3u_playlist("user", "pass")
    print(f"\nM3U generated: {'YES' if m3u else 'NO'}")
    if m3u:
        lines = m3u.split("\n")
        print(f"  First 3 lines: {lines[:3]}")

    # Get stats
    stats = adapter.get_stats()
    print("\nAdapter Stats:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    print("=" * 50)
    print("Test complete!")


if __name__ == "__main__":
    # Run test if executed directly
    test_adapter()