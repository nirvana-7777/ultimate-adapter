#!/usr/bin/env python3
"""
Ultimate Adapter - API adapter for Ultimate Backend
Maps API calls to Ultimate Backend endpoints

IMPROVEMENTS v2.0:
- Deterministic stream_id generation using hash of provider:channel_id
- Easy URL reconstruction from stream_id without cache lookup
- Persistent mapping between stream_id and backend identifiers
- Channel IDs are stable across restarts and cache flushes
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

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
    Adapter that translates API calls to Ultimate Backend API

    Key Feature: Uses deterministic stream_id generation based on provider:channel_id
    This means the same channel always gets the same ID, making URLs easily reconstructable.
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

        # Channel mapping - now using deterministic IDs
        # These are rebuilt on demand and always produce the same results
        self.channel_map = {}  # stream_id -> (provider, channel_id)
        self.reverse_channel_map = {}  # (provider, channel_id) -> stream_id

        # Initialize
        self._load_providers()

        logger.info(f"Ultimate Adapter initialized for backend: {ultimate_backend_url}")

    @staticmethod
    def _generate_stream_id(provider_name: str, channel_id: str) -> int:
        """
        Generate a deterministic stream_id from provider name and channel ID.

        This ensures:
        1. Same channel always gets same stream_id (deterministic)
        2. Stream IDs survive restarts and cache flushes
        3. No dependency on load order
        4. Easy reverse lookup via channel_map

        Args:
            provider_name: Name of the provider
            channel_id: Original channel ID from backend

        Returns:
            Positive integer stream_id (max 2^31-1 for compatibility)
        """
        # Create unique string combining provider and channel
        unique_str = f"{provider_name}:{channel_id}"

        # Generate MD5 hash and convert to integer
        hash_obj = hashlib.md5(unique_str.encode())
        # Use first 4 bytes of hash to create an integer
        stream_id = int.from_bytes(hash_obj.digest()[:4], byteorder="big")

        # Ensure it's a positive integer (max 2^31 - 1)
        stream_id = stream_id & 0x7FFFFFFF

        # Ensure we don't get 0 (unlikely but possible)
        if stream_id == 0:
            stream_id = 1

        return stream_id

    def _parse_stream_id(self, stream_id: int) -> Optional[Tuple[str, str]]:
        """
        Parse a stream_id back to (provider_name, channel_id).

        Uses the mapping stored in channel_map which is built
        deterministically from the backend data.

        Args:
            stream_id: The stream ID to look up

        Returns:
            Tuple of (provider_name, channel_id) or None if not found
        """
        return self.channel_map.get(stream_id)

    def _build_stream_url(self, provider_name: str, channel_id: str) -> str:
        """
        Build the stream URL from provider name and channel ID.

        Format: /api/providers/{provider}/channels/{channel_id}/stream/decrypted/ffmpeg/index.mpd

        This is the SINGLE SOURCE OF TRUTH for stream URL format.

        Args:
            provider_name: Name of the provider
            channel_id: Original channel ID from backend

        Returns:
            Complete stream URL
        """
        return (
            f"{self.ultimate_backend_url}/api/providers/"
            f"{provider_name}/channels/{channel_id}/"
            f"stream/decrypted/ffmpeg/index.mpd"
        )

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
            logger.debug(f"Making request to backend: {method} {url}")

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
                    logger.debug(f"Backend returned non-JSON response for {url}")
                    return {"_raw": response.text}
            else:
                logger.error(
                    f"Backend request failed: {response.status_code} for {url}"
                )
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error making request: {e}")
            return None

    def _load_providers(self, force_refresh: bool = False) -> bool:
        """Load providers from Ultimate Backend"""
        cache_key = "providers"

        # Check cache (5 minute TTL)
        if not force_refresh and self.cache[cache_key]["data"]:
            cache_age = time.time() - self.cache[cache_key]["timestamp"]
            if cache_age < 300:  # 5 minutes
                logger.debug(f"Using cached providers (age: {cache_age:.1f}s)")
                return True

        try:
            logger.info("Loading providers from backend...")
            response = self._make_request("/api/providers")
            if not response:
                logger.error("Failed to fetch providers from backend")
                return False

            self.cache[cache_key]["data"] = response.get("providers", [])
            self.cache[cache_key]["timestamp"] = time.time()

            providers_count = len(self.cache[cache_key]["data"])
            logger.info(f"Loaded {providers_count} providers from backend")
            return True

        except Exception as e:
            logger.error(f"Error loading providers: {e}")
            return False

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user (simplified - in production use database)"""
        # For demo purposes, accept default credentials
        if username == self.default_username and password == self.default_password:
            logger.debug(f"Authentication successful for user: {username}")
            return User(
                username=username,
                password=password,
                exp_date="1999999999",  # Never expires
                max_connections=1,
                is_trial=0,
                active_cons=0,
                status="Active",
            )
        logger.warning(f"Authentication failed for user: {username}")
        return None

    def get_server_info(self) -> Dict:
        """Get server information for API"""
        logger.debug("Returning server info")
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
                logger.debug(f"Using cached categories (age: {cache_age:.1f}s)")
                return self.cache[cache_key]["data"]

        if not self._load_providers():
            logger.warning("Cannot load categories: providers not loaded")
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

        logger.info(f"Generated {len(categories)} categories")
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
                logger.debug(f"Using cached channels (age: {cache_age:.1f}s)")
                return self.cache[cache_key]["data"]

        return self._load_all_channels(category_id)

    def _get_channels_for_category(self, category_id: str) -> List[Channel]:
        """Load channels only for specific category (lazy loading)"""
        cache_key = f"channels_cat_{category_id}"

        # Check cache
        if cache_key not in self.cache:
            self.cache[cache_key] = {"data": None, "timestamp": 0}

        if self.cache[cache_key]["data"]:
            cache_age = time.time() - self.cache[cache_key]["timestamp"]
            if cache_age < 300:  # 5 minutes
                logger.debug(
                    f"Using cached channels for category {category_id} (age: {cache_age:.1f}s)"
                )
                return self.cache[cache_key]["data"]

        if not self._load_providers():
            logger.warning(
                f"Cannot load channels: providers not loaded for category {category_id}"
            )
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

        logger.info(
            f"Loading channels for category {category_id} (provider: {provider_name})"
        )

        channels = []

        try:
            endpoint = f"/api/providers/{provider_name}/channels"
            response = self._make_request(endpoint)

            if not response or "channels" not in response:
                logger.warning(f"No channels found for provider: {provider_name}")
                return []

            for channel_data in response["channels"]:
                # Backend uses capitalized fields: Name, Id, LogoUrl
                channel_id = channel_data.get("Id", "")
                channel_name = channel_data.get("Name", "Unknown")
                channel_logo = channel_data.get("LogoUrl", "")

                # Skip radio channels (optional)
                if channel_data.get("IsRadio", False):
                    continue

                # Generate deterministic stream_id from provider and channel
                stream_id = self._generate_stream_id(provider_name, channel_id)

                # Store mapping for reverse lookup
                self.channel_map[stream_id] = (provider_name, channel_id)
                self.reverse_channel_map[(provider_name, channel_id)] = stream_id

                # Build stream URL using helper method
                stream_url = self._build_stream_url(provider_name, channel_id)

                # Create channel object
                channel = Channel(
                    stream_id=stream_id,
                    num=stream_id,  # Use same value for consistency
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

        except Exception as e:
            logger.error(f"Error fetching channels for {provider_name}: {e}")
            return []

        # Cache the result
        self.cache[cache_key]["data"] = channels
        self.cache[cache_key]["timestamp"] = time.time()

        logger.info(f"Loaded {len(channels)} channels for category {category_id}")

        return channels

    def _load_all_channels(self, category_id: Optional[str] = None) -> List[Channel]:
        """Load channels from all providers"""
        cache_key = "channels"

        if not self._load_providers():
            logger.warning("Cannot load all channels: providers not loaded")
            return []

        all_channels = []

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
                    logger.warning(f"No channels found for provider: {provider_name}")
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

                    # Generate deterministic stream_id
                    stream_id = self._generate_stream_id(provider_name, channel_id)

                    # Store mapping
                    self.channel_map[stream_id] = (provider_name, channel_id)
                    self.reverse_channel_map[(provider_name, channel_id)] = stream_id

                    # Build stream URL
                    stream_url = self._build_stream_url(provider_name, channel_id)

                    # Create channel object
                    channel = Channel(
                        stream_id=stream_id,
                        num=stream_id,
                        name=channel_name,
                        stream_icon=channel_logo,
                        category_id=str(provider_idx),
                        category_name=provider_label,
                        epg_channel_id=channel_id,
                        direct_source=stream_url,
                        provider_name=provider_name,
                        original_channel_id=channel_id,
                    )

                    all_channels.append(channel)

            except Exception as e:
                logger.error(f"Error fetching channels for {provider_name}: {e}")
                continue

        # Cache results
        self.cache[cache_key]["data"] = all_channels
        self.cache[cache_key]["timestamp"] = time.time()

        logger.info(f"Loaded {len(all_channels)} channels from all providers")

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
            logger.warning(f"Category name not found for ID: {category_id}")
            return channels

        # Filter channels by category name
        filtered = [c for c in channels if c.category_name == category_name]
        logger.debug(f"Filtered channels: {len(filtered)} for category {category_name}")
        return filtered

    def get_channel_by_id(self, stream_id: int) -> Optional[Channel]:
        """
        Get a specific channel by stream ID.

        This method now supports reconstruction from the mapping if needed.
        Even if the channel isn't in cache, we can rebuild it from the mapping.
        """
        # First try to find in cached channels
        channels = self.get_channels()
        for channel in channels:
            if channel.stream_id == stream_id:
                logger.debug(f"Found channel {stream_id}: {channel.name}")
                return channel

        # If not in cache, try to reconstruct from mapping
        mapping = self._parse_stream_id(stream_id)
        if mapping:
            provider_name, channel_id = mapping
            logger.info(
                f"Reconstructing channel from mapping: {provider_name}:{channel_id}"
            )

            # Fetch just this channel's data
            try:
                endpoint = f"/api/providers/{provider_name}/channels"
                response = self._make_request(endpoint)
                if response and "channels" in response:
                    for channel_data in response["channels"]:
                        if channel_data.get("Id") == channel_id:
                            # Found it! Build the channel object
                            stream_url = self._build_stream_url(
                                provider_name, channel_id
                            )

                            channel = Channel(
                                stream_id=stream_id,
                                num=stream_id,
                                name=channel_data.get("Name", "Unknown"),
                                stream_icon=channel_data.get("LogoUrl", ""),
                                epg_channel_id=channel_id,
                                direct_source=stream_url,
                                provider_name=provider_name,
                                original_channel_id=channel_id,
                            )
                            logger.debug(
                                f"Reconstructed channel {stream_id}: {channel.name}"
                            )
                            return channel
            except Exception as e:
                logger.error(f"Error reconstructing channel {stream_id}: {e}")

        logger.warning(f"Channel not found for stream_id: {stream_id}")
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
                logger.debug(f"Using cached EPG (age: {cache_age:.1f}s)")
                epg_data = self.cache[cache_key]["data"]
                if stream_id:
                    filtered = [p for p in epg_data if p.channel_id == str(stream_id)]
                    logger.debug(
                        f"Returning {len(filtered[:limit])} EPG entries for stream {stream_id}"
                    )
                    return filtered[:limit]
                logger.debug(f"Returning {len(epg_data[:limit])} EPG entries")
                return epg_data[:limit]

        # Try to get EPG from Ultimate Backend
        try:
            # Check if Ultimate Backend has EPG endpoint
            response = self._make_request("/api/epg/xmltv-channels")
            if not response:
                # Return empty EPG
                logger.info("No EPG data available from backend")
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
                        description=f"Description for program {program_counter}",
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
                logger.info(
                    f"Generated {len(filtered[:limit])} dummy EPG entries for stream {stream_id}"
                )
                return filtered[:limit]

            logger.info(f"Generated {len(epg_programs[:limit])} dummy EPG entries")
            return epg_programs[:limit]

        except Exception as e:
            logger.error(f"Error fetching EPG: {e}")
            return []

    def generate_m3u_playlist(self, username: str, password: str) -> str:
        """Generate M3U playlist for user"""
        # Verify authentication
        user = self.authenticate(username, password)
        if not user:
            logger.warning(
                f"M3U generation failed: authentication failed for {username}"
            )
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

        logger.info(
            f"Generated M3U playlist with {len(channels)} channels for {username}"
        )
        return m3u_content

    def generate_xmltv_epg(self, username: str, password: str) -> str:
        """Generate XMLTV EPG data"""
        # Verify authentication
        user = self.authenticate(username, password)
        if not user:
            logger.warning(
                f"XMLTV generation failed: authentication failed for {username}"
            )
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

        logger.info(
            f"Generated XMLTV EPG with {len(channels)} channels and "
            f"{len(epg_data)} programs for {username}"
        )
        return xmltv

    def get_stream_url(
        self, username: str, password: str, stream_id: int
    ) -> Optional[str]:
        """
        Get actual stream URL for a channel.

        Now supports fast reconstruction from stream_id even if channel not in cache.
        This is the key benefit of deterministic IDs.
        """
        # Verify authentication
        user = self.authenticate(username, password)
        if not user:
            logger.warning(
                f"Stream URL request failed: authentication failed for {username}"
            )
            return None

        # Try to get from mapping first (faster - no channel lookup needed)
        mapping = self._parse_stream_id(stream_id)
        if mapping:
            provider_name, channel_id = mapping
            stream_url = self._build_stream_url(provider_name, channel_id)
            logger.info(
                f"Built stream URL from mapping for stream {stream_id}: "
                f"{provider_name}:{channel_id}"
            )
            return stream_url

        # Fallback to channel lookup (slower but works if mapping not built yet)
        channel = self.get_channel_by_id(stream_id)
        if not channel:
            logger.warning(
                f"Stream URL request failed: channel {stream_id} not found for {username}"
            )
            return None

        logger.info(f"Returning stream URL for channel {stream_id} to user {username}")
        return channel.direct_source

    def handle_api_request(self, action: str, params: Dict) -> Dict:
        """Handle API request and return appropriate response"""

        # Log all incoming requests
        username = params.get("username", "unknown")
        logger.info(
            f"Adapter handling API request - Action: '{action}', User: {username}"
        )

        # Extract common parameters
        username = params.get("username")
        password = params.get("password")

        # Verify authentication for most actions
        if action not in ["", "test"]:
            user = self.authenticate(username, password)
            if not user:
                logger.warning(f"API authentication failed for user: {username}")
                return {"error": "Invalid credentials"}

        # Handle different actions
        try:
            if action == "" or action == "test":
                # Return user info
                user = self.authenticate(username, password)
                if not user:
                    return {"error": "Invalid credentials"}

                logger.info(f"Test action completed for user: {username}")
                return {
                    "user_info": asdict(user),
                    "server_info": self.get_server_info(),
                }

            elif action == "get_live_categories":
                categories = self.get_categories(StreamType.LIVE)
                logger.info(
                    f"Returning {len(categories)} live categories for user: {username}"
                )
                return [asdict(cat) for cat in categories]

            elif action == "get_live_streams":
                category_id = params.get("category_id")
                channels = self.get_channels(category_id)
                logger.info(
                    f"Returning {len(channels)} live streams "
                    f"(category_id: {category_id}) for user: {username}"
                )
                return [asdict(channel) for channel in channels]

            elif action == "get_vod_categories":
                # VOD not currently supported
                logger.info(
                    f"VOD categories requested (not supported) by user: {username}"
                )
                return []

            elif action == "get_vod_streams":
                # VOD not currently supported
                logger.info(
                    f"VOD streams requested (not supported) by user: {username}"
                )
                return []

            elif action == "get_series_categories":
                # Series not currently supported
                logger.info(
                    f"Series categories requested (not supported) by user: {username}"
                )
                return []

            elif action == "get_series":
                # Series not currently supported
                logger.info(f"Series requested (not supported) by user: {username}")
                return []

            elif action == "get_short_epg":
                stream_id = params.get("stream_id")
                limit = int(params.get("limit", 12))

                if stream_id:
                    try:
                        stream_id_int = int(stream_id)
                        epg = self.get_epg(stream_id_int, limit)
                        logger.info(
                            f"Returning EPG for stream_id: {stream_id_int}, "
                            f"limit: {limit} for user: {username}"
                        )
                    except ValueError:
                        logger.warning(
                            f"Invalid stream_id format: {stream_id} from user: {username}"
                        )
                        epg = []
                else:
                    epg = self.get_epg(limit=limit)
                    logger.info(
                        f"Returning EPG for all streams, limit: {limit} for user: {username}"
                    )

                return [asdict(program) for program in epg]

            elif action == "get_simple_data_table":
                # Common but unsupported action
                logger.info(
                    f"get_simple_data_table requested (not implemented) by user: {username}"
                )
                return []

            elif action == "get_vod_info":
                logger.info(
                    f"get_vod_info requested (not supported) by user: {username}"
                )
                return {}

            elif action == "get_series_info":
                logger.info(
                    f"get_series_info requested (not supported) by user: {username}"
                )
                return {}

            elif action in ["get_languages", "get_countries"]:
                # Common actions that return empty for now
                logger.info(
                    f"Action '{action}' requested (returning empty) by user: {username}"
                )
                return []

            else:
                # Log unknown/unhandled action with full details
                logger.warning(f"UNHANDLED ACTION: '{action}' with params: {params}")

                # Log to a separate file for analysis
                try:
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/unhandled_requests.log", "a") as log_file:
                        log_file.write(
                            f"{datetime.now().isoformat()} - Action: {action}, "
                            f"Params: {params}\n"
                        )
                except Exception as e:
                    logger.error(f"Could not write to unhandled requests log: {e}")

                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            # Log any unexpected errors
            logger.error(
                f"Error handling action '{action}' for user {username}: {str(e)}",
                exc_info=True,
            )
            return {"error": f"Internal server error: {str(e)}"}

    def flush_cache(self, cache_type: Optional[str] = None) -> Dict:
        """Flush cache"""
        logger.info(f"Flushing cache - type: {cache_type}")

        if cache_type:
            if cache_type in self.cache:
                self.cache[cache_type] = {"data": None, "timestamp": 0}
                logger.info(f"Cache {cache_type} cleared")
                return {"message": f"Cache {cache_type} cleared"}
            else:
                logger.warning(f"Unknown cache type: {cache_type}")
                return {"error": f"Unknown cache type: {cache_type}"}
        else:
            # Clear all caches
            for key in self.cache:
                self.cache[key] = {"data": None, "timestamp": 0}
            # Don't clear the mappings - they're deterministic and will rebuild the same
            logger.info(
                "All caches cleared (mappings preserved as they're deterministic)"
            )
            return {"message": "All caches cleared"}

    def get_stats(self) -> Dict:
        """Get adapter statistics"""
        logger.debug("Generating adapter statistics")
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
            "mapping_info": {
                "total_mapped_channels": len(self.channel_map),
                "deterministic_ids": True,
                "id_generation": "MD5 hash of provider:channel_id",
            },
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
    """Test the adapter with focus on deterministic stream ID generation"""
    adapter = UltimateAdapter(
        ultimate_backend_url="http://localhost:7777",
        default_username="user",
        default_password="pass",
    )

    print("Testing Ultimate Adapter v2.0 (Deterministic Stream IDs)")
    print("=" * 70)

    # Test stream_id generation
    print("\n1. Testing Stream ID Generation:")
    print("-" * 70)
    test_cases = [
        ("provider1", "channel123"),
        ("provider1", "channel456"),
        ("provider2", "channel123"),  # Same channel ID, different provider
    ]

    for provider, channel_id in test_cases:
        stream_id = adapter._generate_stream_id(provider, channel_id)
        url = adapter._build_stream_url(provider, channel_id)
        print(f"  {provider}:{channel_id}")
        print(f"    → stream_id: {stream_id}")
        print(f"    → URL: {url}")
        print()

    # Verify determinism
    print("2. Testing Determinism:")
    print("-" * 70)
    id1 = adapter._generate_stream_id("test_provider", "test_channel")
    id2 = adapter._generate_stream_id("test_provider", "test_channel")
    print(f"  First call:  {id1}")
    print(f"  Second call: {id2}")
    if id1 == id2:
        print("  ✓ Deterministic: IDs match")
    else:
        print("  ✗ ERROR: IDs don't match!")

    # Test authentication
    print("\n3. Testing Authentication:")
    print("-" * 70)
    user = adapter.authenticate("user", "pass")
    print(f"  Result: {'✓ SUCCESS' if user else '✗ FAILED'}")

    # Test categories
    print("\n4. Testing Categories:")
    print("-" * 70)
    categories = adapter.get_categories()
    print(f"  Total categories: {len(categories)}")
    for cat in categories[:3]:
        print(f"    - {cat.category_name} (ID: {cat.category_id})")

    # Test channels
    print("\n5. Testing Channels:")
    print("-" * 70)
    channels = adapter.get_channels()
    print(f"  Total channels: {len(channels)}")
    if channels:
        print("  Showing first 3 channels:")
        for ch in channels[:3]:
            print(f"    • {ch.name}")
            print(f"      stream_id: {ch.stream_id}")
            print(f"      provider: {ch.provider_name}")
            print(f"      channel_id: {ch.original_channel_id}")
            print(f"      URL: {ch.direct_source}")
            print()

    # Test reverse lookup
    if channels:
        print("6. Testing Reverse Lookup:")
        print("-" * 70)
        test_channel = channels[0]
        mapping = adapter._parse_stream_id(test_channel.stream_id)
        if mapping:
            provider, channel_id = mapping
            print(f"  stream_id {test_channel.stream_id} maps to:")
            print(f"    Provider: {provider}")
            print(f"    Channel ID: {channel_id}")
            reconstructed_url = adapter._build_stream_url(provider, channel_id)
            print(f"    Reconstructed URL: {reconstructed_url}")
            matches = reconstructed_url == test_channel.direct_source
            print(f"    Matches original: {'✓ YES' if matches else '✗ NO'}")

    # Get stats
    print("\n7. Adapter Statistics:")
    print("-" * 70)
    stats = adapter.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    print("=" * 70)
    print("Test complete!")


if __name__ == "__main__":
    # Run test if executed directly
    test_adapter()
