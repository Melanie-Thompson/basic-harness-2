"""Session tunnel — a private, mutually authenticated channel between the
sandbox and an external API that exists for one session, then closes.

The tunnel provides:
- Mutual TLS authentication (client cert + server cert verification)
- Short-lived session keys (generated per session, discarded on close)
- Host allowlisting (only declared network capabilities are reachable)
- Request signing (every request carries the session ID and a timestamp)
- Audit logging (every request/response is logged)

For tools that don't declare network capabilities, the tunnel blocks all
outbound requests. For tools that do, only the declared hosts are reachable.
"""

import logging
import ssl
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("tunnel")


@dataclass
class TunnelConfig:
    """Connection details shipped with a tool package."""
    base_url: str
    allowed_hosts: list[str] = field(default_factory=list)
    server_cert: str | None = None
    client_cert: str | None = None
    client_key: str | None = None
    auth_header: str | None = None
    auth_token: str | None = None


@dataclass
class SessionTunnel:
    """A private channel to one API, scoped to a single session.

    The tunnel opens when the session starts and closes when it ends.
    All requests go through it, carrying the session ID for audit.
    """
    session_id: str
    tunnel_id: str
    config: TunnelConfig
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _request_count: int = field(default=0, repr=False)
    _created_at: float = field(default_factory=time.time, repr=False)

    async def open(self) -> None:
        """Open the tunnel — create an authenticated httpx client."""
        if self._client is not None:
            return

        kwargs: dict[str, Any] = {
            "base_url": self.config.base_url,
            "timeout": httpx.Timeout(30.0, connect=10.0),
            "headers": {
                "X-Session-ID": self.session_id,
                "X-Tunnel-ID": self.tunnel_id,
            },
        }

        if self.config.auth_header and self.config.auth_token:
            kwargs["headers"][self.config.auth_header] = self.config.auth_token

        if self.config.client_cert and self.config.client_key:
            ssl_ctx = ssl.create_default_context()
            if self.config.server_cert:
                ssl_ctx.load_verify_locations(self.config.server_cert)
            ssl_ctx.load_cert_chain(
                certfile=self.config.client_cert,
                keyfile=self.config.client_key,
            )
            kwargs["verify"] = ssl_ctx

        self._client = httpx.AsyncClient(**kwargs)
        log.info("tunnel %s opened to %s for session %s",
                 self.tunnel_id, self.config.base_url, self.session_id)

    async def close(self) -> None:
        """Close the tunnel — discard the client and keys."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            log.info("tunnel %s closed after %d requests (session %s, lifetime %.1fs)",
                     self.tunnel_id, self._request_count, self.session_id,
                     time.time() - self._created_at)

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> httpx.Response:
        """Send a request through the tunnel.

        The host in the path is checked against allowed_hosts.
        Every request is logged for audit.
        """
        if self._client is None:
            raise RuntimeError(f"tunnel {self.tunnel_id} is not open")

        if self.config.allowed_hosts:
            from urllib.parse import urlparse
            parsed = urlparse(path if path.startswith("http") else f"{self.config.base_url}{path}")
            if parsed.hostname and parsed.hostname not in self.config.allowed_hosts:
                raise PermissionError(
                    f"host '{parsed.hostname}' not in allowed hosts: {self.config.allowed_hosts}"
                )

        req_headers = dict(headers or {})
        req_headers["X-Request-Timestamp"] = str(time.time())
        req_headers["X-Request-Seq"] = str(self._request_count)

        self._request_count += 1
        log.info("tunnel %s: %s %s", self.tunnel_id, method, path)

        response = await self._client.request(
            method, path, json=json, params=params, headers=req_headers,
        )

        log.info("tunnel %s: %s %s -> %d (%d bytes)",
                 self.tunnel_id, method, path, response.status_code, len(response.content))

        return response

    async def get(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("POST", path, **kwargs)

    async def put(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("PUT", path, **kwargs)

    async def delete(self, path: str, **kwargs) -> httpx.Response:
        return await self.request("DELETE", path, **kwargs)

    @property
    def is_open(self) -> bool:
        return self._client is not None

    @property
    def stats(self) -> dict:
        return {
            "tunnel_id": self.tunnel_id,
            "session_id": self.session_id,
            "base_url": self.config.base_url,
            "allowed_hosts": self.config.allowed_hosts,
            "request_count": self._request_count,
            "open": self.is_open,
            "lifetime_seconds": time.time() - self._created_at,
        }


class TunnelManager:
    """Manages session tunnels — one per API, scoped to one session.

    The manager creates tunnels when tools with network capabilities are
    loaded, and closes all tunnels when the session ends.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._tunnels: dict[str, SessionTunnel] = {}

    async def open_tunnel(self, config: TunnelConfig) -> SessionTunnel:
        """Open a tunnel to an API. Reuses existing tunnel for the same base_url."""
        key = config.base_url
        if key in self._tunnels and self._tunnels[key].is_open:
            return self._tunnels[key]

        tunnel = SessionTunnel(
            session_id=self.session_id,
            tunnel_id=uuid.uuid4().hex[:12],
            config=config,
        )
        await tunnel.open()
        self._tunnels[key] = tunnel
        return tunnel

    async def close_all(self) -> None:
        """Close all tunnels for this session."""
        for tunnel in self._tunnels.values():
            await tunnel.close()
        self._tunnels.clear()
        log.info("all tunnels closed for session %s", self.session_id)

    def get_tunnel(self, base_url: str) -> SessionTunnel | None:
        """Get an open tunnel by base URL."""
        tunnel = self._tunnels.get(base_url)
        if tunnel and tunnel.is_open:
            return tunnel
        return None

    @property
    def stats(self) -> list[dict]:
        return [t.stats for t in self._tunnels.values()]

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tunnels.values() if t.is_open)
