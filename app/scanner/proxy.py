"""Validating Egress Forward Proxy for PhishVision AI Scanner.

Enforces actual network-level connection safety and IP pinning by intercepting
all outbound HTTP and HTTPS CONNECT requests from the browser, validating every
destination against Anti-SSRF policies, and binding outbound TCP sockets strictly
to pre-validated public IP addresses while preserving TLS SNI and Host headers.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
from typing import Any

from app.scanner.anti_ssrf import validate_url_and_resolve

logger = logging.getLogger(__name__)


class ValidatingEgressProxy:
    """Ephemeral local forward proxy enforcing connection-level Anti-SSRF and IP pinning."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.requested_port = port
        self.port: int = 0
        self.server: asyncio.Server | None = None
        self._active_tasks: set[asyncio.Task] = set()
        self.intercepted_requests: list[dict[str, Any]] = []
        self.blocked_requests: list[dict[str, Any]] = []

    async def start(self) -> int:
        """Start the proxy server on an available local port and return the assigned port."""
        self.server = await asyncio.start_server(
            self._handle_client,
            host=self.host,
            port=self.requested_port
        )
        sockets = self.server.sockets
        if not sockets:
            raise RuntimeError("Failed to bind proxy server to any socket")
        self.port = sockets[0].getsockname()[1]
        logger.debug("Validating egress proxy listening on %s:%d", self.host, self.port)
        return self.port

    async def stop(self) -> None:
        """Stop the proxy server and close all active forwarding pipes."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

        # Cancel any active bidirectional streaming tasks
        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()

    async def __aenter__(self) -> ValidatingEgressProxy:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    @property
    def proxy_url(self) -> str:
        """Return the proxy server URL for Playwright configuration."""
        if not self.port:
            raise RuntimeError("Proxy has not been started")
        return f"http://{self.host}:{self.port}"

    async def _handle_client(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task:
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)

        try:
            # Read HTTP request line
            line = await client_reader.readline()
            if not line:
                return

            line_str = line.decode("iso-8859-1", errors="replace").strip()
            parts = line_str.split()
            if len(parts) < 2:
                client_writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                await client_writer.drain()
                return

            method, target = parts[0].upper(), parts[1]

            if method == "CONNECT":
                await self._handle_connect(target, client_reader, client_writer)
            else:
                await self._handle_http_forward(method, target, line, client_reader, client_writer)

        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            logger.debug("Proxy client handler error: %s", e)
        finally:
            try:
                client_writer.close()
                await client_writer.wait_closed()
            except Exception:
                pass

    async def _handle_connect(
        self,
        target: str,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter
    ) -> None:
        """Handle HTTPS CONNECT tunnels with strict destination validation and IP pinning."""
        # Target format: host:port
        if ":" in target:
            target_host, port_str = target.rsplit(":", 1)
            try:
                target_port = int(port_str)
            except ValueError:
                target_port = 443
        else:
            target_host = target
            target_port = 443

        # Read remaining headers (up to double CRLF)
        while True:
            header_line = await client_reader.readline()
            if not header_line or header_line == b"\r\n" or header_line == b"\n":
                break

        # Validate destination target against Anti-SSRF policy
        validation_url = f"https://{target_host}:{target_port}"
        val_result = validate_url_and_resolve(validation_url)

        self.intercepted_requests.append({
            "type": "CONNECT",
            "target": target,
            "target_host": target_host,
            "target_port": target_port,
            "is_valid": val_result.is_valid,
            "pinned_ip": val_result.pinned_ip,
            "error_code": val_result.error_code,
        })

        if not val_result.is_valid:
            self.blocked_requests.append({
                "target": target,
                "reason": val_result.error_message,
                "code": val_result.error_code,
            })
            client_writer.write(
                b"HTTP/1.1 403 Forbidden\r\n"
                b"Content-Type: text/plain\r\n"
                b"Connection: close\r\n\r\n"
                b"Blocked by PhishVision Anti-SSRF Egress Policy\r\n"
            )
            await client_writer.drain()
            return

        # Connect directly to the validated, pinned IP address!
        pinned_ip = val_result.pinned_ip
        if not pinned_ip:
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\nNo IP resolved\r\n")
            await client_writer.drain()
            return

        try:
            remote_reader, remote_writer = await asyncio.wait_for(
                asyncio.open_connection(pinned_ip, target_port),
                timeout=12.0
            )
        except Exception as e:
            logger.debug("Failed connecting to pinned IP %s:%d: %s", pinned_ip, target_port, e)
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\nConnection to remote host failed\r\n")
            await client_writer.drain()
            return

        # Reply 200 Connection Established to the browser
        client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await client_writer.drain()

        # Pipe raw encrypted TLS stream bidirectionally
        await self._pipe_bidirectional(client_reader, client_writer, remote_reader, remote_writer)

    async def _handle_http_forward(
        self,
        method: str,
        target: str,
        initial_line: bytes,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter
    ) -> None:
        """Handle standard plain HTTP proxy requests."""
        # Target may be full URL: http://example.com/path
        parsed = urllib.parse.urlsplit(target)
        target_host = parsed.hostname or ""
        target_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"

        # Read headers
        headers_bytes = bytearray()
        while True:
            line = await client_reader.readline()
            if not line or line in (b"\r\n", b"\n"):
                break
            headers_bytes.extend(line)

        # Validate URL
        val_result = validate_url_and_resolve(target)
        self.intercepted_requests.append({
            "type": method,
            "target": target,
            "target_host": target_host,
            "target_port": target_port,
            "is_valid": val_result.is_valid,
            "pinned_ip": val_result.pinned_ip,
            "error_code": val_result.error_code,
        })

        if not val_result.is_valid:
            self.blocked_requests.append({
                "target": target,
                "reason": val_result.error_message,
                "code": val_result.error_code,
            })
            client_writer.write(
                b"HTTP/1.1 403 Forbidden\r\n"
                b"Content-Type: text/plain\r\n"
                b"Connection: close\r\n\r\n"
                b"Blocked by PhishVision Anti-SSRF Egress Policy\r\n"
            )
            await client_writer.drain()
            return

        pinned_ip = val_result.pinned_ip
        if not pinned_ip:
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await client_writer.drain()
            return

        try:
            remote_reader, remote_writer = await asyncio.wait_for(
                asyncio.open_connection(pinned_ip, target_port),
                timeout=12.0
            )
        except Exception:
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await client_writer.drain()
            return

        # Forward rewritten request line and headers to remote server
        forward_line = f"{method} {path} HTTP/1.1\r\n".encode("iso-8859-1")
        remote_writer.write(forward_line)
        remote_writer.write(bytes(headers_bytes))
        remote_writer.write(b"\r\n")
        await remote_writer.drain()

        # Pipe responses
        await self._pipe_bidirectional(client_reader, client_writer, remote_reader, remote_writer)

    async def _pipe_bidirectional(
        self,
        c_reader: asyncio.StreamReader,
        c_writer: asyncio.StreamWriter,
        r_reader: asyncio.StreamReader,
        r_writer: asyncio.StreamWriter
    ) -> None:
        """Stream bytes bidirectionally between client and remote sockets."""
        async def forward(src: asyncio.StreamReader, dst: asyncio.StreamWriter):
            try:
                while True:
                    chunk = await src.read(32768)
                    if not chunk:
                        break
                    dst.write(chunk)
                    await dst.drain()
            except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
                pass
            except Exception:
                pass
            finally:
                try:
                    dst.close()
                except Exception:
                    pass

        try:
            await asyncio.gather(
                forward(c_reader, r_writer),
                forward(r_reader, c_writer)
            )
        finally:
            for w in (c_writer, r_writer):
                try:
                    w.close()
                except Exception:
                    pass
