"""Exercise TLS configuration reuse against a real local WSS server."""

import asyncio
import os
import shutil
import ssl
import subprocess
import sys

import pytest
import websockets


@pytest.mark.asyncio
@pytest.mark.parametrize("trusted", [True, False], ids=["trusted-ca", "untrusted-ca"])
async def test_default_tls_context_reuse_preserves_verification(tmp_path, trusted):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is required to generate the local TLS certificate")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
         "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost",
         "-keyout", str(key), "-out", str(cert)],
        check=True, capture_output=True, timeout=15,
    )
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert, key)
    accepted = []

    async def server(ws):
        accepted.append(ws.remote_address)
        async for _message in ws:
            pass

    # A separate interpreter gives each case a fresh default trust configuration.
    script = r'''
import asyncio, ssl, sys
from concurrent.futures import ThreadPoolExecutor
from kotoba._ws_base import AsyncSession, _default_ssl_context
from kotoba.errors import APIError

async def main():
    url, trusted = sys.argv[1], sys.argv[2] == 'True'
    with ThreadPoolExecutor(max_workers=16) as pool:
        contexts = list(pool.map(lambda _: _default_ssl_context(), range(32)))
    context = contexts[0]
    assert all(c is context for c in contexts)
    assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
    if not trusted:
        try:
            async with AsyncSession(url):
                raise AssertionError('An untrusted certificate was accepted')
        except APIError as exc:
            assert isinstance(exc.__cause__, ssl.SSLCertVerificationError)
        return
    async with AsyncSession(url) as first, AsyncSession(url) as second:
        one = first._ws.transport
        two = second._ws.transport
        assert one.get_extra_info('sockname') != two.get_extra_info('sockname')
        assert one.get_extra_info('ssl_object') is not two.get_extra_info('ssl_object')
        assert one.get_extra_info('ssl_object').context is context
        assert two.get_extra_info('ssl_object').context is context
    try:
        async with AsyncSession(url.replace('localhost', '127.0.0.1')):
            raise AssertionError('A certificate hostname mismatch was accepted')
    except APIError as exc:
        assert isinstance(exc.__cause__, ssl.SSLCertVerificationError)
    explicit = ssl.create_default_context()
    async with AsyncSession(url, connect_kwargs={'ssl': explicit}) as session:
        assert session._ws.transport.get_extra_info('ssl_object').context is explicit

asyncio.run(main())
'''
    environment = os.environ.copy()
    environment.pop("SSL_CERT_FILE", None)
    environment.pop("SSL_CERT_DIR", None)
    if trusted:
        environment["SSL_CERT_FILE"] = str(cert)
    async with websockets.serve(server, "127.0.0.1", 0, ssl=server_context) as listener:
        url = f"wss://localhost:{listener.sockets[0].getsockname()[1]}"
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script, url, str(trusted), env=environment,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
            assert process.returncode == 0, (stdout + stderr).decode()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
    assert len(accepted) == (3 if trusted else 0)
