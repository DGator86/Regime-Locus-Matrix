"""Minimal IBKR connectivity check: TCP port, then API handshake (prints IB error lines)."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Farm status spam; still useful for debugging but optional to print
_IB_INFO = frozenset({2104, 2106, 2107, 2108, 2158, 2174})


def _stop_api_thread(app: object, run_thread: threading.Thread | None) -> None:
    """Disconnect and wait for ``app.run()`` so daemon threads do not die during interpreter shutdown."""
    try:
        app.disconnect()
    except Exception:
        pass
    if run_thread is not None and run_thread.is_alive():
        run_thread.join(timeout=15.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose IBKR TWS/Gateway socket + API handshake")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--client-id", type=int, default=None)
    args = parser.parse_args()

    from rlm.data.ibkr_stocks import load_ibkr_socket_config

    h, p, cid = load_ibkr_socket_config()
    h = args.host if args.host is not None else h
    p = args.port if args.port is not None else p
    cid = args.client_id if args.client_id is not None else cid

    print(f"Target: {h}:{p}  clientId={cid}")

    try:
        sock = socket.create_connection((h, p), timeout=5.0)
        sock.close()
        print("TCP: OK (port is accepting connections)")
    except OSError as e:
        print(f"TCP: FAILED — {e}")
        print("  → Start TWS or IB Gateway, log in, and match this port in Global Configuration → API → Settings.")
        sys.exit(1)

    try:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper
    except ImportError:
        print('ibapi: not installed. Run: python -m pip install -e ".[ibkr]"')
        sys.exit(2)

    errors: list[tuple[int, int, str]] = []
    ready = threading.Event()

    class App(EWrapper, EClient):
        def __init__(self) -> None:
            EClient.__init__(self, self)

        def nextValidId(self, orderId: int) -> None:  # noqa: N802
            ready.set()

        def error(  # noqa: N802
            self,
            reqId: int,
            errorCode: int,
            errorString: str,
            advancedOrderRejectJson: str = "",
        ) -> None:
            errors.append((reqId, errorCode, errorString))
            if errorCode not in _IB_INFO:
                print(f"  IB message: reqId={reqId} code={errorCode} {errorString}")

    app = App()
    print("Connecting API client...")
    app.connect(h, p, cid)
    # Non-daemon: we always join after disconnect so shutdown does not kill the thread mid-print
    t = threading.Thread(target=app.run, daemon=False, name="ibapi-run")
    t.start()

    if ready.wait(timeout=30.0):
        _stop_api_thread(app, t)
        print("API: OK (received nextValidId — TWS accepted this client)")
        sys.exit(0)

    print("API: no nextValidId within 30s")
    if errors:
        print("IB messages seen (use text above to fix TWS / client id / trusted IPs):")
    else:
        print("No IB error lines — often: TWS popup 'Accept incoming connection?' not clicked,")
        print("or another app is blocking the API thread. Try a new IBKR_CLIENT_ID (e.g. 101).")
    _stop_api_thread(app, t)
    sys.exit(3)


if __name__ == "__main__":
    main()
