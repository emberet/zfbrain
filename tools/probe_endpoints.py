#!/usr/bin/env python
"""The table's HTTP surface, with no browser and no brain.

`/pong/join` and `/pong/move` are the first inbound endpoints this project has
had since `/say` was pulled, so they get their own test rather than being
exercised by hand once and trusted. The Roamer is stubbed down to the four
attributes `make_handler` actually touches, which means this runs in under a
second and cannot be broken by anything in the sim.

What it pins, in order of how much it would hurt to get wrong:

  * with `ZF_PONG` off, none of it exists - not the paths, not the `pong` key
    on `/state`, not `POST` in the preflight.
  * one seat. A second joiner is refused, a wrong token cannot move the paddle,
    and - the one that was actually broken - an empty body cannot free somebody
    else's seat.
  * an oversized body is refused on the `Content-Length` header, before it is
    read, so we never allocate what we are about to refuse.
  * the preflight answers zfbrain.online and does not answer anyone else.

    .venv/bin/python tools/probe_endpoints.py
"""
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ZF_LIVE_ORIGINS", "https://zfbrain.online")

import pong  # noqa: E402
import roam  # noqa: E402


class Stub:
    """Everything make_handler reads, and nothing else."""

    def __init__(self, table):
        self.table = table
        self.feed = roam.Feed()
        self.graph_doc = b"{}"
        self.graph_bin = b""
        self.graph_ver = "1"
        self.meta = {"built_at": "0"}


def serve(table):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), roam.make_handler(Stub(table)))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def call(url, data=None, method=None, headers=None, raw=None):
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url, data=body, method=method or ("POST" if body else "GET"))
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def main():
    # ---- with the flag off: none of it exists ---------------------------
    _, base = serve(None)
    assert call(base + "/pong")[0] == 404, "/pong answered with ZF_PONG off"
    assert call(base + "/pong/join", {})[0] == 404, "/pong/join answered with ZF_PONG off"
    code, body, _ = call(base + "/state")
    assert code == 200 and "pong" not in json.loads(body), "/state carries pong with the flag off"
    _, _, hdrs = call(base, method="OPTIONS", headers={"Origin": "https://zfbrain.online"})
    assert "POST" not in hdrs["Access-Control-Allow-Methods"], hdrs
    print("  [PASS] flag off  /pong 404 · /pong/join 404 · no pong key · no POST in OPTIONS")

    # ---- with the flag on ------------------------------------------------
    table = pong.Table(seat_idle_s=30.0, per_ip_s=0.0)
    _, base = serve(table)
    code, body, _ = call(base + "/pong")
    assert code == 200 and "points" in json.loads(body), body

    code, body, _ = call(base + "/pong/join", {})
    tok = json.loads(body)["token"]
    assert code == 200 and tok, body
    # one seat, whatever the IP - every request here comes from 127.0.0.1
    code2, body2, _ = call(base + "/pong/join", {})
    assert code2 == 409, (code2, body2)

    code, body, _ = call(base + "/pong/move", {"token": tok, "y": 250})
    assert code == 200 and json.loads(body)["ok"], body
    assert call(base + "/pong/move", {"token": "nope", "y": 250})[0] == 403
    assert call(base + "/pong/move", {"token": tok, "y": "700; DROP"})[0] == 400
    print("  [PASS] one seat   join 200 · second join 409 · bad token 403 · non-number 400")

    # the seat cannot be taken from the person holding it. Table.release(None)
    # frees it unconditionally for the rally path, so an empty body reaching
    # that argument would let any watcher boot the visitor.
    code, body, _ = call(base + "/pong/leave", {})
    assert code == 403, ("an empty /pong/leave freed somebody else's seat", code, body)
    assert call(base + "/pong/leave", {"token": "nope"})[0] == 403
    assert table.occupied(), "the seat was released by a request that had no token"
    code, body, _ = call(base + "/pong/leave", {"token": tok})
    assert code == 200 and json.loads(body)["ok"], body
    assert not table.occupied(), "the right token did not free the seat"
    print("  [PASS] leave      empty body 403 · wrong token 403 · right token 200")

    # ---- an oversized body is refused on the header ----------------------
    tok = json.loads(call(base + "/pong/join", {})[1])["token"]
    big = json.dumps({"token": tok, "y": 1, "pad": "x" * 3000}).encode()
    code, body, _ = call(base + "/pong/move", raw=big)
    assert code == 413, (code, body)
    print(f"  [PASS] body cap   {len(big)} B refused 413, on the header, unread")

    # ---- CORS preflight --------------------------------------------------
    st, _, hdrs = call(base, method="OPTIONS", headers={"Origin": "https://zfbrain.online"})
    assert st == 204 and "POST" in hdrs["Access-Control-Allow-Methods"], hdrs
    assert hdrs["Access-Control-Allow-Origin"] == "https://zfbrain.online", hdrs
    _, _, h2 = call(base, method="OPTIONS", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in h2, h2
    print(f"  [PASS] preflight  {hdrs['Access-Control-Allow-Methods']} for zfbrain.online · "
          f"nothing for evil.example")

    print("\nendpoints OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
