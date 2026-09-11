#!/bin/sh
# live.zfbrain.online -> 127.0.0.1:4660 over the zfbrain-live named tunnel.
# ZF_TUNNEL_TOKEN comes from .env and is never echoed.
cd "$(dirname "$0")/.." || exit 1
set -a; [ -f .env ] && . ./.env; set +a
[ -n "$ZF_TUNNEL_TOKEN" ] || { echo "bin/tunnel.sh: ZF_TUNNEL_TOKEN missing in .env"; exit 1; }
exec /opt/homebrew/bin/cloudflared tunnel --config tunnel.yml run --token "$ZF_TUNNEL_TOKEN"
