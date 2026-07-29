import os
import re
import sys
import time
import datetime
import subprocess
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

REGISTRY_URL = "https://jsonblob.com/api/jsonBlob/019fad63-2894-7b2d-bc29-b1856d316967"
TARGET_LOCAL_PORT = "http://localhost:5000"

def update_registry(tunnel_url):
    payload = {
        "url": tunnel_url,
        "updated_at": datetime.datetime.now().isoformat()
    }
    try:
        res = requests.put(REGISTRY_URL, json=payload, timeout=10)
        if res.status_code in (200, 201):
            print("✅ [Auto-Discovery] Registry successfully updated on JSONBlob!")
            print(f"📡 Registered Backend URL: {tunnel_url}")
            return True
        else:
            print(f"⚠️ [Auto-Discovery] Failed to update registry. Status: {res.status_code}")
    except Exception as e:
        print(f"❌ [Auto-Discovery] Error connecting to registry: {e}")
    return False

def launch_tunnel():
    print("🚀 [Cloudflare] Starting Cloudflare Tunnel for http://localhost:5000...")
    
    cmd = ["cloudflared", "tunnel", "--url", TARGET_LOCAL_PORT]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )

    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        match = url_pattern.search(line)
        if match and not tunnel_url:
            tunnel_url = match.group(0)
            print("\n" + "=" * 65)
            print(f"🌐 LIVE CLOUDFLARE TUNNEL URL: {tunnel_url}")
            print("=" * 65)
            update_registry(tunnel_url)
            print("✨ Clients will automatically auto-discover this URL when opening the website!\n")

    process.wait()

if __name__ == "__main__":
    launch_tunnel()
