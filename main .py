"""Backend: wraps xhamster_api.Client and serves the frontend.
Run: uvicorn main:app --host 0.0.0.0 --port 8000
Put xhamster_api (your api.py package) on the Python path.
"""
import asyncio
import inspect
import traceback
import ipaddress
import socket
import time
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urljoin, quote, urlparse
import re

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from curl_cffi import AsyncSession

from xhamster_api import Client
from xhamster_api.modules.errors import NotFound, BotDetection, ProxyError, NetworkError

SEED_QUERY = "best"      # xhamster search needs a query; used for rows with no user query
CACHE_TTL = 300          # seconds
RATE_LIMIT = 120         # requests per minute per IP
BASE = Path(__file__).parent

app = FastAPI(title="Streaming API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Client()
_cache: dict = {}
_hits: dict = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip = request.client.host if request.client else "?"
    q = _hits[ip]
    now = time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return Response("Too many requests", status_code=429)
    q.append(now)
    return await call_next(request)


def cached(key):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]


def store(key, val):
    _cache[key] = (time.time(), val)
    return val


def unwrap(r):
    """ScrapeResult -> underlying Short/Video object (defensive about field names)."""
    for attr in ("video", "result", "data", "item"):
        v = getattr(r, attr, None)
        if v is not None:
            return v
    return r


def g(o, name, default=None):
    try:
        return getattr(o, name)
    except Exception:
        return default


def card(o) -> dict:
    return {
        "id": g(o, "video_id"),
        "title": g(o, "title", ""),
        "thumb": g(o, "poster_url") or g(o, "thumb_url") or g(o, "thumbnail") or "",
        "author": g(o, "author") or g(o, "uploader_name") or "",
        "views": g(o, "views", 0),
        "likes": g(o, "likes", 0),
        "dislikes": g(o, "dislikes", 0),
        "duration": g(o, "duration", 0),
        "tags": list(g(o, "tags", []) or []),
        "m3u8": g(o, "m3u8_base_url", ""),
    }


def map_error(e: Exception):
    if isinstance(e, NotFound):
        return HTTPException(404, "Not found")
    if isinstance(e, BotDetection):
        return HTTPException(503, "Upstream bot protection triggered")
    if isinstance(e, ProxyError):
        return HTTPException(502, "Proxy error")
    if isinstance(e, NetworkError):
        return HTTPException(502, "Network error")
    return HTTPException(500, f"{type(e).__name__}: {e}")


def supported(fn, kw: dict) -> dict:
    """Keep only keyword args the installed library version accepts."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kw
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


async def collect(limit: int = 40, **kw):
    items = []
    kw = supported(client.search_videos, kw)
    try:
        async for r in client.search_videos(**kw):
            obj = unwrap(r)
            if getattr(r, "error", None) and obj is r:
                continue
            c = card(obj)
            if c["m3u8"]:
                items.append(c)
            if len(items) >= limit:
                break
    except Exception as e:
        traceback.print_exc()
        raise map_error(e)
    return items


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/api/search")
async def search(q: str = SEED_QUERY, sort: str | None = None, quality: str = "720p",
                 category: str | None = None, vr: bool = False, date: str | None = None,
                 min_duration: str | None = None, page: int = Query(1, ge=1, le=20)):
    key = ("s", q, sort, quality, category, vr, date, min_duration, page)
    if (c := cached(key)) is not None:
        return c
    kw = dict(query=q, minimum_quality=quality, sort_by=sort or None,
              category=category or None, vr=vr, date=date or None,
              min_duration=min_duration or None, pages=1, videos_concurrency=4)
    if page > 1:
        # search_videos fetches pages 1..pages; fetch up to `page` and keep the last slice
        kw["pages"] = page
    items = await collect(**kw)
    if page > 1:
        items = items[(page - 1) * 20:]
    return store(key, {"page": page, "items": items})


@app.get("/api/trending")
async def trending():
    if (c := cached("trending")) is not None:
        return c
    items = await collect(query=SEED_QUERY, sort_by="views", date="weekly", pages=1, videos_concurrency=4)
    return store("trending", {"items": items})


@app.get("/api/video")
async def video(url: str):
    try:
        v = await client.get_video(url)
    except Exception as e:
        raise map_error(e)
    return {
        "id": v.video_id, "title": v.title, "likes": v.likes, "dislikes": v.dislikes,
        "rating": v.rating_percentage, "uploader": v.uploader_name,
        "tags": v.tags, "categories": v.categories, "pornstars": v.pornstars,
        "thumb": v.thumbnail, "m3u8": v.m3u8_base_url,
    }


@app.get("/api/shorts")
async def short(url: str):
    try:
        s = await client.get_short(url)
    except Exception as e:
        raise map_error(e)
    return card(s)


# ---------- HLS proxy (fixes CORS / referrer blocks) ----------
_URI_RE = re.compile(r'URI="([^"]+)"')


async def _is_public(host: str) -> bool:
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
        return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)
    except Exception:
        return False


@app.get("/api/stream")
async def stream(u: str):
    p = urlparse(u)
    if p.scheme not in ("http", "https") or not p.hostname or not await _is_public(p.hostname):
        raise HTTPException(400, "Bad URL")
    async with AsyncSession(impersonate="chrome") as s:
        r = await s.get(u, headers={"Referer": "https://xhamster.com/"}, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, "Upstream error")
    ctype = r.headers.get("content-type", "application/octet-stream")
    if ".m3u8" in p.path or "mpegurl" in ctype.lower():
        out = []
        for line in r.text.splitlines():
            if line.startswith("#"):
                line = _URI_RE.sub(lambda m: f'URI="/api/stream?u={quote(urljoin(u, m.group(1)), safe="")}"', line)
            elif line.strip():
                line = f"/api/stream?u={quote(urljoin(u, line.strip()), safe='')}"
            out.append(line)
        return Response("\n".join(out), media_type="application/vnd.apple.mpegurl")
    return Response(r.content, media_type=ctype)


@app.get("/")
async def index():
    return FileResponse(BASE / "index.html")
