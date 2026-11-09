"""Backend: wraps xhamster_api.Client and serves the frontend.
Run: uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import asyncio
import inspect
import ipaddress
import re
import socket
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urljoin, quote, urlparse

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from curl_cffi import AsyncSession

from xhamster_api import Client
from xhamster_api.modules.errors import NotFound, BotDetection, ProxyError, NetworkError

SEED_QUERY = "best"        # the site's search needs a query; used for rows without one
CACHE_TTL = 600            # fresh for 10 min
STALE_TTL = 6 * 3600       # still served if upstream fails
API_LIMIT = 150            # JSON API requests per minute per IP
STREAM_LIMIT = 4000        # video segment requests per minute per IP
UPSTREAM_PARALLEL = 3      # simultaneous scraping jobs
MAX_PAGE = 5
BASE = Path(__file__).parent

app = FastAPI(title="Streaming API")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Client()
_cache: dict = {}
_inflight: dict = {}
_upstream = asyncio.Semaphore(UPSTREAM_PARALLEL)
_hits: dict = defaultdict(deque)
_sess: AsyncSession | None = None


# ---------- rate limit (real client IP behind Render's proxy) ----------
def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


@app.middleware("http")
async def middleware(request: Request, call_next):
    path = request.url.path
    stream = path.startswith("/api/stream")
    bucket = _hits[(client_ip(request), stream)]
    now = time.time()
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= (STREAM_LIMIT if stream else API_LIMIT):
        return Response("Too many requests", status_code=429, headers={"Retry-After": "20"})
    bucket.append(now)
    if len(_hits) > 5000:
        for k in [k for k, v in _hits.items() if not v][:2500]:
            _hits.pop(k, None)
    resp = await call_next(request)
    if path in ("/api/search", "/api/trending") and resp.status_code == 200:
        resp.headers["Cache-Control"] = "public, max-age=120, stale-while-revalidate=600"
    return resp


# ---------- cache helpers ----------
def cache_get(key, max_age=CACHE_TTL):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < max_age:
        return hit[1]


def cache_put(key, val):
    if len(_cache) > 400:  # drop the oldest entries
        for k in sorted(_cache, key=lambda k: _cache[k][0])[:100]:
            _cache.pop(k, None)
    _cache[key] = (time.time(), val)
    return val


async def once(key, factory):
    """Collapse identical concurrent requests into one upstream job."""
    if key in _inflight:
        return await _inflight[key]
    task = asyncio.ensure_future(factory())
    _inflight[key] = task
    try:
        return await task
    finally:
        _inflight.pop(key, None)


# ---------- scraping ----------
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
        "tags": list(g(o, "tags", []) or [])[:20],
        "m3u8": g(o, "m3u8_base_url", ""),
    }


def map_error(e: Exception):
    if isinstance(e, HTTPException):
        return e
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
    async with _upstream:
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


async def cached_job(key, **kw):
    """Fresh cache -> else one shared upstream job -> else stale cache on failure."""
    if (c := cache_get(key)) is not None:
        return c
    try:
        items = await once(key, lambda: collect(**kw))
    except HTTPException:
        stale = cache_get(key, STALE_TTL)
        if stale is not None:
            return stale
        raise
    return cache_put(key, items)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/api/search")
async def search(q: str = SEED_QUERY, sort: str | None = None, quality: str = "720p",
                 category: str | None = None, vr: bool = False, date: str | None = None,
                 min_duration: str | None = None, page: int = Query(1, ge=1, le=MAX_PAGE)):
    q = (q or SEED_QUERY).strip()[:100] or SEED_QUERY
    key = ("s", q, sort, quality, category, vr, date, min_duration, page)
    items = await cached_job(
        key, limit=40 * page, query=q, minimum_quality=quality, sort_by=sort or None,
        category=category or None, vr=vr, date=date or None,
        min_duration=min_duration or None, pages=page, videos_concurrency=4)
    return {"page": page, "items": items}


@app.get("/api/trending")
async def trending():
    items = await cached_job("trending", query=SEED_QUERY, sort_by="views", date="weekly", pages=1,
                             videos_concurrency=4)
    return {"items": items}


@app.get("/api/video")
async def video(url: str):
    if urlparse(url).hostname not in ("xhamster.com", "www.xhamster.com"):
        raise HTTPException(400, "Bad URL")
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
    if urlparse(url).hostname not in ("xhamster.com", "www.xhamster.com"):
        raise HTTPException(400, "Bad URL")
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


def session() -> AsyncSession:
    """One shared connection pool: avoids a new TLS handshake for every segment."""
    global _sess
    if _sess is None:
        _sess = AsyncSession(impersonate="chrome")
    return _sess


@app.on_event("shutdown")
async def _close():
    if _sess is not None:
        await _sess.close()


@app.get("/api/stream")
async def stream(u: str, request: Request):
    p = urlparse(u)
    if p.scheme not in ("http", "https") or not p.hostname or not await _is_public(p.hostname):
        raise HTTPException(400, "Bad URL")
    headers = {"Referer": "https://xhamster.com/"}
    if rng := request.headers.get("range"):
        headers["Range"] = rng
    try:
        r = await session().get(u, headers=headers, timeout=30)
    except Exception:
        raise HTTPException(502, "Upstream unreachable")
    if r.status_code >= 400:
        raise HTTPException(r.status_code, "Upstream error")
    ctype = r.headers.get("content-type", "application/octet-stream")
    if ".m3u8" in p.path or "mpegurl" in ctype.lower():
        out = []
        for line in r.text.splitlines():
            if line.startswith("#"):
                line = _URI_RE.sub(
                    lambda m: f'URI="/api/stream?u={quote(urljoin(u, m.group(1)), safe="")}"', line)
            elif line.strip():
                line = f"/api/stream?u={quote(urljoin(u, line.strip()), safe='')}"
            out.append(line)
        return Response("\n".join(out), media_type="application/vnd.apple.mpegurl",
                        headers={"Cache-Control": "public, max-age=10", "Content-Encoding": "identity"})
    # media segments: already compressed, immutable -> let browsers/CDN cache them, skip gzip
    resp_headers = {"Cache-Control": "public, max-age=3600", "Content-Encoding": "identity",
                    "Accept-Ranges": "bytes"}
    status = r.status_code if r.status_code in (200, 206) else 200
    if cr := r.headers.get("content-range"):
        resp_headers["Content-Range"] = cr
    return Response(r.content, status_code=status, media_type=ctype, headers=resp_headers)


@app.get("/")
async def index():
    return FileResponse(BASE / "index.html", headers={"Cache-Control": "no-cache"})
