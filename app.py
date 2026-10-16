"""
FastAPI backend
run -> uvicorn app:app --reload --port 8000
Open: http://localhost:8000
"""
import base64
import io
import json
import logging
import mimetypes
import os
import re
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline import pipeline
from src.helper import load_whisper_model
from src.chat_model import get_openai_client
from src.retrive import index_files
from src.logger.production_logger import ProductionLogger
from langchain_community.callbacks import get_openai_callback


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"; UPLOAD_DIR.mkdir(exist_ok=True)
FRONTEND_DIR = BASE_DIR / "frontend"
IMAGES_DIR = BASE_DIR / "vision_img" / "images" 
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_BACKEND = "local"   # local | openai

MEDIA_ROOTS = [Path(p).resolve() for p in os.getenv("MEDIA_ROOTS", str(IMAGES_DIR)).split(os.pathsep) if p]

STREAM_NODES: set = set()  

prod_logger = ProductionLogger(log_file=str(BASE_DIR / "logs" / "rag_production.log"))


# ---- metrics (token/cost) ----------------------------------------------------
MODEL_PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}


def price_for(model: str):
    for name in sorted(MODEL_PRICES, key=len, reverse=True): 
        if str(model).startswith(name):
            return MODEL_PRICES[name]
    return None


def add_usage(usage: dict, ev: dict, meta: dict) -> None:
    out = ev["data"].get("output")
    um = getattr(out, "usage_metadata", None) or {}
    rm = getattr(out, "response_metadata", None) or {}
    tu = rm.get("token_usage") or {}
    pt = um.get("input_tokens") or tu.get("prompt_tokens") or 0
    ct = um.get("output_tokens") or tu.get("completion_tokens") or 0
    model = meta.get("ls_model_name") or rm.get("model_name") or rm.get("model") or "unknown"
    usage[model][0] += pt
    usage[model][1] += ct



def build_metrics(usage: dict, started: float, project_id: str, thread_id: str, callback=None, status="success", error=None) -> dict:
    # Fallback to callback if usage dict is empty but callback has data
    if not usage and callback and callback.total_tokens > 0:
        p = callback.prompt_tokens
        c = callback.completion_tokens
        cost = callback.total_cost
    else:
        p = sum(v[0] for v in usage.values())
        c = sum(v[1] for v in usage.values())
        cost, unpriced = 0.0, []
        for model, (pt, ct) in usage.items():
            pr = price_for(model)
            if pr:
                cost += pt / 1e6 * pr[0] + ct / 1e6 * pr[1]
            elif pt or ct:
                unpriced.append(model)
        if unpriced:
            logging.warning(f"query_cost- 0.0000000: {unpriced}")

    return {
        "status": status, 
        "error": error,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "prompt_tokens": p, 
        "completion_tokens": c, 
        "tokens_used": p + c,
        "query_cost": round(cost, 6),
        "project_id": project_id, 
        "thread_id": thread_id,
    }


def safe_log(state: dict, metrics: dict) -> None:
    try:
        prod_logger.log_query(state, metrics)
    except Exception as e:           
        logging.error(f"production log failed: {e}")
# ---- end metrics -------------------------------------------------------------


# ------------------------------------------------------------------
# Graph adapter 
# ------------------------------------------------------------------
def build_input_state(question, image_data_url, skip_cache):
    return {"question": question, "query_image": image_data_url, "skip_cache": skip_cache}


def safe_media_path(raw: str) -> Optional[Path]:
    p = Path(raw)
    if not p.is_absolute():
        p = BASE_DIR / p
    try:
        p = p.resolve()
    except OSError:
        return None
    for root in MEDIA_ROOTS:
        try:
            p.relative_to(root)
            return p if p.is_file() else None
        except ValueError:
            continue
    return None


def parse_table(content) -> Optional[dict]:
    """markdown / dict / list-of-dict → {"columns": [...], "rows": [[...]]}"""
    if isinstance(content, dict) and "columns" in content:
        return {"columns": content["columns"], "rows": content.get("rows", [])}
    if isinstance(content, list) and content and isinstance(content[0], dict):
        cols = list(content[0].keys())
        return {"columns": cols, "rows": [[r.get(c, "") for c in cols] for r in content]}
    if isinstance(content, str):
        lines = [l.strip() for l in content.splitlines() if l.strip().startswith("|")]
        rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
        rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
        if len(rows) >= 2:
            return {"columns": rows[0], "rows": rows[1:]}
    return None


def extract_media(state: dict) -> list:
    out, seen = [], set()
    for m in state.get("media") or []:
        if not isinstance(m, dict):
            continue
        t = m.get("type", "image")
        cap = (m.get("caption") or "").strip()
        if m.get("page") is not None:
            cap = f"{cap} (p.{m['page']})".strip()
        if t == "table":
            tbl = parse_table(m.get("content"))
            if tbl:
                out.append({"type": "table", **tbl})
            continue
        url = m.get("url") or ""
        if not url and m.get("path"):
            p = safe_media_path(str(m["path"]))
            url = f"/api/media?path={quote(str(p))}" if p else ""
        if not url or url in seen:
            continue
        seen.add(url)
        src = m.get("source") or ""
        out.append({"type": "chart" if t == "chart" else "image", "url": url,
                    "caption": cap or t, "source": src if str(src).startswith("http") else None})
    return out


def extract_sources(state: dict) -> list:
    out, seen = [], set()

    def add(title, url=""):
        title = str(title).strip()
        if title and title not in seen:
            seen.add(title)
            out.append({"title": title, "url": url if str(url).startswith("http") else ""})

    for s in state.get("sources") or []:
        if isinstance(s, str):
            add(s, s) if s.startswith("http") else add(Path(s).name)
    for c in state.get("citations") or []:
        if isinstance(c, dict):
            name = str(c.get("title") or c.get("source") or c.get("file") or "")
            page = c.get("page")
            title = f"{Path(name).name} p.{page}" if name and page is not None else (Path(name).name if name else "")
            add(title, c.get("url") or (name if name.startswith("http") else ""))
    return out[:8]


def resolve_context_image(u: str) -> str:
    if u.startswith(("data:", "http://", "https://")):
        return u
    raw = None
    if u.startswith("/api/media"):
        raw = (parse_qs(urlparse(u).query).get("path") or [""])[0]
    elif u.startswith("/images/"):
        raw = str(IMAGES_DIR / unquote(u[len("/images/"):]))
    p = safe_media_path(raw) if raw else None
    if not p:
        raise HTTPException(400, "context_image পাওয়া যায়নি")
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


# ------------------------------------------------------------------
# App + helpers
# ------------------------------------------------------------------
app = FastAPI(title="Langgraph-Docmind")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def ndjson(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode()


async def save_upload(f: UploadFile, subdir: str = "") -> Path:
    d = UPLOAD_DIR / re.sub(r"[^\w.-]", "_", subdir) 
    d.mkdir(parents=True, exist_ok=True)
    name = os.path.basename((f.filename or "file").replace("\\", "/")) or uuid.uuid4().hex
    path = d / name
    path.write_bytes(await f.read())
    return path


def openai_or_400():
    try:
        return get_openai_client()
    except ValueError:
        raise HTTPException(400, "OpenAI is not configured. Please set your OpenAI API Key first.")


@app.get("/api/media")
async def media_file(path: str = Query(...)):
    p = safe_media_path(path)
    if not p:
        raise HTTPException(404, "not found")
    return FileResponse(p)


# ------------------------------------------------------------------
# /api/chat — streaming NDJSON
# ------------------------------------------------------------------
@app.post("/api/chat")
async def chat(
    query: str = Form(""),
    chat_id: str = Form("default"),
    project_id: str = Form(""),
    context_image: Optional[str] = Form(None),
    skip_cache: bool = Form(False),
    files: List[UploadFile] = File(default=[]),
    images: List[UploadFile] = File(default=[]),
):
    if not query.strip() and not files and not images and not context_image:
        raise HTTPException(400, "empty request")

    project_id = project_id.strip() or chat_id
    file_paths = [str(await save_upload(f, project_id)) for f in files]
    image_data_url = None
    if images: 
        raw = await images[0].read()
        mime = images[0].content_type or mimetypes.guess_type(images[0].filename or "")[0] or "image/png"
        image_data_url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    elif context_image:
        image_data_url = resolve_context_image(context_image)

    question = query.strip() or ("Describe this image." if image_data_url else "Summarize the uploaded file(s).")
    state_in = build_input_state(question, image_data_url, skip_cache)
    thread_id = f"{project_id}:{chat_id}"
    config = {"configurable": {"thread_id": thread_id}}

    async def stream():
        started = time.perf_counter()
        usage = defaultdict(lambda: [0, 0])      # model -> [prompt_tokens, completion_tokens]
        final_state: dict = {}
        notice = ""
        try:
            if file_paths:
                yield ndjson({"type": "node", "name": "index_files", "status": "start"})
                n = await run_in_threadpool(index_files, project_id, file_paths)
                yield ndjson({"type": "node", "name": "index_files", "status": "end"})
                if n == 0:
                    notice = "⚠️ File text empty।\n\n"
            # Execute pipeline and track token callbacks properly
            async with pipeline(project_id) as graph_app:
                with get_openai_callback() as callback:
                    async for ev in graph_app.astream_events(state_in, config=config, version="v2"):
                        kind, name = ev["event"], ev.get("name") or ""
                        meta = ev.get("metadata") or {}
                        is_node = bool(name) and not name.startswith("__") and meta.get("langgraph_node") == name
                        
                        if kind == "on_chat_model_end":
                            add_usage(usage, ev, meta)
                        elif is_node and kind == "on_chain_start":
                            yield ndjson({"type": "node", "name": name, "status": "start"})
                        elif is_node and kind == "on_chain_end":
                            out = ev["data"].get("output")
                            if isinstance(out, dict):
                                final_state.update(out)
                            yield ndjson({"type": "node", "name": name, "status": "end"})
                        elif kind == "on_chat_model_stream" and meta.get("langgraph_node") in STREAM_NODES:
                            chunk = ev["data"]["chunk"].content
                            if isinstance(chunk, str) and chunk:
                                yield ndjson({"type": "token", "text": chunk})
                                
                    try:   
                        snap = await graph_app.aget_state(config)
                        final_state.update(snap.values or {})
                    except Exception:
                        pass

            # Calculate actual total run latency
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            
            metrics = {
                "Prompt tokens": callback.prompt_tokens,
                "Completion tokens": callback.completion_tokens,
                "latency_ms": latency_ms,
                "tokens_used": callback.total_tokens,
                "query_cost": callback.total_cost,
            }

            answer = final_state.get("answer")
            answer = answer if isinstance(answer, str) and answer.strip() else "Sorry, No answer found"
            payload = {
                "type": "final",
                "answer": notice + answer,
                "media": extract_media(final_state),
                "sources": extract_sources(final_state),
            }
            safe_log(final_state, metrics)
            
            yield ndjson(payload)
            
        except Exception as e:
            error_metrics = build_metrics(usage=usage, started=started, project_id=project_id, thread_id=thread_id, 
                callback=callback, 
                status="error", 
                error=str(e)
            )
            safe_log(final_state, error_metrics)
            yield ndjson({"type": "final", "answer": f"⚠️ Server error: {e}", "media": [], "sources": []})
    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------
# Voice → text
# ------------------------------------------------------------------
def _transcribe_sync(path: str) -> str:
    if WHISPER_BACKEND == "local":
        return load_whisper_model().transcribe(path, fp16=False)["text"].strip()
    client = openai_or_400()
    with open(path, "rb") as f:
        return client.audio.transcriptions.create(model="whisper-1", file=f).text.strip()


async def transcribe_upload(f: UploadFile) -> str:
    path = await save_upload(f, uuid.uuid4().hex)     
    try:
        if path.stat().st_size == 0:
            raise HTTPException(400, "empty audio")
        return await run_in_threadpool(_transcribe_sync, str(path))
    finally:
        shutil.rmtree(path.parent, ignore_errors=True)


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    return {"text": await transcribe_upload(audio)}


# ------------------------------------------------------------------
# Text → speech
# ------------------------------------------------------------------
class TTSBody(BaseModel):
    text: str


def _tts_sync(text: str) -> bytes:
    r = openai_or_400().audio.speech.create(model="tts-1", voice="alloy", input=text[:4000], response_format="mp3")
    return r.content


async def _tts_response(text: str):
    if not text.strip():
        raise HTTPException(400, "Text parameter cannot be empty.")
    try:
        audio = await run_in_threadpool(_tts_sync, text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"TTS synthesis error: {e}")
    return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")


@app.post("/api/tts")
async def tts_post(body: TTSBody):
    return await _tts_response(body.text)


@app.get("/api/tts")
async def tts_get(text: str = Query(...)):
    return await _tts_response(text)


@app.get("/api/health")
async def health():
    return {"ok": True}


app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
