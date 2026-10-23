#!/usr/bin/env python3
"""Sloth Study Digest

Drop lecture recordings, PDFs and notes into ./inbox, go to sleep, and wake up
to a summary page plus flashcards for each file. Everything runs locally:

  audio -> whisper.cpp -> transcript -> Ollama (open-weight model) -> HTML digest

Standard library only (pypdf is optional, for PDFs).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".webm", ".mkv", ".aac"}
TEXT_EXT = {".txt", ".md"}
PDF_EXT = {".pdf"}
SUPPORTED = AUDIO_EXT | TEXT_EXT | PDF_EXT

NOTES_PROMPT = """You are a careful study assistant. Turn this excerpt of a lecture into concise bullet-point notes.
Keep technical terms exactly as written. Use only what is in the excerpt; do not add facts.

EXCERPT:
{chunk}

NOTES (Markdown bullets only):"""

SUMMARY_PROMPT = """Below are bullet-point notes from one lecture. Write a study digest in Markdown using exactly these headings:

## Summary
(3 sentences)

## Key points
(at most 8 bullets)

## Key terms
(bullets in the form **term**: one-line definition)

Use only the notes. Do not add facts.

NOTES:
{notes}
"""

MERGE_PROMPT = """Merge these bullet-point notes into one shorter set of bullet-point notes.
Remove repeats, keep technical terms exact, and add nothing new.

NOTES:
{notes}

MERGED NOTES (Markdown bullets only):"""

CARDS_PROMPT = """Create {n} flashcards that test the most important ideas in these lecture notes.
Questions must be answerable from the notes alone. Keep answers under 30 words.
Return ONLY JSON in this shape: {{"cards": [{{"q": "...", "a": "..."}}]}}

NOTES:
{notes}
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Ollama client (plain HTTP, no SDK)
# --------------------------------------------------------------------------
class Ollama:
    def __init__(self, host: str, model: str, num_ctx: int = 4096):
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx

    def _request(self, path: str, payload: dict | None = None, timeout: int = 1800):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.host + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def generate(self, prompt: str, json_mode: bool = False, retries: int = 3) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": self.num_ctx},
        }
        if json_mode:
            payload["format"] = "json"
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return self._request("/api/generate", payload)["response"].strip()
            except (urllib.error.URLError, TimeoutError, KeyError) as err:
                last_err = err
                log(f"  model call failed ({err}); retry {attempt}/{retries}")
                time.sleep(5 * attempt)
        raise RuntimeError(f"Ollama is not answering at {self.host}: {last_err}")

    def loaded_model_memory_gb(self) -> float | None:
        """RAM/VRAM the loaded model uses, as reported by Ollama itself."""
        try:
            models = self._request("/api/ps", timeout=10).get("models", [])
            if models:
                return round(sum(m.get("size", 0) for m in models) / 1e9, 2)
        except Exception:
            pass
        return None


# --------------------------------------------------------------------------
# Getting text out of files
# --------------------------------------------------------------------------
def transcribe(path: Path, whisper_bin: str, whisper_model: str) -> str:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install it to transcribe audio or video.")
    if not shutil.which(whisper_bin):
        raise RuntimeError(f"'{whisper_bin}' not found. Build whisper.cpp and pass --whisper-bin.")
    if not Path(whisper_model).exists():
        raise RuntimeError(f"Whisper model not found: {whisper_model}")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(wav)],
            check=True, capture_output=True,
        )
        out_base = Path(tmp) / "transcript"
        subprocess.run(
            [whisper_bin, "-m", whisper_model, "-f", str(wav), "-nt", "-otxt", "-of", str(out_base)],
            check=True, capture_output=True,
        )
        return out_base.with_suffix(".txt").read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        if shutil.which("pdftotext"):
            out = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                                 check=True, capture_output=True, text=True)
            return out.stdout
        raise RuntimeError("PDF support needs `pip install pypdf` (or poppler's pdftotext).")
    return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


def extract_text(path: Path, args) -> str:
    ext = path.suffix.lower()
    if ext in AUDIO_EXT:
        return transcribe(path, args.whisper_bin, args.whisper_model)
    if ext in PDF_EXT:
        return read_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


# --------------------------------------------------------------------------
# Chunking and the map-reduce summary
# --------------------------------------------------------------------------
def chunk_text(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    pieces = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current)
            current = ""
        # A single huge sentence (e.g. a transcript with no punctuation): hard split.
        while len(piece) > max_chars:
            chunks.append(piece[:max_chars])
            piece = piece[max_chars:]
        current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def build_notes(llm: Ollama, text: str, chunk_chars: int) -> str:
    chunks = chunk_text(text, chunk_chars)
    log(f"  {len(chunks)} chunk(s) to read")
    notes = []
    for i, chunk in enumerate(chunks, 1):
        log(f"  notes {i}/{len(chunks)}")
        notes.append(llm.generate(NOTES_PROMPT.format(chunk=chunk)))
    merged = "\n".join(notes)
    # Very long lectures: shrink the notes until they fit comfortably in context.
    while len(merged) > chunk_chars * 2:
        log("  merging notes")
        groups = chunk_text(merged, chunk_chars)
        if len(groups) <= 1:
            break
        merged = "\n".join(llm.generate(MERGE_PROMPT.format(notes=g)) for g in groups)
    return merged


def parse_cards(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    cards = data.get("cards", []) if isinstance(data, dict) else data
    clean = []
    for c in cards if isinstance(cards, list) else []:
        if isinstance(c, dict) and str(c.get("q", "")).strip() and str(c.get("a", "")).strip():
            clean.append({"q": str(c["q"]).strip(), "a": str(c["a"]).strip()})
    return clean


# --------------------------------------------------------------------------
# HTML output (self-contained: no CDN, no web fonts, so it works offline)
# --------------------------------------------------------------------------
def md_to_html(md: str) -> str:
    out, in_list = [], False

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        return re.sub(r"(?<!\*)\*(?!\s)(.+?)\*(?!\*)", r"<em>\1</em>", s)

    for line in md.splitlines():
        stripped = line.strip()
        bullet = re.match(r"^[-*•]\s+(.*)", stripped)
        if bullet:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(bullet.group(1))}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        heading = re.match(r"^#{1,6}\s+(.*)", stripped)
        if heading:
            out.append(f"<h2>{inline(heading.group(1))}</h2>")
        elif stripped:
            out.append(f"<p>{inline(stripped)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


CSS = """
:root{--paper:#eef1ea;--ink:#1e2b24;--soft:#52615a;--moss:#4f7a4c;--bark:#6b503d;--leaf:#dfe8d8;--line:#c5d0bf;--card:#f8faf5}
@media(prefers-color-scheme:dark){:root{--paper:#16201b;--ink:#e6ece3;--soft:#9fb0a5;--moss:#8fbf86;--bark:#c9a98c;--leaf:#22322a;--line:#34473c;--card:#1c2a23}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:18px/1.65 Georgia,'Times New Roman',serif}
main{max-width:46rem;margin:0 auto;padding:2.5rem 1.25rem 5rem}
h1{font-size:2rem;line-height:1.2;margin:.2rem 0 .4rem}
h2{font-size:1.25rem;margin:2rem 0 .4rem;color:var(--bark)}
.meta,.back{font:14px/1.4 system-ui,sans-serif;color:var(--soft)}
a{color:var(--moss)}a:focus-visible,button:focus-visible{outline:3px solid var(--moss);outline-offset:3px}
ul{padding-left:1.2rem}li{margin:.3rem 0}
.cards{display:grid;gap:.9rem;margin-top:1rem}
.card{all:unset;display:block;cursor:pointer;background:var(--card);border:1px solid var(--line);border-left:5px solid var(--moss);border-radius:4px;padding:1rem 1.2rem}
.card .a{display:none;margin-top:.7rem;padding-top:.7rem;border-top:1px dashed var(--line);color:var(--soft)}
.card[aria-expanded=true] .a{display:block}
.card .hint{font:13px system-ui,sans-serif;color:var(--soft);margin-top:.5rem}
.card[aria-expanded=true] .hint{display:none}
.list{list-style:none;padding:0}.list li{border-bottom:1px solid var(--line);padding:.8rem 0}
.list a{font-size:1.15rem;text-decoration:none}
.empty{color:var(--soft)}
@media(prefers-reduced-motion:no-preference){.card{transition:background .15s}.card:hover{background:var(--leaf)}}
"""

CARD_JS = """
document.querySelectorAll('.card').forEach(c=>c.addEventListener('click',()=>{
  c.setAttribute('aria-expanded',c.getAttribute('aria-expanded')==='true'?'false':'true')}));
"""


def render_digest(record: dict) -> str:
    cards = "".join(
        f'<button class="card" aria-expanded="false"><div class="q">{html.escape(c["q"])}</div>'
        f'<div class="hint">Tap to reveal the answer</div><div class="a">{html.escape(c["a"])}</div></button>'
        for c in record["cards"]
    ) or '<p class="empty">No flashcards could be generated for this file.</p>'
    s = record["stats"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(record['title'])}</title><style>{CSS}</style></head><body><main>
<a class="back" href="index.html">All digests</a>
<h1>{html.escape(record['title'])}</h1>
<p class="meta">{html.escape(record['source'])}. Made in {s['seconds']} s with {html.escape(s['model'])}.</p>
{md_to_html(record['digest'])}
<h2>Flashcards</h2><div class="cards">{cards}</div>
</main><script>{CARD_JS}</script></body></html>"""


def render_index(records: list[dict]) -> str:
    items = "".join(
        f'<li><a href="{html.escape(r["page"])}">{html.escape(r["title"])}</a>'
        f'<div class="meta">{html.escape(r["when"])}. {len(r["cards"])} flashcards.</div></li>'
        for r in sorted(records, key=lambda r: r["when"], reverse=True)
    ) or '<li class="empty">Nothing yet. Put a lecture in the inbox folder.</li>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sloth Study Digest</title><style>{CSS}</style></head><body><main>
<h1>Good morning. Here is what got done overnight.</h1>
<ul class="list">{items}</ul></main></body></html>"""


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def file_key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "lecture"


def process(path: Path, args, llm: Ollama, out_dir: Path) -> dict:
    started = time.time()
    log(f"Working on {path.name}")
    text = extract_text(path, args)
    if len(text.strip()) < 200:
        raise RuntimeError("Very little text found (scanned PDF or silent audio?).")
    notes = build_notes(llm, text, args.chunk_chars)
    log("  writing digest")
    digest = llm.generate(SUMMARY_PROMPT.format(notes=notes))
    log("  making flashcards")
    cards = parse_cards(llm.generate(CARDS_PROMPT.format(n=args.cards, notes=notes), json_mode=True))
    if not cards:  # small models sometimes fumble JSON once; try again
        cards = parse_cards(llm.generate(CARDS_PROMPT.format(n=args.cards, notes=notes), json_mode=True))

    cards = cards[: args.cards]  # small models sometimes ignore the requested count

    slug = slugify(path.stem)
    record = {
        "title": re.sub(r"[_\s]\d{6,}.*$", "", path.stem).replace("_", " ").replace("-", " ").strip().title(),
        "source": path.name,
        "page": f"{slug}.html",
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "digest": digest,
        "cards": cards,
        "stats": {
            "seconds": round(time.time() - started),
            "model": llm.model,
            "words": len(text.split()),
            "model_memory_gb": llm.loaded_model_memory_gb(),
        },
    }
    (out_dir / f"{slug}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out_dir / record["page"]).write_text(render_digest(record), encoding="utf-8")
    log(f"  done in {record['stats']['seconds']} s, {record['stats']['words']} words, "
        f"{len(cards)} cards, model memory {record['stats']['model_memory_gb']} GB")
    return record


def rebuild_index(out_dir: Path) -> None:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in out_dir.glob("*.json")]
    (out_dir / "index.html").write_text(render_index(records), encoding="utf-8")


def run_once(args, llm: Ollama) -> int:
    inbox, out_dir = Path(args.inbox), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = Path(args.state)
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    done = 0
    for path in sorted(p for p in inbox.iterdir() if p.suffix.lower() in SUPPORTED):
        key = file_key(path)
        if state.get(path.name) == key:
            continue
        try:
            process(path, args, llm, out_dir)
            state[path.name] = key
            state_file.write_text(json.dumps(state, indent=2))
            done += 1
        except Exception as err:  # one bad file must not stop the night shift
            log(f"  skipped {path.name}: {err}")
    rebuild_index(out_dir)
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description="Turn lectures into a digest and flashcards, locally.")
    ap.add_argument("--inbox", default="inbox")
    ap.add_argument("--out", default="digests")
    ap.add_argument("--state", default=".state.json")
    ap.add_argument("--model", default="qwen2.5:3b", help="any model you have pulled in Ollama")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--whisper-bin", default="whisper-cli")
    ap.add_argument("--whisper-model", default="models/ggml-base.en.bin")
    ap.add_argument("--chunk-chars", type=int, default=6000)
    ap.add_argument("--cards", type=int, default=8)
    ap.add_argument("--watch", type=int, metavar="SECONDS", help="keep running, checking every N seconds")
    args = ap.parse_args()

    if not Path(args.inbox).is_dir():
        print(f"Inbox folder not found: {args.inbox}", file=sys.stderr)
        return 1
    llm = Ollama(args.host, args.model)
    if args.watch:
        log(f"Watching {args.inbox} every {args.watch} s. Ctrl+C to stop.")
        while True:
            run_once(args, llm)
            time.sleep(args.watch)
    done = run_once(args, llm)
    log(f"Finished. {done} new file(s). Open {Path(args.out) / 'index.html'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)