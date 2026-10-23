# Sloth Study Digest

Drop lecture recordings, PDFs and notes into `inbox/`, go to sleep, wake up to a
summary page and flashcards for each one. All local, all open-source:

    audio -> whisper.cpp -> transcript -> Ollama (open-weight model) -> HTML digest

No API keys, no uploads, no internet needed after setup. Python standard library only
(`pypdf` is optional, for PDFs).

## Setup (once)

1. Install [Ollama](https://ollama.com) and pull a small model:
   `ollama pull qwen2.5:3b` (or `llama3.2:3b`)
2. For audio/video: install `ffmpeg`, build [whisper.cpp](https://github.com/ggml-org/whisper.cpp),
   and download a model, e.g. `ggml-base.en.bin`.
3. For PDFs: `pip install -r requirements.txt`

## Use

    python sloth_digest.py                       # process everything new in inbox/, then exit
    python sloth_digest.py --watch 300           # keep running, check every 5 minutes
    python sloth_digest.py --model llama3.2:3b   # swap the model, nothing else changes
    python sloth_digest.py --whisper-bin C:\whisper\whisper-cli.exe --whisper-model C:\whisper\ggml-base.en.bin

Open `digests/index.html` in the morning. Click a flashcard to reveal its answer.
Files already processed are skipped, and a file that fails is logged and skipped
without stopping the rest.

For a true overnight run, schedule it: Windows Task Scheduler, or cron
(`0 23 * * * python sloth_digest.py`).

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `qwen2.5:3b` | Any model you have pulled in Ollama |
| `--cards` | `8` | Flashcards per lecture |
| `--chunk-chars` | `6000` | Transcript slice per model call; lower it for small context windows |
| `--whisper-bin` / `--whisper-model` | `whisper-cli` / `models/ggml-base.en.bin` | whisper.cpp binary and model |

## Numbers for your write-up

Each `digests/<name>.json` records seconds taken, word count, and the memory the model
used (read from Ollama's `/api/ps`). Compare against the length of the recording.

## How it works

1. Audio/video is converted to 16 kHz mono and transcribed by whisper.cpp. PDFs and text are read directly.
2. The text is cut into chunks; the model writes bullet notes per chunk, merged if the lecture is very long.
3. The notes become a short summary, key points and key terms, plus flashcards as JSON.
4. The output pages are plain HTML with no CDN or web fonts, so they open offline.

The prompts tell the model to use only the source text. Small models can still get terms
wrong, so skim the digest before relying on it.
