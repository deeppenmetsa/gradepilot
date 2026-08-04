#!/usr/bin/env python3
"""
GradePilot — grading server.

Serves the app and exposes POST /api/grade-one, which grades ONE
question with all four models whose API keys are configured, and
returns all four results in one response. Any model without a key
set gets score: null for that slot — the browser fills that slot
in with a mock score, so the app always works with zero, some, or
all four keys present.

Context sent to every grading call:
  - The whole question paper (uploaded once for the exam) — image/PDF
    goes to vision-capable models directly; plain text goes to all
    four models.
  - The whole rubric / answer key (uploaded once for the exam) — same
    image-vs-text handling.
  - The specific question's label + max points.
  - The student's answer (typed text or an uploaded photo/scan/PDF).

All four models here are vision-capable (ChatGPT, Gemini, Claude, and
Qwen via Groq) — they can all read the actual question paper, rubric,
and student answer sheet directly from an uploaded photo/PDF, no OCR
step and no retyping needed.

Setup — set whichever keys you have, mix and match:
    export OPENAI_API_KEY=sk-...          # https://platform.openai.com/api-keys
    export GOOGLE_API_KEY=...             # https://aistudio.google.com/apikey (free tier)
    export ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com/
    export GROQ_API_KEY=gsk_...           # https://console.groq.com/keys (free tier)

    pip install flask flask-cors requests
    python server.py
    Open http://127.0.0.1:5050/
"""

import os
import re

import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

APP_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = "gradepilot.html"

GEMINI_MODEL = "gemini-3.5-flash-lite"  # cheapest current Gemini tier per Google's own docs (July 2026):
                                          # "the fastest, lowest-cost model in the 3.5 family," multimodal.
OPENAI_MODEL = "gpt-5.6-luna"  # OpenAI's own docs recommend this as the current cheap/fast tier,
                                # replacing the now-legacy gpt-4o-mini.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GROQ_MODEL = "qwen/qwen3.6-27b"  # Groq deprecated the old Llama chat models (June 2026); this is their
                                  # current replacement, and unlike the old Llama models, it's vision-capable.

app = Flask(__name__, static_folder=APP_DIR, static_url_path="")
CORS(app)


@app.get("/")
def index():
    return send_from_directory(APP_DIR, INDEX_FILE)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(question_label, max_points, student_answer_text, paper_text, rubric_text, has_student_image, has_paper_image, has_rubric_image):
    lines = [
        "You are grading one question from a student's exam.",
        "",
        f"Question: {question_label}",
        f"Max points: {max_points}",
        "",
    ]
    if has_paper_image:
        lines.append("The full question paper is attached as an image/PDF — find this specific question in it.")
    elif paper_text:
        lines.append(f"Full question paper text:\n{paper_text}")

    if has_rubric_image:
        lines.append("The full rubric / answer key is attached as an image/PDF — find the scoring guidance for this specific question in it.")
    elif rubric_text:
        lines.append(f"Rubric / answer key text:\n{rubric_text}")
    if not (has_rubric_image or rubric_text):
        lines.append("(No rubric was provided — use your best judgment for what a correct answer looks like.)")

    lines.append("")
    if has_student_image:
        lines.append("The student's answer is attached as an image/PDF — find their answer to this specific question in it (it may contain answers to other questions too; only grade this one).")
    else:
        lines.append(f"Student's answer:\n{student_answer_text or '(no answer submitted)'}")

    lines.append("")
    lines.append(f"Respond with ONLY a number between 0 and {max_points} representing the score. No words, no explanation — just the number.")
    return "\n".join(lines)


def extract_score(text, max_points):
    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        raise ValueError(f"couldn't parse a score from the reply: {text!r}")
    return max(0.0, min(max_points, float(match.group())))


def ctx(data, prefix):
    """Pull a context blob (paper or rubric) out of the request: either
    plain text, or an image/PDF as base64+mime. Returns (text, b64, mime)."""
    return (
        data.get(f"{prefix}_text", ""),
        data.get(f"{prefix}_base64", ""),
        data.get(f"{prefix}_mime", ""),
    )


# ---------------------------------------------------------------------------
# Per-provider graders
# ---------------------------------------------------------------------------
def grade_gemini(q_label, max_pts, ans_text, ans_b64, ans_mime, paper_text, paper_b64, paper_mime, rubric_text, rubric_b64, rubric_mime):
    api_key = os.environ["GOOGLE_API_KEY"]
    prompt = build_prompt(q_label, max_pts, ans_text, paper_text, rubric_text, bool(ans_b64), bool(paper_b64), bool(rubric_b64))
    parts = []
    if paper_b64:
        parts.append({"inline_data": {"mime_type": paper_mime or "image/png", "data": paper_b64}})
    if rubric_b64:
        parts.append({"inline_data": {"mime_type": rubric_mime or "image/png", "data": rubric_b64}})
    if ans_b64:
        parts.append({"inline_data": {"mime_type": ans_mime or "image/png", "data": ans_b64}})
    parts.append({"text": prompt})
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}",
        json={"contents": [{"parts": parts}]}, timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return extract_score(text, max_pts)


def grade_openai(q_label, max_pts, ans_text, ans_b64, ans_mime, paper_text, paper_b64, paper_mime, rubric_text, rubric_b64, rubric_mime):
    api_key = os.environ["OPENAI_API_KEY"]
    prompt = build_prompt(q_label, max_pts, ans_text, paper_text, rubric_text, bool(ans_b64), bool(paper_b64), bool(rubric_b64))
    images = []
    if paper_b64:
        images.append({"type": "image_url", "image_url": {"url": f"data:{paper_mime or 'image/png'};base64,{paper_b64}"}})
    if rubric_b64:
        images.append({"type": "image_url", "image_url": {"url": f"data:{rubric_mime or 'image/png'};base64,{rubric_b64}"}})
    if ans_b64:
        images.append({"type": "image_url", "image_url": {"url": f"data:{ans_mime or 'image/png'};base64,{ans_b64}"}})
    content = ([{"type": "text", "text": prompt}] + images) if images else prompt
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": OPENAI_MODEL, "messages": [{"role": "user", "content": content}]}, timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return extract_score(text, max_pts)


def grade_claude(q_label, max_pts, ans_text, ans_b64, ans_mime, paper_text, paper_b64, paper_mime, rubric_text, rubric_b64, rubric_mime):
    api_key = os.environ["ANTHROPIC_API_KEY"]
    prompt = build_prompt(q_label, max_pts, ans_text, paper_text, rubric_text, bool(ans_b64), bool(paper_b64), bool(rubric_b64))
    images = []
    if paper_b64:
        images.append({"type": "image", "source": {"type": "base64", "media_type": paper_mime or "image/png", "data": paper_b64}})
    if rubric_b64:
        images.append({"type": "image", "source": {"type": "base64", "media_type": rubric_mime or "image/png", "data": rubric_b64}})
    if ans_b64:
        images.append({"type": "image", "source": {"type": "base64", "media_type": ans_mime or "image/png", "data": ans_b64}})
    content = (images + [{"type": "text", "text": prompt}]) if images else prompt
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": CLAUDE_MODEL, "max_tokens": 50, "messages": [{"role": "user", "content": content}]}, timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = next(b["text"] for b in data["content"] if b.get("type") == "text")
    return extract_score(text, max_pts)


def grade_qwen(q_label, max_pts, ans_text, ans_b64, ans_mime, paper_text, paper_b64, paper_mime, rubric_text, rubric_b64, rubric_mime):
    api_key = os.environ["GROQ_API_KEY"]
    prompt = build_prompt(q_label, max_pts, ans_text, paper_text, rubric_text, bool(ans_b64), bool(paper_b64), bool(rubric_b64))
    images = []
    if paper_b64:
        images.append({"type": "image_url", "image_url": {"url": f"data:{paper_mime or 'image/png'};base64,{paper_b64}"}})
    if rubric_b64:
        images.append({"type": "image_url", "image_url": {"url": f"data:{rubric_mime or 'image/png'};base64,{rubric_b64}"}})
    if ans_b64:
        images.append({"type": "image_url", "image_url": {"url": f"data:{ans_mime or 'image/png'};base64,{ans_b64}"}})
    content = ([{"type": "text", "text": prompt}] + images) if images else prompt
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": content}]}, timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return extract_score(text, max_pts)


GRADERS = [
    ("OPENAI_API_KEY", grade_openai),
    ("GOOGLE_API_KEY", grade_gemini),
    ("ANTHROPIC_API_KEY", grade_claude),
    ("GROQ_API_KEY", grade_qwen),
]


@app.post("/api/grade-one")
def grade_one():
    data = request.get_json(force=True) or {}
    q_label = data.get("question_label", "")
    max_pts = float(data.get("max_points", 10))
    ans_text = data.get("student_answer", "")
    ans_b64 = data.get("image_base64", "")
    ans_mime = data.get("image_mime", "")
    paper_text, paper_b64, paper_mime = ctx(data, "paper")
    rubric_text, rubric_b64, rubric_mime = ctx(data, "rubric")

    scores, errors = [], []
    for env_key, grader_fn in GRADERS:
        if not os.environ.get(env_key):
            scores.append(None)
            continue
        try:
            score = grader_fn(q_label, max_pts, ans_text, ans_b64, ans_mime, paper_text, paper_b64, paper_mime, rubric_text, rubric_b64, rubric_mime)
            scores.append(score)
        except Exception as e:
            scores.append(None)
            msg = f"{grader_fn.__name__}: {e}"
            errors.append(msg)
            print(f"[grade-one] {msg}", flush=True)

    return jsonify({"scores": scores, "errors": errors})


if __name__ == "__main__":
    configured = [env_key for env_key, _ in GRADERS if os.environ.get(env_key)]
    if not configured:
        print("WARNING: no API keys are set. Every model will fall back to mock scores.")
        print("Set any of: OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY\n")
    else:
        print(f"Real grading enabled for: {', '.join(k.replace('_API_KEY','') for k in configured)}")
        missing = [k for k, _ in GRADERS if k not in configured]
        if missing:
            print(f"Still mocked (no key set): {', '.join(k.replace('_API_KEY','') for k in missing)}\n")
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
