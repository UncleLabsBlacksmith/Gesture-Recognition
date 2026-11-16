#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS Voice Assistant (Powered by Hermes Agent)
================================================
Pipeline:  ไมค์ -> VAD -> STT (Whisper) -> Hermes Agent -> TTS -> ลำโพง

ติดตั้ง:
    pip install sounddevice numpy requests edge-tts playsound==1.2.2

ตั้งค่า API key (แนะนำให้ใช้ environment variable แทนการฝังไว้ในไฟล์):
    setx OPENROUTER_API_KEY "sk-or-v1-..."

ตัวอย่างการใช้งาน:
    python jarvis_hermes_unified.py                 # เมนูเลือกโหมด
    python jarvis_hermes_unified.py --loop          # ลูปต่อเนื่อง
    python jarvis_hermes_unified.py --once          # รอบเดียว
    python jarvis_hermes_unified.py --text "สวัสดี" # ข้ามไมค์ ทดสอบสมอง + เสียง
    python jarvis_hermes_unified.py --calibrate     # วัดระดับเสียงรบกวนของห้อง
    python jarvis_hermes_unified.py --check         # ตรวจความพร้อมของระบบ
"""

import os
import re
import sys
import time
import wave
import queue
import base64
import shutil
import asyncio
import argparse
import tempfile
import platform
import threading
import subprocess
from collections import deque
from datetime import datetime

import numpy as np
import requests

# บังคับ stdout/stderr เป็น UTF-8 กัน UnicodeEncodeError ตอนพิมพ์ภาษาไทยบนคอนโซล Windows
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import sounddevice as sd
except Exception as _sd_err:  # pragma: no cover - ขึ้นกับเครื่องของผู้ใช้
    sd = None
    _SD_IMPORT_ERROR = _sd_err
else:
    _SD_IMPORT_ERROR = None


# ==========================================================================
# 0. ยูทิลิตี้สำหรับอ่านค่า config จาก environment variable
# ==========================================================================
def _env_str(name, default):
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name, default):
    try:
        return int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


# ==========================================================================
# 1. การตั้งค่าระบบหลัก (ทุกค่าปรับผ่าน environment variable ได้)
# ==========================================================================
IS_WINDOWS = platform.system() == "Windows"

# --- เสียง / การอัด ---
SAMPLE_RATE = _env_int("JARVIS_SAMPLE_RATE", 16000)   # Whisper ชอบ 16 kHz mono
CHUNK_SIZE = _env_int("JARVIS_CHUNK_SIZE", 1024)      # ~64 ms ต่อบล็อกที่ 16 kHz
MAX_RECORD_SECONDS = _env_float("JARVIS_MAX_RECORD", 30.0)
SILENCE_DURATION = _env_float("JARVIS_SILENCE_DURATION", 1.2)    # เงียบนานเท่านี้ = พูดจบ
NO_SPEECH_TIMEOUT = _env_float("JARVIS_NO_SPEECH_TIMEOUT", 8.0)  # ไม่พูดเลย = ยกเลิกรอบ
PREROLL_SECONDS = _env_float("JARVIS_PREROLL", 0.5)   # เก็บเสียงก่อนเริ่มพูด กันคำแรกขาด
MIN_SPEECH_SECONDS = _env_float("JARVIS_MIN_SPEECH", 0.35)  # สั้นกว่านี้ถือว่าเป็นเสียงรบกวน
TAIL_SILENCE_SECONDS = _env_float("JARVIS_TAIL_SILENCE", 0.3)  # ความเงียบท้ายที่เก็บไว้
NORMALIZE_TARGET_PEAK = _env_float("JARVIS_NORMALIZE_PEAK", 0.85)
NORMALIZE_MAX_GAIN = _env_float("JARVIS_NORMALIZE_MAX_GAIN", 16.0)

# --- VAD แบบปรับตัวตามห้อง ---
CALIBRATION_SECONDS = _env_float("JARVIS_CALIBRATION", 0.7)
NOISE_MULTIPLIER = _env_float("JARVIS_NOISE_MULTIPLIER", 3.0)
MIN_SILENCE_THRESHOLD = _env_float("JARVIS_MIN_THRESHOLD", 180.0)
MAX_SILENCE_THRESHOLD = _env_float("JARVIS_MAX_THRESHOLD", 4000.0)
FIXED_THRESHOLD = _env_float("JARVIS_FIXED_THRESHOLD", 0.0)  # > 0 = ปิด auto-calibrate

# --- เครือข่าย / STT ---
OPENROUTER_STT_URL = _env_str(
    "JARVIS_STT_URL", "https://openrouter.ai/api/v1/audio/transcriptions"
)
STT_MODEL = _env_str("JARVIS_STT_MODEL", "openai/whisper-large-v3")
STT_LANGUAGE = _env_str("JARVIS_STT_LANGUAGE", "th")
HTTP_TIMEOUT = _env_float("JARVIS_HTTP_TIMEOUT", 60.0)
HTTP_RETRIES = _env_int("JARVIS_HTTP_RETRIES", 3)

# --- Hermes ---
HERMES_TIMEOUT = _env_float("JARVIS_HERMES_TIMEOUT", 180.0)
HERMES_MAX_PROMPT_CHARS = _env_int("JARVIS_HERMES_MAX_PROMPT", 6000)

# --- TTS ---
TTS_VOICE = _env_str("JARVIS_TTS_VOICE", "th-TH-NiwatNeural")  # หญิง: th-TH-PremwadeeNeural
TTS_RATE = _env_str("JARVIS_TTS_RATE", "+0%")
TTS_VOLUME = _env_str("JARVIS_TTS_VOLUME", "+0%")
TTS_MAX_CHARS = _env_int("JARVIS_TTS_MAX_CHARS", 1200)

# --- ไฟล์ทำงาน ---
WORK_DIR = _env_str(
    "JARVIS_WORK_DIR", os.path.join(tempfile.gettempdir(), "jarvis_hermes")
)
KEEP_AUDIO = _env_bool("JARVIS_KEEP_AUDIO", False)
LOG_FILE = _env_str("JARVIS_LOG_FILE", os.path.join(WORK_DIR, "jarvis_session.log"))

# --- คำสั่งเสียงสำหรับปิดระบบ ---
EXIT_PHRASES = (
    "จบการทำงาน", "ปิดระบบ", "หยุดทำงาน", "เลิกทำงาน", "ปิดจาร์วิส",
    "goodbye jarvis", "exit jarvis", "shutdown jarvis",
)

SYSTEM_HINT = (
    "[คำแนะนำระบบ: คุณกำลังทำงานเป็นสมองของ JARVIS ตอบกลับเป็นภาษาไทยแบบสั้น กระชับ เป็นกันเอง "
    "ไม่ต้องใส่ Markdown อิโมจิ หรือโค้ดบล็อก เพราะข้อความจะถูกอ่านออกเสียงผ่าน TTS "
    "หากมีการใช้ Tools ทำงานสำเร็จ ให้สรุปผลสั้นๆ ไม่เกิน 3 ประโยค]"
)


# ==========================================================================
# 2. ระบบ log: พิมพ์ลงจอ + เก็บไฟล์ไว้ย้อนดูภายหลัง
# ==========================================================================
_log_lock = threading.Lock()


def _ensure_work_dir():
    try:
        os.makedirs(WORK_DIR, exist_ok=True)
    except OSError:
        pass


def log(message, to_file=True):
    """พิมพ์ข้อความลง stdout และผนวกลงไฟล์ log พร้อม timestamp"""
    print(message, flush=True)
    if not to_file:
        return
    try:
        _ensure_work_dir()
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with _log_lock, open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {message}\n")
    except OSError:
        pass  # log ล้มเหลวต้องไม่ทำให้ระบบหลักพัง


def warn(message):
    log(f"⚠️  {message}")


def fail(message):
    log(f"❌ {message}")


# ==========================================================================
# 3. API key: environment variable มาก่อน แล้วค่อย fallback ไปที่ k.py
# ==========================================================================
def resolve_api_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if key and key.strip():
        return key.strip(), "environment variable"

    try:
        from k import API_KEY  # type: ignore
    except Exception:
        return "", "not found"

    if API_KEY and str(API_KEY).strip():
        return str(API_KEY).strip(), "k.py"
    return "", "not found"


OPENROUTER_API_KEY, API_KEY_SOURCE = resolve_api_key()


def mask_key(key):
    if not key:
        return "(ไม่พบ)"
    return f"{key[:8]}...{key[-4:]}" if len(key) > 14 else "***"


# ==========================================================================
# 4. ค้นหา hermes.exe ในเครื่องอัตโนมัติ
# ==========================================================================
_HERMES_PATH_CACHE = None


def get_hermes_path(force_refresh=False):
    """คืน path ของ hermes ถ้าเจอ, คืน None ถ้าไม่เจอ (จะได้แจ้งเตือนตั้งแต่ต้น)"""
    global _HERMES_PATH_CACHE
    if _HERMES_PATH_CACHE is not None and not force_refresh:
        return _HERMES_PATH_CACHE

    candidates = []

    override = os.getenv("JARVIS_HERMES_PATH")
    if override:
        candidates.append(override.strip('"'))

    found = shutil.which("hermes") or shutil.which("hermes.exe")
    if found:
        candidates.append(found)

    exe_dir = os.path.dirname(sys.executable)
    candidates.append(os.path.join(exe_dir, "hermes.exe"))
    candidates.append(os.path.join(exe_dir, "hermes"))
    candidates.append(os.path.join(exe_dir, "Scripts", "hermes.exe"))

    candidates.append(
        os.path.expanduser(r"~\AppData\Local\hermes\hermes-agent\venv\Scripts\hermes.exe")
    )
    candidates.append(os.path.expanduser("~/.local/bin/hermes"))

    for path in candidates:
        if path and os.path.isfile(path):
            _HERMES_PATH_CACHE = path
            return path

    _HERMES_PATH_CACHE = None
    return None


# ==========================================================================
# 5. ทำความสะอาดเอาต์พุตจาก Hermes CLI
# ==========================================================================
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_BOX_CHARS_RE = re.compile(r"[─-╿]")
_SPINNER_RE = re.compile(r"[⠁-⣿⏳⌛◐-◓|/\\-]{2,}")

_NOISE_PREFIXES = (
    "Query:", "Initializing", "↻ Resumed", "Resume this", "Session:",
    "Duration:", "Messages:", "Tokens:", "Model:", "Cost:", "Tool:",
    "Thinking", "Loading", "Connecting",
)


def clean_hermes_output(text):
    """ดึงเฉพาะคำตอบจริงออกจากเอาต์พุต CLI ที่เต็มไปด้วยสี กรอบ และ progress bar"""
    if not text:
        return ""

    # ยุบผลของ carriage return (progress bar) เหลือเฉพาะสถานะสุดท้ายของแต่ละบรรทัด
    text = "\n".join(line.split("\r")[-1] for line in text.split("\n"))
    clean_text = _ANSI_RE.sub("", text)

    # กรณีมีกรอบ ╭─ ⚕ Hermes ─╮ ... ╰─╯ ให้ดึงเนื้อในกรอบ (รองรับหลายกรอบ เอากรอบสุดท้าย)
    matches = re.findall(
        r"╭─\s*⚕?\s*Hermes.*?╮\s*\n(.*?)\n\s*╰[─]*╯", clean_text, re.DOTALL
    )
    if matches:
        body = matches[-1]
        body = "\n".join(
            _BOX_CHARS_RE.sub("", ln).strip().strip("│").strip()
            for ln in body.split("\n")
        )
        cleaned = "\n".join(ln for ln in body.split("\n") if ln).strip()
        if cleaned:
            return cleaned

    # ไม่มีกรอบ: กรอง header/footer ของ CLI ออก
    lines = []
    for raw_line in clean_text.split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in _NOISE_PREFIXES):
            continue
        if _BOX_CHARS_RE.sub("", stripped).strip() == "":
            continue  # บรรทัดที่มีแต่เส้นกรอบ
        if _SPINNER_RE.fullmatch(stripped):
            continue
        lines.append(_BOX_CHARS_RE.sub("", line).strip())

    return "\n".join(ln for ln in lines if ln).strip()


# ==========================================================================
# 6. ทำความสะอาดข้อความก่อนส่งเข้า TTS
# ==========================================================================
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


def sanitize_for_tts(text):
    """เอา Markdown / อิโมจิ / URL / โค้ดบล็อก ออก เพื่อไม่ให้ TTS อ่านสัญลักษณ์ออกมา"""
    if not text:
        return ""

    text = _ANSI_RE.sub("", text)
    text = _CODE_BLOCK_RE.sub(" ", text)           # โค้ดบล็อกอ่านออกเสียงไม่รู้เรื่อง
    text = _URL_RE.sub(" ลิงก์ ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _BOX_CHARS_RE.sub(" ", text)

    text = re.sub(r"[*_`~#>]+", " ", text)          # สัญลักษณ์ Markdown
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)  # bullet
    text = text.replace("|", " ")                   # เส้นตาราง
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.strip()

    if len(text) > TTS_MAX_CHARS:
        cut = text[:TTS_MAX_CHARS]
        # ตัดที่ท้ายประโยคถ้าทำได้ จะได้ไม่ค้างกลางคำ
        boundary = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"), cut.rfind("\n"))
        if boundary > TTS_MAX_CHARS * 0.6:
            cut = cut[: boundary + 1]
        text = cut.rstrip() + " ... (ตัดข้อความที่ยาวเกินไป)"

    return text


# ==========================================================================
# 7. เครื่องมือจัดการเสียง
# ==========================================================================
def block_rms(block):
    """RMS ของบล็อกเสียง int16 (คำนวณบน float64 กัน overflow)"""
    if block.size == 0:
        return 0.0
    samples = block.astype(np.float64)
    return float(np.sqrt(np.mean(samples * samples)))


def write_wav(path, samples, sample_rate=SAMPLE_RATE):
    """เขียน WAV 16-bit mono ด้วย stdlib (ไม่ต้องพึ่ง scipy)"""
    pcm = np.asarray(samples, dtype=np.int16).reshape(-1)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def normalize_audio(samples, target_peak=NORMALIZE_TARGET_PEAK):
    """ปรับความดังให้พอเหมาะ ช่วยให้ Whisper ถอดเสียงเบาๆ ได้แม่นขึ้น"""
    pcm = np.asarray(samples, dtype=np.float32).reshape(-1)
    if pcm.size == 0:
        return pcm.astype(np.int16)

    pcm = pcm - float(np.mean(pcm))  # ตัด DC offset ของไมค์ราคาถูก
    peak = float(np.max(np.abs(pcm)))
    if peak < 1.0:
        return np.zeros_like(pcm, dtype=np.int16)

    gain = min((32767.0 * target_peak) / peak, NORMALIZE_MAX_GAIN)  # จำกัด gain กันขยายเสียงรบกวน
    return np.clip(pcm * gain, -32768, 32767).astype(np.int16)


def measure_noise_floor(seconds=CALIBRATION_SECONDS):
    """วัดระดับเสียงรบกวนของห้อง เพื่อตั้ง threshold แบบปรับตัวได้"""
    if sd is None:
        return 0.0
    frames = max(int(seconds * SAMPLE_RATE), CHUNK_SIZE)
    try:
        buffer = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
    except Exception as exc:
        warn(f"วัดเสียงรบกวนไม่สำเร็จ ({exc}) จะใช้ค่าเริ่มต้นแทน")
        return 0.0
    return block_rms(np.asarray(buffer).reshape(-1))


def compute_threshold(noise_floor):
    if FIXED_THRESHOLD > 0:
        return FIXED_THRESHOLD
    threshold = max(noise_floor * NOISE_MULTIPLIER, MIN_SILENCE_THRESHOLD)
    return min(threshold, MAX_SILENCE_THRESHOLD)


# ==========================================================================
# 8. อัดเสียงแบบตรวจจับการพูดอัตโนมัติ (VAD)
# ==========================================================================
def record_audio_vad(output_path, threshold=None):
    """
    อัดเสียงจนกว่าจะเงียบตามเวลาที่กำหนด แล้วคืน path ของไฟล์ WAV
    คืน None ถ้าไม่พบเสียงพูดที่ใช้งานได้
    """
    if sd is None:
        fail(f"ไม่พบไลบรารี sounddevice: {_SD_IMPORT_ERROR}")
        return None

    if threshold is None:
        noise_floor = measure_noise_floor()
        threshold = compute_threshold(noise_floor)
        log(f"🎚️  เสียงรบกวนห้อง ≈ {noise_floor:.0f} → ตั้งเกณฑ์เสียงพูดที่ {threshold:.0f}")

    log("\n🎙️  JARVIS กำลังฟัง... (พูดได้เลย เงียบเมื่อพูดจบ)")

    preroll_blocks = max(int(PREROLL_SECONDS * SAMPLE_RATE / CHUNK_SIZE), 1)
    preroll = deque(maxlen=preroll_blocks)
    speech_blocks = []

    state = {
        "speaking": False,
        "silence_since": None,
        "reason": "timeout",
        "voiced_blocks": 0,   # นับเฉพาะบล็อกที่ดังเกินเกณฑ์ = ความยาวเสียงพูดจริง
        "last_voiced": 0,     # ตำแหน่งบล็อกสุดท้ายที่มีเสียง ใช้ตัดความเงียบท้ายทิ้ง
    }
    stop_event = threading.Event()
    status_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            # ไม่พิมพ์ทันทีเพื่อไม่ให้รบกวนจอ แต่เก็บไว้รายงานตอนจบ
            status_queue.put(str(status))

        block = np.asarray(indata, dtype=np.int16).reshape(-1).copy()
        rms = block_rms(block)
        now = time.monotonic()

        if rms > threshold:
            if not state["speaking"]:
                state["speaking"] = True
                speech_blocks.extend(preroll)  # ใส่เสียงก่อนหน้ากลับไป กันคำแรกขาด
                preroll.clear()
                print("🗣️  ตรวจพบเสียงพูด... กำลังบันทึก", flush=True)
            state["silence_since"] = None
            speech_blocks.append(block)
            state["voiced_blocks"] += 1
            state["last_voiced"] = len(speech_blocks)
            return

        if state["speaking"]:
            speech_blocks.append(block)  # เก็บช่วงเงียบสั้นๆ ระหว่างประโยคไว้ด้วย
            if state["silence_since"] is None:
                state["silence_since"] = now
            elif now - state["silence_since"] >= SILENCE_DURATION:
                state["reason"] = "silence"
                stop_event.set()
                raise sd.CallbackStop()
        else:
            preroll.append(block)

    start = time.monotonic()
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=callback,
        ):
            # รอด้วย Event แทนการ sleep วนลูป -> หยุดทันทีที่ตรวจพบความเงียบ
            while not stop_event.wait(0.05):
                elapsed = time.monotonic() - start
                if elapsed > MAX_RECORD_SECONDS:
                    state["reason"] = "max_length"
                    break
                if not state["speaking"] and elapsed > NO_SPEECH_TIMEOUT:
                    state["reason"] = "no_speech"
                    break
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        fail(f"เกิดข้อผิดพลาดขณะอัดเสียง: {exc}")
        return None

    if state["reason"] == "silence":
        log("🛑 ตรวจพบความเงียบ... จบการอัดเสียง")
    elif state["reason"] == "max_length":
        log("⏱️  ครบกำหนดเวลาอัดสูงสุดแล้ว")
    elif state["reason"] == "no_speech":
        log("😴 ไม่ได้ยินเสียงพูดเลย ข้ามรอบนี้")
        return None

    if not state["speaking"] or not speech_blocks:
        log("❌ ไม่พบเสียงพูดในช่วงเวลาที่กำหนด")
        return None

    # วัดจากบล็อกที่มีเสียงพูดจริง ไม่รวม preroll กับความเงียบท้าย
    # ทำให้เสียงไอ/จาม/เคาะโต๊ะ ถูกคัดทิ้งก่อนเปลืองโควตา STT
    block_seconds = CHUNK_SIZE / SAMPLE_RATE
    speech_seconds = state["voiced_blocks"] * block_seconds
    if speech_seconds < MIN_SPEECH_SECONDS:
        log(f"❌ เสียงพูดสั้นเกินไป ({speech_seconds:.2f} วิ) น่าจะเป็นเสียงรบกวน")
        return None

    # ตัดความเงียบท้ายประโยคออก เหลือไว้พอเป็นหางเสียง -> ไฟล์เล็กลง STT เร็วขึ้น
    tail_blocks = max(int(TAIL_SILENCE_SECONDS / block_seconds), 1)
    keep_until = min(state["last_voiced"] + tail_blocks, len(speech_blocks))
    recording = np.concatenate(speech_blocks[:keep_until])
    duration = len(recording) / SAMPLE_RATE

    write_wav(output_path, normalize_audio(recording))

    size_kb = os.path.getsize(output_path) / 1024
    log(f"💾 บันทึกเสียงแล้ว: {duration:.1f} วินาที "
        f"(พูดจริง {speech_seconds:.1f} วิ, {size_kb:.0f} KB)")

    warnings = set()
    while not status_queue.empty():
        warnings.add(status_queue.get())
    if warnings:
        warn(f"สถานะไมโครโฟน: {', '.join(sorted(warnings))}")

    return output_path


# ==========================================================================
# 9. แปลงเสียงเป็นข้อความ (STT)
# ==========================================================================
_session = requests.Session()


def _post_with_retry(**kwargs):
    """POST พร้อม retry + exponential backoff เฉพาะ error ที่ลองใหม่แล้วมีโอกาสสำเร็จ"""
    last_error = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            response = _session.post(timeout=HTTP_TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            last_error = exc
        else:
            if response.status_code < 500 and response.status_code != 429:
                return response, None
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"

        if attempt < HTTP_RETRIES:
            delay = 1.5 ** attempt
            warn(f"เชื่อมต่อไม่สำเร็จ (ครั้งที่ {attempt}/{HTTP_RETRIES}) ลองใหม่ใน {delay:.1f} วิ")
            time.sleep(delay)

    return None, last_error


def _extract_transcript(response):
    """รองรับรูปแบบ response ได้หลายแบบตามที่แต่ละ provider ส่งกลับมา"""
    if response is None or response.status_code != 200:
        return ""
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()

    if isinstance(data, str):
        return data.strip()

    for key in ("text", "transcript", "transcription"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # เผื่อ provider ส่งกลับในรูปแบบ chat completion
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def speech_to_text(audio_path):
    """ถอดเสียงเป็นข้อความ ลองแบบ JSON base64 ก่อน ถ้าไม่ผ่านค่อยลองแบบ multipart"""
    if not OPENROUTER_API_KEY:
        fail("ไม่พบ OPENROUTER_API_KEY (ตั้งค่าผ่าน environment variable หรือไฟล์ k.py)")
        return ""

    if not audio_path or not os.path.exists(audio_path):
        fail("ไม่พบไฟล์เสียงที่จะถอดข้อความ")
        return ""

    log("🤖 กำลังแปลงเสียงเป็นข้อความ...")
    started = time.monotonic()

    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()

    auth = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

    # --- วิธีที่ 1: JSON + base64 ---
    response, error = _post_with_retry(
        url=OPENROUTER_STT_URL,
        headers={**auth, "Content-Type": "application/json"},
        json={
            "model": STT_MODEL,
            "input_audio": {
                "data": base64.b64encode(audio_bytes).decode("utf-8"),
                "format": "wav",
            },
            "language": STT_LANGUAGE,
        },
    )

    text = _extract_transcript(response)
    if text:
        log(f"⏱️  STT ใช้เวลา {time.monotonic() - started:.1f} วินาที")
        return text

    detail = f"HTTP {response.status_code}" if response is not None else error
    warn(f"STT แบบ JSON ไม่สำเร็จ ({detail}) กำลังลองแบบ multipart...")

    # --- วิธีที่ 2: multipart/form-data (รูปแบบมาตรฐาน OpenAI-compatible) ---
    response, error = _post_with_retry(
        url=OPENROUTER_STT_URL,
        headers=auth,  # ห้ามตั้ง Content-Type เอง ต้องปล่อยให้ requests ใส่ boundary
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"model": STT_MODEL, "language": STT_LANGUAGE},
    )

    text = _extract_transcript(response)
    if text:
        log(f"⏱️  STT ใช้เวลา {time.monotonic() - started:.1f} วินาที")
        return text

    if response is not None:
        fail(f"STT Error {response.status_code}: {response.text[:300]}")
    else:
        fail(f"STT Error: {error}")
    return ""


# ==========================================================================
# 10. ส่งให้ Hermes Agent คิดและทำตามคำสั่ง
# ==========================================================================
def ask_hermes_brain(user_text):
    log(f"👤 คุณพูดว่า: '{user_text}'")
    log("🧠 กำลังส่งให้ Hermes Agent ประมวลผล (พร้อม Memory และ Tools)...")

    hermes_path = get_hermes_path()
    if not hermes_path:
        fail("ไม่พบโปรแกรม hermes ในเครื่อง (ตั้งค่า JARVIS_HERMES_PATH เพื่อระบุตำแหน่งเอง)")
        return "ขออภัยครับ ยังไม่ได้ติดตั้งระบบ Hermes ในเครื่องนี้"

    trimmed = user_text.strip()
    if len(trimmed) > HERMES_MAX_PROMPT_CHARS:
        trimmed = trimmed[:HERMES_MAX_PROMPT_CHARS] + " ..."
    prompt = f"{trimmed}\n\n{SYSTEM_HINT}"

    kwargs = {}
    if IS_WINDOWS and hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # ไม่ให้หน้าต่างคอนโซลกระพริบ

    started = time.monotonic()
    try:
        result = subprocess.run(
            [hermes_path, "chat", "--continue", "-q", prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",           # กัน UnicodeDecodeError จากเอาต์พุตภาษาไทยบน Windows
            timeout=HERMES_TIMEOUT,
            stdin=subprocess.DEVNULL,   # กัน CLI ค้างรอ input
            **kwargs,
        )
    except subprocess.TimeoutExpired:
        fail(f"Hermes ใช้เวลานานเกิน {HERMES_TIMEOUT:.0f} วินาที จึงยกเลิกคำสั่ง")
        return "ขออภัยครับ คำสั่งนี้ใช้เวลานานเกินไป ผมจึงยกเลิกให้ก่อน"
    except FileNotFoundError:
        get_hermes_path(force_refresh=True)
        fail(f"เรียกใช้ไม่ได้: {hermes_path}")
        return "ขออภัยครับ ไม่สามารถเรียกใช้โปรแกรม Hermes ได้"
    except Exception as exc:
        fail(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Hermes: {exc}")
        return "ขออภัยครับ ไม่สามารถเชื่อมต่อระบบ Hermes ได้"

    log(f"⏱️  Hermes ใช้เวลา {time.monotonic() - started:.1f} วินาที")

    if result.returncode != 0:
        detail = clean_hermes_output(result.stderr or result.stdout)
        fail(f"Hermes ตอบกลับด้วย exit code {result.returncode}")
        if detail:
            warn(detail[:500])
        return "ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผลคำสั่ง"

    cleaned = clean_hermes_output(result.stdout)
    if not cleaned:
        cleaned = clean_hermes_output(result.stderr)  # บาง CLI พิมพ์ผลลัพธ์ลง stderr
    return cleaned or "ดำเนินการเรียบร้อยครับ"


# ==========================================================================
# 11. แปลงข้อความเป็นเสียง (TTS) และเล่นเสียง
# ==========================================================================
async def _generate_tts_async(text, output_path):
    import edge_tts

    communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE, volume=TTS_VOLUME)
    await communicate.save(output_path)


def generate_tts(text, output_path):
    try:
        asyncio.run(_generate_tts_async(text, output_path))
    except ImportError:
        fail("ไม่พบไลบรารี edge-tts (ติดตั้งด้วย: pip install edge-tts)")
        return False
    except Exception as exc:
        fail(f"สร้างเสียงไม่สำเร็จ: {exc}")
        return False

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 512:
        fail("ไฟล์เสียงที่สร้างได้ว่างเปล่า")
        return False
    return True


def _play_with_playsound(path):
    from playsound import playsound

    playsound(path)
    return True


def _play_with_ffplay(path):
    ffplay = shutil.which("ffplay")
    if not ffplay:
        return False
    subprocess.run(
        [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", path],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return True


def _play_with_powershell(path):
    if not IS_WINDOWS:
        return False
    script = (
        "Add-Type -AssemblyName presentationCore; "
        "$p = New-Object System.Windows.Media.MediaPlayer; "
        f"$p.Open([uri]'{path}'); "
        "Start-Sleep -Milliseconds 500; "
        "$p.Play(); "
        "$d = 30; try { $d = $p.NaturalDuration.TimeSpan.TotalSeconds } catch {}; "
        "Start-Sleep -Seconds $d; $p.Close();"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    return True


PLAYBACK_BACKENDS = (
    ("playsound", _play_with_playsound),
    ("ffplay", _play_with_ffplay),
    ("powershell", _play_with_powershell),
)


def play_audio(path):
    """ลองเล่นเสียงด้วยหลาย backend ระบบจะไม่ล่มถ้า backend ใดใช้ไม่ได้"""
    errors = []
    for name, backend in PLAYBACK_BACKENDS:
        try:
            if backend(path):
                return True
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    warn("เล่นเสียงไม่สำเร็จ | " + (" | ".join(errors) if errors else "ไม่พบตัวเล่นเสียง"))
    return False


def text_to_speech(ai_response_text, output_path):
    log(f"🤖 คำตอบจาก Hermes: '{ai_response_text}'")

    spoken = sanitize_for_tts(ai_response_text)
    if not spoken:
        warn("ไม่มีข้อความที่จะอ่านออกเสียง")
        return False

    log("🔊 กำลังสังเคราะห์เสียง...")
    started = time.monotonic()
    if not generate_tts(spoken, output_path):
        return False
    log(f"⏱️  TTS ใช้เวลา {time.monotonic() - started:.1f} วินาที")

    played = play_audio(output_path)
    if not KEEP_AUDIO:
        try:
            os.remove(output_path)
        except OSError:
            pass
    return played


# ==========================================================================
# 12. ตรวจความพร้อมของระบบก่อนเริ่มทำงาน
# ==========================================================================
def preflight_check(need_mic=True):
    log("\n🔍 ตรวจสอบความพร้อมของระบบ")
    log("-" * 50)
    ready = True

    # API key
    if OPENROUTER_API_KEY:
        log(f"  ✅ API Key      : {mask_key(OPENROUTER_API_KEY)} (จาก {API_KEY_SOURCE})")
        if API_KEY_SOURCE == "k.py":
            log("     ⚠️  แนะนำให้ย้ายไปใช้ environment variable OPENROUTER_API_KEY")
    else:
        log("  ❌ API Key      : ไม่พบ (ตั้ง OPENROUTER_API_KEY หรือสร้างไฟล์ k.py)")
        ready = False

    # Hermes
    hermes_path = get_hermes_path()
    if hermes_path:
        log(f"  ✅ Hermes CLI   : {hermes_path}")
    else:
        log("  ❌ Hermes CLI   : ไม่พบในเครื่อง")
        ready = False

    # edge-tts
    try:
        import edge_tts  # noqa: F401
        log(f"  ✅ edge-tts     : พร้อมใช้งาน (เสียง {TTS_VOICE})")
    except ImportError:
        log("  ❌ edge-tts     : ไม่พบ (pip install edge-tts)")
        ready = False

    # ตัวเล่นเสียง
    players = []
    try:
        import playsound  # noqa: F401
        players.append("playsound")
    except ImportError:
        pass
    if shutil.which("ffplay"):
        players.append("ffplay")
    if IS_WINDOWS:
        players.append("powershell")
    if players:
        log(f"  ✅ ตัวเล่นเสียง  : {', '.join(players)}")
    else:
        log("  ❌ ตัวเล่นเสียง  : ไม่พบ (pip install playsound==1.2.2)")
        ready = False

    # ไมโครโฟน
    if need_mic:
        if sd is None:
            log(f"  ❌ ไมโครโฟน     : sounddevice ใช้งานไม่ได้ ({_SD_IMPORT_ERROR})")
            ready = False
        else:
            try:
                device = sd.query_devices(kind="input")
                log(f"  ✅ ไมโครโฟน     : {device['name']}")
            except Exception as exc:
                log(f"  ❌ ไมโครโฟน     : ไม่พบอุปกรณ์อินพุต ({exc})")
                ready = False

    log(f"  📁 โฟลเดอร์งาน  : {WORK_DIR}")
    log("-" * 50)
    return ready


# ==========================================================================
# 13. Pipeline หลัก
# ==========================================================================
def _session_paths(turn):
    _ensure_work_dir()
    stamp = datetime.now().strftime("%H%M%S")
    return (
        os.path.join(WORK_DIR, f"user_{stamp}_{turn}.wav"),
        os.path.join(WORK_DIR, f"jarvis_{stamp}_{turn}.mp3"),
    )


def run_pipeline(turn=0, threshold=None, forced_text=None, speak=True):
    """
    รันหนึ่งรอบสนทนา
    คืน True ถ้าผู้ใช้สั่งให้ปิดระบบ, คืน False ถ้าให้ทำงานต่อ
    """
    input_path, output_path = _session_paths(turn)
    turn_started = time.monotonic()

    try:
        if forced_text:
            user_text = forced_text
        else:
            if not record_audio_vad(input_path, threshold=threshold):
                return False
            user_text = speech_to_text(input_path)

        if not user_text or not user_text.strip():
            fail("ไม่ได้ยินเสียงพูด หรือฟังข้อความไม่ออก")
            return False

        normalized = user_text.strip().lower()
        if any(phrase in normalized for phrase in EXIT_PHRASES):
            log("👋 รับทราบครับ กำลังปิดระบบ")
            if speak:
                text_to_speech("รับทราบครับ แล้วพบกันใหม่", output_path)
            return True

        ai_reply = ask_hermes_brain(user_text)
        if speak:
            text_to_speech(ai_reply, output_path)
        else:
            log(f"🤖 คำตอบจาก Hermes: '{ai_reply}'")

        log(f"✅ จบรอบที่ {turn + 1} (รวม {time.monotonic() - turn_started:.1f} วินาที)")
        return False

    finally:
        if not KEEP_AUDIO and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass


def run_loop(threshold=None, speak=True):
    log("\n🔄 โหมดลูปต่อเนื่อง (กด Ctrl+C หรือพูดว่า 'ปิดระบบ' เพื่อเลิกรัน)")
    turn = 0
    consecutive_errors = 0

    while True:
        try:
            should_exit = run_pipeline(turn=turn, threshold=threshold, speak=speak)
            consecutive_errors = 0
            if should_exit:
                break
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            consecutive_errors += 1
            fail(f"รอบนี้ล้มเหลว: {exc}")
            if consecutive_errors >= 3:
                fail("เกิดข้อผิดพลาดติดกัน 3 ครั้ง หยุดการทำงานเพื่อความปลอดภัย")
                break
            time.sleep(2 * consecutive_errors)  # backoff กันลูปไหม้ CPU

        turn += 1
        time.sleep(0.6)  # เว้นจังหวะให้ลำโพงเงียบก่อนเปิดไมค์รอบใหม่


# ==========================================================================
# 14. Entry point
# ==========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="JARVIS Voice Assistant powered by Hermes Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="รันรอบเดียวแล้วจบ")
    mode.add_argument("--loop", action="store_true", help="รันแบบลูปต่อเนื่อง")
    mode.add_argument("--check", action="store_true", help="ตรวจความพร้อมของระบบแล้วจบ")
    mode.add_argument("--calibrate", action="store_true", help="วัดระดับเสียงรบกวนของห้อง")
    parser.add_argument("--text", help="ข้ามไมค์และ STT ส่งข้อความนี้ให้ Hermes โดยตรง")
    parser.add_argument("--no-tts", action="store_true", help="ไม่ต้องอ่านออกเสียง แสดงผลเป็นข้อความ")
    parser.add_argument("--threshold", type=float, help="กำหนดเกณฑ์เสียงพูดเอง (ปิด auto-calibrate)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    speak = not args.no_tts

    log("=" * 50)
    log("🤖 JARVIS VOICE ASSISTANT (Powered by Hermes Agent)")
    log("=" * 50)

    if args.check:
        return 0 if preflight_check(need_mic=True) else 1

    if args.calibrate:
        if sd is None:
            fail(f"sounddevice ใช้งานไม่ได้: {_SD_IMPORT_ERROR}")
            return 1
        log("\n🤫 กรุณาเงียบสักครู่ กำลังวัดเสียงรบกวนของห้อง...")
        noise = measure_noise_floor(max(CALIBRATION_SECONDS, 2.0))
        suggested = compute_threshold(noise)
        log(f"📊 ระดับเสียงรบกวน : {noise:.0f}")
        log(f"📊 เกณฑ์ที่แนะนำ    : {suggested:.0f}")
        log(f'💡 ล็อกค่านี้ได้ด้วย: setx JARVIS_FIXED_THRESHOLD "{suggested:.0f}"')
        return 0

    # โหมดข้อความ: ทดสอบสมองกับเสียงได้โดยไม่ต้องใช้ไมค์
    if args.text:
        if not preflight_check(need_mic=False):
            fail("ระบบยังไม่พร้อมใช้งาน")
            return 1
        run_pipeline(turn=0, forced_text=args.text, speak=speak)
        return 0

    if not preflight_check(need_mic=True):
        fail("ระบบยังไม่พร้อมใช้งาน กรุณาแก้ไขรายการที่มีเครื่องหมาย ❌ ก่อน")
        return 1

    # เลือกโหมด: จาก flag ถ้ามี ไม่งั้นถามผ่านเมนู
    if args.once:
        choice = "1"
    elif args.loop:
        choice = "2"
    elif not sys.stdin or not sys.stdin.isatty():
        choice = "2"  # รันผ่าน pipe/สคริปต์ ให้ default เป็นลูป ไม่ค้างรอ input
    else:
        log("\nเลือกรอบการทำงาน:")
        log("  1. รันรอบเดียว (One-time Execution)")
        log("  2. รันแบบลูปต่อเนื่อง (Continuous Loop)")
        try:
            choice = input("กรุณาเลือก (1 หรือ 2) [ค่าเริ่มต้น: 2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            log("\n👋 ยกเลิกการทำงาน")
            return 0

    # คำนวณเกณฑ์เสียงครั้งเดียวแล้วใช้ซ้ำทั้ง session ไม่ต้องวัดใหม่ทุกรอบ
    threshold = args.threshold
    if threshold is None and FIXED_THRESHOLD > 0:
        threshold = FIXED_THRESHOLD
    if threshold is None and sd is not None:
        log("\n🤫 กำลังวัดเสียงรบกวนของห้องสักครู่...")
        noise = measure_noise_floor()
        threshold = compute_threshold(noise)
        log(f"🎚️  เสียงรบกวน ≈ {noise:.0f} → เกณฑ์เสียงพูด {threshold:.0f}")

    try:
        if choice == "1":
            log("\n▶️  เริ่มต้นรันรอบเดียว...")
            run_pipeline(turn=0, threshold=threshold, speak=speak)
        else:
            run_loop(threshold=threshold, speak=speak)
    except KeyboardInterrupt:
        log("\n👋 ยุติการทำงานของ JARVIS เรียบร้อยแล้ว")
    finally:
        try:
            if sd is not None:
                sd.stop()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
