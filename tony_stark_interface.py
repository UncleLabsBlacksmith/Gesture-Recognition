#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TONY STARK INTERFACE  (JARVIS + Gesture Virtual Mouse)
======================================================
* หน้าจอ fullscreen พื้นหลังโปร่งใส (tkinter) เห็นเดสก์ท็อปทะลุได้
* แชทกับ AI ผ่าน OpenRouter แบบ streaming  (พิมพ์ หรือ พูดผ่านไมค์)
* เสียงตอบกลับแบบ streaming: อ่านทีละประโยคทันทีที่ AI พิมพ์ครบประโยค
* ควบคุมเมาส์ด้วยท่ามือผ่านเว็บแคม (ระบบเดียวกับ gesture_v2.py)

ท่ามือ (เหมือน gesture_v2.py)
    ชูนิ้วชี้              -> เลื่อนเคอร์เซอร์
    นิ้วโป้ง+นิ้วชี้แตะกัน   -> กดเมาส์ซ้าย (ขยับ = ลาก, ปล่อย = คลิก)
    ชู 3 นิ้ว (ชี้ กลาง นาง) -> คลิกขวา
    ชู 2 นิ้ว (ชี้ กลาง)    -> โหมดเลื่อนหน้า (ขยับมือขึ้น/ลง)
    ชูนิ้วก้อยนิ้วเดียว      -> กด Enter
    กำมือ                 -> หยุด (Pause)      แบมือ -> ทำงานต่อ (Resume)

ติดตั้ง:
    pip install sounddevice numpy requests edge-tts opencv-python mediapipe pyautogui

ตั้งค่า API key (หรือสร้างไฟล์ k.py ที่มี API_KEY = "sk-or-..."):
    setx OPENROUTER_API_KEY "sk-or-v1-..."

ใช้งาน:
    python tony_stark_interface.py                # เปิดหน้าจอเต็ม
    python tony_stark_interface.py --no-gesture   # ไม่เปิดระบบควบคุมเมาส์
    python tony_stark_interface.py --no-tts       # ไม่อ่านออกเสียง
    python tony_stark_interface.py --check        # ตรวจความพร้อมของระบบ

ปุ่มลัด: ESC = ออก | Enter = ส่งข้อความ
"""

import os
import re
import sys
import json
import math
import time
import wave
import queue
import base64
import shutil
import asyncio
import ctypes
import argparse
import tempfile
import platform
import threading
import subprocess
import tkinter as tk
from tkinter import font as tkfont
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

IS_WINDOWS = platform.system() == "Windows"

# ต้องเรียกก่อนสร้าง Tk เพื่อให้พิกัดหน้าจอตรงกับ pyautogui บนจอที่ตั้ง scale
if IS_WINDOWS:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
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
# 0. ยูทิลิตี้อ่านค่า config จาก environment variable
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
# --- เสียง / การอัด ---
SAMPLE_RATE = _env_int("JARVIS_SAMPLE_RATE", 16000)
CHUNK_SIZE = _env_int("JARVIS_CHUNK_SIZE", 1024)
MAX_RECORD_SECONDS = _env_float("JARVIS_MAX_RECORD", 30.0)
SILENCE_DURATION = _env_float("JARVIS_SILENCE_DURATION", 1.2)
NO_SPEECH_TIMEOUT = _env_float("JARVIS_NO_SPEECH_TIMEOUT", 8.0)
PREROLL_SECONDS = _env_float("JARVIS_PREROLL", 0.5)
MIN_SPEECH_SECONDS = _env_float("JARVIS_MIN_SPEECH", 0.35)
TAIL_SILENCE_SECONDS = _env_float("JARVIS_TAIL_SILENCE", 0.3)
NORMALIZE_TARGET_PEAK = _env_float("JARVIS_NORMALIZE_PEAK", 0.85)
NORMALIZE_MAX_GAIN = _env_float("JARVIS_NORMALIZE_MAX_GAIN", 16.0)

# --- VAD แบบปรับตัวตามห้อง ---
CALIBRATION_SECONDS = _env_float("JARVIS_CALIBRATION", 0.7)
NOISE_MULTIPLIER = _env_float("JARVIS_NOISE_MULTIPLIER", 3.0)
MIN_SILENCE_THRESHOLD = _env_float("JARVIS_MIN_THRESHOLD", 180.0)
MAX_SILENCE_THRESHOLD = _env_float("JARVIS_MAX_THRESHOLD", 4000.0)
FIXED_THRESHOLD = _env_float("JARVIS_FIXED_THRESHOLD", 0.0)

# --- เครือข่าย / STT / LLM ---
OPENROUTER_STT_URL = _env_str(
    "JARVIS_STT_URL", "https://openrouter.ai/api/v1/audio/transcriptions"
)
OPENROUTER_CHAT_URL = _env_str(
    "JARVIS_CHAT_URL", "https://openrouter.ai/api/v1/chat/completions"
)
STT_MODEL = _env_str("JARVIS_STT_MODEL", "openai/whisper-large-v3")
STT_LANGUAGE = _env_str("JARVIS_STT_LANGUAGE", "th")
LLM_MODEL = _env_str("JARVIS_LLM_MODEL", "openai/gpt-4o-mini")
LLM_TEMPERATURE = _env_float("JARVIS_LLM_TEMPERATURE", 0.7)
HISTORY_MESSAGES = _env_int("JARVIS_HISTORY", 24)     # จำนวนข้อความย้อนหลังที่ส่งให้ AI
HTTP_TIMEOUT = _env_float("JARVIS_HTTP_TIMEOUT", 60.0)
HTTP_RETRIES = _env_int("JARVIS_HTTP_RETRIES", 3)

SYSTEM_PROMPT = _env_str(
    "JARVIS_SYSTEM_PROMPT",
    "คุณคือ JARVIS ผู้ช่วย AI สไตล์ของ Tony Stark สุภาพ ฉลาด มีอารมณ์ขันเล็กน้อย "
    "ตอบเป็นภาษาเดียวกับที่ผู้ใช้ใช้ (ค่าเริ่มต้นคือภาษาไทย) กระชับ เป็นธรรมชาติ "
    "เพราะคำตอบจะถูกอ่านออกเสียงด้วย TTS จึงหลีกเลี่ยงตาราง อิโมจิ และ Markdown ที่ซับซ้อน "
    "ยกเว้นผู้ใช้ขอโค้ดหรือรายการโดยตรง",
)

# --- TTS ---
TTS_VOICE = _env_str("JARVIS_TTS_VOICE", "th-TH-NiwatNeural")  # หญิง: th-TH-PremwadeeNeural
TTS_RATE = _env_str("JARVIS_TTS_RATE", "+0%")
TTS_VOLUME = _env_str("JARVIS_TTS_VOLUME", "+0%")
TTS_MAX_CHARS = _env_int("JARVIS_TTS_MAX_CHARS", 600)   # ต่อหนึ่งประโยค/ชิ้น

# --- ไฟล์ทำงาน ---
WORK_DIR = _env_str(
    "JARVIS_WORK_DIR", os.path.join(tempfile.gettempdir(), "tony_stark_interface")
)
KEEP_AUDIO = _env_bool("JARVIS_KEEP_AUDIO", False)
LOG_FILE = _env_str("JARVIS_LOG_FILE", os.path.join(WORK_DIR, "session.log"))

# --- กล้อง / ท่ามือ ---
CAMERA_INDEX = _env_int("JARVIS_CAMERA", 0)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
PREVIEW_W, PREVIEW_H = 320, 180

# --- คำสั่งเสียงสำหรับปิดระบบ ---
EXIT_PHRASES = (
    "จบการทำงาน", "ปิดระบบ", "หยุดทำงาน", "เลิกทำงาน", "ปิดจาร์วิส",
    "goodbye jarvis", "exit jarvis", "shutdown jarvis",
)


# ==========================================================================
# 2. ระบบ log: พิมพ์ลงจอ + เก็บไฟล์ + ส่งต่อให้ UI แสดงเป็นแถบสถานะ
# ==========================================================================
_log_lock = threading.Lock()
_log_hook = None   # ฟังก์ชันที่ UI ตั้งไว้ รับข้อความล่าสุด


def _ensure_work_dir():
    try:
        os.makedirs(WORK_DIR, exist_ok=True)
    except OSError:
        pass


def log(message, to_file=True):
    try:
        print(message, flush=True)
    except Exception:
        pass
    if _log_hook is not None:
        try:
            _log_hook(str(message).strip())
        except Exception:
            pass
    if not to_file:
        return
    try:
        _ensure_work_dir()
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with _log_lock, open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {message}\n")
    except OSError:
        pass


def warn(message):
    log(f"⚠️  {message}")


def fail(message):
    log(f"❌ {message}")


# ==========================================================================
# 3. API key: environment variable > คีย์ที่บันทึกจากหน้าโปรแกรม > k.py
#    คีย์ที่บันทึกเก็บนอกโฟลเดอร์โปรเจกต์ (%APPDATA%) และเข้ารหัสด้วย Windows DPAPI
#    ผูกกับบัญชี Windows ของผู้ใช้ จึงไม่หลุดขึ้น GitHub ตอน commit โค้ด
# ==========================================================================
KEY_DIR = os.path.join(os.getenv("APPDATA") or os.path.expanduser("~"), "TonyStarkInterface")
KEY_FILE = os.path.join(KEY_DIR, "api_key.bin")


def _dpapi(data, protect):
    """เข้ารหัส/ถอดรหัสด้วย Windows DPAPI (CryptProtectData / CryptUnprotectData)"""
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    blob_in = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    blob_out = Blob()
    fn = ctypes.windll.crypt32.CryptProtectData if protect else ctypes.windll.crypt32.CryptUnprotectData
    if not fn(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)):
        raise OSError("DPAPI ล้มเหลว")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        k32 = ctypes.windll.kernel32
        k32.LocalFree.argtypes = [ctypes.c_void_p]
        k32.LocalFree(ctypes.cast(blob_out.pbData, ctypes.c_void_p))


def load_saved_key():
    try:
        with open(KEY_FILE, "rb") as fh:
            raw = fh.read()
        if raw[:1] == b"D":
            return _dpapi(raw[1:], protect=False).decode("utf-8").strip()
        if raw[:1] == b"P":
            return raw[1:].decode("utf-8").strip()
    except Exception:
        pass
    return ""


def save_api_key(key):
    """บันทึกคีย์ไว้ใช้ครั้งต่อไป และใช้งานทันที คืน (สำเร็จไหม, ข้อความ error)"""
    key = (key or "").strip()
    if not key:
        return False, "กรุณากรอก API key"
    try:
        os.makedirs(KEY_DIR, exist_ok=True)
        if IS_WINDOWS:
            raw = b"D" + _dpapi(key.encode("utf-8"), protect=True)
        else:
            raw = b"P" + key.encode("utf-8")
        with open(KEY_FILE, "wb") as fh:
            fh.write(raw)
        if not IS_WINDOWS:
            os.chmod(KEY_FILE, 0o600)
    except Exception as exc:
        return False, f"บันทึกไม่สำเร็จ: {exc}"
    refresh_api_key()
    return True, ""


def delete_saved_key():
    try:
        os.remove(KEY_FILE)
    except FileNotFoundError:
        pass
    except OSError as exc:
        return False, f"ลบไม่สำเร็จ: {exc}"
    refresh_api_key()
    return True, ""


def resolve_api_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if key and key.strip():
        return key.strip(), "environment variable"

    saved = load_saved_key()
    if saved:
        return saved, "saved"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from k import API_KEY  # type: ignore
    except Exception:
        return "", "not found"

    if API_KEY and str(API_KEY).strip():
        return str(API_KEY).strip(), "k.py"
    return "", "not found"


OPENROUTER_API_KEY, API_KEY_SOURCE = resolve_api_key()


def refresh_api_key():
    """อ่านคีย์ใหม่ (หลังบันทึก/ลบ) ให้ทั้งโปรแกรมใช้ค่าล่าสุดทันที"""
    global OPENROUTER_API_KEY, API_KEY_SOURCE
    OPENROUTER_API_KEY, API_KEY_SOURCE = resolve_api_key()


API_KEY_SOURCE_LABEL = {
    "environment variable": "environment variable (OPENROUTER_API_KEY)",
    "saved": "คีย์ที่บันทึกไว้ในเครื่องนี้",
    "k.py": "ไฟล์ k.py",
    "not found": "ไม่พบ",
}


def mask_key(key):
    if not key:
        return "(ไม่พบ)"
    return f"{key[:8]}...{key[-4:]}" if len(key) > 14 else "***"


# ==========================================================================
# 4. ทำความสะอาดข้อความก่อนส่งเข้า TTS
# ==========================================================================
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_BOX_CHARS_RE = re.compile(r"[─-╿]")
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
    text = _CODE_BLOCK_RE.sub(" ", text)
    text = _URL_RE.sub(" ลิงก์ ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _BOX_CHARS_RE.sub(" ", text)

    text = re.sub(r"[*_`~#>]+", " ", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    text = text.replace("|", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.strip()

    if len(text) > TTS_MAX_CHARS:
        text = text[:TTS_MAX_CHARS].rstrip()
    return text


# ==========================================================================
# 5. เครื่องมือจัดการเสียง (อัด / VAD)
# ==========================================================================
# --- การตั้งค่าไมค์ที่ผู้ใช้ปรับผ่านเมนู (เก็บในไฟล์ JSON ข้างสคริปต์) ---
MIC_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mic_config.json")
MIC_CFG = {
    "device": None,              # index อุปกรณ์อินพุต (None = ค่าเริ่มต้นของระบบ)
    "auto": True,                # True = วัดเสียงรบกวนห้องแล้วตั้งเกณฑ์เอง
    "threshold": 800.0,          # เกณฑ์เสียงพูดแบบกำหนดเอง (ใช้เมื่อ auto = False)
    "silence": SILENCE_DURATION,  # เงียบนานเท่านี้ (วินาที) = พูดจบ
}


def load_mic_config():
    try:
        with open(MIC_CONFIG_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data.get("auto"), bool):
            MIC_CFG["auto"] = data["auto"]
        if isinstance(data.get("threshold"), (int, float)):
            MIC_CFG["threshold"] = float(data["threshold"])
        if isinstance(data.get("silence"), (int, float)):
            MIC_CFG["silence"] = float(data["silence"])
        dev = data.get("device")
        if isinstance(dev, int) and sd is not None:
            try:
                if sd.query_devices(dev)["max_input_channels"] > 0:
                    MIC_CFG["device"] = dev
            except Exception:
                pass                      # อุปกรณ์หายไป -> กลับไปใช้ค่าเริ่มต้น
    except (OSError, ValueError):
        pass


def save_mic_config():
    try:
        with open(MIC_CONFIG_FILE, "w", encoding="utf-8") as fh:
            json.dump(MIC_CFG, fh, ensure_ascii=False, indent=2)
        return True
    except OSError as exc:
        warn(f"บันทึกการตั้งค่าไมค์ไม่สำเร็จ: {exc}")
        return False


def list_input_devices():
    """คืน [(index, ชื่อ)] ของไมค์ โดยเลือกเฉพาะ host API เดียวกับค่าเริ่มต้น กันชื่อซ้ำหลายรายการ"""
    if sd is None:
        return []
    try:
        default_in = sd.default.device[0]
        default_api = sd.query_devices(default_in)["hostapi"] if default_in is not None and default_in >= 0 else None
    except Exception:
        default_api = None
    devices = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and (default_api is None or d["hostapi"] == default_api):
                devices.append((i, d["name"]))
    except Exception:
        pass
    return devices


load_mic_config()


def block_rms(block):
    if block.size == 0:
        return 0.0
    samples = block.astype(np.float64)
    return float(np.sqrt(np.mean(samples * samples)))


def write_wav(path, samples, sample_rate=SAMPLE_RATE):
    pcm = np.asarray(samples, dtype=np.int16).reshape(-1)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def normalize_audio(samples, target_peak=NORMALIZE_TARGET_PEAK):
    pcm = np.asarray(samples, dtype=np.float32).reshape(-1)
    if pcm.size == 0:
        return pcm.astype(np.int16)

    pcm = pcm - float(np.mean(pcm))
    peak = float(np.max(np.abs(pcm)))
    if peak < 1.0:
        return np.zeros_like(pcm, dtype=np.int16)

    gain = min((32767.0 * target_peak) / peak, NORMALIZE_MAX_GAIN)
    return np.clip(pcm * gain, -32768, 32767).astype(np.int16)


def measure_noise_floor(seconds=CALIBRATION_SECONDS):
    if sd is None:
        return 0.0
    frames = max(int(seconds * SAMPLE_RATE), CHUNK_SIZE)
    try:
        buffer = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        device=MIC_CFG["device"])
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


def record_audio_vad(output_path, threshold=None, should_abort=None):
    """
    อัดเสียงจนกว่าจะเงียบตามเวลาที่กำหนด แล้วคืน path ของไฟล์ WAV
    คืน None ถ้าไม่พบเสียงพูดที่ใช้งานได้ หรือถูกสั่งยกเลิก (should_abort() เป็น True)
    """
    if sd is None:
        fail(f"ไม่พบไลบรารี sounddevice: {_SD_IMPORT_ERROR}")
        return None

    if threshold is None:
        threshold = compute_threshold(measure_noise_floor())

    log("🎙️  JARVIS กำลังฟัง... (พูดได้เลย เงียบเมื่อพูดจบ)")

    preroll_blocks = max(int(PREROLL_SECONDS * SAMPLE_RATE / CHUNK_SIZE), 1)
    preroll = deque(maxlen=preroll_blocks)
    speech_blocks = []

    state = {"speaking": False, "silence_since": None, "reason": "timeout",
             "voiced_blocks": 0, "last_voiced": 0}
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):
        block = np.asarray(indata, dtype=np.int16).reshape(-1).copy()
        rms = block_rms(block)
        now = time.monotonic()

        if rms > threshold:
            if not state["speaking"]:
                state["speaking"] = True
                speech_blocks.extend(preroll)
                preroll.clear()
                log("🗣️  ตรวจพบเสียงพูด... กำลังบันทึก", to_file=False)
            state["silence_since"] = None
            speech_blocks.append(block)
            state["voiced_blocks"] += 1
            state["last_voiced"] = len(speech_blocks)
            return

        if state["speaking"]:
            speech_blocks.append(block)
            if state["silence_since"] is None:
                state["silence_since"] = now
            elif now - state["silence_since"] >= MIC_CFG["silence"]:
                state["reason"] = "silence"
                stop_event.set()
                raise sd.CallbackStop()
        else:
            preroll.append(block)

    start = time.monotonic()
    try:
        with sd.InputStream(device=MIC_CFG["device"], samplerate=SAMPLE_RATE,
                            channels=1, dtype="int16",
                            blocksize=CHUNK_SIZE, callback=callback):
            while not stop_event.wait(0.05):
                elapsed = time.monotonic() - start
                if should_abort and should_abort():
                    state["reason"] = "aborted"
                    break
                if elapsed > MAX_RECORD_SECONDS:
                    state["reason"] = "max_length"
                    break
                if not state["speaking"] and elapsed > NO_SPEECH_TIMEOUT:
                    state["reason"] = "no_speech"
                    break
    except Exception as exc:
        fail(f"เกิดข้อผิดพลาดขณะอัดเสียง: {exc}")
        return None

    if state["reason"] in ("aborted", "no_speech"):
        return None
    if not state["speaking"] or not speech_blocks:
        return None

    block_seconds = CHUNK_SIZE / SAMPLE_RATE
    speech_seconds = state["voiced_blocks"] * block_seconds
    if speech_seconds < MIN_SPEECH_SECONDS:
        log(f"เสียงสั้นเกินไป ({speech_seconds:.2f} วิ) น่าจะเป็นเสียงรบกวน")
        return None

    tail_blocks = max(int(TAIL_SILENCE_SECONDS / block_seconds), 1)
    keep_until = min(state["last_voiced"] + tail_blocks, len(speech_blocks))
    recording = np.concatenate(speech_blocks[:keep_until])
    write_wav(output_path, normalize_audio(recording))
    return output_path


# ==========================================================================
# 6. แปลงเสียงเป็นข้อความ (STT)
# ==========================================================================
_session = requests.Session()


def _post_with_retry(**kwargs):
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
    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()

    auth = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

    response, error = _post_with_retry(
        url=OPENROUTER_STT_URL,
        headers={**auth, "Content-Type": "application/json"},
        json={
            "model": STT_MODEL,
            "input_audio": {"data": base64.b64encode(audio_bytes).decode("utf-8"),
                            "format": "wav"},
            "language": STT_LANGUAGE,
        },
    )
    text = _extract_transcript(response)
    if text:
        return text

    detail = f"HTTP {response.status_code}" if response is not None else error
    warn(f"STT แบบ JSON ไม่สำเร็จ ({detail}) กำลังลองแบบ multipart...")

    response, error = _post_with_retry(
        url=OPENROUTER_STT_URL,
        headers=auth,
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"model": STT_MODEL, "language": STT_LANGUAGE},
    )
    text = _extract_transcript(response)
    if text:
        return text

    if response is not None:
        fail(f"STT Error {response.status_code}: {response.text[:300]}")
    else:
        fail(f"STT Error: {error}")
    return ""


# ==========================================================================
# 7. สมอง AI: OpenRouter chat completions แบบ streaming (SSE)
# ==========================================================================
def stream_chat(messages, cancel):
    """yield ข้อความ (delta) ทีละชิ้นจาก AI จนจบ หรือจน cancel ถูกตั้งค่า"""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("ไม่พบ OPENROUTER_API_KEY (ตั้งค่าผ่าน environment variable หรือไฟล์ k.py)")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Tony Stark Interface",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": LLM_TEMPERATURE,
    }

    with _session.post(OPENROUTER_CHAT_URL, headers=headers, json=payload,
                       stream=True, timeout=(10, HTTP_TIMEOUT)) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        # ถอด UTF-8 เอง (SSE ไม่มี charset requests จะเดาเป็น latin-1 แล้วภาษาไทยเพี้ยน)
        for raw in resp.iter_lines(decode_unicode=False):
            if cancel.is_set():
                return
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue                      # คอมเมนต์ keep-alive เช่น ": OPENROUTER PROCESSING"
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if "error" in chunk:
                err = chunk["error"]
                raise RuntimeError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
            try:
                delta = chunk["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, TypeError, AttributeError):
                continue
            if delta:
                yield delta


_SENT_END = re.compile(r"(?<=[.!?。！？…])[\"')\]]*\s+|\n+")


def split_stream(buf):
    """
    ตัดข้อความที่กำลังสตรีมเป็นประโยคที่ 'จบแล้ว' ส่งไปอ่านได้ทันที
    คืน (รายการประโยค, ข้อความที่เหลือรอต่อ)
    ภาษาไทยมักไม่มีจุด จึงตัดที่ช่องว่างเมื่อข้อความยาวเกินเกณฑ์ด้วย
    """
    out = []
    while True:
        match = None
        for m in _SENT_END.finditer(buf):
            if m.start() >= 6:               # กันตัดชิ้นสั้นๆ เช่น "1."
                match = m
                break
        if not match:
            break
        chunk = buf[:match.start()].strip()
        buf = buf[match.end():]
        if chunk:
            out.append(chunk)

    while len(buf) > 110:
        cut = buf.rfind(" ", 40, 110)
        if cut == -1:
            break
        out.append(buf[:cut].strip())
        buf = buf[cut + 1:]
    return out, buf


# ==========================================================================
# 8. เสียงตอบกลับแบบ streaming: ประโยค -> TTS -> เล่น (pipeline 2 thread)
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


def _play_with_mci(path, stop):
    """เล่น mp3 ด้วย Windows MCI (ไม่ต้องติดตั้งอะไรเพิ่ม และหยุดกลางคันได้)"""
    if not IS_WINDOWS:
        return False
    winmm = ctypes.windll.winmm
    alias = "jarvisplayer"

    def mci(cmd):
        buf = ctypes.create_unicode_buffer(255)
        err = winmm.mciSendStringW(cmd, buf, 254, 0)
        return err, buf.value

    err, _ = mci(f'open "{path}" type mpegvideo alias {alias}')
    if err:
        return False
    try:
        mci(f"play {alias}")
        while not stop.is_set():
            _, mode = mci(f"status {alias} mode")
            if mode != "playing":
                break
            time.sleep(0.05)
    finally:
        mci(f"stop {alias}")
        mci(f"close {alias}")
    return True


def _play_with_ffplay(path, stop):
    ffplay = shutil.which("ffplay")
    if not ffplay:
        return False
    proc = subprocess.Popen(
        [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", path],
        stdin=subprocess.DEVNULL,
    )
    try:
        while proc.poll() is None:
            if stop.is_set():
                proc.terminate()
                break
            time.sleep(0.05)
    finally:
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
    return True


PLAYBACK_BACKENDS = (("mci", _play_with_mci), ("ffplay", _play_with_ffplay))


def play_audio(path, stop):
    errors = []
    for name, backend in PLAYBACK_BACKENDS:
        try:
            if backend(path, stop):
                return True
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    warn("เล่นเสียงไม่สำเร็จ | " + (" | ".join(errors) if errors else "ไม่พบตัวเล่นเสียง"))
    return False


class SpeechStreamer:
    """
    รับประโยคเข้ามาเรื่อยๆ -> สังเคราะห์เสียงล่วงหน้า -> เล่นต่อเนื่องตามลำดับ
    interrupt() ตัดเสียงที่กำลังเล่นและทิ้งคิวทั้งหมดทันที
    """

    def __init__(self, enabled=True, on_state=None):
        self.enabled = enabled
        self.on_state = on_state or (lambda s: None)
        self.gen = 0
        self.pending = 0
        self.lock = threading.Lock()
        self.idle = threading.Event()
        self.idle.set()
        self.stop_play = threading.Event()
        self.q_text = queue.Queue()
        self.q_play = queue.Queue()
        self.counter = 0
        if enabled:
            threading.Thread(target=self._synth_worker, daemon=True).start()
            threading.Thread(target=self._play_worker, daemon=True).start()

    def say(self, sentence, gen):
        if not self.enabled:
            return
        spoken = sanitize_for_tts(sentence)
        if not spoken or not re.search(r"\w", spoken):
            return
        with self.lock:
            if gen != self.gen:
                return
            self.pending += 1
            self.idle.clear()
        self.q_text.put((gen, spoken))

    def interrupt(self):
        with self.lock:
            self.gen += 1
        self.stop_play.set()

    def current_gen(self):
        with self.lock:
            return self.gen

    def wait_idle(self, should_stop=None, timeout=None):
        started = time.monotonic()
        while not self.idle.wait(0.1):
            if should_stop and should_stop():
                return False
            if timeout is not None and time.monotonic() - started > timeout:
                return False
        return True

    def _done(self):
        with self.lock:
            self.pending = max(0, self.pending - 1)
            if self.pending == 0:
                self.idle.set()

    def _synth_worker(self):
        _ensure_work_dir()
        while True:
            gen, text = self.q_text.get()
            if gen != self.current_gen():
                self._done()
                continue
            self.counter += 1
            path = os.path.join(WORK_DIR, f"tts_{os.getpid()}_{self.counter}.mp3")
            ok = generate_tts(text, path)
            if ok and gen == self.current_gen():
                self.q_play.put((gen, path))
            else:
                self._remove(path)
                self._done()

    def _play_worker(self):
        while True:
            gen, path = self.q_play.get()
            try:
                if gen == self.current_gen():
                    self.stop_play.clear()
                    self.on_state("SPEAKING")
                    play_audio(path, self.stop_play)
            finally:
                self._remove(path)
                self._done()

    @staticmethod
    def _remove(path):
        if KEEP_AUDIO:
            return
        try:
            os.remove(path)
        except OSError:
            pass


# ==========================================================================
# 9. ควบคุมเมาส์ด้วยท่ามือผ่านเว็บแคม (พอร์ตจาก gesture_v2.py)
# ==========================================================================
G_FRAME_MARGIN = 0.10
# ค่าตั้งต้นของ "การคลิก/ความนิ่งของเมาส์" (ปรับได้ที่ ⚙ ตั้งค่า)
CLICK_DEFAULTS = {
    "anchor": "palm",        # จุดที่ใช้พาเคอร์เซอร์: palm = กลางฝ่ามือ (นิ่งที่สุด) / knuckle = โคนนิ้วชี้ / tip = ปลายนิ้วชี้
    "still_px": 12,          # มือต้องขยับเกินกี่พิกเซล (ที่จอ 1080p) เคอร์เซอร์ถึงเริ่มเคลื่อน = กันมือสั่น
    "freeze_on": True,       # ล็อกเคอร์เซอร์เมื่อนิ้วโป้งกับนิ้วชี้เริ่มเข้าใกล้กัน (กำลังจะจีบ)
    "freeze_ratio": 0.65,    # เริ่มล็อกเมื่อระยะนิ้ว < เท่านี้ของความกว้างฝ่ามือ (ต้องมากกว่า pinch_start)
    "lookback_ms": 0,        # ตอนจีบ (0 = ปิด; ใช้คู่กับจุดอ้างอิง "ปลายนิ้วชี้") ให้คลิกที่ตำแหน่งเมื่อกี่มิลลิวินาทีก่อนหน้า (ก่อนนิ้วเริ่มขยับ)
    "drag_px": 40,           # หลังจีบ ต้องขยับมือเกินกี่พิกเซลถึงเริ่มลาก
    "pinch_start": 0.32,     # จีบเมื่อระยะนิ้ว < เท่านี้ของความกว้างฝ่ามือ (ปล่อยที่ +0.16)
}
CLICK_ANCHORS = {"palm": "กลางฝ่ามือ (นิ่งที่สุด)", "knuckle": "โคนนิ้วชี้",
                 "tip": "ปลายนิ้วชี้ (แบบเดิม)"}
G_DEAD_ZONE_PX = 15
G_MIN_CUTOFF, G_BETA, G_D_CUTOFF = 1.2, 0.025, 1.0
G_CURSOR_SPEED = 1.0
G_PINCH_START, G_PINCH_RELEASE = 0.32, 0.48
G_PINCH_DRAG_DEAD_ZONE = 28
G_PINCH_LOCK_CURSOR = False
G_CURSOR_RESUME_DELAY = 0.15
G_GESTURE_COOLDOWN = 0.8
G_SCROLL_THRESHOLD = 0.012
G_SCROLL_SENSITIVITY = 100
G_HAND_LOST_TIMEOUT = 0.3
SPEED_MIN, SPEED_MAX = 1.0, 4.0     # ตัวคูณความเร็วเคอร์เซอร์ (1.0 = แบบ gesture_v2 เดิม)
DEFAULT_MOUSE_SPEED = 1.8           # ค่าเริ่มต้น: มือขยับในกรอบเล็กลง ~45% ก็ครอบคลุมทั้งจอ
G_HELP_HOLD = 1.0          # โชว์ 10 นิ้วค้างกี่วินาทีถึงเปิด/ปิด Help
G_HELP_COOLDOWN = 2.5       # กันสลับรัว


class OneEuroFilter:
    def __init__(self, min_cutoff=1.2, beta=0.025, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.prev_x = None
        self.prev_dx = 0.0
        self.prev_time = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, now):
        if self.prev_x is None:
            self.prev_x, self.prev_time = x, now
            return x

        dt = max(now - self.prev_time, 0.001)
        dx = (x - self.prev_x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.prev_dx

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.prev_x

        self.prev_x, self.prev_dx, self.prev_time = x_hat, dx_hat, now
        return x_hat


def _lm_dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


class GestureMouse:
    """รันในเธรดของตัวเอง: อ่านเว็บแคม -> MediaPipe -> ควบคุมเมาส์ และเตรียมภาพ preview ให้ UI"""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._region = None      # (x, y, w, h) ของจอที่เมาส์จะถูกควบคุมในนั้น
        self.speed = DEFAULT_MOUSE_SPEED   # ตัวคูณความเร็วเคอร์เซอร์ (ปรับจากเมนูตั้งค่า)
        self.preview_size = (PREVIEW_W, PREVIEW_H)   # ขนาดภาพ preview (ขยายตามขนาด UI)
        self.cfg = dict(CLICK_DEFAULTS)    # การตั้งค่าการคลิก/ความนิ่ง (ปรับสดจากเมนูตั้งค่า)
        self._snap = {"seq": 0, "png": None, "gesture": "OFF", "action": "-",
                      "paused": False, "error": "", "help_seq": 0}
        self.help_seq = 0
        self.help_since = None
        self.last_help = 0.0

    # ----- ควบคุมจาก UI -----
    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_safe, daemon=True)
        self._thread.start()

    def set_click_cfg(self, new):
        """รับค่าตั้งค่าการคลิก/ความนิ่ง (กรองให้อยู่ในช่วงที่ปลอดภัย) แล้วใช้ทันที"""
        cfg = dict(self.cfg)
        for key, value in (new or {}).items():
            if key not in CLICK_DEFAULTS:
                continue
            try:
                if key == "anchor":
                    if value in CLICK_ANCHORS:
                        cfg[key] = value
                elif key == "freeze_on":
                    cfg[key] = bool(value)
                elif key == "still_px":
                    cfg[key] = max(0.0, min(80.0, float(value)))
                elif key == "freeze_ratio":
                    cfg[key] = max(0.35, min(1.2, float(value)))
                elif key == "lookback_ms":
                    cfg[key] = max(0.0, min(400.0, float(value)))
                elif key == "drag_px":
                    cfg[key] = max(5.0, min(200.0, float(value)))
                elif key == "pinch_start":
                    cfg[key] = max(0.15, min(0.55, float(value)))
            except (TypeError, ValueError):
                continue
        cfg["freeze_ratio"] = max(cfg["freeze_ratio"], cfg["pinch_start"] + 0.10)  # ต้องเริ่มล็อกก่อนถึงจุดจีบ
        self.cfg = cfg
        self.anchor = None                 # เริ่มติดตามใหม่ ไม่ให้เคอร์เซอร์กระโดด

    def _scale_k(self):
        """ค่า px ในเมนูคิดที่จอสูง 1080 พิกเซล — ปรับตามความสูงของจอที่ควบคุมอยู่ (4K = ×2)"""
        return max(0.5, self._region[3] / 1080.0) if self._region else 1.0

    def set_preview_size(self, w, h):
        self.preview_size = (max(160, int(w)), max(90, int(h)))

    def set_speed(self, speed):
        """ยิ่งสูง ยิ่งใช้พื้นที่ในภาพกล้องน้อยลงเพื่อเลื่อนเคอร์เซอร์ข้ามทั้งจอ"""
        self.speed = max(SPEED_MIN, min(SPEED_MAX, float(speed)))
        if self.running and hasattr(self, "fx"):
            self.anchor = None
            self.fx.reset()
            self.fy.reset()

    def active_margin(self):
        """ขอบของกรอบ 'พื้นที่ใช้งาน' ในภาพกล้อง (สัดส่วน 0-0.5) ที่สอดคล้องกับความเร็วปัจจุบัน"""
        return 0.5 - (1 - 2 * G_FRAME_MARGIN) / (2.0 * self.speed)

    def set_region(self, x, y, w, h):
        """กำหนดจอที่ท่ามือจะแมปไปถึง (พิกัดเดสก์ท็อปรวมทุกจอ)"""
        self._region = (int(x), int(y), int(w), int(h))
        if self.running and hasattr(self, "fx"):   # กันเคอร์เซอร์กระโดดจากค่าเดิมของจอก่อนหน้า
            self.anchor = None
            self.fx.reset()
            self.fy.reset()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None
        with self._lock:
            self._snap.update(gesture="OFF", action="-", paused=False, png=None,
                              seq=self._snap["seq"] + 1)

    def snapshot(self):
        with self._lock:
            return dict(self._snap)

    def _publish(self, **kw):
        with self._lock:
            self._snap.update(kw)

    # ----- เธรดหลัก -----
    def _run_safe(self):
        try:
            self._run()
        except Exception as exc:
            self._publish(error=f"{type(exc).__name__}: {exc}", gesture="ERROR", action="-")

    def _reset_state(self):
        self.fx = OneEuroFilter(G_MIN_CUTOFF, G_BETA, G_D_CUTOFF)
        self.fy = OneEuroFilter(G_MIN_CUTOFF, G_BETA, G_D_CUTOFF)
        self.paused = False
        self.pinching = False
        self.mouse_down = False
        self.drag_started = False
        self.last_gesture_time = 0.0
        self.last_scroll_y = None
        self.resume_cursor_time = 0.0
        self.pinch_start = (0.0, 0.0)
        self.pinch_start_cursor = (0, 0)
        self.anchor = None            # จุดผูกเคอร์เซอร์ (leash) ในพิกัดที่แมปแล้ว: ขยับเมื่อมือพ้นระยะนิ่ง
        self.cursor_f = None          # ตำแหน่งเคอร์เซอร์แบบทศนิยม (สะสมการขยับเล็กๆ ไม่ให้หาย)
        self.last_set = None          # ตำแหน่งสุดท้ายที่เราสั่งให้เคอร์เซอร์ไป
        self.cursor_hist = deque(maxlen=90)   # (เวลา, ตำแหน่ง) ใช้ย้อนตำแหน่งก่อนจีบ
        self.gesture = "WAITING"
        self.action = "READY"

    def _release_mouse(self):
        if self.mouse_down:
            self.pyautogui.mouseUp()
        self.mouse_down = False
        self.pinching = False
        self.drag_started = False

    def _run(self):
        import cv2
        import mediapipe as mp
        import pyautogui

        pyautogui.FAILSAFE = False      # มุมจอเข้าถึงได้โดยตั้งใจ; ปิดโปรแกรมด้วย ESC/ปุ่มปิด
        pyautogui.PAUSE = 0.005
        self.pyautogui = pyautogui
        if self._region is None:                     # ยังไม่เลือกจอ = จอหลัก
            sw, sh = pyautogui.size()
            self._region = (0, 0, sw, sh)
        self._reset_state()
        self._publish(error="")

        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands(max_num_hands=2, model_complexity=1,
                               min_detection_confidence=0.75,
                               min_tracking_confidence=0.75)

        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) if IS_WINDOWS \
            else cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if not cap.isOpened():
            self._publish(error="เปิดเว็บแคมไม่ได้", gesture="NO CAMERA")
            hands.close()
            return

        last_seen = time.time()
        frame_no = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                now = time.time()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                self.gesture, self.action = "NO HAND", "WAITING"

                found = results.multi_hand_landmarks or []
                for hand in found:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                if found:
                    last_seen = now

                if len(found) >= 2:
                    # สองมือในเฟรม: ไม่ควบคุมเมาส์ (กันคลิกพลาด) ใช้ตรวจท่า "10 นิ้ว" เปิด Help
                    self._two_hands(found[:2], now)
                elif len(found) == 1:
                    self.help_since = None
                    self._process_hand(found[0].landmark, now)
                elif now - last_seen > G_HAND_LOST_TIMEOUT:
                    self.help_since = None
                    self._release_mouse()
                    self.last_scroll_y = None
                    # กันเคอร์เซอร์กระโดดเมื่อมือกลับเข้าเฟรม
                    self.fx.reset()
                    self.fy.reset()
                    self.anchor = None

                frame_no += 1
                if frame_no % 2 == 0:
                    h, w, _ = frame.shape
                    m = self.active_margin()
                    cv2.rectangle(frame, (int(w * m), int(h * m)),
                                  (int(w * (1 - m)), int(h * (1 - m))), (255, 150, 0), 2)
                    small = cv2.resize(frame, self.preview_size)
                    okp, png = cv2.imencode(".png", small)
                    self._publish(png=base64.b64encode(png.tobytes()) if okp else None,
                                  seq=self._snap["seq"] + 1,
                                  gesture=self.gesture, action=self.action,
                                  paused=self.paused)
                else:
                    self._publish(gesture=self.gesture, action=self.action,
                                  paused=self.paused)
        finally:
            try:
                self._release_mouse()
            except Exception:
                pass
            cap.release()
            hands.close()

    # ----- ท่า "โชว์มือ 10 นิ้ว" = เปิด/ปิดหน้า Help -----
    @staticmethod
    def _is_open_hand(lm):
        """ครบ 5 นิ้ว: 4 นิ้วชูขึ้น + นิ้วโป้งกางออกจากฝ่ามือ"""
        palm = max(_lm_dist(lm[5], lm[17]), 0.001)
        four_up = (lm[8].y < lm[6].y and lm[12].y < lm[10].y
                   and lm[16].y < lm[14].y and lm[20].y < lm[18].y)
        thumb_out = _lm_dist(lm[4], lm[5]) / palm > 0.6
        return four_up and thumb_out

    def _two_hands(self, hands, now):
        self._release_mouse()
        self.last_scroll_y = None
        self.anchor = None             # กันเคอร์เซอร์กระโดดเมื่อกลับมาใช้มือเดียว
        self.gesture, self.action = "TWO HANDS", "WAITING"

        if now - self.last_help < G_HELP_COOLDOWN:
            self.help_since = None
            self.action = "HELP"
            return

        if all(self._is_open_hand(h.landmark) for h in hands):
            self.gesture = "10 FINGERS"
            if self.help_since is None:
                self.help_since = now
            progress = (now - self.help_since) / G_HELP_HOLD
            if progress >= 1.0:
                self.help_since = None
                self.last_help = now
                self.help_seq += 1
                self._publish(help_seq=self.help_seq)
                self.action = "HELP"
            else:
                self.action = f"HELP {int(progress * 100)}%"
        else:
            self.help_since = None

    # ----- ตรรกะท่ามือ (เหมือน gesture_v2.py) -----
    def _map_xy(self, px, py):
        # แมปรอบจุดกึ่งกลางภาพ คูณด้วย speed (speed = 1 เท่ากับสูตรเดิม (p - margin) / (1 - 2*margin))
        gain = self.speed / (1 - 2 * G_FRAME_MARGIN)
        x = 0.5 + (px - 0.5) * gain
        y = 0.5 + (py - 0.5) * gain
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        rx, ry, rw, rh = self._region
        return rx + x * rw, ry + y * rh

    def _pointer_xy(self, lm):
        """จุดบนมือที่ใช้พาเคอร์เซอร์ — กลางฝ่ามือ/โคนนิ้วไม่ขยับตามการงอนิ้วตอนจีบ ปลายนิ้วจะขยับ"""
        anchor = self.cfg["anchor"]
        if anchor == "tip":
            return lm[8].x, lm[8].y
        if anchor == "knuckle":
            return lm[5].x, lm[5].y
        pts = [lm[i] for i in (0, 5, 9, 13, 17)]            # ข้อมือ + โคนนิ้วทั้งสี่
        return sum(p.x for p in pts) / 5.0, sum(p.y for p in pts) / 5.0

    def _leash(self, fx, fy):
        """
        'สายจูง' กันมือสั่น: จุดผูกอยู่นิ่งตราบใดที่มือยังอยู่ในวงกลมรัศมี R รอบมัน
        เมื่อมือหลุดวง จุดผูกจะถูกลากตามให้มืออยู่ขอบวง → ขยับเคอร์เซอร์เฉพาะส่วนที่เกินวง (ไม่มีการกระโดด)
        คืนระยะที่ควรขยับ (mx, my)
        """
        radius = float(self.cfg["still_px"]) * self._scale_k()
        if self.anchor is None:
            self.anchor = (fx, fy)
            return 0.0, 0.0
        ax, ay = self.anchor
        dx, dy = fx - ax, fy - ay
        dist = math.hypot(dx, dy)
        if dist <= radius or dist == 0:
            return 0.0, 0.0
        s = (dist - radius) / dist
        mx, my = dx * s, dy * s
        self.anchor = (ax + mx, ay + my)
        return mx, my

    def _clamp_to_region(self, x, y):
        rx, ry, rw, rh = self._region
        return max(rx, min(rx + rw - 1, x)), max(ry, min(ry + rh - 1, y))

    def _move_cursor_by(self, mx, my, now):
        """ขยับเคอร์เซอร์ตามระยะ (mx, my) โดยสะสมเศษทศนิยม ไม่ให้การขยับช้าๆ หายไปตอนปัดเป็นพิกเซล"""
        if mx == 0 and my == 0:
            return
        pg = self.pyautogui
        real = pg.position()
        if (self.cursor_f is None or self.last_set is None
                or abs(real[0] - self.last_set[0]) > 3 or abs(real[1] - self.last_set[1]) > 3):
            self.cursor_f = [float(real[0]), float(real[1])]     # มีการขยับด้วยเมาส์จริง/ครั้งแรก: ซิงก์ใหม่
        cx, cy = self._clamp_to_region(self.cursor_f[0] + mx * G_CURSOR_SPEED,
                                       self.cursor_f[1] + my * G_CURSOR_SPEED)
        self.cursor_f = [cx, cy]
        pos = (int(round(cx)), int(round(cy)))
        if pos != self.last_set:
            pg.moveTo(*pos)
            self.last_set = pos

    def _restore_lookback(self, now):
        """ก่อนกดเมาส์ ย้อนไปตำแหน่งเมื่อ lookback_ms ที่แล้ว (ก่อนนิ้วเริ่มขยับเพื่อจีบ)"""
        lookback = self.cfg["lookback_ms"] / 1000.0
        if lookback <= 0 or not self.cursor_hist:
            return
        if now - self.cursor_hist[-1][0] > 0.35:       # ไม่ได้กำลังชี้อยู่ ไม่ต้องย้อน
            self.cursor_hist.clear()
            return
        target = None
        for t, pos in self.cursor_hist:                # เก่า → ใหม่ เอาอันล่าสุดที่เก่าพอ
            if t <= now - lookback:
                target = pos
        self.cursor_hist.clear()
        if target is None or self.last_set is None:
            return
        if math.hypot(target[0] - self.last_set[0], target[1] - self.last_set[1]) > 90 * self._scale_k():
            return                                      # ขยับไกลเกินไป = ตั้งใจเลื่อนอยู่ ไม่ย้อน
        self.pyautogui.moveTo(*target)
        self.last_set = target
        self.cursor_f = [float(target[0]), float(target[1])]

    def _process_hand(self, lm, now):
        pg = self.pyautogui
        index, thumb = lm[8], lm[4]
        i = lm[8].y < lm[6].y
        m = lm[12].y < lm[10].y
        r = lm[16].y < lm[14].y
        p = lm[20].y < lm[18].y

        palm_width = _lm_dist(lm[5], lm[17])
        pinch_ratio = _lm_dist(index, thumb) / max(palm_width, 0.001)

        px, py = self._pointer_xy(lm)
        tx, ty = self._map_xy(px, py)
        fx, fy = self.fx.filter(tx, now), self.fy.filter(ty, now)

        pinch_start = self.cfg["pinch_start"]
        is_pinch = pinch_ratio < (pinch_start + 0.16 if self.pinching else pinch_start)

        if is_pinch:
            self.gesture = "PINCH"
            if not self.pinching:
                self._restore_lookback(now)            # คลิกที่ตำแหน่งก่อนนิ้วเริ่มขยับ
                self.pinching = True
                self.drag_started = False
                self.pinch_start = (fx, fy)
                self.pinch_start_cursor = pg.position()
                pg.mouseDown()
                self.mouse_down = True

            dx, dy = fx - self.pinch_start[0], fy - self.pinch_start[1]
            dist = math.hypot(dx, dy)
            drag_px = self.cfg["drag_px"] * self._scale_k()
            if dist > drag_px:
                self.drag_started = True
            if self.drag_started and dist > 0:
                # ลากเริ่มจากศูนย์ (หักระยะนิ่งออก) เคอร์เซอร์จึงไม่กระโดดตอนเริ่มลาก
                s = max(0.0, (dist - drag_px) / dist)
                nx, ny = self._clamp_to_region(
                    self.pinch_start_cursor[0] + dx * s * G_CURSOR_SPEED,
                    self.pinch_start_cursor[1] + dy * s * G_CURSOR_SPEED)
                pos = (int(round(nx)), int(round(ny)))
                if pos != self.last_set:
                    pg.moveTo(*pos)
                    self.last_set = pos
                self.cursor_f = [nx, ny]
                self.action = "DRAGGING"
            else:
                self.action = "PINCH - HOLD STILL"     # จีบอยู่ แต่ยังไม่ถึงระยะลาก: ตำแหน่งนิ่ง
            self.anchor = (fx, fy)                     # ปล่อยแล้วต่อจากตรงนี้ ไม่กระโดด
            return

        if self.pinching:                      # ปล่อย pinch = คลิก / วาง
            self._release_mouse()
            self.resume_cursor_time = now + G_CURSOR_RESUME_DELAY
            self.anchor = (fx, fy)
            self.cursor_f = None
            self.cursor_hist.clear()
            self.action = "CLICK / RELEASE"
            return

        if not any((i, m, r, p)):              # กำมือ = pause
            self.paused = True
            self.gesture, self.action = "FIST", "PAUSED"
            self._release_mouse()
            self.last_scroll_y = None

        elif all((i, m, r, p)):                # แบมือ = resume
            self.paused = False
            self.gesture, self.action = "OPEN HAND", "RESUME"
            self.last_scroll_y = None

        elif i and m and r and not p:          # 3 นิ้ว = คลิกขวา
            self.gesture, self.action = "THREE FINGERS", "RIGHT CLICK"
            if not self.paused and now - self.last_gesture_time > G_GESTURE_COOLDOWN:
                pg.rightClick()
                self.last_gesture_time = now
            self.last_scroll_y = None

        elif i and m and not r and not p:      # 2 นิ้ว = เลื่อนหน้า
            self.gesture = "SCROLL MODE"
            if not self.paused:
                center_y = (index.y + lm[12].y) / 2
                if self.last_scroll_y is not None:
                    delta = self.last_scroll_y - center_y
                    if abs(delta) > G_SCROLL_THRESHOLD:
                        amount = int(delta * G_SCROLL_SENSITIVITY)
                        if amount:
                            pg.scroll(amount)
                            self.action = "SCROLL"
                self.last_scroll_y = center_y
            else:
                self.action = "PAUSED"

        elif not i and not m and not r and p:  # นิ้วก้อยนิ้วเดียว = Enter
            self.gesture, self.action = "PINKY UP", "ENTER"
            if not self.paused and now - self.last_gesture_time > G_GESTURE_COOLDOWN:
                pg.press("enter")
                self.last_gesture_time = now
            self.last_scroll_y = None

        elif i and not m and not r and not p:  # นิ้วชี้ = เลื่อนเคอร์เซอร์
            self.gesture = "MOUSE CONTROL"
            self.last_scroll_y = None
            if not self.paused and now >= self.resume_cursor_time:
                if self.cfg["freeze_on"] and pinch_ratio < self.cfg["freeze_ratio"]:
                    # นิ้วกำลังเข้าใกล้กัน = กำลังจะจีบ: ตรึงเคอร์เซอร์ไว้ ไม่ตามการขยับของนิ้ว
                    self.anchor = (fx, fy)
                    self.action = "LOCK (กำลังจะจีบ)"
                else:
                    mx, my = self._leash(fx, fy)
                    self._move_cursor_by(mx, my, now)
                    if self.last_set is not None:
                        self.cursor_hist.append((now, self.last_set))
                    self.action = "MOVE CURSOR" if (mx or my) else "STILL"

        else:
            self.gesture, self.action = "OTHER GESTURE", "WAITING"
            self.last_scroll_y = None


# ==========================================================================
# 9.5 จอภาพหลายจอ (เช่น เสียบ XREAL One เป็นจอที่สอง)
# ==========================================================================
UI_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_config.json")


def list_monitors():
    """
    คืนรายการจอทั้งหมด [{x, y, w, h, primary, name}] (พิกัดแบบเดสก์ท็อปรวม จอซ้ายของหลักอาจติดลบ)
    เรียงให้จอหลักมาก่อน นอก Windows หรือถ้าถามระบบไม่ได้ จะคืนค่าจากขนาดหน้าจอที่ Tk เห็น
    """
    monitors = []
    if IS_WINDOWS:
        try:
            from ctypes import wintypes

            class MONITORINFOEXW(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                            ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                            ("szDevice", wintypes.WCHAR * 32)]

            proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
            user32 = ctypes.windll.user32
            user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MONITORINFOEXW)]
            user32.EnumDisplayMonitors.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   proc_type, wintypes.LPARAM]

            def callback(hmon, hdc, rect, lparam):
                info = MONITORINFOEXW()
                info.cbSize = ctypes.sizeof(MONITORINFOEXW)
                if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
                    r = info.rcMonitor
                    monitors.append({"x": r.left, "y": r.top, "w": r.right - r.left,
                                     "h": r.bottom - r.top, "primary": bool(info.dwFlags & 1),
                                     "name": info.szDevice})
                return True

            user32.EnumDisplayMonitors(None, None, proc_type(callback), 0)
        except Exception as exc:
            warn(f"อ่านรายการจอภาพไม่สำเร็จ: {exc}")
            monitors = []
    monitors.sort(key=lambda m: (not m["primary"], m["x"], m["y"]))
    return monitors


def monitor_label(index, m):
    tag = " (หลัก)" if m["primary"] else ""
    return f"จอ {index + 1}{tag} — {m['w']}×{m['h']} @ {m['x']},{m['y']}"


def load_ui_config():
    try:
        with open(UI_CONFIG_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_ui_config(data):
    try:
        with open(UI_CONFIG_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        warn(f"บันทึกการตั้งค่าหน้าจอไม่สำเร็จ: {exc}")


# ==========================================================================
# 9.7 สั่งเปิดโปรแกรม (เฉพาะรายการที่อนุญาตไว้ล่วงหน้า — ไม่รันคำสั่งอิสระใดๆ)
# ==========================================================================
APPS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps.json")

# id -> ชื่อที่แสดง / คำเรียก (ไทย+อังกฤษ) / สิ่งที่จะเปิด (ชื่อโปรแกรม, URL หรือ ms-settings:)
APP_REGISTRY = {
    "browser": {"name": "เว็บเบราว์เซอร์",
                "aliases": ["เว็บเบราว์เซอร์", "เบราว์เซอร์", "เว็บ", "อินเทอร์เน็ต", "browser",
                            "web browser", "internet"],
                "target": "https://www.google.com"},
    "chrome": {"name": "Chrome", "aliases": ["chrome", "โครม", "กูเกิลโครม"], "target": "chrome"},
    "edge": {"name": "Microsoft Edge", "aliases": ["edge", "เอดจ์", "ไมโครซอฟต์เอดจ์"],
             "target": "msedge"},
    "firefox": {"name": "Firefox", "aliases": ["firefox", "ไฟร์ฟอกซ์"], "target": "firefox"},
    "paint": {"name": "Paint", "aliases": ["paint", "เพนต์", "เพ้นท์", "เพนท์", "วาดรูป",
                                           "โปรแกรมวาดรูป", "mspaint"], "target": "mspaint"},
    "notepad": {"name": "Notepad", "aliases": ["notepad", "โน้ตแพด", "โน๊ตแพด", "สมุดจด"],
                "target": "notepad"},
    "calculator": {"name": "เครื่องคิดเลข", "aliases": ["เครื่องคิดเลข", "calculator", "calc"],
                   "target": "calc"},
    "explorer": {"name": "File Explorer",
                 "aliases": ["file explorer", "explorer", "เอ็กซ์พลอเรอร์", "จัดการไฟล์", "โฟลเดอร์",
                             "คอมพิวเตอร์ของฉัน", "my computer"], "target": "explorer"},
    "word": {"name": "Word", "aliases": ["word", "เวิร์ด", "ไมโครซอฟต์เวิร์ด"], "target": "winword"},
    "excel": {"name": "Excel", "aliases": ["excel", "เอ็กเซล"], "target": "excel"},
    "powerpoint": {"name": "PowerPoint", "aliases": ["powerpoint", "พาวเวอร์พอยท์", "พาวเวอร์พอยต์"],
                   "target": "powerpnt"},
    "settings": {"name": "ตั้งค่า Windows", "aliases": ["การตั้งค่า", "ตั้งค่าวินโดว์", "settings",
                                                         "windows settings"], "target": "ms-settings:"},
    "taskmgr": {"name": "Task Manager", "aliases": ["task manager", "taskmgr", "ตัวจัดการงาน"],
                "target": "taskmgr"},
    "snipping": {"name": "Snipping Tool", "aliases": ["snipping tool", "snippingtool", "แคปหน้าจอ",
                                                        "จับภาพหน้าจอ"], "target": "snippingtool"},
    "youtube": {"name": "YouTube", "aliases": ["youtube", "ยูทูป", "ยูทูบ"],
                "target": "https://www.youtube.com"},
    "google": {"name": "Google", "aliases": ["google", "กูเกิล"], "target": "https://www.google.com"},
    "gmail": {"name": "Gmail", "aliases": ["gmail", "จีเมล", "อีเมล"], "target": "https://mail.google.com"},
}


def load_custom_apps():
    """เพิ่มโปรแกรมของตัวเองได้ใน apps.json: {"id": {"name": "...", "aliases": [...], "target": "C:\\\\...exe"}}"""
    try:
        with open(APPS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    for app_id, spec in data.items():
        if (isinstance(spec, dict) and isinstance(spec.get("target"), str)
                and isinstance(spec.get("aliases", []), list)):
            APP_REGISTRY[str(app_id)] = {
                "name": str(spec.get("name", app_id)),
                "aliases": [str(a) for a in spec.get("aliases", [])] + [str(spec.get("name", app_id))],
                "target": spec["target"],
            }


load_custom_apps()


def _squash(text):
    return re.sub(r"\s+", "", text.lower())


_OPEN_VERB_RE = re.compile(
    r"^\s*(?:ช่วย|กรุณา|ขอ|รบกวน|please)?\s*(?:เปิด|รัน|เริ่ม|open|launch|start|run)\s*"
    r"(?:โปรแกรม|แอปพลิเคชัน|แอปฯ?|แอป|application|program|app)?\s*(.+?)\s*$",
    re.IGNORECASE,
)
_FILLER_TAIL_RE = re.compile(r"(?:ให้หน่อย|ให้ผมหน่อย|ให้ฉันหน่อย|ให้ที|หน่อย|ให้|ที|ด้วย|ครับ|ค่ะ|คะ|นะ|please)\s*$",
                             re.IGNORECASE)


def find_app(text):
    """หาโปรแกรมจากคำเรียกที่ยาวที่สุดที่ปรากฏในข้อความ คืน app_id หรือ None"""
    target = _squash(text)
    best, best_len = None, 0
    for app_id, spec in APP_REGISTRY.items():
        for alias in spec["aliases"]:
            a = _squash(alias)
            if len(a) >= 3 and a in target and len(a) > best_len:
                best, best_len = app_id, len(a)
    return best


def parse_open_command(text):
    """ตรวจว่าเป็นคำสั่ง 'เปิด <โปรแกรม>' ชัดเจนหรือไม่ คืน app_id หรือ None (ไม่ใช่ -> ส่งให้ AI ตามปกติ)"""
    m = _OPEN_VERB_RE.match(text or "")
    if not m:
        return None
    target = m.group(1)
    previous = None
    while previous != target:                 # ตัดคำลงท้าย เช่น "ให้หน่อยครับ"
        previous = target
        target = _FILLER_TAIL_RE.sub("", target).strip()
    if not target or len(target) > 40:
        return None
    return find_app(target)


def launch_app(app_id):
    """เปิดโปรแกรมในรายการที่อนุญาต คืน (สำเร็จไหม, ข้อความตอบผู้ใช้)"""
    spec = APP_REGISTRY.get(app_id)
    if not spec:
        return False, f"ไม่รู้จักโปรแกรม '{app_id}' ครับ"
    name, target = spec["name"], spec["target"]
    try:
        if target.startswith(("http://", "https://")):
            import webbrowser
            if not webbrowser.open(target):
                raise OSError("เปิดเบราว์เซอร์ไม่ได้")
        elif IS_WINDOWS:
            os.startfile(target)              # ใช้ ShellExecute จึงหา App Paths / ms-settings ได้
        else:
            subprocess.Popen([target], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    except Exception as exc:
        fail(f"เปิด {name} ไม่สำเร็จ: {exc}")
        return False, f"เปิด {name} ไม่ได้ครับ อาจยังไม่ได้ติดตั้งโปรแกรมนี้"
    log(f"🚀 เปิดโปรแกรม: {name}")
    return True, f"เปิด {name} ให้แล้วครับ"


def apps_hint():
    """ข้อความต่อท้าย system prompt สอนให้ AI ขอเปิดโปรแกรมด้วยแท็ก [[OPEN:id]]"""
    ids = ", ".join(f"{k} ({v['name']})" for k, v in APP_REGISTRY.items())
    return ("\n\nความสามารถพิเศษ: เปิดโปรแกรมให้ผู้ใช้ได้ เมื่อผู้ใช้ขอให้เปิดโปรแกรมหรือเว็บในรายการ "
            "ให้ใส่แท็ก [[OPEN:id]] ในคำตอบ (หนึ่งแท็กต่อหนึ่งโปรแกรม) แล้วตอบสั้นๆ ว่าเปิดให้แล้ว "
            "ห้ามใส่แท็กหากผู้ใช้ไม่ได้ขอให้เปิดโดยตรง และห้ามใช้ id นอกรายการ "
            f"รายการที่เปิดได้: {ids}")


class OpenTagFilter:
    """กรองแท็ก [[OPEN:id]] ออกจากข้อความที่กำลังสตรีม (แท็กอาจถูกตัดครึ่งข้ามหลาย token) เก็บคำสั่งไว้ให้ไปเปิด"""

    def __init__(self):
        self.pending = ""
        self.commands = []

    def feed(self, delta):
        text, out = self.pending + delta, ""
        self.pending = ""
        while text:
            i = text.find("[[")
            if i == -1:
                if text.endswith("["):             # อาจเป็นต้นแท็ก รอ token ถัดไป
                    self.pending, text = "[", text[:-1]
                out += text
                break
            out += text[:i]
            rest = text[i:]
            j = rest.find("]]")
            if j == -1:
                if len(rest) > 80:                 # ยาวเกินกว่าจะเป็นแท็ก ถือเป็นข้อความธรรมดา
                    out += "[["
                    text = rest[2:]
                    continue
                self.pending = rest                # แท็กยังมาไม่ครบ
                break
            tag = rest[2:j]
            if tag.strip().upper().startswith("OPEN:"):
                self.commands.append(tag.split(":", 1)[1].strip().lower())
            else:
                out += rest[:j + 2]
            text = rest[j + 2:]
        return out

    def flush(self):
        out, self.pending = self.pending, ""
        return out

    def take_commands(self):
        cmds, self.commands = self.commands, []
        return cmds


# ==========================================================================
# 10. ตัวควบคุมการสนทนา (ประวัติแชท + สตรีมคำตอบ + เสียงตอบกลับ)
# ==========================================================================
class Assistant:
    def __init__(self, ui_q, speech):
        self.ui_q = ui_q
        self.speech = speech
        self.history = []                  # เฉพาะ user/assistant (system เติมตอนส่ง)
        self.cancel = threading.Event()
        self.turn_lock = threading.Lock()

    def _ui(self, kind, payload=None):
        self.ui_q.put((kind, payload))

    def interrupt(self):
        self.cancel.set()
        self.speech.interrupt()

    def _run_open_tags(self, app_ids):
        """เปิดโปรแกรมตามแท็กที่ AI ส่งมา (เปิดได้เฉพาะรายการที่อนุญาตเท่านั้น)"""
        for app_id in app_ids:
            ok, msg = launch_app(app_id) if app_id in APP_REGISTRY else (False, f"ไม่รู้จักโปรแกรม '{app_id}'")
            self._ui("sys" if ok else "error", ("🚀 " if ok else "") + msg)

    def _open_app_directly(self, app_id, user_text):
        ok, msg = launch_app(app_id)
        gen = self.speech.current_gen()
        self._ui("ai_start")
        self._ui("ai_delta", msg)
        self._ui("ai_end")
        self.speech.say(msg, gen)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": msg})
        self.history = self.history[-HISTORY_MESSAGES:]

    def handle_text(self, user_text):
        """ส่งข้อความหนึ่งรอบ: สตรีมคำตอบขึ้นจอ พร้อมอ่านออกเสียงทีละประโยค"""
        user_text = (user_text or "").strip()
        if not user_text:
            return

        self.interrupt()                    # ข้อความใหม่ตัดรอบเก่าทันที
        with self.turn_lock:
            self.cancel.clear()
            self._ui("user", user_text)

            normalized = user_text.lower()
            if any(phrase in normalized for phrase in EXIT_PHRASES):
                gen = self.speech.current_gen()
                self._ui("sys", "รับทราบครับ กำลังปิดระบบ")
                self.speech.say("รับทราบครับ แล้วพบกันใหม่", gen)
                self.speech.wait_idle(timeout=10)
                self._ui("quit")
                return

            # คำสั่ง "เปิด <โปรแกรม>" ที่ชัดเจน: ทำทันทีโดยไม่ต้องรอ AI (เร็วและใช้ได้แม้ไม่มี API key)
            app_id = parse_open_command(user_text)
            if app_id:
                self._open_app_directly(app_id, user_text)
                return

            if not OPENROUTER_API_KEY:
                # ไม่มีคีย์: ไม่เรียก AI (ส่วนอื่นของโปรแกรมใช้ได้ตามปกติ) แนะนำให้ตั้งค่าแทนการขึ้น error ดิบ
                self._ui("error", "ยังไม่ได้ตั้งค่า API key จึงคุยกับ AI ไม่ได้ — ตั้งค่าที่เมนู ⚙ ตั้งค่า "
                                  "(สั่ง \"เปิด <โปรแกรม>\" และควบคุมเมาส์ด้วยมือใช้ได้โดยไม่ต้องมีคีย์)")
                self._ui("need_key")
                self._ui("state", "STANDBY")
                return

            self.history.append({"role": "user", "content": user_text})
            self.history = self.history[-HISTORY_MESSAGES:]
            messages = ([{"role": "system", "content": SYSTEM_PROMPT + apps_hint()}]
                        + self.history)

            gen = self.speech.current_gen()
            self._ui("state", "THINKING")
            self._ui("ai_start")

            full, buf = "", ""
            started = time.monotonic()
            first_token = None
            tags = OpenTagFilter()
            try:
                for raw_delta in stream_chat(messages, self.cancel):
                    if self.cancel.is_set():
                        break
                    if first_token is None:
                        first_token = time.monotonic() - started
                        self._ui("state", "RESPONDING")
                    delta = tags.feed(raw_delta)          # ตัดแท็ก [[OPEN:id]] ออกจากข้อความ
                    self._run_open_tags(tags.take_commands())
                    if not delta:
                        continue
                    full += delta
                    self._ui("ai_delta", delta)
                    buf += delta
                    sentences, buf = split_stream(buf)
                    for s in sentences:
                        self.speech.say(s, gen)
                tail = tags.flush()
                if tail and not self.cancel.is_set():
                    full += tail
                    self._ui("ai_delta", tail)
                    buf += tail
                if not self.cancel.is_set() and buf.strip():
                    self.speech.say(buf.strip(), gen)
            except Exception as exc:
                self._ui("error", f"ติดต่อ AI ไม่สำเร็จ: {exc}")
                fail(f"LLM error: {exc}")
            finally:
                self._ui("ai_end")

            if full.strip():
                self.history.append({"role": "assistant", "content": full})
            elif self.history and self.history[-1]["role"] == "user":
                self.history.pop()          # รอบที่ไม่ได้คำตอบ ไม่ต้องจำ

            if first_token is not None:
                log(f"⏱️  token แรกใน {first_token:.1f} วิ, ทั้งหมด {time.monotonic() - started:.1f} วิ")


# ==========================================================================
# 11. อินเทอร์เฟซ tkinter: fullscreen พื้นหลังโปร่งใส
# ==========================================================================
KEY_COLOR = "#010203"       # สีนี้จะโปร่งใสทะลุเห็นเดสก์ท็อป (Windows)
C_PANEL = "#06131c"
C_PANEL2 = "#0a1f2c"
C_CYAN = "#19e3ff"
C_TEXT = "#d9f7ff"
C_DIM = "#5fa9bd"
C_USER = "#ffd166"
C_ERR = "#ff6b6b"
C_OK = "#5dffb0"
FONT_FAMILY = "Leelawadee UI" if IS_WINDOWS else "Noto Sans Thai"

STATE_COLORS = {
    "STANDBY": C_DIM, "LISTENING": C_OK, "CALIBRATING": C_USER,
    "TRANSCRIBING": C_USER, "THINKING": C_USER, "RESPONDING": C_CYAN,
    "SPEAKING": C_CYAN,
}


GESTURES_SVG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gestures.svg")

HELP_LINES = (
    ("☝  ชูนิ้วชี้", "เลื่อนเคอร์เซอร์"),
    ("🤏  โป้ง + ชี้แตะกัน", "คลิก / ค้างไว้แล้วขยับ = ลาก"),
    ("🤟  ชู 3 นิ้ว (ชี้ กลาง นาง)", "คลิกขวา"),
    ("✌  ชู 2 นิ้ว (ชี้ กลาง)", "ขยับมือขึ้น/ลง = เลื่อนหน้า"),
    ("🤙  ชูนิ้วก้อยนิ้วเดียว", "กด Enter"),
    ("✊  กำมือ", "หยุดควบคุมชั่วคราว (Pause)"),
    ("🖐  แบมือ", "ควบคุมต่อ (Resume)"),
)


def _rounded_rect(canvas, x1, y1, x2, y2, r, **kw):
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2,
           x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


FONT_BASE = {            # ชื่อ -> (ขนาดฐาน, น้ำหนัก, family)
    "title": (18, "bold", None),
    "body": (13, "normal", None),
    "bold": (13, "bold", None),
    "small": (10, "normal", None),
    "mono": (10, "normal", "Consolas"),
}
FONT_SCALE_MIN, FONT_SCALE_MAX = 0.8, 4.0      # ปรับได้ถึง 400% (จอ 4K/จอใหญ่/แว่นระยะไกล)


def draw_svg_on_canvas(canvas, svg_path, scale, text_scale=1.0):
    """
    วาดไฟล์ SVG ลง tkinter Canvas (รองรับเฉพาะ rect / line / circle / text ที่ใช้ใน gestures.svg)
    คืน (กว้าง, สูง) ของภาพที่วาด  ธีมสีถูกแทนที่ให้เข้ากับพื้นหลังมืดของ interface
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(svg_path).getroot()
    vb = [float(v) for v in root.get("viewBox", "0 0 680 480").split()]
    s = scale

    def num(el, name, default=0.0):
        try:
            return float(el.get(name, default))
        except (TypeError, ValueError):
            return default

    def color(value, default):
        if not value or value == "none" or value.startswith(("var(", "context-")):
            return default
        return value

    def walk(el):
        tag = el.tag.split("}")[-1]
        cls = el.get("class", "")
        if tag in ("defs", "title", "desc", "style"):
            return
        if tag == "rect":
            x, y, w, h = num(el, "x"), num(el, "y"), num(el, "width"), num(el, "height")
            r = num(el, "rx") * s
            fill = C_PANEL2 if "box" in cls else ""
            _rounded_rect(canvas, (x - vb[0]) * s, (y - vb[1]) * s,
                          (x + w - vb[0]) * s, (y + h - vb[1]) * s, r,
                          fill=fill, outline=C_DIM, width=1)
        elif tag == "line":
            width = 2 if "arr" in cls else num(el, "stroke-width", 1)
            canvas.create_line((num(el, "x1") - vb[0]) * s, (num(el, "y1") - vb[1]) * s,
                               (num(el, "x2") - vb[0]) * s, (num(el, "y2") - vb[1]) * s,
                               fill=color(el.get("stroke"), C_DIM), width=max(1, width * s),
                               capstyle="round",
                               arrow="last" if el.get("marker-end") else None,
                               arrowshape=(10 * s, 12 * s, 4 * s))
        elif tag == "circle":
            cx, cy, r = (num(el, "cx") - vb[0]) * s, (num(el, "cy") - vb[1]) * s, num(el, "r") * s
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="",
                               outline=color(el.get("stroke"), C_DIM),
                               width=max(1, num(el, "stroke-width", 1) * s))
        elif tag == "text":
            title = "th" in cls
            font = tkfont.Font(family=FONT_FAMILY, size=-max(9, int((15 if title else 12) * s * text_scale)),
                               weight="bold" if title else "normal")
            if not hasattr(canvas, "_fonts"):
                canvas._fonts = []
            canvas._fonts.append(font)          # กัน Font ถูกเก็บกวาดจนตัวอักษรหาย
            anchor = {"middle": "s", "end": "se"}.get(el.get("text-anchor"), "sw")
            canvas.create_text((num(el, "x") - vb[0]) * s, (num(el, "y") - vb[1]) * s,
                               text=(el.text or "").strip(), font=font, anchor=anchor,
                               fill=C_CYAN if title else C_TEXT)
        for child in el:
            walk(child)

    walk(root)
    return int(vb[2] * s), int(vb[3] * s)


DEFAULT_MODELS = (      # รายการตัวอย่าง (กด "โหลดรายการ" ในเมนูตั้งค่าเพื่อดูโมเดลที่มีจริงทั้งหมด)
    "openai/gpt-4o-mini", "openai/gpt-4o", "openai/gpt-4.1-mini", "openai/gpt-4.1",
    "anthropic/claude-3.5-haiku", "anthropic/claude-3.7-sonnet", "anthropic/claude-sonnet-4",
    "google/gemini-2.0-flash-001", "google/gemini-2.5-flash", "google/gemini-2.5-pro",
    "deepseek/deepseek-chat", "meta-llama/llama-3.3-70b-instruct", "qwen/qwen-2.5-72b-instruct",
)


def set_llm_model(model):
    """เปลี่ยนโมเดลแชทที่ใช้ในการสนทนาถัดไปทันที"""
    global LLM_MODEL
    model = (model or "").strip()
    if model:
        LLM_MODEL = model


CLAP_MIN_PEAK = _env_float("JARVIS_CLAP_MIN", 2500.0)   # ยอดคลื่นเสียงขั้นต่ำ (int16) ที่นับเป็นเสียงปรบมือ
CLAP_RATIO = _env_float("JARVIS_CLAP_RATIO", 8.0)        # ต้องดังกว่าเสียงรบกวนพื้นหลังกี่เท่า


class ClapDetector:
    """
    ตรวจเสียงปรบมือ 1 ครั้งจากไมค์ — เปิดไว้เฉพาะตอนหน้า Help แสดงอยู่
    หลักการ: เสียงปรบมือคือ 'ยอดแหลมสั้นๆ' ดังขึ้นทันทีแล้วหายเงียบภายใน ~0.25 วินาที
    เสียงพูดหรือเสียงต่อเนื่องจะดังค้างนานกว่านั้น จึงไม่ถูกนับ
    """

    BLOCK = 512                  # ~32 มิลลิวินาทีที่ 16 kHz
    MAX_CLAP_SECONDS = 0.25
    QUIET_BLOCKS = 3             # ต้องเงียบต่อเนื่องหลังยอดเสียงกี่บล็อก ถึงยืนยันว่าเป็นการปรบมือ
    COOLDOWN = 1.0

    def __init__(self, on_clap):
        self.on_clap = on_clap
        self.stream = None
        self._reset()

    def _reset(self):
        self.noise = 200.0       # ระดับเสียงรบกวนพื้นหลัง (rms เฉลี่ยแบบเคลื่อนที่)
        self.prev_peak = 0.0
        self.onset = None        # (เวลาเริ่ม, ยอดสูงสุด)
        self.quiet = 0
        self.cooldown_until = 0.0

    def start(self):
        """คืน True ถ้าเปิดไมค์สำเร็จ"""
        if sd is None or self.stream is not None:
            return self.stream is not None
        self._reset()
        try:
            self.stream = sd.InputStream(device=MIC_CFG["device"], samplerate=SAMPLE_RATE,
                                         channels=1, dtype="int16", blocksize=self.BLOCK,
                                         callback=self._callback)
            self.stream.start()
            return True
        except Exception as exc:
            self.stream = None
            warn(f"เปิดไมค์ตรวจเสียงปรบมือไม่ได้: {exc}")
            return False

    def stop(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _callback(self, indata, frames, time_info, status):
        block = np.asarray(indata, dtype=np.int16).reshape(-1)
        if block.size == 0:
            return
        peak = float(np.max(np.abs(block.astype(np.int32))))
        self.process(peak, block_rms(block), time.monotonic())

    def process(self, peak, rms, now):
        if now < self.cooldown_until:
            self.prev_peak = peak
            return
        threshold = max(CLAP_MIN_PEAK, self.noise * CLAP_RATIO)

        if self.onset is None:
            if peak >= threshold and self.prev_peak < threshold * 0.5:
                self.onset, self.quiet = (now, peak), 0       # เจอยอดแหลมที่พุ่งขึ้นจากความเงียบ
            else:
                self.noise = 0.95 * self.noise + 0.05 * max(rms, 50.0)
        else:
            t0, pk = self.onset
            pk = max(pk, peak)
            self.onset = (t0, pk)
            if peak < pk * 0.3:                               # ยอดเสียงตกลงแล้ว
                self.quiet += 1
                if self.quiet >= self.QUIET_BLOCKS:           # เงียบต่อเนื่อง = ปรบมือจริง
                    self.onset = None
                    self.cooldown_until = now + self.COOLDOWN
                    try:
                        self.on_clap()
                    except Exception:
                        pass
            elif self.quiet > 0 or now - t0 > self.MAX_CLAP_SECONDS:
                self.onset = None                             # ดังต่อเนื่อง = เสียงพูด/เสียงอื่น ไม่ใช่ปรบมือ
        self.prev_peak = peak


class MicMenu:
    """เมนูตั้งค่าไมค์: เลือกอุปกรณ์ / ความไว / เวลาเงียบ พร้อมมิเตอร์ระดับเสียงสดและปุ่มวัดเสียงห้อง"""

    METER_MAX = 4000.0

    def __init__(self, ui):
        self.ui = ui
        self.stream = None
        self.level = 0.0
        self.calib = None            # list ของ rms ระหว่างวัดเสียงห้อง
        self.calib_end = 0.0
        self.closed = False

        self.devices = list_input_devices()
        self.default_label = "ค่าเริ่มต้นของระบบ"
        self.labels = {self.default_label: None}
        for idx, name in self.devices:
            self.labels[f"{idx}: {name}"] = idx
        current = next((lbl for lbl, idx in self.labels.items() if idx == MIC_CFG["device"]),
                       self.default_label)

        f_body, f_bold, f_small = ui.fonts["body"], ui.fonts["bold"], ui.fonts["small"]
        self._positioned = False
        self.mw = ui.px(560)               # ความกว้างเนื้อหา (ขยายตามสเกลของ UI)
        self.orig_font = ui.font_scale     # ยกเลิก = คืนขนาดตัวอักษรเดิม
        self.orig_auto = ui.auto_scale
        self.saved = False

        self.win = tk.Toplevel(ui.root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg=C_CYAN)
        self.orig_speed = ui.gesture_mouse.speed
        outer = tk.Frame(self.win, bg=C_PANEL)
        outer.pack(fill="both", expand=True, padx=2, pady=2)

        self.head = tk.Label(outer, text="⚙  ตั้งค่า", font=f_bold, fg=C_CYAN, bg=C_PANEL2,
                             anchor="w", padx=16, pady=10)
        self.head.pack(fill="x", side="top")
        # แถบปุ่มล่าง: pack ก่อนเนื้อหาเพื่อให้ได้พื้นที่ก่อนเสมอ (ขยายตัวอักษรแล้วปุ่มจะไม่หาย)
        bar = tk.Frame(outer, bg=C_PANEL)
        bar.pack(fill="x", padx=18, pady=12, side="bottom")
        self.bar = bar

        # เนื้อหาเลื่อนขึ้น/ลงได้ (กันล้นจอเมื่อตัวอักษรใหญ่) — ใช้ลูกกลิ้งเมาส์หรือแถบด้านขวา
        mid = tk.Frame(outer, bg=C_PANEL)
        mid.pack(fill="both", expand=True, side="top")
        self.scroll_canvas = tk.Canvas(mid, bg=C_PANEL, highlightthickness=0, bd=0)
        vbar = tk.Scrollbar(mid, orient="vertical", command=self.scroll_canvas.yview,
                            troughcolor=C_PANEL, bg=C_PANEL2, bd=0)
        self.scroll_canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        body = tk.Frame(self.scroll_canvas, bg=C_PANEL)
        self.body = body
        self._win_id = self.scroll_canvas.create_window(0, 0, window=body, anchor="nw")
        body.bind("<Configure>", lambda e: self.scroll_canvas.configure(
            scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.bind("<Configure>", lambda e: self.scroll_canvas.itemconfigure(
            self._win_id, width=e.width))

        def label(text):
            tk.Label(body, text=text, font=f_body, fg=C_TEXT, bg=C_PANEL, anchor="w"
                     ).pack(fill="x", padx=18, pady=(12, 2))

        # --- ขนาดตัวอักษร: spinner (กดลูกศร/ปุ่ม − + หรือพิมพ์ตัวเลขแล้วกด Enter) ---
        label(f"ขนาดตัวอักษร (%) — {int(FONT_SCALE_MIN * 100)} ถึง {int(FONT_SCALE_MAX * 100)}")
        self._font_after = None
        self.font_var = tk.StringVar(value=str(int(round(ui.font_scale * 100))))
        frow = tk.Frame(body, bg=C_PANEL)
        frow.pack(fill="x", padx=18)
        step_btn = dict(font=f_bold, fg=C_CYAN, bg=C_PANEL2, activebackground=C_CYAN,
                        activeforeground=C_PANEL, bd=0, padx=14, pady=2, cursor="hand2")
        tk.Button(frow, text="−", command=lambda: self._font_step(-10), **step_btn
                  ).pack(side="left", padx=(0, 6))
        self.font_spin = tk.Spinbox(frow, from_=int(FONT_SCALE_MIN * 100),
                                    to=int(FONT_SCALE_MAX * 100), increment=5, width=5,
                                    textvariable=self.font_var, justify="center", font=f_body,
                                    command=self._font_changed, bg=C_PANEL2, fg=C_TEXT,
                                    buttonbackground=C_PANEL2, insertbackground=C_CYAN, bd=0,
                                    highlightthickness=1, highlightbackground=C_DIM,
                                    highlightcolor=C_CYAN)
        self.font_spin.pack(side="left", ipady=4)
        tk.Label(frow, text="%", font=f_body, fg=C_TEXT, bg=C_PANEL).pack(side="left", padx=(4, 6))
        tk.Button(frow, text="+", command=lambda: self._font_step(+10), **step_btn
                  ).pack(side="left", padx=(0, 6))
        self.font_spin.bind("<Return>", lambda e: self._font_commit())
        self.font_spin.bind("<FocusOut>", lambda e: self._font_commit())
        self.auto_scale_var = tk.BooleanVar(value=ui.auto_scale)
        tk.Checkbutton(body, text="ปรับขนาดตามความละเอียดของจออัตโนมัติ (จอ 4K จะใหญ่ขึ้น 2 เท่า)",
                       variable=self.auto_scale_var, command=self._on_auto_scale, font=f_small,
                       fg=C_TEXT, bg=C_PANEL, selectcolor=C_PANEL2, activebackground=C_PANEL,
                       activeforeground=C_CYAN, anchor="w", justify="left", wraplength=self.mw
                       ).pack(fill="x", padx=14, pady=(6, 0))
        self.scale_info = tk.Label(body, text="", font=f_small, fg=C_DIM, bg=C_PANEL, anchor="w",
                                   justify="left", wraplength=self.mw)
        self.scale_info.pack(fill="x", padx=18)
        self._update_scale_info()

        # --- ความเร็วเคอร์เซอร์เมื่อคุมด้วยมือ (เห็นผลทันที) ---
        tk.Frame(body, bg=C_DIM, height=1).pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(body, text="🖐 เมาส์ด้วยมือ", font=f_bold, fg=C_CYAN, bg=C_PANEL, anchor="w"
                 ).pack(fill="x", padx=18, pady=(8, 0))
        label("ความเร็วเคอร์เซอร์ — ยิ่งสูง ขยับนิ้วนิดเดียวเคอร์เซอร์ก็ไปไกล (1.0 = แบบเดิม)")
        self.speed_w = tk.Scale(body, from_=SPEED_MIN, to=SPEED_MAX, orient="horizontal",
                                resolution=0.1, bg=C_PANEL, fg=C_TEXT, troughcolor=C_PANEL2,
                                highlightthickness=0, activebackground=C_CYAN, bd=0)
        self.speed_w.set(ui.gesture_mouse.speed)
        self.speed_w.config(command=self._on_speed)
        self.speed_w.pack(fill="x", padx=18)
        tk.Label(body, text="กรอบสีส้มในภาพกล้องมุมซ้ายบนจะเล็กลงเมื่อเพิ่มความเร็ว — "
                            "มือต้องขยับในกรอบนั้นเท่านั้นเพื่อครอบคลุมทั้งจอ",
                 font=f_small, fg=C_DIM, bg=C_PANEL, anchor="w", justify="left", wraplength=self.mw
                 ).pack(fill="x", padx=18, pady=(2, 0))

        # --- การคลิกและความนิ่งของเมาส์ (ใช้สดทันที) ---
        tk.Frame(body, bg=C_DIM, height=1).pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(body, text="👆 การคลิก / ความนิ่งของเคอร์เซอร์", font=f_bold, fg=C_CYAN,
                 bg=C_PANEL, anchor="w").pack(fill="x", padx=18, pady=(8, 0))
        tk.Label(body, wraplength=self.mw, justify="left", anchor="w", font=f_small, fg=C_DIM,
                 bg=C_PANEL,
                 text="ลดอาการเคอร์เซอร์เคลื่อนตอนจีบนิ้วเพื่อคลิก: เคอร์เซอร์จะขยับเมื่อ \"ทั้งมือ\" ขยับเท่านั้น "
                      "และจะถูกตรึงไว้เมื่อนิ้วเริ่มเข้าใกล้กัน"
                 ).pack(fill="x", padx=18, pady=(2, 0))

        self.click_cfg = ui.gesture_mouse.cfg
        self.orig_click = dict(self.click_cfg)
        self.click_vars = {}

        label("จุดที่ใช้พาเคอร์เซอร์")
        self.anchor_var = tk.StringVar(value=CLICK_ANCHORS[self.click_cfg["anchor"]])
        anchor_menu = tk.OptionMenu(body, self.anchor_var, *CLICK_ANCHORS.values(),
                                    command=lambda _v: self._click_changed())
        anchor_menu.config(font=f_small, fg=C_TEXT, bg=C_PANEL2, activebackground=C_CYAN,
                           activeforeground=C_PANEL, bd=0, highlightthickness=1,
                           highlightbackground=C_DIM, anchor="w")
        anchor_menu["menu"].config(font=f_small, bg=C_PANEL2, fg=C_TEXT,
                                   activebackground=C_CYAN, activeforeground=C_PANEL)
        anchor_menu.pack(fill="x", padx=18)

        self.freeze_var = tk.BooleanVar(value=self.click_cfg["freeze_on"])
        tk.Checkbutton(body, text="ตรึงเคอร์เซอร์เมื่อนิ้วโป้งกับนิ้วชี้เริ่มเข้าใกล้กัน (กำลังจะจีบ)",
                       variable=self.freeze_var, command=self._click_changed, font=f_body,
                       fg=C_TEXT, bg=C_PANEL, selectcolor=C_PANEL2, activebackground=C_PANEL,
                       activeforeground=C_CYAN, anchor="w", justify="left", wraplength=self.mw
                       ).pack(fill="x", padx=14, pady=(10, 0))

        # (คีย์, ชื่อ, ต่ำสุด, สูงสุด, ครั้งละ, ตัวคูณแสดงผล, คำอธิบาย)
        specs = (
            ("still_px", "ความนิ่ง (พิกเซล) — มือต้องขยับเกินระยะนี้เคอร์เซอร์ถึงจะเริ่มเคลื่อน", 0, 80, 2, 1,
             "สูง = นิ่งขึ้น แต่ต้องขยับมือมากขึ้นก่อนเคอร์เซอร์เริ่มวิ่ง · 0 = ปิด"),
            ("freeze_ratio", "ระยะเริ่มตรึง (%) — นิ้วโป้งห่างนิ้วชี้เท่าไรของความกว้างฝ่ามือ", 40, 120, 5, 100,
             "สูง = ตรึงเร็วขึ้นตั้งแต่นิ้วยังห่างกัน"),
            ("lookback_ms", "ย้อนตำแหน่งก่อนจีบ (มิลลิวินาที)", 0, 400, 20, 1,
             "คลิกที่ตำแหน่งเมื่อกี่มิลลิวินาทีก่อน (ก่อนนิ้วเริ่มขยับ) · 0 = ปิด (แนะนำ 80–120 ถ้าใช้ปลายนิ้วชี้)"),
            ("drag_px", "ระยะก่อนเริ่มลาก (พิกเซล)", 5, 200, 5, 1,
             "หลังจีบ ต้องขยับมือเกินระยะนี้ถึงจะลาก · สูง = คลิกนิ่งขึ้น ลากยากขึ้น"),
            ("pinch_start", "ความแน่นของการจีบ (%)", 15, 55, 1, 100,
             "ต่ำ = ต้องจีบนิ้วชิดกันถึงจะคลิก (กันคลิกพลาด) · สูง = จีบหลวมๆ ก็คลิก"),
        )
        self.click_specs = {}
        step_btn = dict(font=f_bold, fg=C_CYAN, bg=C_PANEL2, activebackground=C_CYAN,
                        activeforeground=C_PANEL, bd=0, padx=12, pady=1, cursor="hand2")
        for key, title, lo, hi, step, mult, hint in specs:
            label(title)
            var = tk.StringVar(value=str(int(round(self.click_cfg[key] * mult))))
            self.click_vars[key] = var
            self.click_specs[key] = (lo, hi, step, mult)
            row = tk.Frame(body, bg=C_PANEL)
            row.pack(fill="x", padx=18)
            tk.Button(row, text="−", command=lambda k=key: self._click_step(k, -1), **step_btn
                      ).pack(side="left", padx=(0, 6))
            sp = tk.Spinbox(row, from_=lo, to=hi, increment=step, width=6, textvariable=var,
                            justify="center", font=f_body, command=self._click_changed,
                            bg=C_PANEL2, fg=C_TEXT, buttonbackground=C_PANEL2,
                            insertbackground=C_CYAN, bd=0, highlightthickness=1,
                            highlightbackground=C_DIM, highlightcolor=C_CYAN)
            sp.pack(side="left", ipady=3)
            sp.bind("<Return>", lambda e: self._click_changed(normalize=True))
            sp.bind("<FocusOut>", lambda e: self._click_changed(normalize=True))
            tk.Button(row, text="+", command=lambda k=key: self._click_step(k, +1), **step_btn
                      ).pack(side="left", padx=(6, 10))
            tk.Label(row, text=hint, font=f_small, fg=C_DIM, bg=C_PANEL, anchor="w",
                     justify="left", wraplength=max(200, self.mw - 330)).pack(side="left", fill="x")
        tk.Button(body, text="↺ รีเซ็ตเป็นค่าแนะนำ", command=self._click_reset, font=f_small,
                  fg=C_USER, bg=C_PANEL2, activebackground=C_USER, activeforeground=C_PANEL, bd=0,
                  padx=12, pady=4, cursor="hand2").pack(anchor="w", padx=18, pady=(10, 0))

        # --- โมเดลแชท: ค้นหา/เลือกจากรายการ หรือพิมพ์รหัสโมเดลเอง (มีผลเมื่อกด "บันทึก") ---
        tk.Frame(body, bg=C_DIM, height=1).pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(body, text="🧠 โมเดล AI (แชท)", font=f_bold, fg=C_CYAN, bg=C_PANEL, anchor="w"
                 ).pack(fill="x", padx=18, pady=(8, 0))
        label("พิมพ์ค้นหา หรือกรอกรหัสโมเดลเอง เช่น openai/gpt-4o-mini")
        self.model_var = tk.StringVar(value=LLM_MODEL)
        self.model_entry = tk.Entry(body, textvariable=self.model_var, font=f_small, bg=C_PANEL2,
                                    fg=C_TEXT, bd=0, insertbackground=C_CYAN, highlightthickness=1,
                                    highlightbackground=C_DIM, highlightcolor=C_CYAN)
        self.model_entry.pack(fill="x", padx=18, ipady=6)
        lb_frame = tk.Frame(body, bg=C_PANEL)
        lb_frame.pack(fill="x", padx=18, pady=(6, 0))
        lb_scroll = tk.Scrollbar(lb_frame, troughcolor=C_PANEL, bg=C_PANEL2, bd=0)
        lb_scroll.pack(side="right", fill="y")
        self.model_list = tk.Listbox(lb_frame, height=6, font=f_small, bg=C_PANEL2, fg=C_TEXT,
                                     selectbackground=C_CYAN, selectforeground=C_PANEL, bd=0,
                                     highlightthickness=1, highlightbackground=C_DIM,
                                     exportselection=False, activestyle="none",
                                     yscrollcommand=lb_scroll.set)
        self.model_list.pack(side="left", fill="x", expand=True)
        lb_scroll.config(command=self.model_list.yview)
        self.model_list.bind("<<ListboxSelect>>", self._pick_model)
        self.models = list(dict.fromkeys(ui.model_cache or DEFAULT_MODELS))
        self._filtering = False
        self.model_var.trace_add("write", lambda *_: self._filter_models())
        row2 = tk.Frame(body, bg=C_PANEL)
        row2.pack(fill="x", padx=18, pady=(6, 0))
        tk.Button(row2, text="🔄 โหลดรายการโมเดลทั้งหมดจาก OpenRouter", command=self._load_models,
                  font=f_small, fg=C_CYAN, bg=C_PANEL2, activebackground=C_CYAN,
                  activeforeground=C_PANEL, bd=0, padx=12, pady=4, cursor="hand2").pack(side="left")
        self.model_msg = tk.Label(body, text="", font=f_small, fg=C_DIM, bg=C_PANEL, anchor="w",
                                  justify="left", wraplength=self.mw)
        self.model_msg.pack(fill="x", padx=18, pady=(2, 0))
        self._filter_models(show_all=True)

        # --- OpenRouter API key (บันทึก/ลบได้ ใช้ทันทีไม่ต้องกด "บันทึก" ของเมนูนี้) ---
        tk.Frame(body, bg=C_DIM, height=1).pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(body, text="🔑 OpenRouter API key", font=f_bold, fg=C_CYAN, bg=C_PANEL, anchor="w"
                 ).pack(fill="x", padx=18, pady=(8, 0))
        self.key_status = tk.Label(body, text="", font=f_body, fg=C_TEXT, bg=C_PANEL, anchor="w",
                                   justify="left", wraplength=self.mw)
        self.key_status.pack(fill="x", padx=18, pady=(8, 2))
        row = tk.Frame(body, bg=C_PANEL)
        row.pack(fill="x", padx=18)
        self.key_entry = tk.Entry(row, font=f_small, show="•", bg=C_PANEL2, fg=C_TEXT, bd=0,
                                  insertbackground=C_CYAN, highlightthickness=1,
                                  highlightbackground=C_DIM, highlightcolor=C_CYAN)
        self.key_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        tk.Button(row, text="บันทึกคีย์", command=self._save_key, font=f_small, fg=C_CYAN,
                  bg=C_PANEL2, activebackground=C_CYAN, activeforeground=C_PANEL, bd=0,
                  padx=12, pady=4, cursor="hand2").pack(side="left")
        self.key_msg = tk.Label(body, text="", font=f_small, fg=C_DIM, bg=C_PANEL, anchor="w",
                                wraplength=self.mw, justify="left")
        self.key_msg.pack(fill="x", padx=18, pady=(2, 0))
        self.key_del_btn = tk.Button(body, text="🗑 ลบคีย์ที่บันทึกไว้", command=self._delete_key,
                                     font=f_small, fg=C_ERR, bg=C_PANEL2, activebackground=C_ERR,
                                     activeforeground=C_PANEL, bd=0, padx=12, pady=4,
                                     cursor="hand2", anchor="w")
        self.key_del_btn.pack(anchor="w", padx=18, pady=(6, 0))
        self._del_armed = False
        self._refresh_key_status()

        tk.Frame(body, bg=C_DIM, height=1).pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(body, text="🎙 ไมโครโฟน", font=f_bold, fg=C_CYAN, bg=C_PANEL, anchor="w"
                 ).pack(fill="x", padx=18, pady=(8, 0))

        # --- อุปกรณ์ ---
        label("อุปกรณ์รับเสียง")
        self.dev_var = tk.StringVar(value=current)
        menu = tk.OptionMenu(body, self.dev_var, *self.labels.keys(),
                             command=lambda _v: self._restart_meter())
        menu.config(font=f_small, fg=C_TEXT, bg=C_PANEL2, activebackground=C_CYAN,
                    activeforeground=C_PANEL, bd=0, highlightthickness=1,
                    highlightbackground=C_DIM, anchor="w")
        menu["menu"].config(font=f_small, bg=C_PANEL2, fg=C_TEXT, activebackground=C_CYAN,
                            activeforeground=C_PANEL)
        menu.pack(fill="x", padx=18)

        # --- มิเตอร์ระดับเสียง ---
        label("ระดับเสียงสด (พูดทดสอบได้เลย — เส้นขาวคือเกณฑ์ที่ถือว่า 'กำลังพูด')")
        self.canvas = tk.Canvas(body, width=self.mw, height=ui.px(26), bg=C_PANEL2,
                                highlightthickness=0)
        self.canvas.pack(padx=18, fill="x")
        self.meter_msg = tk.Label(body, text="", font=f_small, fg=C_DIM, bg=C_PANEL, anchor="w")
        self.meter_msg.pack(fill="x", padx=18, pady=(2, 0))

        # --- ความไว ---
        self.auto_var = tk.BooleanVar(value=MIC_CFG["auto"])
        tk.Checkbutton(body, text="ปรับเกณฑ์เสียงอัตโนมัติตามเสียงรบกวนของห้อง",
                       variable=self.auto_var, command=self._sync_enable, font=f_body,
                       fg=C_TEXT, bg=C_PANEL, selectcolor=C_PANEL2, activebackground=C_PANEL,
                       activeforeground=C_CYAN, anchor="w").pack(fill="x", padx=14, pady=(12, 0))

        label("เกณฑ์เสียงพูด (ยิ่งต่ำ = ไวขึ้น แต่จับเสียงรบกวนง่ายขึ้น)")
        self.thr = tk.Scale(body, from_=100, to=int(self.METER_MAX), orient="horizontal",
                            resolution=10, bg=C_PANEL, fg=C_TEXT, troughcolor=C_PANEL2,
                            highlightthickness=0, activebackground=C_CYAN, bd=0)
        self.thr.set(int(MIC_CFG["threshold"]))
        self.thr.pack(fill="x", padx=18)

        label("เงียบนานเท่าไรถือว่าพูดจบ (วินาที)")
        self.sil = tk.Scale(body, from_=0.5, to=3.0, orient="horizontal", resolution=0.1,
                            bg=C_PANEL, fg=C_TEXT, troughcolor=C_PANEL2,
                            highlightthickness=0, activebackground=C_CYAN, bd=0)
        self.sil.set(MIC_CFG["silence"])
        self.sil.pack(fill="x", padx=18)

        # --- ปุ่ม (ใส่ใน bar ที่สร้างไว้ด้านบน) ---
        def btn(text, cmd, fg=C_CYAN):
            tk.Button(bar, text=text, command=cmd, font=f_bold, fg=fg, bg=C_PANEL2,
                      activebackground=C_CYAN, activeforeground=C_PANEL, bd=0, padx=14,
                      pady=6, cursor="hand2").pack(side="left", padx=(0, 8))

        btn("วัดเสียงห้อง (เงียบ 2 วิ)", self._calibrate, fg=C_USER)
        btn("บันทึก", self._save)
        btn("ยกเลิก", self.close, fg=C_ERR)

        self._bind_wheel(self.win)
        self._sync_enable()
        self._start_meter()
        self._fit()
        self._tick()

    # ----- ขนาดหน้าต่าง / เลื่อนเนื้อหา -----
    def _bind_wheel(self, widget):
        """ลูกกลิ้งเมาส์เลื่อนเนื้อหา (ผูกกับทุก widget ในหน้าต่างนี้)"""
        if not isinstance(widget, (tk.Scale, tk.Listbox)):   # สองตัวนี้ใช้ลูกกลิ้งของตัวเอง
            widget.bind("<MouseWheel>", self._on_wheel)
        for child in widget.winfo_children():
            self._bind_wheel(child)

    def _on_wheel(self, event):
        self.scroll_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _fit(self, recenter=False):
        """ปรับขนาดหน้าต่างตามเนื้อหา แต่ไม่เกินจอ ส่วนที่เกินให้เลื่อนดูได้
        ครั้งแรกวางกลางจอ หลังจากนั้นคงมุมซ้ายบนไว้ (หน้าต่างไม่เด้งหนีตอนปรับขนาด)"""
        ui = self.ui
        self.win.update_idletasks()
        content_w = self.body.winfo_reqwidth()
        content_h = self.body.winfo_reqheight()
        chrome_h = self.head.winfo_reqheight() + self.bar.winfo_reqheight() + 24 + 8
        avail_h = max(200, ui.H - 60 - chrome_h)
        self.scroll_canvas.config(width=max(ui.px(580), content_w) + 14,
                                  height=min(content_h, avail_h))
        self.win.update_idletasks()
        w = min(max(ui.px(620), self.win.winfo_reqwidth()), ui.W - 20)
        h = min(self.win.winfo_reqheight(), ui.H - 20)
        if recenter or not self._positioned:
            x, y = ui.X + (ui.W - w) // 2, ui.Y + (ui.H - h) // 2
        else:                                   # คงตำแหน่งเดิม แต่ไม่ให้ล้นขอบจอ
            x = max(ui.X, min(self.win.winfo_x(), ui.X + ui.W - w))
            y = max(ui.Y, min(self.win.winfo_y(), ui.Y + ui.H - h))
        self._positioned = True
        self.win.geometry(f"{w}x{h}+{x}+{y}")

    # ----- spinner ขนาดตัวอักษร -----
    def _font_value(self):
        """อ่านค่า % จากช่อง (ถ้าพิมพ์ผิดคืน None) แล้วบีบให้อยู่ในช่วงที่รองรับ"""
        try:
            v = float(self.font_var.get().strip().replace("%", ""))
        except ValueError:
            return None
        return max(int(FONT_SCALE_MIN * 100), min(int(FONT_SCALE_MAX * 100), round(v)))

    def _font_step(self, delta):
        v = self._font_value()
        if v is None:
            v = int(round(self.ui.font_scale * 100))
        self.font_var.set(str(max(int(FONT_SCALE_MIN * 100),
                                  min(int(FONT_SCALE_MAX * 100), v + delta))))
        self._font_changed()

    def _font_changed(self):
        """กดลูกศร/ปุ่มค้าง: หน่วง 250 มิลลิวินาทีแล้วค่อยปรับจริงครั้งเดียว กัน UI คำนวณขนาดซ้ำรัวๆ"""
        if self.closed:
            return
        if self._font_after is not None:
            self.win.after_cancel(self._font_after)
        self._font_after = self.win.after(250, self._apply_font)

    def _font_commit(self):
        """กด Enter หรือออกจากช่อง: ปรับทันที และแก้ข้อความในช่องให้เป็นตัวเลขที่ถูกต้อง"""
        if self.closed:
            return
        if self._font_after is not None:
            self.win.after_cancel(self._font_after)
            self._font_after = None
        self._apply_font()

    def _apply_font(self):
        self._font_after = None
        if self.closed:
            return
        v = self._font_value()
        if v is None:                           # พิมพ์ไม่ใช่ตัวเลข: คืนค่าเดิม
            v = int(round(self.ui.font_scale * 100))
        self.font_var.set(str(v))
        if abs(v / 100.0 - self.ui.font_scale) > 1e-9:
            self.ui.set_font_scale(v / 100.0)
            self._update_scale_info()
            self.win.after_idle(self._fit)

    def _on_auto_scale(self):
        if self.closed:
            return
        self.ui.set_auto_scale(self.auto_scale_var.get())
        self._update_scale_info()
        self.win.after_idle(self._fit)

    def _update_scale_info(self):
        ui = self.ui
        auto = f"×{ui.screen_factor():.2f}" if ui.auto_scale else "ปิด"
        self.scale_info.config(
            text=f"จอนี้ {ui.W}×{ui.H} · ปรับอัตโนมัติ {auto} · ขนาดจริง {ui.eff_scale * 100:.0f}%")

    def _on_speed(self, value):
        if self.closed:
            return
        self.ui.gesture_mouse.set_speed(float(value))

    # ----- การคลิก / ความนิ่งของเมาส์ -----
    def _click_value(self, key):
        """ค่าจากช่อง (แสดงเป็น % หรือพิกเซล) → ค่าจริงที่ใช้งาน คืน None ถ้ายังพิมพ์ไม่เป็นตัวเลข"""
        lo, hi, _step, mult = self.click_specs[key]
        try:
            v = float(self.click_vars[key].get().strip())
        except ValueError:
            return None
        return max(lo, min(hi, v)) / mult

    def _click_changed(self, normalize=False):
        """ใช้ค่าจากหน้าเมนูทันที  normalize=True (กด Enter/ออกจากช่อง) = แก้ตัวเลขในช่องให้ตรงค่าที่ใช้จริง"""
        if self.closed:
            return
        new = {"anchor": next((k for k, v in CLICK_ANCHORS.items() if v == self.anchor_var.get()),
                              "palm"),
               "freeze_on": bool(self.freeze_var.get())}
        for key in self.click_vars:
            v = self._click_value(key)
            if v is not None:
                new[key] = v
        self.ui.gesture_mouse.set_click_cfg(new)
        applied = self.ui.gesture_mouse.cfg
        focus_var = None
        try:
            focus = self.win.focus_get()
            if focus is not None and focus.winfo_class() == "Spinbox":
                focus_var = str(focus.cget("textvariable"))
        except (tk.TclError, KeyError):
            pass
        for key, var in self.click_vars.items():         # แสดงค่าที่ถูกบีบให้อยู่ในช่วงจริง
            shown = str(int(round(applied[key] * self.click_specs[key][3])))
            if var.get() != shown and (normalize or focus_var != str(var)):
                var.set(shown)                           # ไม่ทับตัวเลขที่ผู้ใช้กำลังพิมพ์อยู่

    def _click_step(self, key, direction):
        lo, hi, step, mult = self.click_specs[key]
        v = self._click_value(key)
        cur = (v * mult) if v is not None else self.ui.gesture_mouse.cfg[key] * mult
        self.click_vars[key].set(str(int(max(lo, min(hi, round(cur + direction * step))))))
        self._click_changed(normalize=True)

    def _click_reset(self):
        self.anchor_var.set(CLICK_ANCHORS[CLICK_DEFAULTS["anchor"]])
        self.freeze_var.set(CLICK_DEFAULTS["freeze_on"])
        for key, var in self.click_vars.items():
            var.set(str(int(round(CLICK_DEFAULTS[key] * self.click_specs[key][3]))))
        self._click_changed(normalize=True)

    # ----- โมเดล AI -----
    def _filter_models(self, show_all=False):
        if self._filtering:
            return
        q = "" if show_all else self.model_var.get().strip().lower()   # ตอนเปิดเมนู แสดงทั้งหมดก่อน
        matches = [m for m in self.models if q in m.lower()] if q else list(self.models)
        self.model_list.delete(0, "end")
        for m in matches[:300]:
            self.model_list.insert("end", m)
        for i, m in enumerate(matches[:300]):          # ไฮไลต์รายการที่ตรงกับที่พิมพ์เป๊ะ
            if m == self.model_var.get().strip():
                self.model_list.selection_set(i)
                self.model_list.see(i)
        count = len(matches)
        self.model_msg.config(
            text=f"พบ {count} รายการ" + (" (แสดง 300 รายการแรก — พิมพ์ค้นหาเพื่อกรอง)" if count > 300 else "")
                 + (" · ไม่พบในรายการ: จะใช้ตามที่พิมพ์" if q and count == 0 else ""),
            fg=C_DIM)

    def _pick_model(self, _event=None):
        sel = self.model_list.curselection()
        if not sel:
            return
        self._filtering = True                 # ตั้งค่าช่องพิมพ์โดยไม่ให้รายการถูกกรองใหม่จนเหลือรายการเดียว
        self.model_var.set(self.model_list.get(sel[0]))
        self._filtering = False

    def _load_models(self):
        self.model_msg.config(text="กำลังโหลดรายการโมเดลจาก OpenRouter...", fg=C_USER)

        def work():
            try:
                resp = requests.get("https://openrouter.ai/api/v1/models", timeout=20)
                resp.raise_for_status()
                ids = sorted(m["id"] for m in resp.json().get("data", []) if m.get("id"))
                if not ids:
                    raise ValueError("ไม่พบรายการโมเดล")
                self.ui.ui_q.put(("models", ids))
            except Exception as exc:
                self.ui.ui_q.put(("models_error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def set_models(self, ids):
        if self.closed:
            return
        self.models = list(dict.fromkeys(ids))
        self._filter_models()
        self.model_msg.config(text=f"โหลดแล้ว {len(self.models)} โมเดล — พิมพ์ค้นหาแล้วคลิกเพื่อเลือก", fg=C_OK)

    def set_models_error(self, msg):
        if not self.closed:
            self.model_msg.config(text=f"โหลดรายการไม่สำเร็จ: {msg} (ยังพิมพ์รหัสโมเดลเองได้)", fg=C_ERR)

    # ----- API key -----
    def _refresh_key_status(self):
        has_saved = bool(load_saved_key())
        if OPENROUTER_API_KEY:
            text = (f"✅ ตั้งค่าแล้ว: {mask_key(OPENROUTER_API_KEY)}\n"
                    f"แหล่งที่มา: {API_KEY_SOURCE_LABEL.get(API_KEY_SOURCE, API_KEY_SOURCE)}")
            if API_KEY_SOURCE == "environment variable" and has_saved:
                text += "\n(มีคีย์ที่บันทึกไว้ด้วย แต่ environment variable ถูกใช้ก่อน)"
            self.key_status.config(text=text, fg=C_OK)
        else:
            self.key_status.config(text="❌ ยังไม่ได้ตั้งค่า API key — AI และการถอดเสียงจะยังใช้ไม่ได้",
                                   fg=C_ERR)
        self.key_del_btn.config(state="normal" if has_saved else "disabled",
                                text="🗑 ลบคีย์ที่บันทึกไว้" if has_saved else "ไม่มีคีย์ที่บันทึกไว้")
        self._del_armed = False

    def _save_key(self):
        ok, err = save_api_key(self.key_entry.get())
        if ok:
            self.key_entry.delete(0, "end")
            self.key_msg.config(text="บันทึกแล้ว — ใช้งานได้ทันที และจะจำไว้ใช้ครั้งต่อไป", fg=C_OK)
            self.ui._sys("บันทึก API key แล้ว (เข้ารหัสเก็บไว้ในเครื่องนี้)")
        else:
            self.key_msg.config(text=err, fg=C_ERR)
        self._refresh_key_status()
        self.win.after_idle(self._fit)

    def _delete_key(self):
        if not self._del_armed:                         # กดครั้งแรก = ขอยืนยัน
            self._del_armed = True
            self.key_del_btn.config(text="⚠ กดอีกครั้งเพื่อยืนยันการลบ")
            self.win.after(4000, lambda: (self._refresh_key_status() if not self.closed else None))
            return
        ok, err = delete_saved_key()
        if ok:
            self.key_msg.config(text="ลบคีย์ที่บันทึกไว้แล้ว", fg=C_OK)
            self.ui._sys("ลบ API key ที่บันทึกไว้แล้ว")
        else:
            self.key_msg.config(text=err, fg=C_ERR)
        self._refresh_key_status()
        self.win.after_idle(self._fit)

    # ----- สถานะของ widget -----
    def _sync_enable(self):
        self.thr.config(state="disabled" if self.auto_var.get() else "normal")

    def _device_index(self):
        return self.labels.get(self.dev_var.get())

    # ----- มิเตอร์ -----
    def _stop_meter(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _restart_meter(self):
        self._stop_meter()
        self.level = 0.0
        self._start_meter()

    def _start_meter(self):
        if sd is None:
            self.meter_msg.config(text=f"ใช้ไมโครโฟนไม่ได้ (sounddevice): {_SD_IMPORT_ERROR}", fg=C_ERR)
            return

        def cb(indata, frames, time_info, status):
            rms = block_rms(np.asarray(indata, dtype=np.int16).reshape(-1))
            self.level = rms
            if self.calib is not None:
                self.calib.append(rms)

        try:
            self.stream = sd.InputStream(device=self._device_index(), samplerate=SAMPLE_RATE,
                                         channels=1, dtype="int16", blocksize=CHUNK_SIZE,
                                         callback=cb)
            self.stream.start()
            self.meter_msg.config(text="", fg=C_DIM)
        except Exception as exc:
            self.stream = None
            self.meter_msg.config(text=f"เปิดอุปกรณ์นี้ไม่ได้: {exc}", fg=C_ERR)

    def _marker_value(self):
        if not self.auto_var.get():
            return float(self.thr.get())
        return self.ui.auto_threshold      # None ถ้ายังไม่เคยวัด

    def _tick(self):
        if self.closed:
            return
        c = self.canvas
        c.delete("all")
        frac = min(1.0, (self.level / self.METER_MAX) ** 0.5)     # สเกลแบบ sqrt ให้เสียงเบาเห็นชัด
        marker = self._marker_value()
        speaking = marker is not None and self.level > marker
        cw, ch = max(1, c.winfo_width()), max(1, c.winfo_height())
        c.create_rectangle(0, 0, int(cw * frac), ch, fill=C_OK if speaking else C_CYAN, width=0)
        if marker is not None:
            mx = int(cw * min(1.0, (marker / self.METER_MAX) ** 0.5))
            c.create_line(mx, 0, mx, ch, fill="#ffffff", width=2)

        if self.calib is not None and time.monotonic() >= self.calib_end:
            samples, self.calib = self.calib, None
            if samples:
                noise = float(np.mean(samples))
                suggested = compute_threshold(noise)
                self.auto_var.set(False)
                self._sync_enable()            # Scale ที่ disabled จะไม่รับ set() ต้องเปิดก่อน
                self.thr.set(int(round(suggested / 10.0) * 10))
                self.meter_msg.config(
                    text=f"เสียงรบกวนห้อง ≈ {noise:.0f} → ตั้งเกณฑ์ที่ {suggested:.0f} (กดบันทึกเพื่อใช้งาน)",
                    fg=C_OK)
            else:
                self.meter_msg.config(text="วัดไม่สำเร็จ ไม่ได้รับสัญญาณจากไมค์", fg=C_ERR)
        self.win.after(50, self._tick)

    def _calibrate(self):
        if self.stream is None:
            self.meter_msg.config(text="ไมค์ยังไม่พร้อม เลือกอุปกรณ์อื่น หรือตรวจสอบการเชื่อมต่อ", fg=C_ERR)
            return
        self.calib = []
        self.calib_end = time.monotonic() + 2.0
        self.meter_msg.config(text="🤫 กรุณาเงียบสักครู่ กำลังวัดเสียงรบกวน...", fg=C_USER)

    # ----- บันทึก / ปิด -----
    def _save(self):
        MIC_CFG["device"] = self._device_index()
        MIC_CFG["auto"] = bool(self.auto_var.get())
        MIC_CFG["threshold"] = float(self.thr.get())
        MIC_CFG["silence"] = round(float(self.sil.get()), 1)
        self.ui.auto_threshold = None          # บังคับวัดใหม่ถ้ายังเป็นโหมดอัตโนมัติ
        ok = save_mic_config()
        mode = "อัตโนมัติ" if MIC_CFG["auto"] else f"กำหนดเอง {MIC_CFG['threshold']:.0f}"
        self.ui.save_font_scale()
        self.ui.save_mouse_speed()
        self.ui.save_click_cfg()
        model = self.model_var.get().strip()
        if model and model != LLM_MODEL:
            set_llm_model(model)
            self.ui.save_llm_model(model)
        self.saved = True
        self.ui._sys(f"บันทึกการตั้งค่าแล้ว — โมเดล {LLM_MODEL} | "
                     f"ตัวอักษร {int(round(self.ui.font_scale * 100))}% | "
                     f"ความเร็วเมาส์ {self.ui.gesture_mouse.speed:.1f}x | "
                     f"อุปกรณ์: {self.dev_var.get()} | เกณฑ์: {mode} | "
                     f"เงียบ {MIC_CFG['silence']} วิ" + ("" if ok else " (บันทึกลงไฟล์ไม่สำเร็จ)"))
        if self.ui.voice_on.is_set():
            self.ui._sys("ถ้าเปลี่ยนอุปกรณ์ ให้กดปิดแล้วเปิดไมค์ใหม่อีกครั้ง")
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if not self.saved:                         # ยกเลิก: คืนขนาดตัวอักษรและความเร็วเมาส์เดิม
            self.ui.auto_scale = self.orig_auto
            self.ui.set_font_scale(self.orig_font)
            self.ui.gesture_mouse.set_speed(self.orig_speed)
            self.ui.gesture_mouse.set_click_cfg(self.orig_click)
        self._stop_meter()
        self.ui.mic_menu = None
        try:
            self.win.destroy()
        except Exception:
            pass


class ApiKeyDialog:
    """ป๊อปอัปขอ OpenRouter API key (แสดงตอนเปิดโปรแกรมถ้ายังไม่มีคีย์) — บันทึกแล้วจำไว้ใช้ครั้งต่อไป"""

    def __init__(self, ui):
        self.ui = ui
        self.closed = False
        f_body, f_bold, f_small = ui.fonts["body"], ui.fonts["bold"], ui.fonts["small"]

        self.win = tk.Toplevel(ui.root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg=C_CYAN)
        body = tk.Frame(self.win, bg=C_PANEL)
        body.pack(fill="both", expand=True, padx=2, pady=2)

        tk.Label(body, text="🔑  ตั้งค่า API key ครั้งแรก", font=f_bold, fg=C_CYAN, bg=C_PANEL2,
                 anchor="w", padx=16, pady=10).pack(fill="x")
        tk.Label(body, wraplength=ui.px(560), justify="left", anchor="w", font=f_body, fg=C_TEXT,
                 bg=C_PANEL,
                 text="ต้องมี OpenRouter API key เพื่อคุยกับ AI และถอดเสียงพูด\n"
                      "สร้างคีย์ได้ที่ openrouter.ai/keys แล้ววางลงช่องด้านล่าง\n\n"
                      "คีย์จะถูกเข้ารหัสและเก็บไว้ในเครื่องนี้ ใช้อัตโนมัติครั้งต่อไป "
                      "และลบได้ที่เมนู ⚙ ตั้งค่า\n\n"
                      "ยังไม่อยากใช้ AI? กด \"ข้ามไปก่อน\" ได้เลย — ควบคุมเมาส์ด้วยมือ เปิดโปรแกรม "
                      "และหน้า Help ยังใช้ได้ตามปกติ (ไม่ต้องใช้คีย์)"
                 ).pack(fill="x", padx=18, pady=(14, 8))

        self.entry = tk.Entry(body, font=f_body, show="•", bg=C_PANEL2, fg=C_TEXT, bd=0,
                              insertbackground=C_CYAN, highlightthickness=1,
                              highlightbackground=C_DIM, highlightcolor=C_CYAN)
        self.entry.pack(fill="x", padx=18, ipady=8)
        self.entry.bind("<Return>", lambda e: self._save())
        self.show_var = tk.BooleanVar(value=False)
        tk.Checkbutton(body, text="แสดงคีย์", variable=self.show_var, command=self._toggle_show,
                       font=f_small, fg=C_DIM, bg=C_PANEL, selectcolor=C_PANEL2,
                       activebackground=C_PANEL, activeforeground=C_CYAN, anchor="w"
                       ).pack(fill="x", padx=14, pady=(4, 0))
        self.msg = tk.Label(body, text="", font=f_small, fg=C_ERR, bg=C_PANEL, anchor="w",
                            wraplength=ui.px(560), justify="left")
        self.msg.pack(fill="x", padx=18)

        bar = tk.Frame(body, bg=C_PANEL)
        bar.pack(fill="x", padx=18, pady=14)
        for text, cmd, fg in (("บันทึก", self._save, C_CYAN), ("ข้ามไปก่อน", self.close, C_TEXT)):
            tk.Button(bar, text=text, command=cmd, font=f_bold, fg=fg, bg=C_PANEL2,
                      activebackground=C_CYAN, activeforeground=C_PANEL, bd=0, padx=28, pady=12,
                      cursor="hand2").pack(side="left", padx=(0, 12))      # ปุ่มใหญ่ จีบนิ้วคลิกง่าย

        self.win.update_idletasks()
        w, h = max(ui.px(620), self.win.winfo_reqwidth()), self.win.winfo_reqheight()
        self.win.geometry(f"{w}x{h}+{ui.X + (ui.W - w) // 2}+{ui.Y + (ui.H - h) // 2}")
        self.win.bind("<Escape>", lambda e: self.close())
        self.win.after(150, self._focus)

    def _focus(self):
        try:
            self.win.focus_force()
            self.entry.focus_force()
        except tk.TclError:
            pass

    def _toggle_show(self):
        self.entry.config(show="" if self.show_var.get() else "•")

    def _save(self):
        ok, err = save_api_key(self.entry.get())
        if not ok:
            self.msg.config(text=err)
            return
        self.ui._sys("บันทึก API key แล้ว — ใช้งานได้ทันที และจะจำไว้ใช้ครั้งต่อไป")
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.ui.api_dialog = None
        try:
            self.win.destroy()
        except Exception:
            pass


class TonyStarkUI:
    def __init__(self, speak=True, gesture=True, threshold=None, monitor=None):
        self.ui_q = queue.Queue()
        self.monitor_arg = monitor      # หมายเลขจอ (เริ่มที่ 1) จาก --monitor
        self.threshold = threshold
        self.state = "STANDBY"
        self.voice_on = threading.Event()
        self.voice_thread = None
        self.closing = False
        self.photo = None
        self.last_seq = -1
        self.ai_open = False
        self.auto_threshold = None      # เกณฑ์ที่วัดจากห้อง (cache) ล้างเมื่อบันทึกเมนูไมค์
        self.mic_menu = None
        self.help_frame = None
        self.last_help_seq = 0
        self.offsets = {"chat": [0, 0], "cam": [0, 0]}   # ระยะที่ผู้ใช้ลากแผงไว้
        saved_off = load_ui_config().get("panel_offsets")
        if isinstance(saved_off, dict):
            for k in self.offsets:
                v = saved_off.get(k)
                if (isinstance(v, list) and len(v) == 2
                        and all(isinstance(n, (int, float)) for n in v)):
                    self.offsets[k] = [int(v[0]), int(v[1])]
        self._drag = None
        self.pinned = True              # True = interface ลอยบนสุดเสมอ
        self.mini_pin = None
        self.api_dialog = None
        self.clap = ClapDetector(lambda: self.ui_q.put(("clap", None)))

        self.speech = SpeechStreamer(enabled=speak, on_state=lambda s: self.ui_q.put(("state", s)))
        self.assistant = Assistant(self.ui_q, self.speech)
        self.gesture_mouse = GestureMouse()
        saved_speed = load_ui_config().get("mouse_speed")
        if isinstance(saved_speed, (int, float)):
            self.gesture_mouse.set_speed(saved_speed)
        saved_click = load_ui_config().get("click_cfg")
        if isinstance(saved_click, dict):
            self.gesture_mouse.set_click_cfg(saved_click)
        saved_model = load_ui_config().get("llm_model")
        if isinstance(saved_model, str) and saved_model.strip():
            set_llm_model(saved_model)          # โมเดลที่เลือกไว้ในเมนูตั้งค่า (ชนะค่าจาก env)
        self.model_cache = None                 # รายการโมเดลที่โหลดมาแล้ว (ใช้ซ้ำตอนเปิดเมนูอีกครั้ง)

        self._build_window()
        self._build_widgets()
        self.gesture_mouse.set_region(self.X, self.Y, self.W, self.H)

        global _log_hook
        _log_hook = lambda msg: self.ui_q.put(("status", msg))

        self.root.after(30, self._pump)
        self._sys("ระบบพร้อมทำงาน — พิมพ์ข้อความ หรือกดปุ่มไมค์เพื่อพูดคุยกับ JARVIS")
        if not speak:
            self._sys("โหมดไม่อ่านออกเสียง (--no-tts)")
        # ระบบท่ามือไม่เกี่ยวกับ API key: เริ่มก่อนเสมอ แล้วค่อยถามคีย์ (ข้ามได้ ใช้มือคุมเมาส์ต่อได้เลย)
        if gesture:
            self._toggle_gesture(force=True)
        if not OPENROUTER_API_KEY:
            self._sys("ยังไม่มี API key — ควบคุมเมาส์ด้วยมือ/เปิดโปรแกรม/หน้า Help ใช้ได้ตามปกติ "
                      "ส่วนการคุยกับ AI ต้องตั้งค่าคีย์ที่เมนู ⚙ ตั้งค่า")
            self.root.after(1500, self._prompt_api_key)

    # ----- หน้าต่าง -----
    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("Tony Stark Interface")
        self.monitors = list_monitors() or [
            {"x": 0, "y": 0, "w": self.root.winfo_screenwidth(),
             "h": self.root.winfo_screenheight(), "primary": True, "name": "default"}]
        mon = self._initial_monitor()
        self.X, self.Y, self.W, self.H = mon["x"], mon["y"], mon["w"], mon["h"]
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.W}x{self.H}+{self.X}+{self.Y}")
        self.root.configure(bg=KEY_COLOR)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-transparentcolor", KEY_COLOR)
            self.root.attributes("-alpha", 0.95)
        except tk.TclError:
            try:
                self.root.attributes("-alpha", 0.88)   # แพลตฟอร์มที่ไม่รองรับสีโปร่งใส
            except tk.TclError:
                pass
        self.root.bind("<Escape>", lambda e: self._on_escape())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(200, self.root.focus_force)

    # ----- ขนาดตัวอักษร (Font object ชุดเดียวใช้ทั้งแอป เปลี่ยนแล้วทุก widget อัปเดตทันที) -----
    def _make_fonts(self):
        cfg = load_ui_config()
        saved = cfg.get("font_scale")
        self.font_scale = 1.0
        if isinstance(saved, (int, float)):
            self.font_scale = max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, float(saved)))
        self.auto_scale = cfg.get("auto_scale") is not False     # ค่าเริ่มต้น: เปิด
        self.fonts = {}
        for name, (size, weight, family) in FONT_BASE.items():
            self.fonts[name] = tkfont.Font(family=family or FONT_FAMILY, weight=weight,
                                           size=max(6, round(size * self.eff_scale)))
        self._update_preview_size()

    # ----- สเกลของ UI: ขนาดที่ผู้ใช้ตั้ง × ปัจจัยตามความละเอียดจอ (จอ 4K = ×2) -----
    def screen_factor(self):
        """จอสูง 1080px = ×1.0 ; 1440p ≈ ×1.33 ; 4K (2160px) = ×2.0"""
        return max(0.75, min(3.0, self.H / 1080.0))

    @property
    def eff_scale(self):
        raw = self.font_scale * (self.screen_factor() if self.auto_scale else 1.0)
        return min(raw, 5.0, self.W / 640.0)     # เพดาน: ไม่ให้ใหญ่จนส่วนหัว/ปุ่มล้นความกว้างจอ

    def px(self, n):
        """ขนาดพิกเซลที่ปรับตามสเกลของ UI"""
        return int(round(n * self.eff_scale))

    def _apply_scale(self):
        """เรียกเมื่อสเกลเปลี่ยน (ผู้ใช้ปรับ / ย้ายไปจอที่ความละเอียดต่างกัน)"""
        for name, (size, _w, _f) in FONT_BASE.items():
            self.fonts[name].configure(size=max(6, round(size * self.eff_scale)))
        self._update_preview_size()
        if getattr(self, "status_label", None) is not None:
            self._layout()

    def _update_preview_size(self):
        k = min(self.eff_scale, 2.2)          # ภาพกล้องโตตามสเกลแต่มีเพดาน ไม่ให้กินพื้นที่แผงแชท
        self.gesture_mouse.set_preview_size(PREVIEW_W * k, PREVIEW_H * k)
        if getattr(self, "cam_label", None) is not None and self.photo is None:
            text = self.cam_label.cget("text")
            if text:
                self._cam_placeholder(text, self.cam_label.cget("fg"))

    def set_font_scale(self, scale):
        self.font_scale = max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, float(scale)))
        self._apply_scale()

    def set_auto_scale(self, flag):
        self.auto_scale = bool(flag)
        self._apply_scale()

    def save_click_cfg(self):
        cfg = load_ui_config()
        cfg["click_cfg"] = dict(self.gesture_mouse.cfg)
        save_ui_config(cfg)

    def save_llm_model(self, model):
        cfg = load_ui_config()
        cfg["llm_model"] = model
        save_ui_config(cfg)

    def save_mouse_speed(self):
        cfg = load_ui_config()
        cfg["mouse_speed"] = round(self.gesture_mouse.speed, 2)
        save_ui_config(cfg)

    def save_font_scale(self):
        cfg = load_ui_config()
        cfg["font_scale"] = round(self.font_scale, 2)
        cfg["auto_scale"] = bool(self.auto_scale)
        save_ui_config(cfg)

    # ----- เลือก / ย้ายไปจอภาพอื่น -----
    def _initial_monitor(self):
        """ลำดับ: --monitor N > จอที่บันทึกไว้ครั้งก่อน (ตามชื่ออุปกรณ์) > จอหลัก"""
        if self.monitor_arg and 1 <= self.monitor_arg <= len(self.monitors):
            return self.monitors[self.monitor_arg - 1]
        saved = load_ui_config().get("monitor_name")
        for m in self.monitors:
            if saved and m["name"] == saved:
                return m
        return self.monitors[0]

    def _layout(self):
        """จัดตำแหน่ง widget ตามขนาดของจอที่เลือก"""
        self.root.update_idletasks()
        # แผงแชทกว้างขึ้นตามขนาดตัวอักษร (ให้ปุ่มด้านล่างไม่ล้น) แต่ไม่ทับแผงกล้องมุมซ้าย
        desired = int(980 * max(1.0, self.eff_scale))
        margin, top = self.px(24), self.px(30)
        left_min = self.cam_panel.winfo_reqwidth() + margin + self.px(20)
        left = max((self.W - desired) // 2, left_min)
        pw = max(self.px(400), min(desired, self.W - left - margin))
        ph = max(self.px(300), self.H - top - self.px(120))
        self._lay = (left, pw, ph)
        self._place_panels()
        # ถ้าช่องพิมพ์ + ปุ่มทั้งหมดไม่พอในแถวเดียว ให้แยกปุ่มลงแถวที่สอง
        self._arrange_bar(False)
        self.root.update_idletasks()
        need = (sum(b.winfo_reqwidth() + 8 for b in self.bar_buttons) + self.px(300) + 40)
        self._arrange_bar(need > pw - 24)
        # relx=0 ต้องระบุชัด: place_configure ไม่ล้างค่า relx เดิม (0.5) ที่เคยตั้งไว้
        self.corner.place_configure(relx=0, x=self.W - margin, y=top, anchor="ne")
        self._place_mini_pin()

    def _arrange_bar(self, compact):
        """แถบพิมพ์: ปกติช่องพิมพ์+ปุ่มอยู่แถวเดียว แต่ถ้าตัวอักษรใหญ่จนไม่พอ ให้ปุ่มลงไปแถวที่สอง"""
        if compact == self._bar_compact:
            return
        self._bar_compact = compact
        widgets = [self.entry] + self.bar_buttons
        for w in widgets:
            w.grid_forget()
        n = len(self.bar_buttons)
        for c in range(n + 1):
            self.bar.columnconfigure(c, weight=0)
        if not compact:
            self.entry.grid(row=0, column=0, sticky="ew", padx=(8, 8), pady=8, ipady=8)
            self.bar.columnconfigure(0, weight=1)
            for i, b in enumerate(self.bar_buttons):
                b.grid(row=0, column=i + 1, padx=(0, 8), pady=8)
        else:
            self.entry.grid(row=0, column=0, columnspan=n, sticky="ew", padx=8, pady=(8, 4), ipady=8)
            for i, b in enumerate(self.bar_buttons):
                self.bar.columnconfigure(i, weight=1)
                b.grid(row=1, column=i, sticky="ew", padx=(8 if i == 0 else 0, 8), pady=(0, 8))

    def _place_panels(self):
        """วางแผงแชท/แผงกล้อง = ตำแหน่งตามเลย์เอาต์ + ระยะที่ผู้ใช้ลากไว้"""
        left, pw, ph = self._lay
        margin, top = self.px(24), self.px(30)
        cdx, cdy = self.offsets["chat"]
        self.chat_panel.place_configure(relx=0, x=left + cdx, y=top + cdy, anchor="nw",
                                        width=pw, height=ph)
        self.status_label.place_configure(relx=0, x=left + cdx, y=top + cdy + ph + self.px(20),
                                          anchor="nw", width=pw, height=self.px(26))
        kdx, kdy = self.offsets["cam"]
        self.cam_panel.place_configure(relx=0, x=margin + kdx, y=top + kdy)

    # ----- ลากย้ายแผง (ลากที่แถบหัวของแผงแชท หรือที่แผงกล้อง; ดับเบิลคลิก = กลับตำแหน่งเดิม) -----
    def _make_draggable(self, grips, key):
        for w in grips:
            w.bind("<ButtonPress-1>", lambda e, k=key: self._drag_start(e, k), add="+")
            w.bind("<B1-Motion>", lambda e, k=key: self._drag_move(e), add="+")
            w.bind("<ButtonRelease-1>", lambda e: self._drag_end(), add="+")
            w.bind("<Double-Button-1>", lambda e, k=key: self._drag_reset(k), add="+")
            try:
                w.config(cursor="fleur")
            except tk.TclError:
                pass

    def _drag_start(self, event, key):
        self._drag = (key, event.x_root, event.y_root, tuple(self.offsets[key]))

    def _drag_move(self, event):
        if not self._drag:
            return
        key, sx, sy, (ox, oy) = self._drag
        left, pw, ph = self._lay
        margin, top = self.px(24), self.px(30)
        panel, base_x, base_y = ((self.chat_panel, left, top) if key == "chat"
                                 else (self.cam_panel, margin, top))
        width = panel.winfo_width() or pw
        nx = base_x + ox + (event.x_root - sx)
        ny = base_y + oy + (event.y_root - sy)
        nx = max(-(width - self.px(160)), min(self.W - self.px(160), nx))   # ต้องเหลือส่วนหัวให้จับลากกลับได้เสมอ
        ny = max(0, min(self.H - self.px(80), ny))
        self.offsets[key] = [nx - base_x, ny - base_y]
        self._place_panels()

    def _drag_end(self):
        if self._drag:
            self._drag = None
            self._save_offsets()

    def _drag_reset(self, key):
        self._drag = None
        self.offsets[key] = [0, 0]
        self._place_panels()
        self._save_offsets()

    def _save_offsets(self):
        cfg = load_ui_config()
        cfg["panel_offsets"] = {k: [int(v[0]), int(v[1])] for k, v in self.offsets.items()}
        save_ui_config(cfg)

    # ----- ปักหมุด / ปลดหมุดหน้าจอ interface -----
    def _toggle_pin(self):
        if self.pinned:
            self._unpin()
        else:
            self._repin()

    def _unpin(self):
        """ปลดหมุด: interface ไม่ลอยทับโปรแกรมอื่นอีก (ลงไปอยู่ล่างสุด) คลิกโปรแกรมอื่นได้ตามปกติ
        จะมีปุ่ม 📌 เล็กๆ ลอยอยู่มุมจอ ไว้กดเรียก interface กลับมา"""
        self.pinned = False
        self.pin_btn.config(text="📌 Pin")
        self.root.attributes("-topmost", False)
        self.root.lower()
        self._show_mini_pin()
        self._sys("ปลดหมุดแล้ว — กดปุ่ม 📌 มุมจอเพื่อเรียก interface กลับมาไว้บนสุด")

    def _repin(self):
        self.pinned = True
        self.pin_btn.config(text="📌 Unpin")
        self._hide_mini_pin()
        self.root.attributes("-topmost", True)
        self.root.lift()
        self.root.after(100, self.root.focus_force)

    def _show_mini_pin(self):
        if self.mini_pin is None:
            win = tk.Toplevel(self.root)
            win.overrideredirect(True)
            win.configure(bg=C_CYAN)
            tk.Button(win, text="📌 Pin", command=self._repin, font=self.fonts["bold"],
                      fg=C_CYAN, bg=C_PANEL, activebackground=C_CYAN, activeforeground=C_PANEL,
                      bd=0, padx=14, pady=6, cursor="hand2").pack(padx=2, pady=2)
            self.mini_pin = win
        self.mini_pin.deiconify()
        self.mini_pin.attributes("-topmost", True)   # ปุ่มนี้ลอยบนสุดเสมอ แม้ interface หลักถูกปลดหมุด
        self._place_mini_pin()

    def _hide_mini_pin(self):
        if self.mini_pin is not None:
            self.mini_pin.withdraw()

    def _place_mini_pin(self):
        if self.mini_pin is None:
            return
        self.mini_pin.update_idletasks()
        w = self.mini_pin.winfo_reqwidth()
        self.mini_pin.geometry(f"+{self.X + self.W - w - self.px(24)}+{self.Y + self.px(30)}")

    def _apply_monitor(self, mon, announce=True):
        resolution_changed = (mon["w"], mon["h"]) != (self.W, self.H)
        self.X, self.Y, self.W, self.H = mon["x"], mon["y"], mon["w"], mon["h"]
        self.root.geometry(f"{self.W}x{self.H}+{self.X}+{self.Y}")
        if resolution_changed:
            self.offsets = {"chat": [0, 0], "cam": [0, 0]}   # ตำแหน่งที่ลากไว้ใช้กับจอเดิม ไม่ใช้ต่อ
            self._save_offsets()
        self._apply_scale()          # ตัวอักษร/ขนาดแผงปรับตามความละเอียดของจอใหม่ (เช่น 4K = ×2)
        self.gesture_mouse.set_region(self.X, self.Y, self.W, self.H)
        cfg = load_ui_config()
        cfg["monitor_name"] = mon["name"]
        save_ui_config(cfg)
        self.root.after(100, self.root.focus_force)
        if announce:
            idx = self.monitors.index(mon)
            extra = f" · สเกล UI ×{self.eff_scale:.2f}" if self.eff_scale != 1 else ""
            self._sys(f"ย้ายหน้าจอไปที่ {monitor_label(idx, mon)}{extra}")

    def _current_monitor_index(self):
        for i, m in enumerate(self.monitors):
            if (m["x"], m["y"], m["w"], m["h"]) == (self.X, self.Y, self.W, self.H):
                return i
        return 0

    def _show_monitor_menu(self):
        """เมนูเลือกจอ — ตรวจจอใหม่ทุกครั้งที่กด จึงเสียบ XREAL ทีหลังแล้วกดเลือกได้เลย"""
        fresh = list_monitors()
        if fresh:
            self.monitors = fresh
        current = self._current_monitor_index()

        menu = tk.Menu(self.root, tearoff=0, bg=C_PANEL2, fg=C_TEXT, activebackground=C_CYAN,
                       activeforeground=C_PANEL, font=self.fonts["body"], bd=0)
        choice = tk.IntVar(value=current)
        for i, m in enumerate(self.monitors):
            menu.add_radiobutton(label=monitor_label(i, m), variable=choice, value=i,
                                 command=lambda m=m: self._apply_monitor(m))
        menu.add_separator()
        menu.add_command(label="↻ ตรวจหาจอใหม่", command=self._show_monitor_menu_later)
        b = self.monitor_btn
        try:
            menu.tk_popup(b.winfo_rootx(), b.winfo_rooty() + b.winfo_height())
        finally:
            menu.grab_release()

    # ----- หน้า Help (เปิดด้วยปุ่ม ❓ หรือโชว์มือ 10 นิ้ว) -----
    def _toggle_help(self):
        if self.help_frame is not None:
            self._close_help()
            return

        f_title, f_body, f_small = self.fonts["title"], self.fonts["body"], self.fonts["small"]

        frame = tk.Frame(self.root, bg=C_PANEL, highlightthickness=2,
                         highlightbackground=C_CYAN, highlightcolor=C_CYAN)
        head = tk.Frame(frame, bg=C_PANEL2)
        head.pack(fill="x")
        tk.Label(head, text="❓  HELP — คำสั่งควบคุมเมาส์ด้วยมือ", font=f_title, fg=C_CYAN,
                 bg=C_PANEL2).pack(side="left", padx=16, pady=10)
        tk.Button(head, text="✕ ปิด", font=f_small, fg=C_ERR, bg=C_PANEL2, bd=0,
                  activebackground=C_ERR, activeforeground=C_PANEL, cursor="hand2",
                  command=self._close_help).pack(side="right", padx=12)

        # ภาพ gestures.svg (ย่อ/ขยายให้พอดีกับจอที่เลือก)
        avail_w = min(self.W - self.px(120), self.px(1000))
        avail_h = max(self.px(240), self.H - self.px(400))   # เหลือที่ให้ชื่อ/คำอธิบายที่ขยายตามสเกลแล้ว
        scale = max(0.5, min(avail_w / 680.0, avail_h / 480.0, 1.6 * max(1.0, self.eff_scale)))
        canvas = tk.Canvas(frame, bg=C_PANEL, highlightthickness=0,
                           width=int(680 * scale), height=int(480 * scale))
        canvas.pack(padx=20, pady=(12, 4))
        try:
            draw_svg_on_canvas(canvas, GESTURES_SVG, scale,
                               text_scale=1.0 if self.font_scale <= 1 else min(1.5, self.font_scale))
        except Exception as exc:
            canvas.config(height=60)
            canvas.create_text(10, 30, anchor="w", fill=C_ERR, font=f_body,
                               text=f"โหลดภาพ gestures.svg ไม่ได้: {exc}")

        grid = tk.Frame(frame, bg=C_PANEL)
        grid.pack(padx=24, pady=(4, 6), fill="x")
        for i, (gesture, action) in enumerate(HELP_LINES):
            r, c = divmod(i, 2)
            tk.Label(grid, text=f"{gesture}  →  {action}", font=f_small, fg=C_TEXT, bg=C_PANEL,
                     anchor="w").grid(row=r, column=c, sticky="w", padx=12, pady=1)
        clap_ok = self.clap.start()      # ฟังเสียงปรบมือเฉพาะตอนหน้า Help เปิดอยู่
        hint = "👏 ปรบมือ 1 ครั้ง · โชว์มือ 10 นิ้วอีกครั้ง · หรือกด ESC เพื่อปิดหน้านี้"
        if not clap_ok:
            hint = "โชว์มือ 10 นิ้วอีกครั้ง หรือกด ESC เพื่อปิดหน้านี้   (ปรบมือปิดไม่ได้: ไมค์ไม่พร้อม)"
        tk.Label(frame, text=hint, font=f_small, fg=C_DIM, bg=C_PANEL).pack(pady=(2, 12))

        frame.place(relx=0.5, rely=0.5, anchor="center")
        frame.lift()
        self.help_frame = frame

    def _close_help(self):
        self.clap.stop()
        if self.help_frame is not None:
            self.help_frame.destroy()
            self.help_frame = None

    def _on_escape(self):
        if self.help_frame is not None:
            self._close_help()
        else:
            self.close()

    def _show_monitor_menu_later(self):
        self.root.after(50, self._show_monitor_menu)

    def _build_widgets(self):
        self._make_fonts()
        f_title, f_body = self.fonts["title"], self.fonts["body"]
        f_bold, f_small, f_mono = self.fonts["bold"], self.fonts["small"], self.fonts["mono"]

        # ===== แผงแชทกลางจอ =====
        self.chat_panel = tk.Frame(self.root, bg=C_PANEL, highlightthickness=2,
                                   highlightbackground=C_CYAN, highlightcolor=C_CYAN)
        self.chat_panel.place(relx=0.5, y=30, anchor="n")   # ขนาดจริงตั้งใน _layout()

        header = tk.Frame(self.chat_panel, bg=C_PANEL2)
        header.pack(fill="x")
        tk.Label(header, text="◉  J.A.R.V.I.S", font=f_title, fg=C_CYAN, bg=C_PANEL2
                 ).pack(side="left", padx=16, pady=10)
        self.state_label = tk.Label(header, text="● STANDBY", font=f_bold, fg=C_DIM, bg=C_PANEL2)
        self.state_label.pack(side="right", padx=16)
        tk.Button(header, text="ล้างแชท", font=f_small, fg=C_TEXT, bg=C_PANEL2, bd=0,
                  activebackground=C_PANEL, activeforeground=C_CYAN, cursor="hand2",
                  command=self._clear_chat).pack(side="right", padx=4)
        self.monitor_btn = tk.Button(header, text="🖥 จอภาพ", font=f_small, fg=C_CYAN,
                                     bg=C_PANEL2, bd=0, activebackground=C_PANEL,
                                     activeforeground=C_CYAN, cursor="hand2",
                                     command=self._show_monitor_menu)
        self.monitor_btn.pack(side="right", padx=4)
        tk.Button(header, text="❓ Help", font=f_small, fg=C_CYAN, bg=C_PANEL2, bd=0,
                  activebackground=C_PANEL, activeforeground=C_CYAN, cursor="hand2",
                  command=self._toggle_help).pack(side="right", padx=4)

        body = tk.Frame(self.chat_panel, bg=C_PANEL)
        body.pack(fill="both", expand=True, padx=12, pady=8)
        scroll = tk.Scrollbar(body, troughcolor=C_PANEL, bg=C_PANEL2, bd=0)
        scroll.pack(side="right", fill="y")
        self.chat = tk.Text(body, wrap="word", bg=C_PANEL, fg=C_TEXT, font=f_body,
                            height=4, bd=0, highlightthickness=0, padx=10, pady=6, spacing3=6,
                            insertbackground=C_CYAN, yscrollcommand=scroll.set,
                            state="disabled", cursor="arrow")
        self.chat.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.chat.yview)
        self.chat.tag_config("who_user", foreground=C_USER, font=f_bold)
        self.chat.tag_config("who_ai", foreground=C_CYAN, font=f_bold)
        self.chat.tag_config("user", foreground=C_TEXT, lmargin1=14, lmargin2=14)
        self.chat.tag_config("ai", foreground=C_TEXT, lmargin1=14, lmargin2=14)
        self.chat.tag_config("sys", foreground=C_DIM, font=f_small)
        self.chat.tag_config("err", foreground=C_ERR, font=f_small)

        # แถบพิมพ์ข้อความ
        # pack ก่อนพื้นที่แชท (before=body) เพื่อให้ช่องพิมพ์/ปุ่มได้พื้นที่ก่อนเสมอ
        # ขยายตัวอักษรแล้วจะได้ไม่ถูกดันตกขอบล่างจนหาย
        bar = tk.Frame(self.chat_panel, bg=C_PANEL2)
        bar.pack(fill="x", side="bottom", before=body, padx=12, pady=(0, 12))
        self.entry = tk.Entry(bar, font=f_body, bg=C_PANEL, fg=C_TEXT, bd=0,
                              insertbackground=C_CYAN, highlightthickness=1,
                              highlightbackground=C_DIM, highlightcolor=C_CYAN)
        self.entry.bind("<Return>", lambda e: self._send())
        self.entry.focus_set()
        self.bar = bar
        self.bar_buttons = []
        self._bar_compact = None

        def btn(text, cmd, fg=C_CYAN):
            b = tk.Button(bar, text=text, command=cmd, font=f_bold, fg=fg, bg=C_PANEL,
                          activebackground=C_CYAN, activeforeground=C_PANEL, bd=0,
                          padx=14, pady=6, cursor="hand2")
            self.bar_buttons.append(b)
            return b

        btn("ส่ง ➤", self._send)
        self.mic_btn = btn("🎙 เปิดไมค์", self._toggle_voice)
        btn("⚙ ตั้งค่า", self._open_mic_menu, fg=C_TEXT)
        btn("■ หยุด", self._stop_all, fg=C_ERR)
        self._arrange_bar(False)

        # ===== แผงกล้อง/ท่ามือ มุมซ้ายบน =====
        self.cam_panel = tk.Frame(self.root, bg=C_PANEL, highlightthickness=2,
                                  highlightbackground=C_CYAN)
        self.cam_panel.place(x=24, y=30)
        self.blank_img = tk.PhotoImage(width=1, height=1)   # ภาพว่าง: ทำให้ width/height ของ Label เป็นพิกเซล
        self.cam_label = tk.Label(self.cam_panel, bg=C_PANEL, fg=C_DIM, font=f_small,
                                  image=self.blank_img, compound="center")
        self.cam_label.pack(padx=6, pady=(6, 0))
        self._cam_placeholder("ระบบควบคุมเมาส์ปิดอยู่")
        self.gesture_label = tk.Label(self.cam_panel, text="GESTURE: OFF", font=f_mono,
                                      fg=C_CYAN, bg=C_PANEL, anchor="w")
        self.gesture_label.pack(fill="x", padx=8)
        self.action_label = tk.Label(self.cam_panel, text="ACTION : -", font=f_mono,
                                     fg=C_TEXT, bg=C_PANEL, anchor="w")
        self.action_label.pack(fill="x", padx=8)
        self.gesture_btn = tk.Button(self.cam_panel, text="เปิดควบคุมเมาส์ด้วยมือ",
                                     command=self._toggle_gesture, font=f_small, fg=C_CYAN,
                                     bg=C_PANEL2, activebackground=C_CYAN,
                                     activeforeground=C_PANEL, bd=0, pady=4, cursor="hand2")
        self.gesture_btn.pack(fill="x", padx=8, pady=8)

        # ===== ปุ่มปิด + แถบสถานะ =====
        self.corner = tk.Frame(self.root, bg=KEY_COLOR)         # มุมขวาบน: ปักหมุด + ปิด
        self.corner.place(x=self.W - 24, y=30, anchor="ne")
        self.pin_btn = tk.Button(self.corner, text="📌 Unpin", command=self._toggle_pin,
                                 font=f_bold, fg=C_CYAN, bg=C_PANEL, activebackground=C_CYAN,
                                 activeforeground=C_PANEL, bd=0, padx=12, pady=4, cursor="hand2")
        self.pin_btn.pack(side="left", padx=(0, 8))
        self.close_btn = tk.Button(self.corner, text="✕", command=self.close, font=f_bold,
                                   fg=C_ERR, bg=C_PANEL, activebackground=C_ERR,
                                   activeforeground=C_PANEL, bd=0, padx=12, pady=4,
                                   cursor="hand2")
        self.close_btn.pack(side="left")
        self.status_label = tk.Label(self.root, text="", font=f_small, fg=C_DIM, bg=C_PANEL,
                                     anchor="w", padx=12)
        self.status_label.place(x=0, y=self.H - 100, anchor="nw")
        self._layout()

        # ลากย้ายได้: แถบหัวแผงแชท (ส่วนที่ไม่ใช่ปุ่ม) และแผงกล้อง
        self._make_draggable([header] + [c for c in header.winfo_children()
                                         if isinstance(c, tk.Label)], "chat")
        self._make_draggable([self.cam_panel, self.cam_label, self.gesture_label,
                              self.action_label], "cam")

    # ----- ช่วยเขียนลงแชท (เรียกจากเธรด UI เท่านั้น) -----
    def _append(self, text, tag):
        self.chat.config(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _sys(self, text):
        self._append(text + "\n", "sys")

    def _error(self, text):
        self._append("⚠ " + text + "\n", "err")

    def _clear_chat(self):
        self.assistant.interrupt()
        self.assistant.history.clear()
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self._sys("ล้างประวัติการสนทนาแล้ว")

    def _set_state(self, name):
        self.state = name
        self.state_label.config(text=f"● {name}", fg=STATE_COLORS.get(name, C_DIM))

    # ----- ตัวดึงเหตุการณ์จากเธรดอื่น -----
    def _pump(self):
        if self.closing:
            return
        try:
            for _ in range(80):
                kind, payload = self.ui_q.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        # เสียงพูดจบแล้ว (และไม่ได้กำลังคิด) -> กลับสู่สถานะพัก
        if self.state == "SPEAKING" and self.speech.idle.is_set() and not self.ai_open:
            self._set_state("LISTENING" if self.voice_on.is_set() else "STANDBY")
        self._update_gesture_panel()
        self.root.after(30, self._pump)

    def _handle(self, kind, payload):
        if kind == "user":
            self._append("คุณ\n", "who_user")
            self._append(payload + "\n", "user")
        elif kind == "ai_start":
            self._append("JARVIS\n", "who_ai")
            self.ai_open = True
        elif kind == "ai_delta":
            self._append(payload, "ai")
        elif kind == "ai_end":
            if self.ai_open:
                self._append("\n", "ai")
                self.ai_open = False
            # ตอบเสร็จ: กลับสู่สถานะของโหมดไมค์ (ถ้ายังมีเสียงพูดค้างอยู่ SpeechStreamer จะเปลี่ยนเป็น SPEAKING)
            if self.speech.idle.is_set():
                self._set_state("LISTENING" if self.voice_on.is_set() else "STANDBY")
        elif kind == "state":
            self._set_state(payload)
        elif kind == "sys":
            self._sys(payload)
        elif kind == "error":
            self._error(payload)
        elif kind == "status":
            self.status_label.config(text=payload[:160])
        elif kind == "mic":
            self.mic_btn.config(text=payload)
        elif kind == "need_key":
            self._prompt_api_key()
        elif kind == "models":
            self.model_cache = payload
            if self.mic_menu is not None:
                self.mic_menu.set_models(payload)
        elif kind == "models_error":
            if self.mic_menu is not None:
                self.mic_menu.set_models_error(payload)
        elif kind == "clap":
            if self.help_frame is not None:          # ปรบมือ 1 ครั้ง = ปิดหน้า Help
                self._close_help()
                self._sys("👏 ตรวจพบเสียงปรบมือ — ปิดหน้า Help")
        elif kind == "quit":
            self.close()

    # ----- ปุ่มต่างๆ -----
    def _send(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        threading.Thread(target=self.assistant.handle_text, args=(text,), daemon=True).start()

    def _stop_all(self):
        self.assistant.interrupt()
        self._set_state("LISTENING" if self.voice_on.is_set() else "STANDBY")

    def _toggle_voice(self):
        if self.voice_on.is_set():
            self.voice_on.clear()
            self.mic_btn.config(text="🎙 กำลังปิด...")
            return
        if sd is None:
            self._error(f"ใช้ไมโครโฟนไม่ได้ (sounddevice): {_SD_IMPORT_ERROR}")
            return
        if not OPENROUTER_API_KEY:                 # ถอดเสียงต้องใช้คีย์ (ท่ามือ/พิมพ์สั่งเปิดโปรแกรมไม่ต้อง)
            self._error("สั่งด้วยเสียงต้องมี API key ก่อน (ใช้ถอดเสียง) — ตั้งค่าได้เลย")
            self._prompt_api_key()
            return
        if self.voice_thread is not None and self.voice_thread.is_alive():
            return    # รอบก่อนยังปิดไม่เสร็จ
        self.voice_on.set()
        self.mic_btn.config(text="🎙 กำลังฟัง (กดเพื่อปิด)")
        self.voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
        self.voice_thread.start()

    def _cam_placeholder(self, text, color=None):
        """ข้อความแทนภาพกล้อง ขนาดเท่าภาพ preview (พิกเซล) จึงไม่โตเกินจริงเมื่อตัวอักษรใหญ่"""
        w, h = self.gesture_mouse.preview_size
        self.cam_label.config(image=self.blank_img, compound="center", text=text,
                              fg=color or C_DIM, width=w, height=h,
                              wraplength=max(120, w - 20))

    def _toggle_gesture(self, force=False):
        if self.gesture_mouse.running and not force:
            self.gesture_mouse.stop()
            self.gesture_btn.config(text="เปิดควบคุมเมาส์ด้วยมือ")
            self.photo = None
            self.last_seq = -1
            self._cam_placeholder("ระบบควบคุมเมาส์ปิดอยู่")
            return
        self.gesture_mouse.start()
        self.gesture_btn.config(text="ปิดควบคุมเมาส์ด้วยมือ")
        self._cam_placeholder("กำลังเปิดกล้อง...")

    def _update_gesture_panel(self):
        snap = self.gesture_mouse.snapshot()
        if snap["help_seq"] != self.last_help_seq:       # โชว์มือ 10 นิ้ว -> สลับหน้า Help
            self.last_help_seq = snap["help_seq"]
            self._toggle_help()

        if not self.gesture_mouse.running:
            if snap["error"]:
                self._cam_placeholder(snap["error"], C_ERR)
                self.gesture_label.config(text="GESTURE: ERROR")
                self.gesture_btn.config(text="เปิดควบคุมเมาส์ด้วยมือ")
                self.gesture_mouse._publish(error="")   # แสดงครั้งเดียว
                self._error(f"ระบบท่ามือ: {snap['error']} (ติดตั้ง: pip install opencv-contrib-python==4.11.0.86 mediapipe==0.10.21 pyautogui)")
            return
        if snap["png"] is not None and snap["seq"] != self.last_seq:
            self.last_seq = snap["seq"]
            try:
                self.photo = tk.PhotoImage(data=snap["png"])
                self.cam_label.config(image=self.photo, text="", width=self.photo.width(),
                                      height=self.photo.height(), compound="none")
            except tk.TclError:
                pass
        self.gesture_label.config(
            text=f"GESTURE: {snap['gesture']}" + ("  [PAUSED]" if snap["paused"] else ""),
            fg=C_ERR if snap["paused"] else C_CYAN)
        self.action_label.config(text=f"ACTION : {snap['action']}")

    # ----- เกณฑ์เสียงพูด: CLI > เมนู(กำหนดเอง) > env > วัดเสียงห้องอัตโนมัติ -----
    def _current_threshold(self):
        if self.threshold is not None:
            return self.threshold
        if not MIC_CFG["auto"]:
            return MIC_CFG["threshold"]
        if FIXED_THRESHOLD > 0:
            return FIXED_THRESHOLD
        if self.auto_threshold is None:
            self.ui_q.put(("state", "CALIBRATING"))
            log("🤫 กำลังวัดเสียงรบกวนของห้อง กรุณาเงียบสักครู่...")
            noise = measure_noise_floor()
            self.auto_threshold = compute_threshold(noise)
            log(f"🎚️  เสียงรบกวน ≈ {noise:.0f} → เกณฑ์เสียงพูด {self.auto_threshold:.0f}")
        return self.auto_threshold

    def _prompt_api_key(self):
        if self.api_dialog is None and not self.closing and not OPENROUTER_API_KEY:
            self.api_dialog = ApiKeyDialog(self)

    def _open_mic_menu(self):
        if self.mic_menu is not None:
            return
        self.mic_menu = MicMenu(self)

    # ----- ลูปเสียง (เธรดแยก) -----
    def _voice_loop(self):
        ui = self.ui_q.put
        try:
            turn = 0
            errors = 0
            while self.voice_on.is_set():
                # รอให้ JARVIS พูดจบก่อน ไม่งั้นไมค์จะอัดเสียงตัวเองกลับเข้ามา
                self.speech.wait_idle(should_stop=lambda: not self.voice_on.is_set())
                if not self.voice_on.is_set():
                    break
                time.sleep(0.4)

                _ensure_work_dir()
                wav = os.path.join(WORK_DIR, f"user_{os.getpid()}_{turn}.wav")
                turn += 1
                try:
                    threshold = self._current_threshold()   # อ่านใหม่ทุกรอบ รับค่าจากเมนูทันที
                    ui(("state", "LISTENING"))
                    path = record_audio_vad(wav, threshold=threshold,
                                            should_abort=lambda: not self.voice_on.is_set())
                    if not path:
                        continue
                    ui(("state", "TRANSCRIBING"))
                    text = speech_to_text(path)
                    errors = 0
                finally:
                    if not KEEP_AUDIO and os.path.exists(wav):
                        try:
                            os.remove(wav)
                        except OSError:
                            pass

                if not text:
                    errors += 1
                    if errors >= 3:
                        ui(("error", "ถอดเสียงไม่สำเร็จติดกัน 3 ครั้ง จึงปิดไมค์"))
                        break
                    continue

                self.assistant.handle_text(text)
        except Exception as exc:
            ui(("error", f"ระบบไมค์ขัดข้อง: {exc}"))
        finally:
            self.voice_on.clear()
            ui(("mic", "🎙 เปิดไมค์"))
            ui(("state", "STANDBY"))

    # ----- ปิดโปรแกรม -----
    def close(self):
        if self.closing:
            return
        self.closing = True
        global _log_hook
        _log_hook = None
        if self.mic_menu is not None:
            self.mic_menu.close()
        self.clap.stop()
        self.voice_on.clear()
        self.assistant.interrupt()
        try:
            self.gesture_mouse.stop()
        except Exception:
            pass
        try:
            if sd is not None:
                sd.stop()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


# ==========================================================================
# 12. ตรวจความพร้อมของระบบ + Entry point
# ==========================================================================
def preflight_check(need_mic=True):
    log("🔍 ตรวจสอบความพร้อมของระบบ")
    log("-" * 50)
    ready = True

    if OPENROUTER_API_KEY:
        log(f"  ✅ API Key      : {mask_key(OPENROUTER_API_KEY)} (จาก {API_KEY_SOURCE})")
    else:
        log("  ❌ API Key      : ไม่พบ (ตั้ง OPENROUTER_API_KEY หรือสร้างไฟล์ k.py)")
        ready = False
    log(f"  🧠 โมเดลแชท     : {LLM_MODEL}")

    try:
        import edge_tts  # noqa: F401
        log(f"  ✅ edge-tts     : พร้อมใช้งาน (เสียง {TTS_VOICE})")
    except ImportError:
        log("  ❌ edge-tts     : ไม่พบ (pip install edge-tts) — จะไม่มีเสียงตอบกลับ")
        ready = False

    players = (["mci"] if IS_WINDOWS else []) + (["ffplay"] if shutil.which("ffplay") else [])
    if players:
        log(f"  ✅ ตัวเล่นเสียง  : {', '.join(players)}")
    else:
        log("  ❌ ตัวเล่นเสียง  : ไม่พบ (ติดตั้ง ffmpeg เพื่อใช้ ffplay)")
        ready = False

    if need_mic:
        if sd is None:
            log(f"  ❌ ไมโครโฟน     : sounddevice ใช้งานไม่ได้ ({_SD_IMPORT_ERROR})")
            ready = False
        else:
            try:
                log(f"  ✅ ไมโครโฟน     : {sd.query_devices(kind='input')['name']}")
            except Exception as exc:
                log(f"  ❌ ไมโครโฟน     : ไม่พบอุปกรณ์อินพุต ({exc})")
                ready = False

    for mod in ("cv2", "mediapipe", "pyautogui"):
        try:
            __import__(mod)
            log(f"  ✅ {mod:<13}: พร้อมใช้งาน")
        except Exception as exc:
            log(f"  ❌ {mod:<13}: ใช้ไม่ได้ ({exc}) — ระบบควบคุมเมาส์ด้วยมือจะไม่ทำงาน")
            ready = False

    for i, m in enumerate(list_monitors()):
        log(f"  🖥  {monitor_label(i, m)}")
    log(f"  📁 โฟลเดอร์งาน  : {WORK_DIR}")
    log("-" * 50)
    return ready


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Tony Stark Interface (JARVIS + Gesture Mouse)")
    parser.add_argument("--check", action="store_true", help="ตรวจความพร้อมของระบบแล้วจบ")
    parser.add_argument("--no-gesture", action="store_true", help="ไม่เปิดระบบควบคุมเมาส์ด้วยมือ")
    parser.add_argument("--no-tts", action="store_true", help="ไม่อ่านออกเสียง")
    parser.add_argument("--monitor", type=int, help="หมายเลขจอที่จะแสดง interface (เริ่มที่ 1, ดูได้จาก --check)")
    parser.add_argument("--threshold", type=float, help="กำหนดเกณฑ์เสียงพูดเอง (ปิด auto-calibrate)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    log("=" * 50)
    log("TONY STARK INTERFACE")
    log("=" * 50)

    if args.check:
        return 0 if preflight_check() else 1

    preflight_check()   # แสดงผลในคอนโซลเฉยๆ ไม่บล็อกการเปิดหน้าจอ

    app = TonyStarkUI(speak=not args.no_tts, gesture=not args.no_gesture,
                      threshold=args.threshold, monitor=args.monitor)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
