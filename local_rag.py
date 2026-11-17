"""Multi-patient Local RAG, PII pseudonymization/re-hydration, and Gemini 3.8 Flash + deterministic fallback engine."""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta
import json
import os
import re
import socket
import threading
import time
from typing import Any

try:
    import urllib3.util.connection as _urllib3_conn

    _urllib3_conn.allowed_gai_family = lambda: socket.AF_INET
except Exception:  # pragma: no cover
    pass

import requests
import requests.adapters
from dotenv import load_dotenv

try:
    import urllib3.connection as _urllib3_c

    _urllib3_c.allowed_gai_family = lambda: socket.AF_INET  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from state_manager import state_manager
from telemetry_provider import BaseTelemetryProvider, get_telemetry_provider

load_dotenv()

try:
    from google import genai
    from google.genai import types

    _GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    _GENAI_AVAILABLE = False


def _parse_bool_env(value: str | None, default: bool = False) -> bool:
    """Parse boolean environment variables safely."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: str = (
    os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
)
OFFLINE_FALLBACK_MODE: bool = _parse_bool_env(os.getenv("OFFLINE_FALLBACK_MODE"), default=False)

PATIENT_TOKEN = "[PATIENT_1429]"
PATIENT_FIRSTNAME_TOKEN = "[PATIENT_1429_FIRSTNAME]"
LOCATION_TOKEN = "[LOCATION_REDACTED]"

_AFFIRMATIVE_PATTERNS = (
    "yes",
    "yeah",
    "yep",
    "yup",
    "sure",
    "please",
    "ok",
    "okay",
    "go ahead",
    "do it",
    "him",
    "her",
    "that's him",
    "that's her",
    "that's right",
    "thats right",
    "you're right",
    "youre right",
    "all right",
    "alright",
    "spot on",
    "exactly",
    "confirmed",
    "sounds good",
    "check in",
    "check on them",
    "talk to them",
    "correct",
    "affirmative",
)


def _is_negative(text: str) -> bool:
    """Return True if the utterance expresses rejection or cancellation."""
    cleaned = re.sub(r"[^\w\s']", " ", text.lower()).strip()
    if not cleaned:
        return False
    tokens = cleaned.split()
    if tokens[0] in {"no", "nope", "nah", "negative"}:
        return True
    return any(
        pat in cleaned
        for pat in ("not him", "not her", "don't", "do not", "cancel", "wrong patient")
    )


def _is_affirmative(text: str) -> bool:
    """Return True if the utterance expresses short confirmation."""
    cleaned = re.sub(r"[^\w\s']", " ", text.lower()).strip()
    if not cleaned or _is_negative(cleaned):
        return False
    tokens = cleaned.split()
    if "?" in text or (
        len(tokens) > 4
        and any(
            kw in cleaned
            for kw in (
                "did they",
                "did he",
                "did she",
                "what",
                "why",
                "how",
                "when",
                "aspirin",
                "medication",
                "headache",
                "symptom",
                "explain",
            )
        )
    ):
        return False
    # Long free-form clinical questions containing words like "sure" or "please"
    # (e.g., "can you please explain her protocol?") are handled as conversational Q&A
    if len(tokens) > 7:
        return False
    if tokens == ["right"]:
        return True
    return any(
        re.search(rf"\b{re.escape(pat)}\b", cleaned) is not None
        for pat in _AFFIRMATIVE_PATTERNS
    )


def _patient_tokens(pid: str) -> tuple[str, str, str]:
    """Return (full_name_token, first_name_token, room_token) for a patient ID."""
    if pid == "1429":
        return PATIENT_TOKEN, PATIENT_FIRSTNAME_TOKEN, LOCATION_TOKEN
    return f"[PATIENT_{pid}]", f"[PATIENT_{pid}_FIRSTNAME]", f"[LOCATION_{pid}]"


def _all_known_patients(state: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Collect all patient records from patients_db.json plus optional explicit state."""
    db_patients = state_manager.load_db().get("patients", {})
    combined: dict[str, dict[str, Any]] = dict(db_patients)
    if state and state.get("patient_id"):
        pid = str(state["patient_id"])
        merged = dict(combined.get(pid, {}))
        merged.update(state)
        combined[pid] = merged
    return combined


def mask_pii(text: str, state: dict[str, Any] | None = None) -> str:
    """
    Strip real patient identifiers across all patients in the database
    before any prompt leaves the local machine.
    """
    if not text:
        return text

    masked = text
    patients = _all_known_patients(state)

    for pid, pdata in patients.items():
        name_tok, first_tok, room_tok = _patient_tokens(pid)
        patient_name = str(pdata.get("name", ""))
        room_label = str(pdata.get("room", ""))

        name_variants = {patient_name}
        if pid == "1429":
            name_variants.update({"Mika Hämäläinen", "Mika Hamalainen"})
        for variant in sorted(name_variants, key=len, reverse=True):
            if variant:
                masked = re.sub(re.escape(variant), name_tok, masked, flags=re.IGNORECASE)

        first_name = patient_name.split()[0] if patient_name else ""
        if first_name and len(first_name) > 2:
            masked = re.sub(
                rf"\b{re.escape(first_name)}\b",
                first_tok,
                masked,
                flags=re.IGNORECASE,
            )

        room_variants = {room_label}
        if pid == "1429":
            room_variants.add("Room 153")
        for variant in sorted(room_variants, key=len, reverse=True):
            if variant:
                masked = re.sub(re.escape(variant), room_tok, masked, flags=re.IGNORECASE)

    return masked


def unmask_pii(text: str, state: dict[str, Any] | None = None) -> str:
    """Re-hydrate local patient identifiers back into model responses."""
    if not text:
        return text

    unmasked = text
    patients = _all_known_patients(state)

    for pid, pdata in patients.items():
        name_tok, first_tok, room_tok = _patient_tokens(pid)
        patient_name = str(pdata.get("name", ""))
        first_name = patient_name.split()[0] if patient_name else ""
        room_label = str(pdata.get("room", ""))

        unmasked = unmasked.replace(name_tok, patient_name)
        unmasked = unmasked.replace(first_tok, first_name)
        unmasked = unmasked.replace(room_tok, room_label)

    unmasked = re.sub(r"\bRoom\s+Room\b", "Room", unmasked, flags=re.IGNORECASE)
    return unmasked


def build_redacted_ehr_context(state: dict[str, Any], telemetry: dict[str, Any]) -> str:
    """Build a local RAG context string for a single patient with all PII pseudonymized."""
    symptoms = state.get("patient_symptoms") or []
    nurse_notes = state.get("nurse_notes") or []
    bedside_msgs = state.get("bedside_messages") or []
    bedside_log = (
        " | ".join(
            f"{m.get('sender', 'unknown')}: {m.get('text', '')}"
            for m in bedside_msgs[-8:]
        )
        if bedside_msgs
        else "No bedside messages yet"
    )
    raw_context = (
        f"Patient ID: {state.get('patient_id', '1429')}\n"
        f"Patient Name: {state.get('name', 'Mika Hämäläinen')}\n"
        f"Unit / Wing: {state.get('wing', 'Wing A')}\n"
        f"Room: {state.get('room', 'Room 153')}\n"
        f"Department: {state.get('department', 'Nephrology — Renal Care')}\n"
        f"Condition: {state.get('condition', 'Dialysis')}\n"
        f"Protocol: {state.get('protocol', 'Routine Dialysis')}\n"
        f"Live Telemetry BPM: {telemetry.get('bpm', state.get('current_bpm', 90))} "
        f"(Baseline: {state.get('baseline_bpm', 90)} BPM)\n"
        f"Wave Pattern: {telemetry.get('wave_state', state.get('wave_state', 'Awake / Normal Sinus Rhythm'))}\n"
        f"Reported Symptoms: {', '.join(symptoms) if symptoms else 'None reported'}\n"
        f"Nurse Round Comments: {', '.join(nurse_notes) if nurse_notes else 'None'}\n"
        f"Bedside Chat Transcript: {bedside_log}\n"
        f"Conversation Stage: {state.get('conversation_stage', 'idle')}\n"
        f"Check-in Active: {state.get('check_in_active', False)}\n"
        f"Nurse Dispatched: {state.get('nurse_dispatched', False)}"
    )
    return mask_pii(raw_context, state)


def build_all_patients_rag_context() -> str:
    """Build a pseudonymized RAG summary of every patient in patients_db.json."""
    db = state_manager.load_db()
    patients = db.get("patients", {})
    active_id = str(db.get("active_patient_id", "1429"))
    blocks: list[str] = [f"Active Patient ID: {active_id}"]
    for pid in sorted(patients.keys()):
        p = patients[pid]
        symptoms = p.get("patient_symptoms") or []
        notes = p.get("nurse_notes") or []
        bedside_msgs = p.get("bedside_messages") or []
        bedside_summary = " | ".join(
            f"{'Patient' if m.get('sender') == 'patient' else 'Malcolm'}: {m.get('text', '')}"
            for m in bedside_msgs[-6:]
            if m.get("text")
        ) or "No bedside chat yet"
        blocks.append(
            f"- Patient {pid}: {p.get('name')} | {p.get('wing')}, {p.get('room')} | "
            f"Department: {p.get('department')} | Condition: {p.get('condition')} | "
            f"Protocol: {p.get('protocol')} | BPM: {p.get('current_bpm', p.get('baseline_bpm'))} "
            f"(Baseline {p.get('baseline_bpm')}) | Wave: {p.get('wave_state')} | "
            f"Symptoms: {', '.join(symptoms) if symptoms else 'None'} | "
            f"Nurse Notes: {', '.join(notes) if notes else 'None'} | "
            f"Bedside Chat: {bedside_summary} | "
            f"Nurse Dispatched: {bool(p.get('nurse_dispatched'))}"
        )
    return mask_pii("\n".join(blocks))


def build_ward_hud_data(db: dict[str, Any] | None = None) -> dict[str, Any]:
    """Construct zero-latency structured Ward Roster rows and live Telemetry Statistics for the HUD pane."""
    raw_db = db if isinstance(db, dict) else state_manager.load_db()
    patients = raw_db.get("patients", {})
    rows: list[dict[str, Any]] = []
    bpms: list[int] = []
    awake_count = 0
    sleeping_count = 0
    symptomatic_count = 0
    tachy_count = 0
    dispatched_count = 0
    wing_groups: dict[str, list[dict[str, Any]]] = {}

    for pid in sorted(patients.keys()):
        p = patients[pid]
        bpm = int(p.get("current_bpm", p.get("baseline_bpm", 90)))
        base_bpm = int(p.get("baseline_bpm", 90))
        wave = str(p.get("wave_state", "Awake / Normal Sinus Rhythm"))
        if "/" in wave:
            wake_label, rhythm_label = [part.strip() for part in wave.split("/", 1)]
        else:
            wake_label = "Sleeping" if "sleep" in wave.lower() else "Awake"
            rhythm_label = wave
        is_sleeping = "sleep" in wake_label.lower()
        if is_sleeping:
            sleeping_count += 1
        else:
            awake_count += 1

        symptoms = [str(s).strip() for s in (p.get("patient_symptoms") or []) if str(s).strip()]
        notes = [str(n).strip() for n in (p.get("nurse_notes") or []) if str(n).strip()]
        dispatched = bool(p.get("nurse_dispatched"))
        if symptoms:
            symptomatic_count += 1
        if bpm >= 100:
            tachy_count += 1
        if dispatched:
            dispatched_count += 1

        if bpm >= 100 or bpm <= 55 or dispatched:
            acuity = "Elevated"
        elif symptoms:
            acuity = "Observation"
        else:
            acuity = "Stable"

        room_raw = str(p.get("room", "Room 153"))
        room_num = re.sub(r"^Room\s+", "", room_raw, flags=re.IGNORECASE).strip() or "153"
        wing = str(p.get("wing", "Wing A"))
        wake_val = "Sleeping" if is_sleeping else "Awake"
        row = {
            "patientId": str(pid),
            "name": str(p.get("name", f"Patient {pid}")),
            "wing": wing,
            "room": room_num,
            "condition": str(p.get("condition", "Monitoring")),
            "department": str(p.get("department", "General Medicine")),
            "protocol": str(p.get("protocol", "Standard Protocol")),
            "bpm": bpm,
            "baselineBpm": base_bpm,
            "waveState": wave,
            "rhythm": rhythm_label,
            "wakeState": wake_val,
            "wakeStatus": wake_val,
            "acuity": acuity,
            "symptom": "; ".join(symptoms) if symptoms else "No acute symptoms reported",
            "nurseNote": " ".join(notes) if notes else "No comments from prior rounds",
            "nurseDispatched": dispatched,
            "checkInActive": bool(p.get("check_in_active")),
        }
        rows.append(row)
        bpms.append(bpm)
        wing_groups.setdefault(wing, []).append(row)

    total = len(rows) or 1
    mean_bpm = round(sum(bpms) / total, 1) if bpms else 90.0
    sorted_rows = sorted(rows, key=lambda r: int(r["bpm"]))
    min_row = sorted_rows[0] if sorted_rows else None
    max_row = sorted_rows[-1] if sorted_rows else None

    wing_breakdown = []
    for w_name in sorted(wing_groups.keys()):
        w_list = wing_groups[w_name]
        w_avg = round(sum(int(x["bpm"]) for x in w_list) / len(w_list), 1)
        wing_breakdown.append(
            {
                "wing": w_name,
                "count": len(w_list),
                "meanBpm": w_avg,
                "avgBpm": w_avg,
                "patients": [str(x["name"]) for x in w_list],
            }
        )

    min_bpm_val = int(min_row["bpm"]) if min_row else 64
    max_bpm_val = int(max_row["bpm"]) if max_row else 104
    min_patient_str = f"{min_row['name']} ({min_row['patientId']})" if min_row else "Sanna Laine (1432)"
    max_patient_str = f"{max_row['name']} ({max_row['patientId']})" if max_row else "Jukka Korhonen (1431)"
    summary_line = (
        f"Ward Mean HR: {mean_bpm} BPM (Range {min_bpm_val}–{max_bpm_val} BPM) • "
        f"{awake_count} Awake / {sleeping_count} Sleeping • "
        f"{symptomatic_count} Symptomatic • {tachy_count} Tachycardic"
    )
    return {
        "patients": rows,
        "stats": {
            "totalPatients": len(rows),
            "meanBpm": mean_bpm,
            "minBpm": min_bpm_val,
            "minBpmPatient": min_patient_str,
            "minPatient": min_patient_str,
            "maxBpm": max_bpm_val,
            "maxBpmPatient": max_patient_str,
            "maxPatient": max_patient_str,
            "bpmSpread": max_bpm_val - min_bpm_val,
            "awakeCount": awake_count,
            "sleepingCount": sleeping_count,
            "symptomaticCount": symptomatic_count,
            "tachycardiaCount": tachy_count,
            "nurseDispatchedCount": dispatched_count,
            "wingBreakdown": wing_breakdown,
            "summaryLine": summary_line,
        },
    }


def _fmt_12h(dt: datetime) -> str:
    """Format a datetime object as '4:30 PM' without leading zero."""
    s = dt.strftime("%I:%M %p")
    return s.lstrip("0") or s


_SPOKEN_NUM_MAP: dict[str, str] = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "twenty": "20",
    "twenty-five": "25",
    "thirty": "30",
    "forty": "40",
    "forty-five": "45",
    "fifty": "50",
    "sixty": "60",
}


def _normalize_spoken_numbers(text: str) -> str:
    """Replace spoken number words ('ten seconds', 'five minutes') with digits ('10 seconds', '5 minutes')."""
    out = text.lower()
    out = re.sub(r"\ba\s+(second|minute)\b", r"1 \1", out)
    out = re.sub(r"\bhalf\s+a\s+minute\b", "30 seconds", out)
    for word, digit in sorted(_SPOKEN_NUM_MAP.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(rf"\b{re.escape(word)}\b", digit, out)
    return out


def _parse_relative_delay_seconds(text: str) -> int | None:
    """
    Detect when the clinician asks to be paged or reminded in X seconds, X minutes, or immediately ('page me now').
    Returns the exact delay in seconds (e.g. 10 for 'page me in 10 seconds'), or None if not a relative timer/page.
    """
    norm = _normalize_spoken_numbers(text)
    if any(k in norm for k in ("time table", "timetable", "time-table", "clock out", "clock-out")):
        return None

    # 1. Explicit seconds ("page me in 10 seconds", "remind me in 15 sec", "for 10 seconds", "10 seconds from now")
    m_sec = re.search(
        r"\b(?:in|for|after)?\s*(\d{1,3})\s*(?:sec|secs|second|seconds)\b",
        norm,
    )
    if m_sec:
        return max(1, min(3600, int(m_sec.group(1))))

    # 2. Immediate page test ("page me now", "page me right now", "test page", "test the paging system", "test loudspeaker")
    if re.search(
        r"\b(?:page\s+me\s+(?:right\s+)?now|test\s+(?:the\s+)?(?:loudspeaker|page|paging\s+system)|broadcast\s+a\s+page(?:\s+now)?)\b",
        norm,
    ):
        return 2

    # 3. Relative minutes ("page me in 1 minute", "remind me in 2 minutes") without "minutes before"
    if not re.search(r"\b\d{1,2}\s*(?:min|minute)s?\s+before\b", norm):
        m_min = re.search(
            r"\b(?:in|for|after)\s+(\d{1,3})\s*(?:min|mins|minute|minutes)\b",
            norm,
        )
        if m_min:
            return max(1, min(180, int(m_min.group(1)))) * 60

    return None


def _is_timetable_or_calendar_query(text: str) -> bool:
    """Return True when the clinician asks to create/view a timetable, schedule, calendar, or loudspeaker reminder."""
    lowered = text.lower().strip()
    if not lowered:
        return False
    if _parse_relative_delay_seconds(text) is not None:
        return True
    schedule_keywords = (
        "time table",
        "timetable",
        "time-table",
        "schedule",
        "calendar",
        "agenda",
        "itinerary",
        "shift plan",
        "timings",
        "clock out",
        "clock-out",
        "remind me",
        "set a reminder",
        "reminder",
        "page me",
        "loudspeaker",
        "paging system",
        "test page",
    )
    if any(kw in lowered for kw in schedule_keywords):
        return True
    if "surgery" in lowered and any(
        k in lowered
        for k in ("pm", "am", "at ", "before", "after", "then", "check in", "check-in", "table", "time")
    ):
        return True
    return False


def build_timetable_events(
    text: str,
    existing_events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Construct or update chronological clinician timetable & calendar events synced to the PC's real clock.
    Default loudspeaker reminder frequency is 5 minutes before each scheduled task, AND supports
    exact second/minute countdown timers (e.g. 'page me in 10 seconds').
    """
    now = datetime.now()
    today_date = now.date()
    lowered = _normalize_spoken_numbers(text.strip())

    # Detect custom reminder offset if explicitly stated (e.g. "remind me 10 minutes before"), default 5 min
    reminder_mins = 5
    m_rem = re.search(r"remind(?:er)?\s+(?:me\s+)?(\d{1,2})\s*(?:min|minute)s?\s+before", lowered)
    if m_rem:
        reminder_mins = max(1, min(60, int(m_rem.group(1))))

    def _make_event(
        idx: int,
        start_dt: datetime,
        end_dt: datetime | None,
        title: str,
        location: str,
        category: str,
        patient_id: str = "",
        rem_mins: int = reminder_mins,
    ) -> dict[str, Any]:
        rem_dt = start_dt - timedelta(minutes=rem_mins)
        disp_time = (
            f"{_fmt_12h(start_dt)} – {_fmt_12h(end_dt)}"
            if end_dt is not None
            else _fmt_12h(start_dt)
        )
        return {
            "id": f"evt_{idx}_{int(start_dt.timestamp()) % 100000}",
            "start_time": start_dt.strftime("%H:%M"),
            "end_time": end_dt.strftime("%H:%M") if end_dt else start_dt.strftime("%H:%M"),
            "display_time": disp_time,
            "scheduled_iso": start_dt.strftime("%Y-%m-%dT%H:%M:00"),
            "reminder_minutes_before": rem_mins,
            "reminder_time": rem_dt.strftime("%H:%M"),
            "reminder_display": f"{_fmt_12h(rem_dt)} ({rem_mins}m prior)",
            "title": title,
            "location": location,
            "category": category,
            "patient_id": patient_id,
            "paged_reminder": False,
            "paged_start": False,
            "status": "Scheduled",
        }

    # 1. Check if clinician is asking for a relative countdown page/reminder ("page me in 10 seconds", "remind me in 2 minutes")
    delay_sec = _parse_relative_delay_seconds(text)
    if delay_sec is not None:
        fire_dt = now + timedelta(seconds=delay_sec)
        fire_epoch = time.time() + float(delay_sec)
        task_desc = "Attending Physician Loudspeaker Page"
        loc = "Hospital PA / Ward Floor"
        pid = ""
        if "mika" in lowered or "1429" in lowered:
            task_desc = "Patient Check-In — Mika Hämäläinen (Patient 1429)"
            loc = "Wing A, Room 153"
            pid = "1429"
        elif "aino" in lowered or "1430" in lowered:
            task_desc = "Patient Check-In — Aino Virtanen (Patient 1430)"
            loc = "Wing B, Room 204"
            pid = "1430"
        elif "jukka" in lowered or "1431" in lowered:
            task_desc = "Patient Check-In — Jukka Korhonen (Patient 1431)"
            loc = "Wing C, Room 312"
            pid = "1431"
        elif "sanna" in lowered or "1432" in lowered:
            task_desc = "Patient Check-In — Sanna Laine (Patient 1432)"
            loc = "Wing A, Room 118"
            pid = "1432"
        elif "surgery" in lowered:
            task_desc = "Scheduled Surgery Page"
            loc = "Operating Room 2"

        fire_clock_str = fire_dt.strftime("%I:%M:%S %p").lstrip("0")
        delay_label = f"{delay_sec}s" if delay_sec < 120 else f"{delay_sec // 60}m"
        current = list(existing_events or [])
        new_evt = {
            "id": f"evt_timer_{int(time.time() * 1000)}",
            "start_time": fire_dt.strftime("%H:%M"),
            "end_time": fire_dt.strftime("%H:%M"),
            "display_time": fire_clock_str,
            "scheduled_iso": fire_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "remind_at_epoch": fire_epoch,
            "delay_seconds": delay_sec,
            "reminder_minutes_before": 0,
            "reminder_time": fire_dt.strftime("%H:%M"),
            "reminder_display": f"In {delay_label} ({fire_clock_str})",
            "title": task_desc,
            "location": loc,
            "category": "Loudspeaker Page",
            "patient_id": pid,
            "paged_reminder": False,
            "paged_start": False,
            "status": f"Armed ({delay_label} Timer)",
        }
        current.append(new_evt)
        current.sort(key=lambda e: str(e.get("start_time", "00:00")))
        return current

    # 2. Parse anchor time (e.g. "5pm surgery", "17:00", "6:30 pm")
    anchor_hour = 17
    anchor_min = 0
    m_time = re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b",
        lowered,
    )
    if m_time:
        hr = int(m_time.group(1))
        mn = int(m_time.group(2) or 0)
        mer = m_time.group(3).lower().replace(".", "")
        if mer == "pm" and hr < 12:
            hr += 12
        elif mer == "am" and hr == 12:
            hr = 0
        anchor_hour, anchor_min = hr, mn
    else:
        m_24 = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", lowered)
        if m_24:
            anchor_hour, anchor_min = int(m_24.group(1)), int(m_24.group(2))
        elif "surgery" not in lowered and "5" not in lowered:
            # Anchor to next 15-minute mark on the PC's real clock
            rounded = now + timedelta(minutes=(15 - now.minute % 15) or 15)
            anchor_hour, anchor_min = rounded.hour, rounded.minute

    surgery_start = datetime(
        today_date.year,
        today_date.month,
        today_date.day,
        anchor_hour,
        anchor_min,
        0,
    )
    # Construct clean chronological timetable:
    # 1) Pre-surgery check-in on Patient 1 (30 mins before surgery)
    t1_start = surgery_start - timedelta(minutes=30)
    t1_end = surgery_start - timedelta(minutes=10)
    # 2) Surgery (60 mins)
    t2_start = surgery_start
    t2_end = surgery_start + timedelta(minutes=60)
    # 3) Post-surgery check-in on Patient 2 (10 mins after surgery)
    t3_start = t2_end + timedelta(minutes=10)
    t3_end = t3_start + timedelta(minutes=20)
    # 4) Final patient check-in on Patient 3 before clock-out
    t4_start = t3_end + timedelta(minutes=5)
    t4_end = t4_start + timedelta(minutes=20)
    # 5) Shift wrap-up & Clock out
    t5_start = t4_end + timedelta(minutes=5)

    events = [
        _make_event(
            1,
            t1_start,
            t1_end,
            "Patient Check-In — Mika Hämäläinen (Patient 1429)",
            "Wing A, Room 153",
            "Check-In",
            patient_id="1429",
        ),
        _make_event(
            2,
            t2_start,
            t2_end,
            "Scheduled Surgery",
            "Operating Room 2 (Surgical Suite)",
            "Surgery",
            patient_id="",
        ),
        _make_event(
            3,
            t3_start,
            t3_end,
            "Patient Check-In — Aino Virtanen (Patient 1430)",
            "Wing B, Room 204",
            "Check-In",
            patient_id="1430",
        ),
        _make_event(
            4,
            t4_start,
            t4_end,
            "Patient Check-In — Jukka Korhonen (Patient 1431)",
            "Wing C, Room 312",
            "Check-In",
            patient_id="1431",
        ),
        _make_event(
            5,
            t5_start,
            None,
            "Shift Handover & Clock Out",
            "Attending Physician Station",
            "Clock-Out",
            patient_id="",
        ),
    ]
    return events


def render_timetable_html_css(events: list[dict[str, Any]] | None = None) -> str:
    """Render a clean, dedicated HTML+CSS Timetable & Hospital Loudspeaker Pager View (no unrelated telemetry clutter)."""
    now = datetime.now()
    items = events if events is not None else state_manager.get_calendar_events()
    if not items:
        items = build_timetable_events("create a time table with a 5pm surgery")

    rows_html = ""
    for idx, ev in enumerate(items, start=1):
        cat = str(ev.get("category", "Task"))
        is_surg = cat.lower() == "surgery"
        is_clockout = "clock" in cat.lower()
        badge_bg = (
            "rgba(244,63,94,0.18);color:#fda4af;border:1px solid rgba(244,63,94,0.45)"
            if is_surg
            else (
                "rgba(16,185,129,0.16);color:#6ee7b7;border:1px solid rgba(16,185,129,0.4)"
                if is_clockout
                else "rgba(56,189,248,0.15);color:#38bdf8;border:1px solid rgba(56,189,248,0.4)"
            )
        )
        pid = str(ev.get("patient_id", ""))
        status_str = str(ev.get("status", "Scheduled"))
        row_click = f"onclick=\"window.selectPatient('{pid}')\"" if pid else ""
        rows_html += (
            f"<tr style='border-bottom:1px solid rgba(255,255,255,0.09)'>"
            f"<td style='padding:12px 14px;white-space:nowrap'>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:14px;font-weight:600;color:#38bdf8'>{ev.get('display_time','')}</div>"
            f"</td>"
            f"<td style='padding:12px 14px' {row_click}>"
            f"<div style='display:flex;align-items:center;gap:8px'>"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:10px;text-transform:uppercase;letter-spacing:0.14em;padding:2px 7px;{badge_bg}'>{cat}</span>"
            f"<span style='font-size:15px;font-weight:600;color:#ffffff'>{ev.get('title','')}</span>"
            f"</div>"
            f"</td>"
            f"<td style='padding:12px 14px;font-family:JetBrains Mono,monospace;font-size:12px;color:#cbd5e1'>{ev.get('location','')}</td>"
            f"<td style='padding:12px 14px;white-space:nowrap'>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;color:#fbbf24'>🔔 {ev.get('reminder_display','5m prior')}</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#8b95a7;margin-top:2px'>Hospital PA Loudspeaker</div>"
            f"</td>"
            f"<td style='padding:12px 14px;text-align:right;white-space:nowrap;font-family:JetBrains Mono,monospace;font-size:11px;color:#6ee7b7'>"
            f"{status_str}"
            f"</td>"
            f"</tr>"
        )

    return f"""<style>
  .tt-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    border: 1px solid rgba(56, 189, 248, 0.35);
    background: rgba(56, 189, 248, 0.07);
    padding: 10px 16px;
  }}
  .tt-clock {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: #38bdf8;
    letter-spacing: 0.12em;
  }}
</style>
<div class="hud-root">
  <div>
    <div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px">
      <p class="hud-kicker">Attending Physician Schedule • Hospital PA Pager Armed</p>
      <span class="tt-clock" id="hud-live-clock">PC CLOCK SYNCED: {_fmt_12h(now)} ({now.strftime('%Y-%m-%d')})</span>
    </div>
    <h2 class="hud-title">Clinician Shift Timetable &amp; Reminder Schedule</h2>
  </div>

  <div class="tt-bar">
    <div class="hud-kicker" style="color:#e2e8f0">
      ⏱ INTERNAL CLOCK: SYNCED TO PC REAL TIME • DEFAULT REMINDER FREQUENCY: <strong style="color:#fbbf24">5 MINUTES BEFORE TASK</strong>
    </div>
    <div class="hud-kicker" style="color:#38bdf8">
      📢 PAGING OUTPUT: HOSPITAL LOUDSPEAKER (LAPTOP SPEAKER)
    </div>
  </div>

  <table class="hud-table">
    <thead>
      <tr>
        <th>Time Slot</th>
        <th>Scheduled Task</th>
        <th>Location / Room</th>
        <th>Pager Reminder (T-5 Min)</th>
        <th style="text-align:right">Status</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <div class="hud-footer">
    <div>
      <p class="hud-kicker" style="color:#fda4af">Hospital Loudspeaker Paging System Active</p>
      <div style="margin-top:3px;font-size:13px;color:#f3f5f8">
        Malcolm will automatically chime and page you over the hospital loudspeaker <strong>5 minutes before</strong> each scheduled task.
      </div>
    </div>
  </div>
</div>
<script>
  setInterval(function() {{
    var el = document.getElementById("hud-live-clock");
    if (el) {{
      var d = new Date();
      el.textContent = "PC CLOCK SYNCED: " + d.toLocaleTimeString([], {{hour: '2-digit', minute: '2-digit', second: '2-digit'}});
    }}
  }}, 1000);
</script>"""


def _detect_pane_intent(text: str) -> dict[str, Any] | None:
    """Detect clinician voice/text requests to construct an on-demand HTML/CSS pane (timetable, table, stats, custom view) or close it."""
    lowered = text.lower().strip()
    if not lowered:
        return None

    # 1. Close / dismiss / hide pane requests
    close_patterns = (
        r"\b(?:close|hide|dismiss|remove|minimize|collapse|clear|shut|exit|turn\s+off|put\s+away)\b.*"
        r"\b(?:pane|panel|table|timetable|schedule|chart|stats|statistics|display|screen|board|roster|sidebar|view|window|it|that)\b",
        r"\b(?:put|take)\s+(?:that|it|the\s+pane|the\s+table|the\s+display)\s+away\b",
        r"\bback\s+to\s+(?:main\s+screen|full\s*screen|idle|standby|circle)\b",
        r"^(?:close|hide|dismiss)\s*(?:pane|table|stats|display|it|that)?[.!?]*$",
    )
    if any(re.search(pat, lowered) for pat in close_patterns):
        return {"action": "close"}

    # Do not open a pane when the request is physically impossible/impractical (e.g. MRI, hologram, camera feed)
    if _detect_impractical_request_fallback(text) is not None:
        return None

    # 2. Timetable / Schedule / Calendar / Reminder requests (MUST run before generic 'table' check!)
    if _is_timetable_or_calendar_query(text):
        return {
            "action": "open",
            "kind": "timetable",
            "generate_html": True,
            "title": "Clinician Shift Timetable & Reminder Schedule",
            "subtitle": "Synced to Real PC Clock • 5-Min Prior Loudspeaker Paging",
        }

    # 2. Custom Generative HTML/CSS layout (ONLY when asking for a custom/bespoke visual matrix/comparison
    #    that is NOT one of the readily available native views: ward_table, ward_stats, timetable, patient)
    is_custom_bespoke_ui = (
        any(v in lowered for v in ("create", "construct", "build", "generate", "render", "design"))
        and any(
            w in lowered
            for w in (
                "custom",
                "side-by-side",
                "side by side",
                "matrix",
                "comparison card",
                "protocol breakdown",
                "medication breakdown",
                "triage board",
                "acuity board",
                "heat map",
                "heatmap",
            )
        )
    )
    if is_custom_bespoke_ui:
        return {
            "action": "open",
            "kind": "custom_html",
            "generate_html": True,
            "title": "AI-Synthesized Clinical View",
            "subtitle": "Custom Generative HTML & CSS Display",
            "filterWing": "",
        }

    # 3. Readily available Statistics / analytics / metrics pane requests (uses native React <WardStatsView />)
    stats_keywords = (
        "statistic",
        "statistics",
        "ward stats",
        "vitals stats",
        "telemetry stats",
        "patient stats overview",
        "analytics",
        "metrics",
        "average heart rate",
        "mean heart rate",
        "heart rate distribution",
        "compare heart rates",
        "compare vitals",
        "compare patients",
        "comparison",
        "bar chart",
        "graph",
        "plot",
        "ward numbers",
        "statistical",
        "breakdown of the ward",
        "give me stats",
        "show me stats",
        "pull up stats",
    )
    if any(kw in lowered for kw in stats_keywords):
        return {
            "action": "open",
            "kind": "ward_stats",
            "generate_html": False,
            "title": "Ward Telemetry & Acuity Statistics",
            "subtitle": "Live Statistical Breakdown • Monitored Ward Beds",
        }

    # 4. Readily available Ward Table / All-patients roster (uses native React <WardTableView />, NO AI regeneration!)
    has_table_word = any(
        w in lowered
        for w in (
            "table",
            "roster",
            "spreadsheet",
            "grid",
            "board",
            "ledger",
            "dashboard",
            "columns",
            "cards",
        )
    )
    has_show_verb = any(
        v in lowered
        for v in (
            "show",
            "display",
            "pull up",
            "bring up",
            "open",
            "create",
            "construct",
            "build",
            "make",
            "generate",
            "render",
            "draw",
            "design",
            "give me",
            "put up",
            "view",
            "see",
        )
    )
    has_multi_target = any(
        t in lowered
        for t in (
            "all patients",
            "all patience",
            "every patient",
            "everyone",
            "the ward",
            "people in the ward",
            "patients in the ward",
            "patients on the floor",
            "our patients",
            "patient list",
            "list of patients",
            "four patients",
            "all four",
            "wing a",
            "wing b",
            "wing c",
            "awake patients",
            "sleeping patients",
            "symptoms",
            "medications",
            "protocols",
        )
    )
    if has_table_word or (has_show_verb and has_multi_target):
        filter_wing = ""
        title = "Active Ward Roster — All Monitored Patients"
        subtitle = "Real-Time Multi-Patient EHR & Telemetry Matrix"
        if "wing a" in lowered and "wing b" not in lowered and "wing c" not in lowered:
            filter_wing = "Wing A"
            title = "Wing A Patient Roster"
            subtitle = "Real-Time Multi-Patient EHR • Wing A Beds"
        elif "wing b" in lowered and "wing a" not in lowered:
            filter_wing = "Wing B"
            title = "Wing B Patient Roster"
            subtitle = "Real-Time Multi-Patient EHR • Wing B Beds"
        elif "wing c" in lowered and "wing a" not in lowered:
            filter_wing = "Wing C"
            title = "Wing C Patient Roster"
            subtitle = "Real-Time Multi-Patient EHR • Wing C Beds"
        return {
            "action": "open",
            "kind": "ward_table",
            "generate_html": False,
            "title": title,
            "subtitle": subtitle,
            "filterWing": filter_wing,
        }

    # 5. Explicit request to open/show a single patient's readily available pane/chart
    if has_show_verb and any(w in lowered for w in ("pane", "panel", "chart", "profile", "card", "summary")):
        return {
            "action": "open",
            "kind": "patient",
            "generate_html": False,
            "title": "",
            "subtitle": "",
        }

    return None


def _detect_export_request(text: str, current_pane_kind: str = "patient") -> dict[str, str] | None:
    """
    Detect when the clinician asks Malcolm to generate, export, download, or save
    an .xlsx (Excel spreadsheet) or .png (formatted plain-white image) file.
    """
    lowered = text.lower()
    wants_xlsx = bool(
        re.search(
            r"\b(xlsx|xls|x\s*l\s*s\s*x|x\s*l\s*s|excel|spreadsheet\s+file|excel\s+sheet|excel\s+file|csv)\b",
            lowered,
        )
        or (
            any(v in lowered for v in ("export", "download", "save", "generate", "create", "make", "give me"))
            and any(f in lowered for f in ("spreadsheet", "excel", "xlsx", "xls"))
        )
    )
    wants_png = bool(
        re.search(
            r"\b(png|p\s*n\s*g|p\s+and\s+g|p&g|image\s+file|picture\s+file|screenshot|snapshot)\b",
            lowered,
        )
        or (
            any(v in lowered for v in ("export", "download", "save", "generate", "create", "make", "give me", "turn"))
            and any(f in lowered for f in (".png", " png", "image", "picture", "photo", "ping"))
        )
    )
    has_generic_export_verb = bool(
        re.search(r"\b(export|download)\b", lowered)
        and any(
            w in lowered
            for w in (
                "file",
                "table",
                "timetable",
                "time table",
                "schedule",
                "summary",
                "stats",
                "statistics",
                "chart",
                "report",
                "roster",
                "pane",
                "it",
                "this",
                "that",
                "them",
            )
        )
    )

    if not (wants_xlsx or wants_png or has_generic_export_verb):
        return None

    if wants_xlsx and wants_png:
        fmt = "both"
    elif wants_png:
        fmt = "png"
    else:
        fmt = "xlsx"

    # Determine which view to export
    if any(k in lowered for k in ("timetable", "time table", "time-table", "schedule", "calendar", "timings", "agenda")):
        kind = "timetable"
    elif any(k in lowered for k in ("statistic", "statistics", "stats", "analytics", "metrics")):
        kind = "ward_stats"
    elif any(k in lowered for k in ("ward table", "roster", "all patients", "four patients", "patient table")):
        kind = "ward_table"
    elif "table" in lowered and current_pane_kind == "patient":
        kind = "ward_table"
    else:
        kind = current_pane_kind or "patient"

    return {"format": fmt, "kind": kind}


def _detect_patient_crud_request(text: str, db: dict[str, Any]) -> dict[str, Any] | None:
    """
    Detect when the clinician asks to Add/Admit a new patient or Delete/Remove/Discharge a patient
    in patients_db.json (or asks if Malcolm can add/delete patients).
    """
    lowered = text.lower().strip()
    if not lowered:
        return None

    # Ignore pane close commands like "remove the pane" or "clear the screen"
    if re.search(r"\b(?:remove|clear|delete|close)\s+(?:the\s+)?(?:pane|panel|table|display|screen|chart|view)\b", lowered):
        return None
    # Ignore destructive server-wipe commands ("wipe the database", "drop all tables")
    if re.search(r"\b(?:wipe|destroy|drop|format|nuke)\s+(?:the\s+)?(?:server|database|system|hard\s+drive|backend)\b", lowered):
        return None

    # 1. Capability question: "can you add or delete patients?", "can you delete patients?"
    if re.search(r"\b(?:can\s+you|are\s+you\s+able\s+to|do\s+you\s+have\s+the\s+ability\s+to)\b", lowered) and re.search(
        r"\b(?:add|admit|create|register|delete|remove|discharge)\b.*\b(?:patient|patients|people)\b",
        lowered,
    ):
        target_pid = _detect_target_patient_id(text, db)
        has_named_new = bool(re.search(r"\b(?:named|called)\s+[a-zA-Z]", text, flags=re.IGNORECASE))
        if not target_pid and not has_named_new:
            return {"action": "capability"}

    # 2. Delete / Remove / Discharge patient
    target_pid_for_del = _detect_target_patient_id(text, db)
    is_delete = bool(
        re.search(
            r"\b(?:delete|remove|discharge|drop|erase|clear\s+out|take\s+off)\b.*\b(?:patient|patients|him|her|them|mika|aino|jukka|sanna|14\d\d|from\s+the\s+(?:ward|roster|table|list|database))\b",
            lowered,
        )
        or re.search(
            r"\b(?:delete|remove|discharge)\s+(?:mika|aino|jukka|sanna|patient\s*\d+|#\d+)\b",
            lowered,
        )
        or (
            target_pid_for_del is not None
            and re.search(r"\b(?:delete|remove|discharge)\b", lowered)
        )
    )
    if is_delete:
        target_pid = target_pid_for_del
        if not target_pid and any(w in lowered for w in ("this patient", "current patient", "active patient", "him", "her")):
            target_pid = str(db.get("active_patient_id", "1429"))
        return {
            "action": "delete",
            "patient_id": target_pid,
        }

    # 3. Add / Admit / Register / Create a new patient
    is_add = bool(
        re.search(
            r"\b(?:add|admit|register|create|enroll|put\s+in|insert)\s+(?:a\s+)?(?:new\s+)?patient\b",
            lowered,
        )
        or re.search(
            r"\bnew\s+patient\s+(?:named|called|in\s+wing|in\s+room)\b",
            lowered,
        )
    )
    if is_add:
        # Extract Name (prefer explicit 'named/called X' first, then 'patient X')
        name = ""
        m_name = re.search(
            r"\b(?:named|called|name\s+is)\s+([a-zA-ZäöåÄÖÅ]+(?:\s+[a-zA-ZäöåÄÖÅ]+){0,2}?)(?=\s+(?:in|to|at|with|for|room|wing|who|having|on|bpm|heart)\b|[,.!?]|$)",
            text,
            flags=re.IGNORECASE,
        )
        if not m_name:
            m_name = re.search(
                r"\bpatient\s+(?!named\b|called\b|in\b|to\b|with\b|for\b|at\b|who\b)([a-zA-ZäöåÄÖÅ]+(?:\s+[a-zA-ZäöåÄÖÅ]+){0,2}?)(?=\s+(?:in|to|at|with|for|room|wing|who|having|on|bpm|heart)\b|[,.!?]|$)",
                text,
                flags=re.IGNORECASE,
            )
        if m_name:
            cand = re.sub(r"^(?:named|called|patient)\s+", "", m_name.group(1).strip(), flags=re.IGNORECASE).strip()
            if cand and cand.lower() not in {"named", "called", "in", "to", "with", "for", "wing", "room", "new"}:
                name = " ".join(w.capitalize() for w in cand.split())
        if not name:
            name = "Alex Rivera"

        # Extract Wing
        wing = "Wing B"
        m_wing = re.search(r"\bwing\s+([a-cA-C])\b", text, flags=re.IGNORECASE)
        if m_wing:
            wing = f"Wing {m_wing.group(1).upper()}"

        # Extract Room
        room = "Room 210"
        m_room = re.search(r"\broom\s+(\d{2,4})\b", text, flags=re.IGNORECASE)
        if m_room:
            room = f"Room {m_room.group(1)}"

        # Extract BPM
        bpm = 82
        m_bpm = re.search(r"\b(\d{2,3})\s*bpm\b", lowered)
        if m_bpm:
            bpm = max(45, min(180, int(m_bpm.group(1))))

        # Extract Condition / Department
        condition = "General Clinical Observation"
        department = "Internal Medicine"
        protocol = "Continuous Telemetry & Q4H Vitals"
        m_cond = re.search(
            r"\b(?:with|for|having|suffering\s+from|diagnosed\s+with)\s+([a-zA-Z0-9\s\-]+?)(?=\s+(?:in\s+wing|in\s+room|at\s+\d+\s*bpm|and\s+)|[,.!?]|$)",
            text,
            flags=re.IGNORECASE,
        )
        if m_cond:
            raw_c = m_cond.group(1).strip()
            if raw_c and len(raw_c) <= 48:
                condition = raw_c[0].upper() + raw_c[1:]
                c_low = condition.lower()
                if any(k in c_low for k in ("cardiac", "heart", "chest", "tachycardia", "arrhythmia", "valve")):
                    department = "Cardiology & Telemetry"
                    protocol = "Continuous 12-Lead ECG & Cardiac Enzymes"
                elif any(k in c_low for k in ("pneumonia", "asthma", "copd", "respiratory", "breath", "lung")):
                    department = "Pulmonology & Respiratory Care"
                    protocol = "SpO2 Monitoring & Nebulizer Therapy"
                elif any(k in c_low for k in ("renal", "dialysis", "kidney")):
                    department = "Nephrology — Renal Care"
                    protocol = "Routine Dialysis & Fluid Balance"
                elif any(k in c_low for k in ("neuro", "concussion", "stroke", "seizure", "head")):
                    department = "Neurology — Trauma Unit"
                    protocol = "Q2H Neuro Checks & Cranial Telemetry"

        return {
            "action": "add",
            "name": name,
            "wing": wing,
            "room": room,
            "condition": condition,
            "department": department,
            "protocol": protocol,
            "baseline_bpm": bpm,
        }

    return None


def _detect_impractical_request_fallback(text: str, state: dict[str, Any] | None = None) -> str | None:
    """Offline/deterministic Jarvis fallback when the clinician asks for something genuinely impractical or impossible."""
    lowered = text.lower()
    name = (state or {}).get("name", "Mika Hämäläinen")
    room = (state or {}).get("room", "Room 153")

    if any(
        k in lowered
        for k in (
            "mri",
            "ct scan",
            "cat scan",
            "x-ray",
            "xray",
            "ultrasound",
            "3d scan",
            "hologram",
            "brain scan",
            "pet scan",
            "echocardiogram",
            "angiogram",
        )
    ):
        return (
            f"I'm afraid rendering a live radiology scan or 3D hologram is outside my console hardware, doctor. "
            f"However, I can immediately pull up {name}'s clinical chart and live telemetry pane, or display the ward statistics—would that work instead?"
        )
    if any(
        k in lowered
        for k in (
            "draw blood",
            "intubate",
            "defibrillate",
            "perform surgery",
            "operate on",
            "inject",
            "administer",
            "infuse",
            "push epinephrine",
            "give morphine",
            "morphine",
            "fentanyl",
            "epinephrine",
            "insert catheter",
            "biopsy",
            "ventilator",
            "prescribe controlled",
            "dispense medication",
            "amputate",
            "appendectomy",
            "transplant",
            "teleport",
            "clone",
            "edit dna",
        )
    ):
        return (
            f"As much as I'd like to lend a hand, I cannot physically perform bedside procedures or administer medications from the console, sir. "
            f"Shall I dispatch a floor nurse to {room} right away, or pull up {name}'s protocol chart for you?"
        )
    if any(k in lowered for k in ("camera", "video feed", "cctv", "livestream", "watch the room", "spy on")):
        return (
            f"We don't stream live video feeds from the patient rooms for privacy reasons, doctor, so that's a no-go. "
            f"If you'd like eyes on {name}, I can initiate a live bedside voice check-in or open the real-time ECG telemetry pane instead—would you prefer one of those?"
        )
    if any(k in lowered for k in ("wake everyone", "wake all patients", "check in on all four at once")):
        return (
            "Waking the entire ward at once—especially Sanna Laine while she's resting under concussion precautions—would be rather impractical, doctor. "
            "Would you like me to bring up the ward patient table first so we can pick which awake patient to check in on?"
        )
    if any(
        k in lowered
        for k in (
            "make coffee",
            "brew coffee",
            "bring me coffee",
            "espresso",
            "order lunch",
            "order food",
            "order pizza",
            "call an uber",
            "book a flight",
            "play music",
        )
    ):
        return (
            "I'm still waiting on the hospital board to wire me into the breakroom espresso machine and outside delivery services, doctor, so I can't quite pull that off yet. "
            "In the meantime, would you like me to put the ward patient table or live telemetry statistics up on your pane?"
        )
    if re.search(r"\b(?:wipe|destroy|drop|format|nuke|hack)\s+(?:the\s+)?(?:server|database|system|hard\s+drive|backend|hospital)\b", lowered):
        return (
            "I'm afraid wiping the hospital backend server or tampering with core infrastructure is strictly off-limits, doctor. "
            "If you'd like to discharge a specific patient from the active ward roster or reset the demo state, just let me know which patient."
        )
    return None


def _detect_target_patient_id(text: str, db: dict[str, Any]) -> str | None:
    """Detect which patient ID in patients_db.json the clinician is referring to (including phonetic ASR variants and newly added patients)."""
    lowered = text.lower()
    patients = db.get("patients", {})

    # 1. Explicit patient ID match (e.g. 1429, 1430, 1431, 1432, 1433...)
    for pid in patients.keys():
        if re.search(rf"\b{re.escape(pid)}\b", lowered):
            return pid

    # 2. Room number match (e.g. 153, 204, 312, 118, 210...)
    for pid, pdata in patients.items():
        room = str(pdata.get("room", ""))
        room_num = re.sub(r"^Room\s+", "", room, flags=re.IGNORECASE).strip()
        if room_num and re.search(rf"\broom\s+{re.escape(room_num)}\b", lowered):
            return pid

    # 3. Dynamic exact/partial name match for ANY patient currently in db["patients"] (including newly added patients!)
    for pid, pdata in patients.items():
        full_name = str(pdata.get("name", "")).strip().lower()
        if full_name and full_name in lowered:
            return pid
        parts = [p for p in full_name.split() if len(p) >= 3 and p not in {"patient", "new"}]
        for part in parts:
            if re.search(rf"\b{re.escape(part)}\b", lowered):
                return pid

    # 4. Phonetic Finnish patient name match (1429-1432)
    phonetic_name_map: dict[str, tuple[str, ...]] = {
        "1429": (
            "mika",
            "meeka",
            "micah",
            "hamalainen",
            "hämäläinen",
            "hamelainen",
            "aham linen",
            "ham linen",
            "hama linen",
            "hama line",
            "mr linen",
        ),
        "1430": (
            "aino",
            "virtanen",
            "vertonen",
            "i know virtanen",
            "ino virtanen",
        ),
        "1431": (
            "jukka",
            "yuka",
            "yucca",
            "luka",
            "korhonen",
            "core honen",
        ),
        "1432": (
            "sanna",
            "sana",
            "laine",
            "sanna line",
            "sana line",
            "sanna lane",
            "sana lane",
            "hannah laine",
        ),
    }
    for pid, variants in phonetic_name_map.items():
        if pid in patients:
            for token in variants:
                if re.search(rf"\b{re.escape(token)}\b", lowered):
                    return pid

    # 5. Unique Wing / Condition cues when clinician refers to a patient by clinical feature
    if "1431" in patients and any(k in lowered for k in ("wing c", "wing see", "tachycardia", "pneumonia", "ceftriaxone", "104 bpm")):
        return "1431"
    if "1430" in patients and any(k in lowered for k in ("wing b", "mitral valve", "cardiothoracic")):
        return "1430"
    if "1432" in patients and any(k in lowered for k in ("concussion", "photophobia", "room 118", "neuro check")):
        return "1432"
    if "1429" in patients and ("dialysis" in lowered or "nephrology" in lowered):
        return "1429"

    return None


def _is_multi_patient_overview_query(text: str) -> bool:
    """Return True when the clinician asks about multiple/all patients, a ward table, or ward statistics."""
    lowered = text.lower()
    patterns = (
        "all patients",
        "all patience",
        "our patients",
        "our patience",
        "list patients",
        "list patience",
        "list our",
        "list all",
        "every patient",
        "patients in the database",
        "patient database",
        "who are the patients",
        "who do we have",
        "who is on the floor",
        "who's on the floor",
        "on the ward",
        "in the ward",
        "about the ward",
        "the ward",
        "people in the",
        "people on the",
        "about the people",
        "more about the people",
        "about the patients",
        "more about the patients",
        "tell me about the patients",
        "what patients",
        "floor summary",
        "summarize the floor",
        "highest heart rate",
        "lowest heart rate",
        "average heart rate",
        "mean heart rate",
        "who is sleeping",
        "who has",
        "which patient",
        "compare patients",
        "compare vitals",
        "compare heart",
        "overview",
        "ward status",
        "all four",
        "four patients",
        "everyone on",
        "everyone in",
        "table",
        "roster",
        "spreadsheet",
        "statistic",
        "statistics",
        "ward stats",
        "vitals stats",
        "telemetry stats",
        "analytics",
        "metrics",
    )
    return any(pat in lowered for pat in patterns)


def _is_direct_question(text: str) -> bool:
    """Return True when the doctor asks a specific clinical question rather than generic status."""
    lowered = text.lower()
    question_keywords = (
        "nurse note",
        "nurse round",
        "round comment",
        "protocol",
        "department",
        "condition",
        "diagnosis",
        "symptom",
        "heart rate",
        "bpm",
        "vitals",
        "wave",
        "sleeping",
        "awake",
        "oxygen",
        "o2",
        "aspirin",
        "medication",
        "request",
        "ask for",
        "asked",
        "headache",
        "bedside",
        "did they",
        "did he",
        "did she",
        "show chart",
        "open chart",
        "pull up",
        "tell me about",
        "what is",
        "what's",
        "what did",
        "how is",
        "how he",
        "how she",
        "how they",
        "how's",
        "doing",
        "why is",
        "any comments",
        "any notes",
        "summary for",
        "interested",
        "refresher",
        "surgery",
        "scheduled",
        "minutes",
        "can you",
        "could you",
        "would you",
        "will you",
        "should we",
        "mri",
        "ct scan",
        "x-ray",
        "xray",
        "ultrasound",
        "hologram",
        "draw blood",
        "blood test",
        "lab",
        "administer",
        "inject",
        "intubate",
        "camera",
        "video",
        "coffee",
        "espresso",
    )
    return any(kw in lowered for kw in question_keywords) or _detect_impractical_request_fallback(text) is not None


def _extract_symptom_summary(patient_input: str) -> str:
    """Extract a clean symptom summary for EHR logging."""
    lowered = patient_input.lower()
    if "itch" in lowered and "right foot" in lowered:
        return "Persistent itch in right foot (uncomfortable)"
    cleaned = patient_input.strip()
    cleaned = re.sub(
        r"^(?:(?:yes|yeah|no|well|okay|ok),?\s*)?(?:i\s*(?:am|'m)\s*doing\s*(?:well|good|fine|okay|alright)[.,!\s]*)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:yes|yeah|well|but|so),?\s*(?:i\s*(?:have\s*been\s*facing|have|'ve\s*got|got|feel|am\s*feeling|am\s*having)\s*)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r",?\s*\bthough\b[.!\s]*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^i\s+(?:have\s+a\s+|have\s+|feel\s+|'ve\s+got\s+a\s+|'ve\s+got\s+)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" .,!")
    if not cleaned:
        return "Patient reported physical discomfort"
    return cleaned[0].upper() + cleaned[1:]


def _extract_patient_followup_note(patient_input: str) -> str | None:
    """Extract a specific patient request (e.g., medication or comfort item) from a nurse confirmation message."""
    cleaned = patient_input.strip()
    if not cleaned:
        return None
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = re.sub(
            r"^(?:yes|yeah|yep|yup|sure|ok|okay|please|thank\s+you|thanks|alright|go\s+ahead)[,.\s!]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(
            r"[,.\s!]*(?:please|thank\s+you|thanks)[,.\s!]*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
    if not cleaned:
        return None
    if re.fullmatch(
        r"(?:call\s+(?:a\s+|over\s+a\s+)?nurse|send\s+(?:a\s+|the\s+)?nurse|need\s+a\s+nurse|nurse)",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return None
    nurse_action = re.sub(
        r"^(?:can\s+you\s+|could\s+you\s+|would\s+you\s+)?(?:tell|ask|have)\s+(?:them|her|him|the\s+nurse)\s+(?:to\s+)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" .,!")
    if nurse_action and nurse_action.lower() != cleaned.lower():
        return f"Requested nurse to {nurse_action[0].lower() + nurse_action[1:]}"
    if len(cleaned.split()) >= 2 or any(
        k in cleaned.lower()
        for k in ("aspirin", "tylenol", "ibuprofen", "med", "pain", "water", "blanket", "ice", "doctor", "nurse")
    ):
        return f"Patient request: {cleaned[0].upper() + cleaned[1:]}"
    return None


class LocalRAGAgent:
    """
    Hybrid Cloud + Local RAG triage engine across the multi-patient database.

    Routes pseudonymized prompts to Gemini 3.8 Flash (<=6s timeout) and falls back
    seamlessly to deterministic local RAG templates when offline or unconfigured.
    """

    def __init__(
        self,
        telemetry_provider: BaseTelemetryProvider | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        offline_fallback: bool | None = None,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.telemetry_provider = telemetry_provider or get_telemetry_provider()
        self.model_name = model_name or GEMINI_MODEL
        self.api_key = GEMINI_API_KEY if api_key is None else api_key
        self.offline_fallback = (
            OFFLINE_FALLBACK_MODE if offline_fallback is None else bool(offline_fallback)
        )
        self.timeout_seconds = timeout_seconds
        self._client: Any = None
        self._cloud_backoff_until: float = 0.0
        self._http = requests.Session()
        self._http.trust_env = False
        _adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=1)
        self._http.mount("https://", _adapter)
        self._http.mount("http://", _adapter)
        self._clinician_history: list[tuple[str, str]] = []
        self._last_rescored_transcript: str = ""
        self._last_pane_directive: str = "keep"

    def warm_connection(self) -> None:
        """Pre-warm TLS connection pool to generativelanguage.googleapis.com so the first turn is ~600ms."""
        if not self.is_cloud_configured():
            return
        try:
            self._http.get(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite?key={self.api_key}",
                timeout=(3.0, 4.0),
            )
        except Exception:
            pass

    def is_cloud_configured(self) -> bool:
        """Return True when Gemini API key and online mode are active."""
        if self.offline_fallback:
            return False
        if not self.api_key or self.api_key == "your_api_key_here":
            return False
        return True

    def _resolve_cloud_model(self) -> str:
        """Map UI/legacy model names onto active ultra-low-latency Gemini 3.x REST model IDs."""
        model = (self.model_name or "").strip()
        if not model or model in {"gemini-3.8-flash", "gemini-2.5-flash", "gemini-3.1-flash-lite"}:
            return "gemini-3.5-flash-lite"
        return model

    def get_engine_label(self) -> str:
        """Return active AI engine status label."""
        if self.is_cloud_configured():
            return f"Gemini {self._resolve_cloud_model()} + Local RAG Fallback"
        return "Local Deterministic RAG Fallback (Offline/Demo Safe)"

    def _get_client(self) -> Any:
        """Lazily initialize the google-genai client if installed."""
        if self._client is None and _GENAI_AVAILABLE and self.is_cloud_configured():
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _format_recent_history(self) -> str:
        """Format the last 6 clinician-Malcolm dialogue turns for conversational continuity."""
        if not self._clinician_history:
            return "No prior turns in this session."
        lines: list[str] = []
        for doc_msg, mal_msg in self._clinician_history[-6:]:
            lines.append(f"Doctor: {doc_msg}\nMalcolm: {mal_msg}")
        return "\n".join(lines)

    def _record_clinician_turn(self, doc_msg: str, mal_msg: str) -> None:
        """Append a turn to rolling conversational memory."""
        if doc_msg and mal_msg:
            self._clinician_history.append((doc_msg.strip(), mal_msg.strip()))
            if len(self._clinician_history) > 10:
                self._clinician_history = self._clinician_history[-10:]

    def _query_gemini(
        self,
        system_instruction: str,
        user_prompt: str,
        state: dict[str, Any] | None = None,
        thinking_budget: int = 384,
    ) -> tuple[str | None, str]:
        """
        Send a strictly PII-redacted prompt to generativelanguage.googleapis.com
        using gemini-3.5-flash-lite / gemini-3-flash-preview / gemini-3.5-flash
        with MINIMAL thinking (~0.6s latency) and fast failover.

        Returns (rehydrated_text_or_None, engine_used_label).
        """
        if not self.is_cloud_configured():
            return None, "Local Deterministic RAG Fallback"
        if time.time() < self._cloud_backoff_until:
            return None, "Local Deterministic RAG Fallback (Circuit Breaker)"

        del thinking_budget
        cloud_model = self._resolve_cloud_model()
        models_to_try = [cloud_model]
        for candidate in ("gemini-3.5-flash-lite", "gemini-3-flash-preview", "gemini-3.5-flash"):
            if candidate not in models_to_try:
                models_to_try.append(candidate)

        masked_system = mask_pii(system_instruction, state)
        masked_prompt = mask_pii(user_prompt, state)
        headers = {"Content-Type": "application/json"}

        last_reason = "Local Deterministic RAG Fallback"
        for idx, model_id in enumerate(models_to_try):
            gen_config: dict[str, Any] = {
                "temperature": 0.25,
                "maxOutputTokens": 600,
            }
            if model_id in {
                "gemini-3.5-flash-lite",
                "gemini-3-flash-preview",
                "gemini-3.1-flash-lite",
                "gemini-3.1-flash-lite-preview",
            }:
                gen_config["thinkingConfig"] = {"thinkingLevel": "MINIMAL"}
            elif model_id == "gemini-3.5-flash":
                gen_config["thinkingConfig"] = {"thinkingBudget": 0}
            payload = {
                "systemInstruction": {"parts": [{"text": masked_system}]},
                "contents": [{"role": "user", "parts": [{"text": masked_prompt}]}],
                "generationConfig": gen_config,
            }
            endpoint = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_id}:generateContent?key={self.api_key}"
            )
            timeout = (3.5, 6.5) if idx == 0 else (3.0, 5.5)
            try:
                resp = self._http.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates") or []
                    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
                    raw_reply = "".join(
                        str(p.get("text", ""))
                        for p in parts
                        if not p.get("thought")
                    ).strip()
                    raw_reply = re.sub(r"[*#`]+", "", raw_reply)
                    raw_reply = re.sub(r"\s+", " ", raw_reply).strip()
                    if raw_reply:
                        return unmask_pii(raw_reply, state), f"Gemini {model_id} + Local RAG"
                    last_reason = "Local Deterministic RAG Fallback (Empty Response)"
                    continue
                if resp.status_code in {401, 403}:
                    self._cloud_backoff_until = time.time() + 15.0
                    return None, f"Local Deterministic RAG Fallback (HTTP {resp.status_code})"
                last_reason = f"Local Deterministic RAG Fallback (HTTP {resp.status_code})"
                continue
            except requests.RequestException:
                last_reason = "Local Deterministic RAG Fallback (Network/Timeout)"
                continue
            except Exception:
                return None, "Local Deterministic RAG Fallback"

        return None, last_reason

    def _query_gemini_html(
        self,
        system_instruction: str,
        user_prompt: str,
        state: dict[str, Any] | None = None,
    ) -> tuple[str | None, str]:
        """
        Query Gemini 3.5 Flash-Lite with MINIMAL thinking to synthesize raw HTML & CSS
        for the right-hand split-screen HUD pane (preserving CSS '#', '*', and markup tags,
        while still applying strict PII pseudonymization and local re-hydration).
        """
        if not self.is_cloud_configured():
            return None, "Local Deterministic HTML/CSS Fallback"
        if time.time() < self._cloud_backoff_until:
            return None, "Local Deterministic HTML/CSS Fallback (Circuit Breaker)"

        cloud_model = self._resolve_cloud_model()
        models_to_try = [cloud_model]
        for candidate in ("gemini-3.5-flash-lite", "gemini-3-flash-preview", "gemini-3.5-flash"):
            if candidate not in models_to_try:
                models_to_try.append(candidate)

        masked_system = mask_pii(system_instruction, state)
        masked_prompt = mask_pii(user_prompt, state)
        headers = {"Content-Type": "application/json"}

        for idx, model_id in enumerate(models_to_try):
            gen_config: dict[str, Any] = {
                "temperature": 0.2,
                "maxOutputTokens": 1100,
            }
            if model_id in {
                "gemini-3.5-flash-lite",
                "gemini-3-flash-preview",
                "gemini-3.1-flash-lite",
                "gemini-3.1-flash-lite-preview",
            }:
                gen_config["thinkingConfig"] = {"thinkingLevel": "MINIMAL"}
            elif model_id == "gemini-3.5-flash":
                gen_config["thinkingConfig"] = {"thinkingBudget": 0}
            payload = {
                "systemInstruction": {"parts": [{"text": masked_system}]},
                "contents": [{"role": "user", "parts": [{"text": masked_prompt}]}],
                "generationConfig": gen_config,
            }
            endpoint = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_id}:generateContent?key={self.api_key}"
            )
            timeout = (3.5, 6.5) if idx == 0 else (3.0, 5.5)
            try:
                resp = self._http.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates") or []
                    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
                    raw_html = "".join(
                        str(p.get("text", ""))
                        for p in parts
                        if not p.get("thought")
                    ).strip()
                    # Strip markdown code fences if the model wrapped the HTML
                    raw_html = re.sub(r"^```(?:html|css)?\s*", "", raw_html, flags=re.IGNORECASE)
                    raw_html = re.sub(r"\s*```$", "", raw_html).strip()
                    if raw_html and "<" in raw_html and ">" in raw_html:
                        return unmask_pii(raw_html, state), f"Gemini {model_id} HTML/CSS Engine"
                    continue
                if resp.status_code in {401, 403}:
                    self._cloud_backoff_until = time.time() + 15.0
                    return None, f"Local Deterministic HTML/CSS Fallback (HTTP {resp.status_code})"
                continue
            except Exception:
                continue

        return None, "Local Deterministic HTML/CSS Fallback"

    def _build_fallback_pane_html_css(
        self,
        clinician_request: str,
        db: dict[str, Any] | None = None,
    ) -> str:
        """Construct a deterministic HTML+CSS HUD view from patients_db.json when offline."""
        if _is_timetable_or_calendar_query(clinician_request):
            return render_timetable_html_css()
        hud = build_ward_hud_data(db)
        patients = hud.get("patients") or []
        stats = hud.get("stats") or {}
        lowered = clinician_request.lower()

        is_stats = any(
            k in lowered
            for k in ("stat", "metric", "analytic", "average", "mean", "compare", "distribution", "chart", "graph")
        )
        style_block = """<style>
  .hud-root {
    height: 100%;
    width: 100%;
    box-sizing: border-box;
    padding: 18px 28px 22px 28px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 14px;
    color: #f3f5f8;
    font-family: 'Inter', system-ui, sans-serif;
    overflow: hidden;
  }
  .hud-kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #8b95a7;
    margin: 0;
  }
  .hud-title {
    font-size: clamp(1.5rem, 2.6vw, 2.35rem);
    font-weight: 500;
    letter-spacing: -0.02em;
    margin: 4px 0 0 0;
    color: #ffffff;
  }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }
  .kpi-card {
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.025);
    padding: 12px 14px;
  }
  .kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: #38bdf8;
    margin: 6px 0 2px 0;
  }
  .kpi-val.alert { color: #fb7185; }
  .kpi-sub { font-size: 11px; color: #8b95a7; margin: 0; }
  .hud-table {
    width: 100%;
    border-collapse: collapse;
    border: 1px solid rgba(255, 255, 255, 0.12);
    font-size: 13px;
  }
  .hud-table th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #8b95a7;
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(255, 255, 255, 0.03);
  }
  .hud-table td {
    padding: 11px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    vertical-align: top;
  }
  .hud-table tr:hover {
    background: rgba(56, 189, 248, 0.07);
    cursor: pointer;
  }
  .bpm-pill {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #38bdf8;
  }
  .bpm-pill.high { color: #fb7185; }
  .bar-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 9px 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.02);
    cursor: pointer;
  }
  .bar-track {
    flex: 1;
    height: 10px;
    background: rgba(255, 255, 255, 0.08);
    overflow: hidden;
  }
  .bar-fill { height: 100%; background: #38bdf8; }
  .bar-fill.high { background: #f43f5e; }
  .hud-footer {
    border-top: 2px solid #f43f5e;
    background: rgba(244, 63, 94, 0.14);
    padding: 12px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
</style>"""

        if is_stats:
            bars_html = ""
            for p in patients:
                pct = max(12, min(100, int(((int(p["bpm"]) - 40) / 100.0) * 100)))
                high_cls = "high" if int(p["bpm"]) >= 100 else ""
                bars_html += (
                    f"<div class='bar-row' onclick=\"window.selectPatient('{p['patientId']}')\">"
                    f"<div style='width:190px'><div style='font-weight:600'>#{p['patientId']} {p['name']}</div>"
                    f"<div class='hud-kicker' style='font-size:10px'>{p['wing']} • Rm {p['room']} • {p['condition']}</div></div>"
                    f"<div class='bar-track'><div class='bar-fill {high_cls}' style='width:{pct}%'></div></div>"
                    f"<div class='bpm-pill {high_cls}' style='width:95px;text-align:right'>{p['bpm']} BPM</div>"
                    f"</div>"
                )
            return (
                style_block
                + f"""<div class="hud-root">
  <div>
    <p class="hud-kicker">AI-Synthesized Telemetry Analytics • Live Ward Feed</p>
    <h2 class="hud-title">Ward Telemetry &amp; Acuity Statistics</h2>
  </div>
  <div class="kpi-grid">
    <div class="kpi-card"><p class="hud-kicker">Mean Heart Rate</p><p class="kpi-val">{stats.get('meanBpm', 83.5)} BPM</p><p class="kpi-sub">4 Active Monitored Beds</p></div>
    <div class="kpi-card"><p class="hud-kicker">BPM Range</p><p class="kpi-val">{stats.get('minBpm', 64)}–{stats.get('maxBpm', 104)} BPM</p><p class="kpi-sub">Spread Δ {stats.get('bpmSpread', 40)} BPM</p></div>
    <div class="kpi-card"><p class="hud-kicker">Wake / Sleep</p><p class="kpi-val">{stats.get('awakeCount', 3)} Awake / {stats.get('sleepingCount', 1)} Sleep</p><p class="kpi-sub">Live ECG Wave Lock</p></div>
    <div class="kpi-card"><p class="hud-kicker">Acuity Alerts</p><p class="kpi-val alert">{stats.get('tachycardiaCount', 1)} Tachycardia</p><p class="kpi-sub">{stats.get('symptomaticCount', 3)} Symptomatic</p></div>
  </div>
  <div style="display:flex;flex-direction:column;gap:8px">{bars_html}</div>
  <div class="hud-footer">
    <div><p class="hud-kicker" style="color:#fda4af">Priority Statistical Outlier</p><div style="margin-top:3px;font-size:14px">Peak HR: <strong>{stats.get('maxBpmPatient', 'Jukka Korhonen (1431)')}</strong> at <strong>{stats.get('maxBpm', 104)} BPM</strong> • Lowest HR: <strong>{stats.get('minBpmPatient', 'Sanna Laine (1432)')}</strong> at <strong>{stats.get('minBpm', 64)} BPM</strong></div></div>
  </div>
</div>"""
            )

        rows_html = ""
        for p in patients:
            high_cls = "high" if int(p["bpm"]) >= 100 else ""
            rows_html += (
                f"<tr onclick=\"window.selectPatient('{p['patientId']}')\">"
                f"<td><div class='hud-kicker'>#{p['patientId']}</div><div style='font-weight:600;font-size:15px;margin-top:2px'>{p['name']}</div></td>"
                f"<td><div class='hud-kicker' style='color:#e2e8f0'>{p['wing']}</div><div style='font-size:12px;color:#8b95a7;margin-top:2px'>Room {p['room']}</div></td>"
                f"<td><div style='font-weight:500'>{p['department']}</div><div style='font-size:12px;color:#8b95a7;margin-top:2px'>{p['protocol']}</div></td>"
                f"<td><div class='bpm-pill {high_cls}'>{p['bpm']} BPM</div><div class='hud-kicker' style='font-size:10px;margin-top:2px'>{p['wakeState']} • {p['rhythm']}</div></td>"
                f"<td><div style='font-size:12px;color:#f1f5f9'>{p['symptom']}</div><div style='font-size:11px;color:#8b95a7;margin-top:3px'>{p['nurseNote']}</div></td>"
                f"</tr>"
            )
        return (
            style_block
            + f"""<div class="hud-root">
  <div>
    <p class="hud-kicker">On-Demand AI-Constructed HTML/CSS View</p>
    <h2 class="hud-title">Active Ward Roster — All Monitored Patients</h2>
  </div>
  <table class="hud-table">
    <thead>
      <tr><th>Patient</th><th>Location</th><th>Department &amp; Protocol</th><th>Live Telemetry</th><th>Symptoms &amp; Nursing Notes</th></tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  <div class="hud-footer">
    <div><p class="hud-kicker" style="color:#fda4af">Ward Summary</p><div style="margin-top:3px;font-size:14px">{stats.get('summaryLine', '')}</div></div>
    <span class="hud-kicker">Click any patient row to inspect chart</span>
  </div>
</div>"""
        )

    def generate_pane_html_css(
        self,
        clinician_request: str,
        db: dict[str, Any] | None = None,
    ) -> str:
        """
        Real-time Generative UI: Ask Gemini 3.5 Flash-Lite to construct bespoke HTML & CSS
        (<style>...</style><div class="hud-root">...</div>) on demand ONLY for custom visual comparisons/matrices
        that are not already covered by the native React views.
        """
        if _is_timetable_or_calendar_query(clinician_request):
            return render_timetable_html_css()
        data = db if isinstance(db, dict) else state_manager.load_db()
        all_context = build_all_patients_rag_context()
        ward_hud = build_ward_hud_data(data)
        stats = ward_hud.get("stats") or {}

        sys_prompt = (
            "You are Malcolm's real-time Generative UI Engine for a hospital clinical HUD console. "
            "Construct raw, compact HTML and CSS (<style>...</style> followed by <div class=\"hud-root\">...</div>) "
            "that immediately renders the custom visual comparison, matrix, or breakdown the doctor asked for.\n\n"
            "PRE-LOADED BASE CLASSES (already styled in the host frame—use them directly and only write <style> for custom additions!):\n"
            "- `.hud-root`: full-height flex column container\n"
            "- `.hud-kicker`: uppercase monospace label (11px, #8b95a7)\n"
            "- `.hud-title`: crisp display heading\n"
            "- `.kpi-grid` (4-col grid), `.kpi-card`, `.kpi-val` (cyan metric, add `.alert` for rose), `.kpi-sub`\n"
            "- `.hud-table` (styled clinical table with `th`/`td` and hover highlight), `.bpm-pill` (add `.high` for >=100 BPM rose alert)\n"
            "- `.bar-row`, `.bar-track`, `.bar-fill` (proportional horizontal bar chart, add `.high` for rose alert)\n"
            "- `.hud-footer`: rose-tinted bottom summary banner\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY a short `<style>...</style>` block (for custom classes/layout, max 12 lines) followed by `<div class=\"hud-root\">...</div>`. No markdown fences, no <html>/<head>/<body>.\n"
            "2. Keep HTML concise and clean so it generates in under 1 second and fits on one screen.\n"
            "3. Interactive Hooks: Add `onclick=\"window.selectPatient('1429')\"` (or the patient's ID) on patient rows/cards/bars.\n"
            "4. READ-ONLY CLINICAL DATA ONLY — NO FAKE BUTTONS: NEVER output `<button>`, `<input>`, `<select>`, or `<form>` elements that pretend to delete records, clear backend state, administer drugs, or run hardware scans. Only display real clinical data from the Multi-Patient Ward Database below.\n"
            "5. Clinical Accuracy: Use ONLY the pseudonymized patient tokens and exact numbers below.\n\n"
            f"Multi-Patient Ward Database:\n{all_context}\n\n"
            f"Computed Ward Statistics:\n"
            f"- Total: {stats.get('totalPatients', 4)} patients | Mean HR: {stats.get('meanBpm', 83.5)} BPM | Range: {stats.get('minBpm', 64)}–{stats.get('maxBpm', 104)} BPM (Spread {stats.get('bpmSpread', 40)} BPM)\n"
            f"- Wakefulness: {stats.get('awakeCount', 3)} Awake / {stats.get('sleepingCount', 1)} Sleeping | Symptomatic: {stats.get('symptomaticCount', 3)} | Tachycardia: {stats.get('tachycardiaCount', 1)}"
        )
        user_msg = f"Construct the read-only HTML and CSS clinical data display for this doctor request: {clinician_request}"
        generated_html, _ = self._query_gemini_html(sys_prompt, user_msg)
        if generated_html and "<div" in generated_html.lower():
            # Sanitize out any hallucinated <button>, <input>, or <form> elements so fake UI controls can never appear
            cleaned_html = re.sub(r"<button\b[^>]*>.*?</button>", "", generated_html, flags=re.IGNORECASE | re.DOTALL)
            cleaned_html = re.sub(r"<(?:input|select|textarea|form)\b[^>]*>", "", cleaned_html, flags=re.IGNORECASE)
            return cleaned_html
        return self._build_fallback_pane_html_css(clinician_request, data)

    def _query_gemini_with_rescore(
        self,
        system_instruction: str,
        user_prompt: str,
        state: dict[str, Any] | None = None,
    ) -> tuple[str, str | None, str]:
        """
        Zero-latency Pass-2 Joint Contextual ASR Rescoring + On-Demand Pane Control + Malcolm Response Generation.
        In a single Gemini Flash-Lite call, reconstructs any clipped/misheard STT homophones
        in the doctor's spoken input ('TRANSCRIPT:'), decides the split-screen HUD pane action ('PANE:'),
        and generates Malcolm's spoken reply ('REPLY:').

        Returns (rescored_transcript, rehydrated_reply_or_None, engine_used_label).
        """
        clean_input = user_prompt.strip()
        self._last_pane_directive = "keep"
        rescore_directive = (
            "\n\nMALCOLM SYSTEM CAPABILITIES, NATIVE HUD PANE CONTROL & IMPOSSIBLE REQUEST RULES:\n"
            "1. BUILT-IN SYSTEM CAPABILITIES (YOU CAN DO ALL OF THESE!):\n"
            "   - NATIVE HUD VIEWS: You control the right-hand split-screen pane with 4 readily-available native views: "
            "Patient Chart (`patient_1429`, `patient_1430`, `patient_1431`, `patient_1432`), Active Ward Roster Table (`ward_table`), "
            "Ward Telemetry & Acuity Statistics (`ward_stats`), and Shift Timetable (`keep` when timetable handler runs).\n"
            "   - CUSTOM HTML/CSS VIEWS (`custom_html`): Only use `PANE: custom_html` when the doctor asks for a bespoke visual comparison/matrix that is NOT a standard patient chart, ward table, or ward statistics view.\n"
            "   - .XLSX & .PNG FILE EXPORT: You HAVE built-in `.xlsx` (Excel spreadsheet) and `.png` (high-res image on a plain white background) export & download capabilities! NEVER say you cannot export or generate `.xlsx` or `.png` files.\n"
            "   - ADD & DELETE PATIENTS: You HAVE built-in database access to admit/add new patients and discharge/delete/remove patients from the live ward roster! NEVER say you cannot add or delete patients.\n"
            "   - LOUDSPEAKER PAGING & TIMERS: You HAVE built-in hospital loudspeaker paging and can page the doctor in X seconds (e.g., 'in 10 seconds'), X minutes, or 5 minutes prior to scheduled timetable tasks.\n"
            "2. IMPOSSIBLE / UNSUPPORTED REQUESTS (NEVER PRETEND OR GENERATE FAKE UI FOR THESE!):\n"
            "   - If the doctor asks you to do something physically impossible or unsupported by our console (e.g., drawing blood, administering IV drugs, performing surgery, showing MRI/CT/X-ray/ultrasound scans, streaming live room CCTV video, projecting a 3D hologram, wiping/hacking the hospital server, or making coffee/food):\n"
            "   - You MUST set `PANE: keep` (NEVER open `custom_html` or pretend to build a UI with buttons for an impossible task!).\n"
            "   - Casually and honestly tell the doctor that you cannot do that from the console, and suggest a real capability you CAN do right now instead.\n\n"
            "IMPORTANT — PASS-2 CONTEXTUAL STT RESCORING + RESPONSE FORMAT:\n"
            "The doctor's input comes from live hospital Speech-to-Text and may have clipped opening words, "
            "spoken hesitations ('er', 'uh', 'um'), homophones ('wing see' -> 'Wing C', 'cardia' -> 'tachycardia', "
            "'said you'll in' -> 'scheduled in', 'p and g' -> 'PNG', 'x l s x' -> 'XLSX'), or phonetic Finnish names "
            "('Aham linen' / 'Hama linen' -> [PATIENT_1429], 'Jukka' -> [PATIENT_1431], 'Aino' -> [PATIENT_1430], 'Sanna' -> [PATIENT_1432]).\n"
            "Output EXACTLY three labeled sections:\n"
            "TRANSCRIPT: <the doctor's intended utterance, grammar/homophones repaired, filler words removed, and properly punctuated>\n"
            "PANE: <keep | close | ward_table | ward_stats | custom_html | patient_1429 | patient_1430 | patient_1431 | patient_1432>\n"
            "REPLY: <your 1 to 3 sentence Jarvis clinical response>"
        )
        raw_out, engine_used = self._query_gemini(
            system_instruction + rescore_directive,
            clean_input,
            state=state,
        )
        if not raw_out:
            return clean_input, None, engine_used

        m_pane = re.search(
            r"PANE:\s*(keep|close|ward_table|ward_stats|custom_html|patient_\d+)",
            raw_out,
            flags=re.IGNORECASE,
        )
        if m_pane:
            self._last_pane_directive = m_pane.group(1).strip().lower()

        m_tr = re.search(
            r"TRANSCRIPT:\s*(.+?)(?=\s*(?:PANE:|REPLY:)|$)",
            raw_out,
            flags=re.IGNORECASE | re.DOTALL,
        )
        m_rep = re.search(
            r"REPLY:\s*(.+)$",
            raw_out,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m_rep:
            reply_text = m_rep.group(1).strip()
            reply_text = re.sub(
                r"\bPANE:\s*(?:keep|close|ward_table|ward_stats|custom_html|patient_\d+)\b",
                "",
                reply_text,
                flags=re.IGNORECASE,
            ).strip()
            rescored = m_tr.group(1).strip().strip('"').strip("'") if m_tr else clean_input
            rescored = re.sub(
                r"\bPANE:\s*(?:keep|close|ward_table|ward_stats|custom_html|patient_\d+)\b",
                "",
                rescored,
                flags=re.IGNORECASE,
            ).strip()
            if clean_input.endswith("?") or not rescored or len(rescored) < 3:
                rescored = clean_input
            if any(
                refusal in reply_text.lower()
                for refusal in ("i cannot", "i can't", "i'm afraid", "outside my", "unable to", "no-go", "off-limits", "not something i can")
            ):
                self._last_pane_directive = "keep"
            return rescored, reply_text, engine_used

        stripped = re.sub(r"^TRANSCRIPT:\s*", "", raw_out, flags=re.IGNORECASE).strip()
        stripped = re.sub(
            r"\bPANE:\s*(?:keep|close|ward_table|ward_stats|custom_html|patient_\d+)\b",
            "",
            stripped,
            flags=re.IGNORECASE,
        ).strip()
        return clean_input, stripped, engine_used

    def _resolve_pane_updates(
        self,
        text: str,
        pane_intent: dict[str, Any] | None,
        target_id: str | None,
        current_state: dict[str, Any],
        generated_html: str = "",
    ) -> dict[str, Any]:
        """Compute ui_mode and custom_pane updates using native React views first and blocking fake UIs on impossible requests."""
        updates: dict[str, Any] = {}
        directive = (self._last_pane_directive or "keep").lower()

        # Hard block: NEVER open or generate a pane when the request is impossible/impractical!
        if _detect_impractical_request_fallback(text, current_state) is not None:
            return updates

        # 1. Explicit deterministic intent takes top priority for instant reliability
        if pane_intent:
            action = pane_intent.get("action")
            if action == "close":
                updates["ui_mode"] = "idle"
                updates["custom_pane"] = {
                    "kind": "patient",
                    "title": "",
                    "subtitle": "",
                    "filterWing": "",
                    "generatedHtml": "",
                }
                return updates
            if action == "open":
                kind = str(pane_intent.get("kind", "patient"))
                # Only generate HTML when kind is 'custom_html' or 'timetable'
                html_code = ""
                if kind == "timetable":
                    html_code = generated_html or render_timetable_html_css()
                elif kind == "custom_html" and pane_intent.get("generate_html"):
                    html_code = generated_html or self.generate_pane_html_css(text)
                updates["ui_mode"] = "patient_stats"
                updates["custom_pane"] = {
                    "kind": kind,
                    "title": str(pane_intent.get("title", "")),
                    "subtitle": str(pane_intent.get("subtitle", "")),
                    "filterWing": str(pane_intent.get("filterWing", "")),
                    "generatedHtml": html_code,
                }
                return updates

        # 2. Gemini's single-pass PANE directive (uses native React views for ward_table, ward_stats, and patient!)
        if directive == "close":
            updates["ui_mode"] = "idle"
            updates["custom_pane"] = {
                "kind": "patient",
                "title": "",
                "subtitle": "",
                "filterWing": "",
                "generatedHtml": "",
            }
            return updates
        if directive == "ward_table":
            updates["ui_mode"] = "patient_stats"
            updates["custom_pane"] = {
                "kind": "ward_table",
                "title": "Active Ward Roster — All Monitored Patients",
                "subtitle": "Real-Time Multi-Patient EHR & Telemetry Matrix",
                "filterWing": "",
                "generatedHtml": "",
            }
            return updates
        if directive == "ward_stats":
            updates["ui_mode"] = "patient_stats"
            updates["custom_pane"] = {
                "kind": "ward_stats",
                "title": "Ward Telemetry & Acuity Statistics",
                "subtitle": "Live Statistical Breakdown • Monitored Ward Beds",
                "filterWing": "",
                "generatedHtml": "",
            }
            return updates
        if directive == "custom_html":
            # Only allow custom_html if the clinician actually asked for a visual display/comparison/matrix
            lowered = text.lower()
            if any(
                w in lowered
                for w in ("compare", "comparison", "matrix", "custom", "breakdown", "side-by-side", "side by side", "chart", "display", "visual", "layout")
            ):
                html_code = generated_html or self.generate_pane_html_css(text)
                updates["ui_mode"] = "patient_stats"
                updates["custom_pane"] = {
                    "kind": "custom_html",
                    "title": "AI-Synthesized Clinical View",
                    "subtitle": "Custom Generative HTML & CSS Display",
                    "filterWing": "",
                    "generatedHtml": html_code,
                }
            return updates
        if directive.startswith("patient_"):
            updates["ui_mode"] = "patient_stats"
            updates["custom_pane"] = {
                "kind": "patient",
                "title": "",
                "subtitle": "",
                "filterWing": "",
                "generatedHtml": "",
            }
            return updates

        # 3. If targeting a specific patient while a ward_table or ward_stats was open, switch to that patient's chart
        existing_pane = current_state.get("custom_pane") or {}
        if target_id and existing_pane.get("kind") in {"ward_table", "ward_stats"}:
            updates["custom_pane"] = {
                "kind": "patient",
                "title": "",
                "subtitle": "",
                "filterWing": "",
                "generatedHtml": "",
            }
        return updates

    def _get_telemetry_for_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Query telemetry provider anchored to the patient's baseline BPM & wave state."""
        baseline = int(state.get("baseline_bpm", 90))
        wave_state = str(state.get("wave_state", "Awake / Normal Sinus Rhythm"))
        return self.telemetry_provider.get_reading(
            baseline_bpm=baseline,
            wave_state=wave_state,
        )

    def synthesize_patient_panel(
        self,
        state: dict[str, Any],
        telemetry: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """
        Construct the right-side split-screen PatientPane data directly from EHR JSON
        via Local RAG + optional Gemini 3.8 Flash JSON synthesis (PII redacted).
        """
        live_telemetry = telemetry or self._get_telemetry_for_state(state)
        condition = str(state.get("condition", "Dialysis"))
        room_raw = str(state.get("room", "Room 153"))
        room_num = re.sub(r"^Room\s+", "", room_raw, flags=re.IGNORECASE).strip() or "153"
        wave_state = str(
            live_telemetry.get("wave_state")
            or state.get("wave_state")
            or "Awake / Normal Sinus Rhythm"
        )
        baseline_bpm = int(state.get("baseline_bpm", 90))
        check_in_active = bool(state.get("check_in_active"))
        nurse_dispatched = bool(state.get("nurse_dispatched"))
        nurse_notes = state.get("nurse_notes") or []
        symptoms = state.get("patient_symptoms") or []

        department = str(
            state.get("department")
            or ("Nephrology — Renal Care" if "dialysis" in condition.lower() else condition)
        )
        protocol = str(
            state.get("protocol")
            or ("Routine Dialysis Protocol" if "dialysis" in condition.lower() else f"{condition} Monitoring")
        )
        if "/" in wave_state:
            wake_label, rhythm_label = [part.strip() for part in wave_state.split("/", 1)]
        else:
            wake_label = "Sleeping" if "sleep" in wave_state.lower() else "Awake"
            rhythm_label = wave_state

        telemetry_detail = (
            f"{wake_label} • Bedside Check-In Active"
            if check_in_active
            else f"{wake_label} • Baseline {baseline_bpm} BPM • Signal Locked"
        )
        nurse_rounds = (
            " ".join(str(note) for note in nurse_notes)
            if nurse_notes
            else "No comments from nurse rounds."
        )

        joined_symptoms = "; ".join(str(s) for s in symptoms).lower()
        if symptoms:
            if "itch" in joined_symptoms and "right foot" in joined_symptoms:
                symptom_text = (
                    "Persistent itch in right foot — won't go away and becoming very uncomfortable."
                )
            else:
                symptom_text = "; ".join(str(s) for s in symptoms)
        elif check_in_active:
            symptom_text = "Live bedside check-in in progress — awaiting patient response..."
        elif nurse_notes:
            symptom_text = "No new acute discomfort reported since last nursing round."
        else:
            symptom_text = "Patient has not reported any discomfort or odd symptoms thus far."

        if nurse_dispatched:
            dispatch_status = f"Nurse dispatched to {room_raw}"
        elif check_in_active:
            dispatch_status = "Bedside check-in active"
        else:
            dispatch_status = ""

        panel: dict[str, str] = {
            "name": str(state.get("name", "Mika Hämäläinen")),
            "patientId": str(state.get("patient_id", "1429")),
            "wing": str(state.get("wing", "Wing A")),
            "room": room_num,
            "department": department,
            "protocol": protocol,
            "rhythm": rhythm_label,
            "telemetryDetail": telemetry_detail,
            "nurseRounds": nurse_rounds,
            "symptom": symptom_text,
            "dispatchStatus": dispatch_status,
        }

        return panel

    def _build_multi_patient_overview(self, question: str) -> tuple[str, str]:
        """Answer cross-patient questions over the entire patients_db.json database."""
        db = state_manager.load_db()
        patients = db.get("patients", {})
        ward_hud = build_ward_hud_data(db)
        ward_stats = ward_hud.get("stats") or {}
        lowered = question.lower()

        impractical_fb = _detect_impractical_request_fallback(question)
        if impractical_fb:
            fallback_reply = impractical_fb
        elif any(k in lowered for k in ("statistic", "statistics", "ward stats", "vitals stats", "telemetry stats", "analytics", "metrics", "average heart", "mean heart", "compare")):
            fallback_reply = (
                f"I've pulled up the live Ward Telemetry & Acuity Statistics pane on your right, doctor. "
                f"Across all {ward_stats.get('totalPatients', 4)} patients, mean heart rate is {ward_stats.get('meanBpm', 83.5)} BPM "
                f"(ranging from {ward_stats.get('minBpm', 64)} BPM on Sanna Laine to {ward_stats.get('maxBpm', 104)} BPM on Jukka Korhonen), "
                f"with {ward_stats.get('awakeCount', 3)} awake and {ward_stats.get('sleepingCount', 1)} sleeping."
            )
        elif any(k in lowered for k in ("table", "roster", "spreadsheet", "grid", "board")):
            summaries = []
            for pid in sorted(patients.keys()):
                p = patients[pid]
                summaries.append(
                    f"{pid} {p['name']} ({p.get('current_bpm', p['baseline_bpm'])} BPM)"
                )
            fallback_reply = (
                f"Right away, doctor—I've constructed the live patient roster table on your display pane covering all {len(patients)} patients: "
                + ", ".join(summaries)
                + ". Click any row or name a patient whenever you'd like to drill into their chart."
            )
        elif "highest" in lowered and ("heart" in lowered or "bpm" in lowered):
            top = max(patients.values(), key=lambda p: int(p.get("current_bpm", p.get("baseline_bpm", 0))))
            fallback_reply = (
                f"Patient {top['patient_id']}, {top['name']} in {top['wing']}, {top['room']}, "
                f"currently has the highest heart rate at {top.get('current_bpm', top['baseline_bpm'])} BPM "
                f"({top['wave_state']}) while under treatment for {top['condition']}."
            )
        elif "lowest" in lowered and ("heart" in lowered or "bpm" in lowered):
            low = min(patients.values(), key=lambda p: int(p.get("current_bpm", p.get("baseline_bpm", 999))))
            fallback_reply = (
                f"Patient {low['patient_id']}, {low['name']} in {low['wing']}, {low['room']}, "
                f"currently has the lowest heart rate at {low.get('current_bpm', low['baseline_bpm'])} BPM "
                f"({low['wave_state']})."
            )
        elif "sleeping" in lowered or "asleep" in lowered:
            sleeping = [p for p in patients.values() if "sleep" in str(p.get("wave_state", "")).lower()]
            if sleeping:
                names = ", ".join(f"{p['name']} (Patient {p['patient_id']}, {p['room']})" for p in sleeping)
                fallback_reply = f"Based on live ECG wave patterns, {names} is currently sleeping."
            else:
                fallback_reply = "All four monitored patients currently show awake wave patterns."
        else:
            summaries = []
            for pid in sorted(patients.keys()):
                p = patients[pid]
                summaries.append(
                    f"{pid} {p['name']} ({p.get('current_bpm', p['baseline_bpm'])} BPM)"
                )
            fallback_reply = (
                f"We have {len(patients)} patients on the floor: "
                + ", ".join(summaries)
                + ". Which patient would you like to examine, doctor?"
            )

        all_context = build_all_patients_rag_context()
        stats_context = ward_stats.get("summaryLine", "")
        history_text = self._format_recent_history()
        sys_prompt = (
            "You are Malcolm, the attending physician's clinical chief-of-staff AI with the brisk cadence, "
            "composed wit, and effortless competence of Jarvis from Iron Man. "
            "Speak like a trusted hospital colleague in 2 to 3 crisp spoken sentences (no markdown or bullet points). "
            "If the doctor asks you to show or construct a table/roster of patients, confirm that you've put the live Ward Table up on the right-hand pane and briefly highlight the 4 patients. "
            "If the doctor asks for statistics/metrics/analytics, confirm that the Ward Statistics pane is now open and cite the live statistical figures (mean BPM, min/max BPM, wake/sleep & acuity counts). "
            "If the doctor includes casual banter (such as getting coffee ready), open with a brief, charming Jarvis remark "
            "before delivering the ward briefing. "
            "Use ONLY the pseudonymized multi-patient EHR database and statistics below. When summarizing the ward or listing all patients, "
            "cover all 4 patients: Patient 1429 ([PATIENT_1429]), Patient 1430 ([PATIENT_1430]), "
            "Patient 1431 ([PATIENT_1431]), and Patient 1432 ([PATIENT_1432]) with their key vitals and status, "
            "and ask which patient the doctor would like to examine first. "
            "Never diagnose or fabricate clinical values.\n\n"
            f"Multi-Patient Ward Database:\n{all_context}\n\n"
            f"Live Ward Telemetry Statistics:\n{stats_context}\n\n"
            f"Recent Conversation:\n{history_text}"
        )
        cloud_Export = self._query_gemini_with_rescore(sys_prompt, question)
        rescored_q, cloud_reply, engine_used = cloud_Export
        self._last_rescored_transcript = rescored_q or question
        is_explicit_db_list = "database" in lowered or "list all" in lowered
        if cloud_reply:
            if is_explicit_db_list and not all(pid in cloud_reply for pid in patients.keys()):
                self._record_clinician_turn(self._last_rescored_transcript, fallback_reply)
                return fallback_reply, engine_used
            mentioned_count = sum(
                1
                for pid, p in patients.items()
                if pid in cloud_reply or str(p.get("name", "")).split()[0].lower() in cloud_reply.lower()
            )
            if mentioned_count >= 3 or not is_explicit_db_list:
                self._record_clinician_turn(self._last_rescored_transcript, cloud_reply)
                return cloud_reply, engine_used
        self._record_clinician_turn(self._last_rescored_transcript, fallback_reply)
        return fallback_reply, engine_used

    def _answer_clinical_question(
        self,
        question: str,
        state: dict[str, Any],
        telemetry: dict[str, Any],
    ) -> tuple[str, str]:
        """Answer an open-ended clinical question about the active patient using Local RAG + Gemini Thinking."""
        lowered = question.lower()
        name = state.get("name", "Mika Hämäläinen")
        pid = state.get("patient_id", "1429")
        wing = state.get("wing", "Wing A")
        room = state.get("room", "Room 153")
        condition = state.get("condition", "Dialysis")
        department = state.get("department", "Nephrology — Renal Care")
        protocol = state.get("protocol", "Routine Dialysis")
        bpm = telemetry.get("bpm", state.get("current_bpm", 90))
        wave = telemetry.get("wave_state", state.get("wave_state", "Awake / Normal Sinus Rhythm"))
        notes = state.get("nurse_notes") or []
        symptoms = state.get("patient_symptoms") or []
        bedside_patient_msgs = [
            str(m.get("text", "")).strip()
            for m in (state.get("bedside_messages") or [])
            if m.get("sender") == "patient" and str(m.get("text", "")).strip()
        ]
        dispatched = bool(state.get("nurse_dispatched"))

        wants_nurse_notes = any(
            k in lowered for k in ("nurse note", "nurse round", "nursing round", "round comment", "prior round")
        ) or ("nurse" in lowered and not any(k in lowered for k in ("ask", "request", "bring", "aspirin", "sent with", "medication", "bedside")))
        wants_bedside_or_symptom = any(
            k in lowered
            for k in (
                "aspirin",
                "ask for",
                "asked",
                "request",
                "bring",
                "sent with",
                "bedside",
                "check-in",
                "check in",
                "what did they",
                "what did he",
                "what did she",
                "headache",
                "symptom",
                "discomfort",
                "pain",
                "itch",
            )
        )

        impractical_fb = _detect_impractical_request_fallback(question, state)
        if impractical_fb:
            fallback = impractical_fb
        elif wants_bedside_or_symptom and wants_nurse_notes:
            notes_str = " ".join(notes) if notes else "no comments recorded from prior nurse rounds"
            sym_str = "; ".join(symptoms) if symptoms else "no new symptoms reported"
            fallback = (
                f"For Patient {pid} ({name} in {wing}, {room}), nursing rounds show {notes_str}, "
                f"and reported symptoms or bedside requests list: {sym_str}."
            )
        elif wants_bedside_or_symptom:
            details: list[str] = []
            if symptoms:
                details.append("; ".join(symptoms))
            extra_quotes = [
                msg
                for msg in bedside_patient_msgs
                if not any(msg.lower() in s.lower() for s in symptoms)
            ]
            if extra_quotes:
                details.append(f'in their bedside chat they also said: "{"; ".join(extra_quotes)}"')
            dispatch_note = (
                " A nurse has been dispatched to the room."
                if dispatched
                else ""
            )
            if details:
                fallback = (
                    f"During the bedside check-in, {name} (Patient {pid} in {wing}, {room}) "
                    f"noted: {', '.join(details)}.{dispatch_note}"
                )
            else:
                fallback = (
                    f"{name} (Patient {pid} in {wing}, {room}) has not reported any discomfort, "
                    f"medication requests, or symptoms so far."
                )
        elif wants_nurse_notes:
            notes_str = " ".join(notes) if notes else "no comments recorded from prior nurse rounds"
            fallback = f"For Patient {pid} ({name} in {wing}, {room}), nursing rounds show: {notes_str}."
        elif "protocol" in lowered or "medication" in lowered or "treatment" in lowered or "department" in lowered:
            fallback = (
                f"{name} (Patient {pid}) is in {department} ({wing}, {room}) for {condition}, "
                f"currently following the {protocol} protocol."
            )
        elif "heart" in lowered or "bpm" in lowered or "rate" in lowered or "vitals" in lowered or "wave" in lowered or "sleep" in lowered:
            fallback = (
                f"Live telemetry for {name} (Patient {pid}) is {bpm} BPM showing {wave}, "
                f"and the patient appears to be {'sleeping' if 'sleep' in wave.lower() else 'awake'}."
            )
        else:
            notes_str = " ".join(notes) if notes else "no prior nursing round comments"
            sym_str = "; ".join(symptoms) if symptoms else "no new symptoms reported"
            fallback = (
                f"{name} (Patient {pid} in {wing}, {room}) is in {department} for {condition} on {protocol}. "
                f"Live telemetry is steady at {bpm} BPM ({wave}), with {sym_str} and {notes_str}."
            )

        name_tok, _, _ = _patient_tokens(str(pid))
        redacted_context = build_redacted_ehr_context(state, telemetry)
        all_context = build_all_patients_rag_context()
        ward_stats_line = (build_ward_hud_data().get("stats") or {}).get("summaryLine", "")
        history_text = self._format_recent_history()
        sys_prompt = (
            "You are Malcolm, the attending physician's hyper-capable clinical chief-of-staff AI with the brisk cadence, "
            "composed confidence, and dry professional camaraderie of Jarvis from Iron Man. "
            "You are a trusted member of the hospital staff speaking live with the doctor: be sharp, concise (1 to 3 spoken sentences, "
            "no markdown bullets or asterisks), warm yet brisk, and feel free to engage in natural hospital-staff banter or ask a smart "
            "proactive clinical follow-up question when appropriate. "
            f"When referring to the active patient, use {name_tok} (or their patient token) so local PII re-hydration restores their real name. "
            "You have full visibility into both the Active Patient EHR (including Reported Symptoms, Nurse Round Comments, and exact Bedside Chat Transcript "
            "requests like aspirin, headaches, or discomfort) AND the entire 4-patient ward database and statistical metrics below. "
            "Never invent clinical values or issue a medical diagnosis.\n\n"
            f"Active Patient Context:\n{redacted_context}\n\n"
            f"All Patients on Ward:\n{all_context}\n\n"
            f"Live Ward Statistics:\n{ward_stats_line}\n\n"
            f"Recent Conversation:\n{history_text}"
        )
        rescored_q, cloud_reply, engine_used = self._query_gemini_with_rescore(sys_prompt, question, state)
        self._last_rescored_transcript = rescored_q or question
        if cloud_reply:
            first_name = str(name).split()[0].lower()
            has_patient_ref = (
                str(name).lower() in cloud_reply.lower()
                or first_name in cloud_reply.lower()
                or str(pid) in cloud_reply
                or any(other_pid in cloud_reply for other_pid in ("1429", "1430", "1431", "1432"))
            )
            if not has_patient_ref and (wants_bedside_or_symptom or wants_nurse_notes or "protocol" in lowered):
                cloud_reply = f"For {name} (Patient {pid}): {cloud_reply}"
            self._record_clinician_turn(self._last_rescored_transcript, cloud_reply)
            return cloud_reply, engine_used
        self._record_clinician_turn(self._last_rescored_transcript, fallback)
        return fallback, engine_used

    def _build_clinician_return_report(
        self,
        state: dict[str, Any],
        telemetry: dict[str, Any],
    ) -> tuple[str, str]:
        """Generate Malcolm's clinician return report based on logged bedside state."""
        symptoms = state.get("patient_symptoms") or []
        nurse_dispatched = bool(state.get("nurse_dispatched"))
        joined_symptoms = "; ".join(symptoms).lower()

        if "itch" in joined_symptoms and "right foot" in joined_symptoms and nurse_dispatched and len(symptoms) == 1:
            fallback_report = (
                "I'm back, sir. The patient said they're doing well, "
                "but they have a persistent itch on their right foot. "
                "I do not know if it's relevant. Nevertheless, they requested that "
                "a nurse be sent to their room so it must be uncomfortable for the patient."
            )
        elif nurse_dispatched:
            symptom_desc = ", ".join(symptoms) if symptoms else "unspecified discomfort"
            fallback_report = (
                f"I'm back, sir. The patient reported: {symptom_desc}. "
                "I have logged all details and dispatched a nurse to their room as requested."
            )
        else:
            symptom_desc = ", ".join(symptoms) if symptoms else "no acute symptoms"
            fallback_report = (
                f"I'm back, sir. The patient reported: {symptom_desc}. "
                "They declined a nurse visit at this time, and I have logged this in the chart."
            )

        redacted_context = build_redacted_ehr_context(state, telemetry)
        sys_prompt = (
            "You are Malcolm, a crisp Jarvis-style hospital clinical triage copilot reporting back to the attending doctor "
            "after a bedside check-in. Speak in 2 to 3 natural spoken sentences starting with 'I'm back, sir.' or 'I'm back, doctor.' "
            "(no markdown, no headings, no bullet points). "
            "Read BOTH Reported Symptoms and Bedside Chat Transcript carefully: summarize ALL symptoms AND any specific requests "
            "the patient made (such as medications like aspirin), state clearly whether a nurse was requested and dispatched, and avoid diagnosing.\n\n"
            f"EHR Context:\n{redacted_context}"
        )
        cloud_report, engine_used = self._query_gemini(
            sys_prompt,
            "Provide the bedside check-in return report to the clinician.",
            state,
            thinking_budget=512,
        )
        if "itch" in joined_symptoms and "right foot" in joined_symptoms and nurse_dispatched and len(symptoms) == 1:
            return fallback_report, engine_used
        return cloud_report or fallback_report, engine_used

    def process_clinician_turn(self, clinician_input: str) -> dict[str, Any]:
        """Execute one turn of the Clinician triage & conversational RAG workflow."""
        text = clinician_input.strip()
        if not text:
            state = state_manager.load_state()
            return {
                "reply": "Standing by, doctor. Speak or enter a patient ID to begin.",
                "engine": self.get_engine_label(),
                "stage": state.get("conversation_stage", "idle"),
                "check_in_active": bool(state.get("check_in_active")),
                "state": state,
            }

        db = state_manager.load_db()
        active_id = str(db.get("active_patient_id", "1429"))
        target_id = _detect_target_patient_id(text, db)
        pane_intent = _detect_pane_intent(text)
        self._last_rescored_transcript = text
        self._last_pane_directive = "keep"

        # Switch active patient automatically if clinician named a specific patient ID/name/room/condition
        if target_id and target_id != active_id:
            state_manager.set_active_patient(target_id)
            active_id = target_id

        # 0. Explicit request to close/hide/dismiss the split-screen pane
        if pane_intent and pane_intent.get("action") == "close":
            state = state_manager.load_state(active_id)
            telemetry = self._get_telemetry_for_state(state)
            panel = self.synthesize_patient_panel(state, telemetry)
            fallback_close = (
                "Right away, doctor. I've closed the display pane and returned the console to standby—"
                "just say the word if you'd like the patient table, statistics, or a chart back on screen."
            )
            sys_close = (
                "You are Malcolm, the attending physician's Jarvis-style clinical AI. "
                "The doctor just asked you to close or put away the split-screen display pane. "
                "Confirm briskly and casually in 1 short spoken sentence that you've closed the pane and are standing by."
            )
            rescored_q, cloud_close, engine_used = self._query_gemini_with_rescore(sys_close, text, state)
            reply = cloud_close or fallback_close
            self._record_clinician_turn(rescored_q or text, reply)
            updated = state_manager.update_state(
                {
                    "ui_mode": "idle",
                    "custom_pane": {
                        "kind": "patient",
                        "title": "",
                        "subtitle": "",
                        "filterWing": "",
                    },
                    "last_clinician_input": rescored_q or text,
                    "last_transcript": reply,
                    "patient_panel": panel,
                },
                patient_id=active_id,
            )
            state_manager.append_audit_log(
                actor="Clinician -> Malcolm",
                action="PANE_CLOSED",
                details="Closed split-screen HUD pane on clinician voice/text command.",
                pii_redacted=True,
                engine=engine_used,
                patient_id=active_id,
            )
            return {
                "reply": reply,
                "engine": engine_used,
                "stage": updated.get("conversation_stage", "idle"),
                "check_in_active": bool(updated.get("check_in_active")),
                "state": state_manager.load_state(active_id),
            }

        # 0.1. Patient Add / Admit & Delete / Discharge CRUD requests
        crud_req = _detect_patient_crud_request(text, db)
        if crud_req:
            c_action = crud_req.get("action")
            if c_action == "capability":
                state = state_manager.load_state(active_id)
                telemetry = self._get_telemetry_for_state(state)
                panel = self.synthesize_patient_panel(state, telemetry)
                reply = (
                    "Yes, doctor—I have full read and write access to the live ward roster. "
                    "Just tell me a patient's name, wing, room, and condition to admit a new patient, "
                    "or tell me which patient ID or name you'd like to discharge and remove."
                )
                self._record_clinician_turn(text, reply)
                updated = state_manager.update_state(
                    {
                        "ui_mode": "patient_stats",
                        "custom_pane": {
                            "kind": "ward_table",
                            "title": "Active Ward Roster — All Monitored Patients",
                            "subtitle": "Real-Time Multi-Patient EHR & Telemetry Matrix",
                            "filterWing": "",
                            "generatedHtml": "",
                        },
                        "last_clinician_input": text,
                        "last_transcript": reply,
                        "patient_panel": panel,
                    },
                    patient_id=active_id,
                )
                return {
                    "reply": reply,
                    "engine": self.get_engine_label(),
                    "stage": updated.get("conversation_stage", "idle"),
                    "check_in_active": bool(updated.get("check_in_active")),
                    "state": state_manager.load_state(active_id),
                }

            if c_action == "add":
                created = state_manager.add_patient(
                    name=str(crud_req.get("name", "Alex Rivera")),
                    wing=str(crud_req.get("wing", "Wing B")),
                    room=str(crud_req.get("room", "Room 210")),
                    condition=str(crud_req.get("condition", "General Clinical Observation")),
                    department=str(crud_req.get("department", "Internal Medicine")),
                    protocol=str(crud_req.get("protocol", "Continuous Telemetry & Q4H Vitals")),
                    baseline_bpm=int(crud_req.get("baseline_bpm", 82)),
                )
                new_id = str(created.get("patient_id", "1433"))
                total_count = len(state_manager.load_db().get("patients", {}))
                reply = (
                    f"Right away, doctor. I've admitted Patient {new_id}, {created.get('name')} "
                    f"to {created.get('wing')}, {created.get('room')} ({created.get('condition')}, "
                    f"{created.get('current_bpm', 82)} BPM). The live ward roster now has {total_count} monitored patients."
                )
                self._record_clinician_turn(text, reply)
                updated = state_manager.update_state(
                    {
                        "ui_mode": "patient_stats",
                        "custom_pane": {
                            "kind": "ward_table",
                            "title": "Active Ward Roster — All Monitored Patients",
                            "subtitle": "Real-Time Multi-Patient EHR & Telemetry Matrix",
                            "filterWing": "",
                            "generatedHtml": "",
                        },
                        "last_clinician_input": text,
                        "last_transcript": reply,
                    },
                    patient_id=new_id,
                )
                return {
                    "reply": reply,
                    "engine": self.get_engine_label(),
                    "stage": updated.get("conversation_stage", "idle"),
                    "check_in_active": False,
                    "state": state_manager.load_state(new_id),
                }

            if c_action == "delete":
                del_pid = crud_req.get("patient_id")
                patients_now = db.get("patients", {})
                if not del_pid or del_pid not in patients_now:
                    names_list = ", ".join(f"#{k} {v.get('name')}" for k, v in sorted(patients_now.items()))
                    reply = (
                        f"Which patient would you like me to discharge and remove from the roster, doctor? "
                        f"Currently monitored: {names_list}."
                    )
                    state = state_manager.load_state(active_id)
                    updated = state_manager.update_state(
                        {
                            "ui_mode": "patient_stats",
                            "custom_pane": {
                                "kind": "ward_table",
                                "title": "Active Ward Roster — All Monitored Patients",
                                "subtitle": "Real-Time Multi-Patient EHR & Telemetry Matrix",
                                "filterWing": "",
                                "generatedHtml": "",
                            },
                            "last_clinician_input": text,
                            "last_transcript": reply,
                        },
                        patient_id=active_id,
                    )
                    return {
                        "reply": reply,
                        "engine": self.get_engine_label(),
                        "stage": updated.get("conversation_stage", "idle"),
                        "check_in_active": False,
                        "state": state_manager.load_state(active_id),
                    }

                deleted_entry, next_active = state_manager.delete_patient(str(del_pid))
                next_id = str(next_active.get("patient_id", state_manager.get_active_patient_id()))
                rem_count = len(state_manager.load_db().get("patients", {}))
                del_name = (deleted_entry or {}).get("name", f"Patient {del_pid}")
                reply = (
                    f"Done, doctor. I've discharged and removed Patient {del_pid}, {del_name} "
                    f"from the active ward database. We now have {rem_count} monitored patient{'s' if rem_count != 1 else ''} on the floor."
                )
                self._record_clinician_turn(text, reply)
                telemetry = self._get_telemetry_for_state(next_active)
                panel = self.synthesize_patient_panel(next_active, telemetry)
                updated = state_manager.update_state(
                    {
                        "ui_mode": "patient_stats",
                        "custom_pane": {
                            "kind": "ward_table",
                            "title": "Active Ward Roster — All Monitored Patients",
                            "subtitle": "Real-Time Multi-Patient EHR & Telemetry Matrix",
                            "filterWing": "",
                            "generatedHtml": "",
                        },
                        "last_clinician_input": text,
                        "last_transcript": reply,
                        "patient_panel": panel,
                    },
                    patient_id=next_id,
                )
                return {
                    "reply": reply,
                    "engine": self.get_engine_label(),
                    "stage": updated.get("conversation_stage", "idle"),
                    "check_in_active": False,
                    "state": state_manager.load_state(next_id),
                }

        # 0.2. Explicit .xlsx / .png file generation & export requests
        current_state_for_exp = state_manager.load_state(active_id)
        current_pane_kind = str((current_state_for_exp.get("custom_pane") or {}).get("kind") or "patient")
        export_req = _detect_export_request(text, current_pane_kind=current_pane_kind)
        if export_req:
            fmt = export_req["format"]
            exp_kind = export_req["kind"]
            if exp_kind == "timetable" or _is_timetable_or_calendar_query(text):
                exp_kind = "timetable"
                existing_events = state_manager.get_calendar_events()
                events = (
                    build_timetable_events(text, existing_events)
                    if _is_timetable_or_calendar_query(text) or not existing_events
                    else existing_events
                )
                state_manager.set_calendar_events(events)
                pane_obj = {
                    "kind": "timetable",
                    "title": "Clinician Shift Timetable & Reminder Schedule",
                    "subtitle": "Synced to Real PC Clock • 5-Min Prior Loudspeaker Paging",
                    "filterWing": "",
                    "generatedHtml": render_timetable_html_css(events),
                }
            elif exp_kind == "ward_stats":
                pane_obj = {
                    "kind": "ward_stats",
                    "title": "Ward Telemetry & Acuity Statistics",
                    "subtitle": "Live Statistical Breakdown • Monitored Ward Beds",
                    "filterWing": "",
                    "generatedHtml": "",
                }
            elif exp_kind == "ward_table":
                pane_obj = {
                    "kind": "ward_table",
                    "title": "Active Ward Roster — All Monitored Patients",
                    "subtitle": "Real-Time Multi-Patient EHR & Telemetry Matrix",
                    "filterWing": "",
                    "generatedHtml": "",
                }
            else:
                pane_obj = current_state_for_exp.get("custom_pane") or {
                    "kind": "patient",
                    "title": "",
                    "subtitle": "",
                    "filterWing": "",
                    "generatedHtml": "",
                }

            fmt_label = (
                "Excel (.xlsx) and PNG (.png) files"
                if fmt == "both"
                else ("Excel (.xlsx) spreadsheet" if fmt == "xlsx" else "high-resolution PNG (.png) image")
            )
            kind_label = {
                "timetable": "clinician shift timetable",
                "ward_stats": "ward telemetry statistics summary",
                "ward_table": "ward roster table",
                "patient": f"clinical summary for {current_state_for_exp.get('name', 'the active patient')}",
            }.get(exp_kind, "clinical table")

            reply = (
                f"Right away, doctor. I've generated the formatted {fmt_label} of the {kind_label} "
                "on a clean plain white background and started your download now."
            )
            telemetry = self._get_telemetry_for_state(current_state_for_exp)
            panel = self.synthesize_patient_panel(current_state_for_exp, telemetry)
            export_trigger = {
                "id": f"exp_{int(time.time() * 1000)}",
                "format": fmt,
                "kind": exp_kind,
            }
            self._record_clinician_turn(text, reply)
            updated = state_manager.update_state(
                {
                    "ui_mode": "patient_stats",
                    "custom_pane": pane_obj,
                    "export_request": export_trigger,
                    "last_clinician_input": text,
                    "last_transcript": reply,
                    "patient_panel": panel,
                },
                patient_id=active_id,
            )
            return {
                "reply": reply,
                "engine": self.get_engine_label(),
                "stage": updated.get("conversation_stage", "idle"),
                "check_in_active": bool(updated.get("check_in_active")),
                "export_request": export_trigger,
                "state": state_manager.load_state(active_id),
            }

        # 0.3. Hard Gate for Genuinely Impossible / Impractical Requests (Never generate fake UIs!)
        impractical_reply = _detect_impractical_request_fallback(text, current_state_for_exp)
        if impractical_reply is not None:
            telemetry = self._get_telemetry_for_state(current_state_for_exp)
            panel = self.synthesize_patient_panel(current_state_for_exp, telemetry)
            self._record_clinician_turn(text, impractical_reply)
            updated = state_manager.update_state(
                {
                    "last_clinician_input": text,
                    "last_transcript": impractical_reply,
                    "patient_panel": panel,
                },
                patient_id=active_id,
            )
            return {
                "reply": impractical_reply,
                "engine": self.get_engine_label(),
                "stage": updated.get("conversation_stage", "idle"),
                "check_in_active": bool(updated.get("check_in_active")),
                "state": state_manager.load_state(active_id),
            }

        # 0.5. Timetable / Schedule / Calendar / Loudspeaker Pager Reminder requests
        if _is_timetable_or_calendar_query(text):
            existing_events = state_manager.get_calendar_events()
            events = build_timetable_events(text, existing_events)
            state_manager.set_calendar_events(events)
            tt_html = render_timetable_html_css(events)

            state = state_manager.load_state(active_id)
            telemetry = self._get_telemetry_for_state(state)
            panel = self.synthesize_patient_panel(state, telemetry)

            delay_sec = _parse_relative_delay_seconds(text)
            if delay_sec is not None:
                fire_dt = datetime.now() + timedelta(seconds=delay_sec)
                fire_str = fire_dt.strftime("%I:%M:%S %p").lstrip("0")
                if delay_sec <= 3:
                    reply = "Right away, doctor. Broadcasting your hospital loudspeaker page now."
                elif delay_sec < 120:
                    reply = (
                        f"Right away, doctor. I've armed the hospital loudspeaker paging system "
                        f"to page you in {delay_sec} seconds at {fire_str}—stand by."
                    )
                else:
                    mins = delay_sec // 60
                    reply = (
                        f"Right away, doctor. I've armed the hospital loudspeaker paging system "
                        f"to page you in {mins} minute{'s' if mins != 1 else ''} at {_fmt_12h(fire_dt)}."
                    )
                engine_used = self.get_engine_label()
                rescored_q = text
            else:
                now_str = _fmt_12h(datetime.now())
                slot_summaries = [
                    f"{ev.get('display_time')}: {ev.get('title')} ({ev.get('location')}, loudspeaker page at {ev.get('reminder_display')})"
                    for ev in events
                ]
                fallback_tt_reply = (
                    f"I've constructed your shift timetable on the right-hand pane, synced to the system clock ({now_str}). "
                    f"Your schedule is set as follows: {'; '.join(slot_summaries[:4])}"
                    + (f"; followed by {slot_summaries[4]}." if len(slot_summaries) > 4 else ".")
                    + " Hospital loudspeaker paging is armed for 5 minutes before each scheduled task."
                )
                sys_tt = (
                    "You are Malcolm, the attending physician's Jarvis-style clinical chief-of-staff AI. "
                    "The doctor just asked you to create or update their shift timetable / calendar reminder schedule. "
                    "Speak in 2 to 3 crisp spoken sentences (no markdown or bullets): confirm that the timetable is now displayed on the right-hand pane, "
                    "briefly read out the scheduled task timings in chronological order (check-ins, surgery, and clock-out), "
                    "and note that hospital loudspeaker paging is armed for 5 minutes before each task. "
                    "Do NOT mention unrelated patient heart rates or symptoms.\n\n"
                    f"Current PC System Clock: {now_str}\n"
                    "Scheduled Timetable Tasks:\n" + "\n".join(f"- {s}" for s in slot_summaries)
                )
                rescored_q, cloud_tt, engine_used = self._query_gemini_with_rescore(sys_tt, text, state)
                reply = cloud_tt or fallback_tt_reply
            self._record_clinician_turn(rescored_q or text, reply)

            updated = state_manager.update_state(
                {
                    "ui_mode": "patient_stats",
                    "custom_pane": {
                        "kind": "timetable",
                        "title": "Clinician Shift Timetable & Reminder Schedule",
                        "subtitle": "Synced to Real PC Clock • 5-Min Prior Loudspeaker Paging",
                        "filterWing": "",
                        "generatedHtml": tt_html,
                    },
                    "last_clinician_input": rescored_q or text,
                    "last_transcript": reply,
                    "patient_panel": panel,
                },
                patient_id=active_id,
            )
            state_manager.append_audit_log(
                actor="Clinician -> Malcolm",
                action="TIMETABLE_SCHEDULED",
                details=f"Created/updated {len(events)} calendar timetable tasks with 5-min loudspeaker reminders.",
                pii_redacted=True,
                engine=engine_used,
                patient_id=active_id,
            )
            return {
                "reply": reply,
                "engine": engine_used,
                "stage": updated.get("conversation_stage", "idle"),
                "check_in_active": bool(updated.get("check_in_active")),
                "state": state_manager.load_state(active_id),
            }

        # 1. Cross-patient database queries ("list all patients", "show me a table", "give me statistics", etc.)
        if _is_multi_patient_overview_query(text) or (
            pane_intent and pane_intent.get("kind") in {"ward_table", "ward_stats", "custom_html"}
        ):
            html_box: dict[str, str] = {}
            html_thread: threading.Thread | None = None
            if pane_intent and pane_intent.get("generate_html"):
                def _parallel_html_worker() -> None:
                    synthesized = self.generate_pane_html_css(text, db)
                    html_box["html"] = synthesized
                    curr_id = state_manager.get_active_patient_id()
                    curr_st = state_manager.load_state(curr_id)
                    if curr_st.get("ui_mode") == "patient_stats":
                        cp = dict(curr_st.get("custom_pane") or {})
                        if cp.get("kind") != "patient":
                            cp["generatedHtml"] = synthesized
                            state_manager.update_state({"custom_pane": cp}, patient_id=curr_id)

                html_thread = threading.Thread(target=_parallel_html_worker, daemon=True)
                html_thread.start()

            reply, engine_used = self._build_multi_patient_overview(text)
            if html_thread is not None:
                html_thread.join(timeout=0.25)
            if self._last_pane_directive.startswith("patient_"):
                dir_pid = self._last_pane_directive.replace("patient_", "").strip()
                if dir_pid in (db.get("patients") or {}):
                    state_manager.set_active_patient(dir_pid)
                    active_id = dir_pid
            state = state_manager.load_state(active_id)
            telemetry = self._get_telemetry_for_state(state)
            panel = self.synthesize_patient_panel(state, telemetry)
            overview_updates: dict[str, Any] = {
                "last_clinician_input": self._last_rescored_transcript or text,
                "last_transcript": reply,
                "patient_panel": panel,
            }
            if target_id:
                overview_updates["ui_mode"] = "patient_stats"
            pane_updates = self._resolve_pane_updates(
                text,
                pane_intent,
                target_id,
                state,
                generated_html=html_box.get("html", ""),
            )
            overview_updates.update(pane_updates)
            updated = state_manager.update_state(
                overview_updates,
                patient_id=active_id,
            )
            state_manager.append_audit_log(
                actor="Clinician -> Malcolm",
                action="MULTI_PATIENT_QUERY",
                details="Answered cross-patient clinical RAG query over patients_db.json.",
                pii_redacted=True,
                engine=engine_used,
                patient_id=active_id,
            )
            return {
                "reply": reply,
                "engine": engine_used,
                "stage": updated.get("conversation_stage", "idle"),
                "check_in_active": bool(updated.get("check_in_active")),
                "state": state_manager.load_state(active_id),
            }

        state = state_manager.load_state(active_id)
        stage = state.get("conversation_stage", "idle")
        telemetry = self._get_telemetry_for_state(state)
        redacted_context = build_redacted_ehr_context(state, telemetry)

        # Deliver pending bedside report if ready
        if state.get("clinician_report_ready"):
            report = self.consume_clinician_report()
            if report:
                updated = state_manager.load_state(active_id)
                return {
                    "reply": report,
                    "engine": self.get_engine_label(),
                    "stage": updated.get("conversation_stage", "completed"),
                    "check_in_active": bool(updated.get("check_in_active")),
                    "state": updated,
                }

        # 3. Direct clinical question when targeting a patient (or in open Q&A)
        if target_id and _is_direct_question(text) and "status" not in text.lower():
            qa_html_box: dict[str, str] = {}
            qa_html_thread: threading.Thread | None = None
            if pane_intent and pane_intent.get("generate_html"):
                def _qa_html_worker() -> None:
                    synthesized = self.generate_pane_html_css(text, db)
                    qa_html_box["html"] = synthesized
                    curr_id = state_manager.get_active_patient_id()
                    curr_st = state_manager.load_state(curr_id)
                    if curr_st.get("ui_mode") == "patient_stats":
                        cp = dict(curr_st.get("custom_pane") or {})
                        if cp.get("kind") != "patient":
                            cp["generatedHtml"] = synthesized
                            state_manager.update_state({"custom_pane": cp}, patient_id=curr_id)

                qa_html_thread = threading.Thread(target=_qa_html_worker, daemon=True)
                qa_html_thread.start()

            answer, engine_used = self._answer_clinical_question(text, state, telemetry)
            if qa_html_thread is not None:
                qa_html_thread.join(timeout=0.25)
            display_input = self._last_rescored_transcript or text
            lowered = text.lower()
            is_impractical = _detect_impractical_request_fallback(text, state) is not None
            next_ui_mode = (
                state.get("ui_mode", "idle")
                if is_impractical and self._last_pane_directive == "keep" and not pane_intent
                else (
                    "heart_rate"
                    if any(k in lowered for k in ("heart rate", "bpm", "pulse"))
                    and not any(k in lowered for k in ("chart", "note", "round", "protocol", "symptom"))
                    else "patient_stats"
                )
            )
            pane_updates = self._resolve_pane_updates(
                text,
                pane_intent,
                target_id,
                state,
                generated_html=qa_html_box.get("html", ""),
            )
            if "ui_mode" in pane_updates:
                next_ui_mode = pane_updates["ui_mode"]
            projected = dict(state)
            projected["current_bpm"] = telemetry["bpm"]
            projected["wave_state"] = telemetry["wave_state"]
            panel = self.synthesize_patient_panel(projected, telemetry)

            def _qa_update(s: dict[str, Any]) -> None:
                s["current_bpm"] = telemetry["bpm"]
                s["wave_state"] = telemetry["wave_state"]
                s["ui_mode"] = next_ui_mode
                if "custom_pane" in pane_updates:
                    s["custom_pane"] = pane_updates["custom_pane"]
                else:
                    s["custom_pane"] = {"kind": "patient", "title": "", "subtitle": "", "filterWing": ""}
                s["last_clinician_input"] = display_input
                s["last_transcript"] = answer
                s["patient_panel"] = panel
                if s.get("conversation_stage") == "idle":
                    s["conversation_stage"] = "awaiting_checkin_confirmation"

            state_manager.update_state(_qa_update, patient_id=active_id)
            state_manager.append_audit_log(
                actor="Clinician -> Malcolm",
                action="CONVERSATIONAL_RAG_QA",
                details=f"Answered direct clinical query for Patient {active_id}.",
                pii_redacted=True,
                engine=engine_used,
                patient_id=active_id,
            )
            return {
                "reply": answer,
                "engine": engine_used,
                "stage": state_manager.load_state(active_id).get("conversation_stage", "awaiting_checkin_confirmation"),
                "check_in_active": bool(state.get("check_in_active")),
                "state": state_manager.load_state(active_id),
            }

        # 4. Stage 1: Initial patient lookup & disambiguation (only on idle, patient switch, or explicit status request)
        if (
            stage == "idle"
            or (
                stage == "completed"
                and (
                    (target_id is not None and target_id != active_id)
                    or "status of patient" in text.lower()
                    or "give me the status" in text.lower()
                )
            )
            or (target_id is not None and ("status" in text.lower() or "switch" in text.lower()))
        ):
            if "status" in text.lower() or (
                (target_id is not None or "patient" in text.lower())
                and len(text.split()) <= 6
                and "?" not in text
                and not _is_direct_question(text)
                and not pane_intent
            ):
                pid = target_id or active_id
                if pid != active_id:
                    state_manager.set_active_patient(pid)
                    active_id = pid
                    state = state_manager.load_state(active_id)
                    telemetry = self._get_telemetry_for_state(state)
                    redacted_context = build_redacted_ehr_context(state, telemetry)

                reply = (
                    f"You're talking about the {state['condition'].lower()} patient in "
                    f"{state['wing']}, {state['room']}? {state['name']}?"
                )
                engine_used = self.get_engine_label()

                projected = dict(state)
                projected["conversation_stage"] = "awaiting_patient_confirmation"
                projected["current_bpm"] = telemetry["bpm"]
                projected["wave_state"] = telemetry["wave_state"]
                panel = self.synthesize_patient_panel(projected, telemetry)

                def _stage1_update(s: dict[str, Any]) -> None:
                    s["conversation_stage"] = "awaiting_patient_confirmation"
                    s["current_bpm"] = telemetry["bpm"]
                    s["wave_state"] = telemetry["wave_state"]
                    s["ui_mode"] = "idle"
                    s["last_clinician_input"] = text
                    s["last_transcript"] = reply
                    s["patient_panel"] = panel

                state_manager.update_state(_stage1_update, patient_id=active_id)
                name_tok, _, room_tok = _patient_tokens(active_id)
                state_manager.append_audit_log(
                    actor="Clinician -> Malcolm",
                    action="PATIENT_DISAMBIGUATION",
                    details=f"Queried EHR index for {name_tok} in {state['wing']}, {room_tok}.",
                    pii_redacted=True,
                    engine=engine_used,
                    patient_id=active_id,
                )
                return {
                    "reply": reply,
                    "engine": engine_used,
                    "stage": "awaiting_patient_confirmation",
                    "check_in_active": False,
                    "state": state_manager.load_state(active_id),
                }

            # General conversational turn in idle/completed
            answer, engine_used = self._answer_clinical_question(text, state, telemetry)
            if self._last_pane_directive.startswith("patient_"):
                dir_pid = self._last_pane_directive.replace("patient_", "").strip()
                if dir_pid in (db.get("patients") or {}) and dir_pid != active_id:
                    state_manager.set_active_patient(dir_pid)
                    active_id = dir_pid
                    state = state_manager.load_state(active_id)
                    telemetry = self._get_telemetry_for_state(state)
            panel = self.synthesize_patient_panel(state, telemetry)
            q_updates: dict[str, Any] = {
                "last_clinician_input": self._last_rescored_transcript or text,
                "last_transcript": answer,
                "patient_panel": panel,
            }
            if target_id:
                q_updates["ui_mode"] = "patient_stats"
            q_updates.update(self._resolve_pane_updates(text, pane_intent, target_id, state))
            updated = state_manager.update_state(q_updates, patient_id=active_id)
            return {
                "reply": answer,
                "engine": engine_used,
                "stage": updated.get("conversation_stage", stage),
                "check_in_active": bool(updated.get("check_in_active")),
                "state": updated,
            }

        # 5. Stage 2: Clinician confirms patient identity -> share telemetry & propose bedside check-in
        if stage == "awaiting_patient_confirmation":
            if _is_negative(text):
                reply = "Understood. Please specify the patient ID (1429–1432) you would like me to check."
                projected = dict(state)
                projected["conversation_stage"] = "idle"
                panel = self.synthesize_patient_panel(projected, telemetry)
                state_manager.update_state(
                    {
                        "conversation_stage": "idle",
                        "ui_mode": "idle",
                        "last_clinician_input": text,
                        "last_transcript": reply,
                        "patient_panel": panel,
                    },
                    patient_id=active_id,
                )
                state_manager.append_audit_log(
                    actor="Clinician -> Malcolm",
                    action="DISAMBIGUATION_REJECTED",
                    details="Clinician rejected patient match; returning to idle.",
                    pii_redacted=True,
                    engine="Local Deterministic RAG",
                    patient_id=active_id,
                )
                return {
                    "reply": reply,
                    "engine": "Local Deterministic RAG",
                    "stage": "idle",
                    "check_in_active": False,
                    "state": state_manager.load_state(active_id),
                }

            if _is_affirmative(text) or active_id in text.lower():
                baseline_bpm = int(state.get("baseline_bpm", 90))
                symptoms = state.get("patient_symptoms") or []
                notes = state.get("nurse_notes") or []
                wave_lower = str(telemetry.get("wave_state", state.get("wave_state", ""))).lower()

                if active_id == "1429" and not symptoms and not notes and "sleep" not in wave_lower:
                    fallback_reply = (
                        f"Stats normal, constant rate of {baseline_bpm} BPM. "
                        "Patient has not reported any discomfort or odd symptoms thus far. "
                        "No comments from nurse rounds. Would you like me to check in on them? "
                        "I don't think they're sleeping based on their wave pattern."
                    )
                else:
                    sym_part = (
                        f"Earlier symptoms noted: {'; '.join(symptoms)}."
                        if symptoms
                        else "Patient has not reported any discomfort or odd symptoms thus far."
                    )
                    note_part = (
                        f"Nurse rounds note: {' '.join(notes)}"
                        if notes
                        else "No comments from nurse rounds."
                    )
                    if "sleep" in wave_lower:
                        sleep_part = (
                            "Based on their wave pattern, they appear to be sleeping right now. "
                            "Would you like me to wake them for a bedside check-in, or keep monitoring?"
                        )
                    else:
                        sleep_part = (
                            "Would you like me to check in on them? "
                            "I don't think they're sleeping based on their wave pattern."
                        )
                    rhythm_desc = str(telemetry.get("wave_state", state.get("wave_state", "Normal")))
                    fallback_reply = (
                        f"Telemetry reads {baseline_bpm} BPM ({rhythm_desc}). "
                        f"{sym_part} {note_part} {sleep_part}"
                    )

                if active_id == "1429" and not symptoms and not notes and "sleep" not in wave_lower:
                    reply = fallback_reply
                    engine_used = self.get_engine_label()
                else:
                    sys_prompt = (
                        "You are Malcolm, a concise clinical triage assistant. "
                        "Report the patient's baseline BPM and wave pattern accurately, summarize symptoms and nurse notes "
                        "strictly from EHR context, and ask whether you should check in on them.\n\n"
                        f"EHR Context:\n{redacted_context}"
                    )
                    cloud_reply, engine_used = self._query_gemini(sys_prompt, text, state)
                    reply = (
                        cloud_reply
                        if cloud_reply
                        and (str(baseline_bpm) in cloud_reply or "BPM" in cloud_reply)
                        and "check in" in cloud_reply.lower()
                        else fallback_reply
                    )

                projected = dict(state)
                projected["conversation_stage"] = "awaiting_checkin_confirmation"
                projected["current_bpm"] = telemetry["bpm"]
                projected["wave_state"] = telemetry["wave_state"]
                panel = self.synthesize_patient_panel(projected, telemetry)

                def _stage2_update(s: dict[str, Any]) -> None:
                    s["conversation_stage"] = "awaiting_checkin_confirmation"
                    s["current_bpm"] = telemetry["bpm"]
                    s["wave_state"] = telemetry["wave_state"]
                    s["ui_mode"] = "heart_rate"
                    s["last_clinician_input"] = text
                    s["last_transcript"] = reply
                    s["patient_panel"] = panel

                state_manager.update_state(_stage2_update, patient_id=active_id)
                name_tok, _, _ = _patient_tokens(active_id)
                state_manager.append_audit_log(
                    actor="Malcolm",
                    action="TELEMETRY_AND_EHR_REPORT",
                    details=(
                        f"Reported {telemetry['bpm']} BPM ({telemetry['wave_state']}) for {name_tok}; "
                        "offered bedside check-in."
                    ),
                    pii_redacted=True,
                    engine=engine_used,
                    patient_id=active_id,
                )
                return {
                    "reply": reply,
                    "engine": engine_used,
                    "stage": "awaiting_checkin_confirmation",
                    "check_in_active": False,
                    "state": state_manager.load_state(active_id),
                }

            # Open-ended follow-up question while waiting for patient confirmation
            answer, engine_used = self._answer_clinical_question(text, state, telemetry)
            p_updates = {
                "last_clinician_input": self._last_rescored_transcript or text,
                "last_transcript": answer,
            }
            p_updates.update(self._resolve_pane_updates(text, pane_intent, target_id, state))
            updated = state_manager.update_state(
                p_updates,
                patient_id=active_id,
            )
            return {
                "reply": answer,
                "engine": engine_used,
                "stage": stage,
                "check_in_active": False,
                "state": updated,
            }

        # 6. Stage 3: Clinician approves bedside check-in -> activate Bedside Web UI chat
        if stage == "awaiting_checkin_confirmation":
            if _is_negative(text):
                reply = "Understood, sir. I will hold off on the bedside check-in and continue passive monitoring."
                projected = dict(state)
                projected["conversation_stage"] = "idle"
                panel = self.synthesize_patient_panel(projected, telemetry)
                state_manager.update_state(
                    {
                        "conversation_stage": "idle",
                        "ui_mode": "patient_stats",
                        "last_clinician_input": text,
                        "last_transcript": reply,
                        "patient_panel": panel,
                    },
                    patient_id=active_id,
                )
                name_tok, _, _ = _patient_tokens(active_id)
                state_manager.append_audit_log(
                    actor="Clinician -> Malcolm",
                    action="CHECKIN_DECLINED",
                    details=f"Clinician declined bedside check-in for {name_tok}.",
                    pii_redacted=True,
                    engine="Local Deterministic RAG",
                    patient_id=active_id,
                )
                return {
                    "reply": reply,
                    "engine": "Local Deterministic RAG",
                    "stage": "idle",
                    "check_in_active": False,
                    "state": state_manager.load_state(active_id),
                }

            if _is_affirmative(text) or "check" in text.lower() or "wake" in text.lower():
                first_name = state.get("name", "Mika Hämäläinen").split()[0]
                _, first_tok, room_tok = _patient_tokens(active_id)
                name_tok, _, _ = _patient_tokens(active_id)
                fallback_bedside_greeting = (
                    f"Hello {first_name}, this is Malcolm checking in for your care team. "
                    "How are you doing, and are you experiencing any discomfort?"
                )
                sys_prompt = (
                    "You are Malcolm, an empathetic hospital bedside assistant. "
                    f"Greet {first_tok} warmly and ask how they are doing and if they are experiencing any discomfort. "
                    "Keep it to two short sentences and never diagnose.\n\n"
                    f"EHR Context:\n{redacted_context}"
                )
                cloud_greeting, engine_used = self._query_gemini(
                    sys_prompt,
                    f"Start bedside check-in with {first_tok}.",
                    state,
                )
                bedside_greeting = (
                    cloud_greeting
                    if cloud_greeting and "discomfort" in cloud_greeting.lower()
                    else fallback_bedside_greeting
                )

                transcript = f"Checking in on {state['name']} in {state['room']} now..."
                projected = dict(state)
                projected["check_in_active"] = True
                projected["clinician_report_ready"] = False
                projected["conversation_stage"] = "bedside_awaiting_symptom"
                projected["current_bpm"] = telemetry["bpm"]
                projected["wave_state"] = telemetry["wave_state"]
                panel = self.synthesize_patient_panel(projected, telemetry)

                def _activate_checkin(s: dict[str, Any]) -> None:
                    s["check_in_active"] = True
                    s["clinician_report_ready"] = False
                    s["conversation_stage"] = "bedside_awaiting_symptom"
                    s["current_bpm"] = telemetry["bpm"]
                    s["wave_state"] = telemetry["wave_state"]
                    s["ui_mode"] = "patient_stats"
                    s["last_clinician_input"] = text
                    s["last_transcript"] = transcript
                    s["patient_panel"] = panel

                state_manager.update_state(_activate_checkin, patient_id=active_id)
                state_manager.add_bedside_message("malcolm", bedside_greeting, patient_id=active_id)
                state_manager.append_audit_log(
                    actor="Clinician -> Malcolm",
                    action="BEDSIDE_CHECKIN_STARTED",
                    details=f"Activated Bedside Web UI check-in for {name_tok} in {room_tok}.",
                    pii_redacted=True,
                    engine=engine_used,
                    patient_id=active_id,
                )
                return {
                    "reply": transcript,
                    "bedside_greeting": bedside_greeting,
                    "engine": engine_used,
                    "stage": "bedside_awaiting_symptom",
                    "check_in_active": True,
                    "state": state_manager.load_state(active_id),
                }

            # Doctor asked a clinical question instead of answering check-in prompt
            answer, engine_used = self._answer_clinical_question(text, state, telemetry)
            projected = dict(state)
            panel = self.synthesize_patient_panel(projected, telemetry)
            c_updates: dict[str, Any] = {
                "ui_mode": "patient_stats",
                "last_clinician_input": self._last_rescored_transcript or text,
                "last_transcript": answer,
                "patient_panel": panel,
            }
            c_updates.update(self._resolve_pane_updates(text, pane_intent, target_id, state))
            updated = state_manager.update_state(
                c_updates,
                patient_id=active_id,
            )
            return {
                "reply": answer,
                "engine": engine_used,
                "stage": stage,
                "check_in_active": False,
                "state": updated,
            }

        # Already in bedside check-in -> still allow doctor to ask RAG questions!
        if stage in {"bedside_awaiting_symptom", "bedside_awaiting_nurse_confirm"}:
            if _is_direct_question(text):
                answer, engine_used = self._answer_clinical_question(text, state, telemetry)
            else:
                answer = (
                    f"Bedside check-in with {state['name']} is currently active. "
                    "Waiting for patient response on the Bedside Web UI."
                )
                engine_used = self.get_engine_label()
            b_updates: dict[str, Any] = {
                "last_clinician_input": self._last_rescored_transcript or text,
                "last_transcript": answer,
            }
            b_updates.update(self._resolve_pane_updates(text, pane_intent, target_id, state))
            updated = state_manager.update_state(
                b_updates,
                patient_id=active_id,
            )
            return {
                "reply": answer,
                "engine": engine_used,
                "stage": stage,
                "check_in_active": True,
                "state": updated,
            }

        answer, engine_used = self._answer_clinical_question(text, state, telemetry)
        panel = self.synthesize_patient_panel(state, telemetry)
        fallback_updates: dict[str, Any] = {
            "last_clinician_input": self._last_rescored_transcript or text,
            "last_transcript": answer,
            "patient_panel": panel,
        }
        if stage == "completed":
            fallback_updates["ui_mode"] = "patient_stats"
        fallback_updates.update(self._resolve_pane_updates(text, pane_intent, target_id, state))
        updated = state_manager.update_state(fallback_updates, patient_id=active_id)
        return {
            "reply": answer,
            "engine": engine_used,
            "stage": stage,
            "check_in_active": bool(updated.get("check_in_active")),
            "state": updated,
        }

    def process_bedside_turn(self, patient_input: str) -> dict[str, Any]:
        """Execute one turn of the active patient's Bedside Web UI interaction."""
        text = patient_input.strip()
        active_id = state_manager.get_active_patient_id()
        state = state_manager.load_state(active_id)
        if not text:
            return {"ok": False, "error": "Message cannot be empty.", "state": state}

        if not state.get("check_in_active"):
            if state.get("conversation_stage") == "completed":
                extra_note = _extract_patient_followup_note(text) or _extract_symptom_summary(text)
                state_manager.add_bedside_message("patient", text, patient_id=active_id)
                refreshed = state_manager.load_state(active_id)
                telemetry = self._get_telemetry_for_state(refreshed)
                projected = dict(refreshed)
                projected_symptoms = list(refreshed.get("patient_symptoms") or [])
                if extra_note and extra_note not in projected_symptoms:
                    projected_symptoms.append(extra_note)
                projected["patient_symptoms"] = projected_symptoms
                panel = self.synthesize_patient_panel(projected, telemetry)
                reply = "Understood, I have added that to your chart for the nurse and doctor."

                def _append_post_checkin(s: dict[str, Any]) -> None:
                    symptoms = s.setdefault("patient_symptoms", [])
                    if extra_note and extra_note not in symptoms:
                        symptoms.append(extra_note)
                    s["ui_mode"] = "patient_stats"
                    s["last_transcript"] = (
                        f"Bedside update for {s.get('name', 'patient')}: {extra_note}."
                    )
                    s["patient_panel"] = panel

                state_manager.update_state(_append_post_checkin, patient_id=active_id)
                state_manager.add_bedside_message("malcolm", reply, patient_id=active_id)
                return {
                    "ok": True,
                    "reply": reply,
                    "engine": self.get_engine_label(),
                    "state": state_manager.load_state(active_id),
                }
            # Patient initiated voice/text directly from the bedside terminal: auto-activate bedside check-in!
            state_manager.update_state(
                {
                    "check_in_active": True,
                    "conversation_stage": "bedside_awaiting_symptom",
                },
                patient_id=active_id,
                announce=False,
            )
            state = state_manager.load_state(active_id)

        state_manager.add_bedside_message("patient", text, patient_id=active_id)
        state = state_manager.load_state(active_id)
        stage = state.get("conversation_stage", "bedside_awaiting_symptom")
        telemetry = self._get_telemetry_for_state(state)
        redacted_context = build_redacted_ehr_context(state, telemetry)
        name_tok, _, room_tok = _patient_tokens(active_id)

        # Bedside Turn 1: Patient reports comfort or discomfort/symptom
        if stage == "bedside_awaiting_symptom":
            symptom_summary = _extract_symptom_summary(text)
            fallback_reply = "Understood, I'll note that down. Would you like me to call over a nurse?"
            if "itch" in text.lower() and "right foot" in text.lower():
                reply = fallback_reply
                engine_used = self.get_engine_label()
            else:
                sys_prompt = (
                    "You are Malcolm, a calm and empathetic hospital bedside assistant. "
                    "The patient just shared how they feel. Acknowledge their discomfort gently, "
                    "NEVER provide a medical diagnosis or treatment advice, state you have noted it down, "
                    "and ask: \"Would you like me to call over a nurse?\"\n\n"
                    f"EHR Context:\n{redacted_context}"
                )
                cloud_reply, engine_used = self._query_gemini(sys_prompt, text, state)
                reply = (
                    cloud_reply
                    if cloud_reply and "nurse" in cloud_reply.lower()
                    else fallback_reply
                )

            projected = dict(state)
            projected_symptoms = list(state.get("patient_symptoms") or [])
            if symptom_summary not in projected_symptoms:
                projected_symptoms.append(symptom_summary)
            projected["patient_symptoms"] = projected_symptoms
            projected["conversation_stage"] = "bedside_awaiting_nurse_confirm"
            panel = self.synthesize_patient_panel(projected, telemetry)

            if "itch" in symptom_summary.lower() and "right foot" in symptom_summary.lower():
                clinician_transcript = (
                    "Bedside update: Patient reported a persistent itch in their right foot. "
                    "Confirming if they want a nurse..."
                )
            else:
                clinician_transcript = (
                    f"Bedside update: Patient reported {symptom_summary}. "
                    "Confirming if they want a nurse..."
                )

            def _record_symptom(s: dict[str, Any]) -> None:
                symptoms = s.setdefault("patient_symptoms", [])
                if symptom_summary not in symptoms:
                    symptoms.append(symptom_summary)
                s["conversation_stage"] = "bedside_awaiting_nurse_confirm"
                s["ui_mode"] = "patient_stats"
                s["last_transcript"] = clinician_transcript
                s["patient_panel"] = panel

            state_manager.update_state(_record_symptom, patient_id=active_id)
            state_manager.add_bedside_message("malcolm", reply, patient_id=active_id)
            state_manager.append_audit_log(
                actor="Patient -> Malcolm (Bedside)",
                action="SYMPTOM_RECORDED",
                details=f"Logged symptom for {name_tok}: {symptom_summary}. Offered nurse dispatch.",
                pii_redacted=True,
                engine=engine_used,
                patient_id=active_id,
            )
            return {
                "ok": True,
                "reply": reply,
                "engine": engine_used,
                "state": state_manager.load_state(active_id),
            }

        # Bedside Turn 2: Patient confirms (or declines) nurse dispatch
        if stage == "bedside_awaiting_nurse_confirm":
            if _is_negative(text):
                reply = (
                    "Understood. I have noted your symptom for the care team, "
                    "and please let me know if you need a nurse at any time."
                )
                projected = dict(state)
                projected["nurse_dispatched"] = False
                projected["check_in_active"] = False
                projected["clinician_report_ready"] = False
                projected["conversation_stage"] = "completed"
                panel = self.synthesize_patient_panel(projected, telemetry)
                clinician_report, report_engine = self._build_clinician_return_report(projected, telemetry)

                def _close_without_dispatch(s: dict[str, Any]) -> None:
                    s["nurse_dispatched"] = False
                    s["check_in_active"] = False
                    s["clinician_report_ready"] = False
                    s["conversation_stage"] = "completed"
                    s["ui_mode"] = "patient_stats"
                    s["last_transcript"] = clinician_report
                    s["patient_panel"] = panel

                state_manager.update_state(_close_without_dispatch, patient_id=active_id)
                state_manager.add_bedside_message("malcolm", reply, patient_id=active_id)
                state_manager.append_audit_log(
                    actor="Malcolm (Bedside)",
                    action="CHECKIN_CLOSED_NO_DISPATCH",
                    details=f"Patient {name_tok} declined nurse dispatch; clinician summary published.",
                    pii_redacted=True,
                    engine=report_engine,
                    patient_id=active_id,
                )
                return {
                    "ok": True,
                    "reply": reply,
                    "engine": report_engine,
                    "state": state_manager.load_state(active_id),
                }

            # Patient confirms nurse dispatch ("Yes please.", "yes please, tell them to bring an aspirin please", etc.)
            followup_note = _extract_patient_followup_note(text)
            fallback_reply = (
                "Understood, I've noted your request for the nurse. Get some rest, they're on their way."
                if followup_note
                else "Understood. Get some rest, they're on their way."
            )
            if followup_note is None:
                reply = fallback_reply
                engine_used = self.get_engine_label()
            else:
                sys_prompt = (
                    "You are Malcolm, an empathetic hospital bedside assistant. "
                    "The patient requested a nurse. If they asked for anything specific (like aspirin, water, or a blanket), "
                    "briefly confirm you have noted it for the nurse, confirm a nurse is on the way, and tell them to get some rest. "
                    "Do not diagnose. Keep it to 1 to 2 calm sentences.\n\n"
                    f"EHR Context:\n{redacted_context}"
                )
                cloud_reply, engine_used = self._query_gemini(sys_prompt, text, state)
                reply = (
                    cloud_reply
                    if cloud_reply and ("on their way" in cloud_reply.lower() or "rest" in cloud_reply.lower())
                    else fallback_reply
                )

            projected = dict(state)
            projected_symptoms = list(state.get("patient_symptoms") or [])
            if not projected_symptoms:
                projected_symptoms.append("Persistent itch in right foot (uncomfortable)")
            if followup_note and followup_note not in projected_symptoms:
                projected_symptoms.append(followup_note)
            projected["patient_symptoms"] = projected_symptoms
            projected["nurse_dispatched"] = True
            projected["check_in_active"] = False
            projected["clinician_report_ready"] = False
            projected["conversation_stage"] = "completed"
            panel = self.synthesize_patient_panel(projected, telemetry)
            clinician_report, _ = self._build_clinician_return_report(projected, telemetry)

            def _dispatch_nurse(s: dict[str, Any]) -> None:
                symptoms = s.setdefault("patient_symptoms", [])
                if not symptoms:
                    symptoms.append("Persistent itch in right foot (uncomfortable)")
                if followup_note and followup_note not in symptoms:
                    symptoms.append(followup_note)
                s["nurse_dispatched"] = True
                s["check_in_active"] = False
                s["clinician_report_ready"] = False
                s["conversation_stage"] = "completed"
                s["ui_mode"] = "patient_stats"
                s["last_transcript"] = clinician_report
                s["patient_panel"] = panel

            state_manager.update_state(_dispatch_nurse, patient_id=active_id)
            state_manager.add_bedside_message("malcolm", reply, patient_id=active_id)
            detail_extra = f" Extra request: {followup_note}." if followup_note else ""
            state_manager.append_audit_log(
                actor="Malcolm (Bedside)",
                action="NURSE_DISPATCHED",
                details=(
                    f"Dispatched nurse to {room_tok} for {name_tok}; "
                    f"symptoms logged and clinician summary published.{detail_extra}"
                ),
                pii_redacted=True,
                engine=engine_used,
                patient_id=active_id,
            )
            return {
                "ok": True,
                "reply": reply,
                "engine": engine_used,
                "state": state_manager.load_state(active_id),
            }

        return {
            "ok": False,
            "error": f"Unexpected bedside stage: {stage}",
            "state": state,
        }

    def consume_clinician_report(self) -> str | None:
        """Consume clinician_report_ready flag and generate Malcolm's verbal return summary."""
        active_id = state_manager.get_active_patient_id()
        state = state_manager.load_state(active_id)
        if not state.get("clinician_report_ready"):
            return None

        telemetry = self._get_telemetry_for_state(state)
        report, engine_used = self._build_clinician_return_report(state, telemetry)
        projected = dict(state)
        projected["clinician_report_ready"] = False
        projected["conversation_stage"] = "completed"
        panel = self.synthesize_patient_panel(projected, telemetry)
        name_tok, _, _ = _patient_tokens(active_id)

        state_manager.update_state(
            {
                "clinician_report_ready": False,
                "conversation_stage": "completed",
                "ui_mode": "patient_stats",
                "last_transcript": report,
                "patient_panel": panel,
            },
            patient_id=active_id,
        )
        state_manager.append_audit_log(
            actor="Malcolm -> Clinician",
            action="CLINICIAN_REPORT_DELIVERED",
            details=(
                f"Delivered bedside summary for {name_tok}. "
                f"Nurse dispatched: {bool(state.get('nurse_dispatched'))}."
            ),
            pii_redacted=True,
            engine=engine_used,
            patient_id=active_id,
        )
        return report


rag_agent = LocalRAGAgent()


def process_clinician_turn(clinician_input: str) -> dict[str, Any]:
    """Module-level wrapper for processing a clinician message."""
    return rag_agent.process_clinician_turn(clinician_input)


def process_bedside_turn(patient_input: str) -> dict[str, Any]:
    """Module-level wrapper for processing a bedside patient message."""
    return rag_agent.process_bedside_turn(patient_input)


def consume_clinician_report() -> str | None:
    """Module-level wrapper for consuming the bedside check-in return report."""
    return rag_agent.consume_clinician_report()
