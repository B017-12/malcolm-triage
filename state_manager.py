"""Thread-safe and process-safe multi-patient database & state manager for Malcolm."""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "patients_db.json"
DB_LOCK_FILE = BASE_DIR / "patients_db.json.lock"
STATE_FILE = BASE_DIR / "patient_1429_state.json"

DEFAULT_PATIENTS: dict[str, dict[str, Any]] = {
    "1429": {
        "patient_id": "1429",
        "name": "Mika Hämäläinen",
        "pronoun_obj": "him",
        "wing": "Wing A",
        "room": "Room 153",
        "condition": "Dialysis",
        "department": "Nephrology — Renal Care",
        "protocol": "Routine Dialysis",
        "baseline_bpm": 90,
        "current_bpm": 90,
        "wave_state": "Awake / Normal Sinus Rhythm",
        "conversation_stage": "idle",
        "ui_mode": "idle",
        "last_clinician_input": "",
        "last_transcript": "Listening for clinician request...",
        "patient_panel": {
            "name": "Mika Hämäläinen",
            "patientId": "1429",
            "wing": "Wing A",
            "room": "153",
            "department": "Nephrology — Renal Care",
            "protocol": "Routine Dialysis",
            "rhythm": "Normal Sinus Rhythm",
            "telemetryDetail": "Awake • Baseline 90 BPM • Signal Locked",
            "nurseRounds": "No comments from nurse rounds.",
            "symptom": "Patient has not reported any discomfort or odd symptoms thus far.",
            "dispatchStatus": "",
        },
        "nurse_notes": [],
        "patient_symptoms": [],
        "nurse_dispatched": False,
        "check_in_active": False,
        "clinician_report_ready": False,
        "alert_level": "normal",
        "doctor_paged": False,
        "crash_cart_active": False,
        "emergency_message": "",
        "bedside_messages": [],
        "audit_log": [],
    },
    "1430": {
        "patient_id": "1430",
        "name": "Aino Virtanen",
        "pronoun_obj": "her",
        "wing": "Wing B",
        "room": "Room 204",
        "condition": "Post-Op Mitral Valve Repair",
        "department": "Cardiothoracic Surgery",
        "protocol": "Post-Op Telemetry & Anticoagulation",
        "baseline_bpm": 76,
        "current_bpm": 76,
        "wave_state": "Awake / Normal Sinus Rhythm",
        "conversation_stage": "idle",
        "ui_mode": "idle",
        "last_clinician_input": "",
        "last_transcript": "Listening for clinician request...",
        "patient_panel": {
            "name": "Aino Virtanen",
            "patientId": "1430",
            "wing": "Wing B",
            "room": "204",
            "department": "Cardiothoracic Surgery",
            "protocol": "Post-Op Telemetry & Anticoagulation",
            "rhythm": "Normal Sinus Rhythm",
            "telemetryDetail": "Awake • Baseline 76 BPM • Signal Locked",
            "nurseRounds": "04:30 RN: Sternal dressing clean and dry; pain 2/10 managed with acetaminophen.",
            "symptom": "No new acute discomfort reported since 04:30 nursing round.",
            "dispatchStatus": "",
        },
        "nurse_notes": [
            "04:30 RN: Sternal dressing clean and dry; pain 2/10 managed with acetaminophen.",
        ],
        "patient_symptoms": [],
        "nurse_dispatched": False,
        "check_in_active": False,
        "clinician_report_ready": False,
        "alert_level": "normal",
        "doctor_paged": False,
        "crash_cart_active": False,
        "emergency_message": "",
        "bedside_messages": [],
        "audit_log": [],
    },
    "1431": {
        "patient_id": "1431",
        "name": "Jukka Korhonen",
        "pronoun_obj": "him",
        "wing": "Wing C",
        "room": "Room 312",
        "condition": "Acute Bacterial Pneumonia",
        "department": "Pulmonology & Respiratory Care",
        "protocol": "2L O2 Nasal Cannula & IV Ceftriaxone",
        "baseline_bpm": 104,
        "current_bpm": 104,
        "wave_state": "Awake / Mild Sinus Tachycardia",
        "conversation_stage": "idle",
        "ui_mode": "idle",
        "last_clinician_input": "",
        "last_transcript": "Listening for clinician request...",
        "patient_panel": {
            "name": "Jukka Korhonen",
            "patientId": "1431",
            "wing": "Wing C",
            "room": "312",
            "department": "Pulmonology & Respiratory Care",
            "protocol": "2L O2 Nasal Cannula & IV Ceftriaxone",
            "rhythm": "Mild Sinus Tachycardia",
            "telemetryDetail": "Awake • Baseline 104 BPM • SpO2 Monitored",
            "nurseRounds": "05:15 RN: SpO2 96% on 2L O2; productive cough improving after nebulizer.",
            "symptom": "Mild shortness of breath on exertion",
            "dispatchStatus": "",
        },
        "nurse_notes": [
            "05:15 RN: SpO2 96% on 2L O2; productive cough improving after nebulizer.",
        ],
        "patient_symptoms": [
            "Mild shortness of breath on exertion",
        ],
        "nurse_dispatched": False,
        "check_in_active": False,
        "clinician_report_ready": False,
        "alert_level": "normal",
        "doctor_paged": False,
        "crash_cart_active": False,
        "emergency_message": "",
        "bedside_messages": [],
        "audit_log": [],
    },
    "1432": {
        "patient_id": "1432",
        "name": "Sanna Laine",
        "pronoun_obj": "her",
        "wing": "Wing A",
        "room": "Room 118",
        "condition": "Post-Concussion Neurological Observation",
        "department": "Neurology — Trauma Unit",
        "protocol": "Q2H Neuro Checks & Cranial Telemetry",
        "baseline_bpm": 64,
        "current_bpm": 64,
        "wave_state": "Sleeping / Slow Sinus Rhythm",
        "conversation_stage": "idle",
        "ui_mode": "idle",
        "last_clinician_input": "",
        "last_transcript": "Listening for clinician request...",
        "patient_panel": {
            "name": "Sanna Laine",
            "patientId": "1432",
            "wing": "Wing A",
            "room": "118",
            "department": "Neurology — Trauma Unit",
            "protocol": "Q2H Neuro Checks & Cranial Telemetry",
            "rhythm": "Slow Sinus Rhythm",
            "telemetryDetail": "Sleeping • Baseline 64 BPM • Signal Locked",
            "nurseRounds": "06:00 RN: Pupils equal and reactive (PERRLA); patient sleeping in darkened room.",
            "symptom": "Mild photophobia reported earlier this shift",
            "dispatchStatus": "",
        },
        "nurse_notes": [
            "06:00 RN: Pupils equal and reactive (PERRLA); patient sleeping in darkened room.",
        ],
        "patient_symptoms": [
            "Mild photophobia reported earlier this shift",
        ],
        "nurse_dispatched": False,
        "check_in_active": False,
        "clinician_report_ready": False,
        "alert_level": "normal",
        "doctor_paged": False,
        "crash_cart_active": False,
        "emergency_message": "",
        "bedside_messages": [],
        "audit_log": [],
    },
}

DEFAULT_STATE: dict[str, Any] = copy.deepcopy(DEFAULT_PATIENTS["1429"])

DEFAULT_DB: dict[str, Any] = {
    "active_patient_id": "1429",
    "patients": copy.deepcopy(DEFAULT_PATIENTS),
    "calendar_events": [],
    "last_page_alert": None,
}


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StateManager:
    """Manages patients_db.json atomically using file locking and mirrors patient_1429_state.json."""

    def __init__(
        self,
        db_path: Path = DB_FILE,
        lock_path: Path = DB_LOCK_FILE,
        legacy_1429_path: Path = STATE_FILE,
        timeout: float = 10.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.lock_path = Path(lock_path)
        self.legacy_1429_path = Path(legacy_1429_path)
        self._lock = FileLock(str(self.lock_path), timeout=timeout)

    def _normalize_db(self, raw_db: Any) -> dict[str, Any]:
        """Ensure every patient in the database has all required schema keys while honoring added/deleted patients."""
        if not isinstance(raw_db, dict):
            return copy.deepcopy(DEFAULT_DB)

        patients_in = raw_db.get("patients")
        if not isinstance(patients_in, dict) or len(patients_in) == 0:
            merged_patients = copy.deepcopy(DEFAULT_PATIENTS)
        else:
            merged_patients = {}
            for pid, existing in patients_in.items():
                if not isinstance(existing, dict):
                    continue
                template = DEFAULT_PATIENTS.get(str(pid), DEFAULT_STATE)
                entry = copy.deepcopy(template)
                entry.update(existing)
                entry["patient_id"] = str(pid)
                merged_patients[str(pid)] = entry
            if not merged_patients:
                merged_patients = copy.deepcopy(DEFAULT_PATIENTS)

        active_id = str(raw_db.get("active_patient_id") or "1429")
        if active_id not in merged_patients:
            active_id = sorted(merged_patients.keys())[0]

        calendar_events = raw_db.get("calendar_events")
        if not isinstance(calendar_events, list):
            calendar_events = []
        last_page_alert = raw_db.get("last_page_alert")
        if not isinstance(last_page_alert, dict):
            last_page_alert = None
        try:
            reply_seq = int(raw_db.get("reply_seq") or 0)
        except (TypeError, ValueError):
            reply_seq = 0

        return {
            "active_patient_id": active_id,
            "patients": merged_patients,
            "calendar_events": calendar_events,
            "last_page_alert": last_page_alert,
            "reply_seq": reply_seq,
        }

    def _read_db_unlocked(self) -> dict[str, Any]:
        """Read patients_db.json without acquiring the lock (caller must hold lock)."""
        if not self.db_path.exists():
            initial = copy.deepcopy(DEFAULT_DB)
            self._write_db_unlocked(initial)
            return initial

        try:
            raw_text = self.db_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            normalized = self._normalize_db(data)
            return normalized
        except (OSError, ValueError, json.JSONDecodeError):
            initial = copy.deepcopy(DEFAULT_DB)
            self._write_db_unlocked(initial)
            return initial

    def _write_db_unlocked(self, db: dict[str, Any]) -> None:
        """Write patients_db.json and mirror Patient 1429 state to patient_1429_state.json."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(db, indent=2, ensure_ascii=False) + "\n"
        self.db_path.write_text(serialized, encoding="utf-8")

        p1429 = db.get("patients", {}).get("1429")
        if isinstance(p1429, dict):
            legacy_serialized = json.dumps(p1429, indent=2, ensure_ascii=False) + "\n"
            self.legacy_1429_path.write_text(legacy_serialized, encoding="utf-8")

    def load_db(self) -> dict[str, Any]:
        """Load and return the full multi-patient database dictionary."""
        with self._lock:
            return copy.deepcopy(self._read_db_unlocked())

    def list_patients(self) -> list[dict[str, Any]]:
        """Return compact summary records for every patient in the database."""
        with self._lock:
            db = self._read_db_unlocked()
            active_id = db["active_patient_id"]
            summaries: list[dict[str, Any]] = []
            for pid in sorted(db["patients"].keys()):
                p = db["patients"][pid]
                summaries.append(
                    {
                        "patient_id": p.get("patient_id", pid),
                        "name": p.get("name", ""),
                        "wing": p.get("wing", ""),
                        "room": p.get("room", ""),
                        "condition": p.get("condition", ""),
                        "department": p.get("department", ""),
                        "current_bpm": p.get("current_bpm", p.get("baseline_bpm", 90)),
                        "wave_state": p.get("wave_state", ""),
                        "check_in_active": bool(p.get("check_in_active")),
                        "nurse_dispatched": bool(p.get("nurse_dispatched")),
                        "alert_level": p.get("alert_level", "normal"),
                        "doctor_paged": bool(p.get("doctor_paged")),
                        "crash_cart_active": bool(p.get("crash_cart_active")),
                        "active": pid == active_id,
                    }
                )
            return summaries

    def get_active_patient_id(self) -> str:
        """Return the currently selected active patient ID."""
        with self._lock:
            db = self._read_db_unlocked()
            return str(db["active_patient_id"])

    def set_active_patient(self, patient_id: str) -> dict[str, Any]:
        """Switch active_patient_id to patient_id and return that patient's state."""
        pid = str(patient_id).strip()
        with self._lock:
            db = self._read_db_unlocked()
            if pid not in db["patients"]:
                raise KeyError(f"Unknown patient_id: {pid}")
            db["active_patient_id"] = pid
            self._write_db_unlocked(db)
            return copy.deepcopy(db["patients"][pid])

    def load_state(self, patient_id: str | None = None) -> dict[str, Any]:
        """Load and return the specified (or active) patient's state dictionary."""
        with self._lock:
            db = self._read_db_unlocked()
            pid = str(patient_id) if patient_id else db["active_patient_id"]
            if pid not in db["patients"]:
                pid = db["active_patient_id"]
            return copy.deepcopy(db["patients"][pid])

    def save_state(self, state: dict[str, Any], patient_id: str | None = None) -> dict[str, Any]:
        """Persist the provided patient state dictionary into patients_db.json."""
        with self._lock:
            db = self._read_db_unlocked()
            pid = str(patient_id or state.get("patient_id") or db["active_patient_id"])
            template = DEFAULT_PATIENTS.get(pid, DEFAULT_STATE)
            to_save = copy.deepcopy(template)
            to_save.update(state)
            db["patients"][pid] = to_save
            self._write_db_unlocked(db)
            return copy.deepcopy(to_save)

    def update_state(
        self,
        mutator: Callable[[dict[str, Any]], None] | dict[str, Any],
        patient_id: str | None = None,
        announce: bool = True,
    ) -> dict[str, Any]:
        """Atomically read, mutate, and save a patient's state under FileLock.

        When the mutation changes `last_transcript` (Malcolm's spoken reply) and `announce`
        is True, a global monotonic `reply_seq` is bumped so the clinician UI speaks it exactly once.
        """
        with self._lock:
            db = self._read_db_unlocked()
            pid = str(patient_id) if patient_id else db["active_patient_id"]
            if pid not in db["patients"]:
                pid = db["active_patient_id"]
            state = db["patients"][pid]
            before_reply = state.get("last_transcript")
            if callable(mutator):
                mutator(state)
            elif isinstance(mutator, dict):
                state.update(mutator)
            else:
                raise TypeError("mutator must be a callable or dict")
            if announce and state.get("last_transcript") != before_reply:
                now_ms = int(time.time() * 1000)
                db["reply_seq"] = max(int(db.get("reply_seq") or 0) + 1, now_ms)
            self._write_db_unlocked(db)
            return copy.deepcopy(state)

    def get_reply_seq(self) -> int:
        """Return the monotonic sequence number of Malcolm's latest spoken reply."""
        with self._lock:
            db = self._read_db_unlocked()
            return int(db.get("reply_seq") or 0)

    def append_audit_log(
        self,
        actor: str,
        action: str,
        details: str,
        pii_redacted: bool = True,
        engine: str = "Local RAG",
        patient_id: str | None = None,
    ) -> dict[str, Any]:
        """Append an immutable entry to the patient's audit_log array."""
        entry = {
            "timestamp": _utc_now_iso(),
            "actor": actor,
            "action": action,
            "details": details,
            "pii_redacted": bool(pii_redacted),
            "engine": engine,
        }

        def _append(state: dict[str, Any]) -> None:
            audit_log = state.setdefault("audit_log", [])
            audit_log.append(entry)

        return self.update_state(_append, patient_id=patient_id)

    def add_bedside_message(
        self,
        sender: str,
        text: str,
        patient_id: str | None = None,
    ) -> dict[str, Any]:
        """Append a message to bedside_messages under the file lock."""
        msg = {
            "sender": sender,
            "text": text,
            "timestamp": _utc_now_iso(),
        }

        def _append(state: dict[str, Any]) -> None:
            messages = state.setdefault("bedside_messages", [])
            messages.append(msg)

        return self.update_state(_append, patient_id=patient_id)

    def reset_state(self) -> dict[str, Any]:
        """Reset all patients in patients_db.json back to clean idle defaults."""
        with self._lock:
            fresh_db = copy.deepcopy(DEFAULT_DB)
            # Keep reply_seq monotonic across resets so the UI never misses a future reply.
            try:
                prev_seq = int(self._read_db_unlocked().get("reply_seq") or 0)
            except Exception:
                prev_seq = 0
            fresh_db["reply_seq"] = prev_seq
            self._write_db_unlocked(fresh_db)
            return copy.deepcopy(fresh_db["patients"]["1429"])

    def get_calendar_events(self) -> list[dict[str, Any]]:
        """Return the current list of scheduled clinician calendar/timetable events."""
        with self._lock:
            db = self._read_db_unlocked()
            return copy.deepcopy(db.get("calendar_events") or [])

    def set_calendar_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace the clinician calendar/timetable events list atomically."""
        with self._lock:
            db = self._read_db_unlocked()
            db["calendar_events"] = copy.deepcopy(events)
            self._write_db_unlocked(db)
            return copy.deepcopy(db["calendar_events"])

    def get_last_page_alert(self) -> dict[str, Any] | None:
        """Return the most recent hospital loudspeaker page alert if any."""
        with self._lock:
            db = self._read_db_unlocked()
            alert = db.get("last_page_alert")
            return copy.deepcopy(alert) if isinstance(alert, dict) else None

    def record_page_alert(self, alert: dict[str, Any]) -> dict[str, Any]:
        """Store the latest hospital loudspeaker page alert in the database."""
        with self._lock:
            db = self._read_db_unlocked()
            db["last_page_alert"] = copy.deepcopy(alert)
            self._write_db_unlocked(db)
            return copy.deepcopy(alert)

    def add_patient(
        self,
        name: str,
        wing: str = "Wing B",
        room: str = "Room 210",
        condition: str = "General Observation",
        department: str = "Internal Medicine",
        protocol: str = "Continuous Telemetry & Q4H Vitals",
        baseline_bpm: int = 82,
        symptom: str = "",
        nurse_note: str = "",
        patient_id: str | None = None,
    ) -> dict[str, Any]:
        """Add/admit a new patient into patients_db.json and return their created state."""
        with self._lock:
            db = self._read_db_unlocked()
            patients = db.setdefault("patients", {})
            if patient_id and str(patient_id).strip() and str(patient_id).strip() not in patients:
                new_id = str(patient_id).strip()
            else:
                numeric_ids = [int(k) for k in patients.keys() if str(k).isdigit()]
                next_num = max(numeric_ids, default=1432) + 1
                new_id = str(next_num)

            clean_name = name.strip() or f"Patient {new_id}"
            clean_wing = wing.strip() or "Wing B"
            if not clean_wing.lower().startswith("wing"):
                clean_wing = f"Wing {clean_wing.upper()}"
            clean_room = room.strip() or "Room 210"
            room_num = clean_room.replace("Room", "").replace("room", "").strip() or "210"
            clean_room_full = f"Room {room_num}"
            bpm_val = max(45, min(180, int(baseline_bpm or 82)))
            wave_state = (
                "Awake / Mild Sinus Tachycardia"
                if bpm_val >= 100
                else ("Sleeping / Slow Sinus Rhythm" if bpm_val < 68 else "Awake / Normal Sinus Rhythm")
            )
            rhythm_short = wave_state.split("/", 1)[-1].strip()
            symptoms_list = [symptom.strip()] if symptom.strip() else []
            notes_list = [nurse_note.strip()] if nurse_note.strip() else []

            entry = copy.deepcopy(DEFAULT_STATE)
            entry.update(
                {
                    "patient_id": new_id,
                    "name": clean_name,
                    "pronoun_obj": "them",
                    "wing": clean_wing,
                    "room": clean_room_full,
                    "condition": condition.strip() or "General Observation",
                    "department": department.strip() or "Internal Medicine",
                    "protocol": protocol.strip() or "Continuous Telemetry & Q4H Vitals",
                    "baseline_bpm": bpm_val,
                    "current_bpm": bpm_val,
                    "wave_state": wave_state,
                    "conversation_stage": "idle",
                    "ui_mode": "patient_stats",
                    "custom_pane": {
                        "kind": "ward_table",
                        "title": "Active Ward Roster — All Monitored Patients",
                        "subtitle": "Real-Time Multi-Patient EHR & Telemetry Matrix",
                        "filterWing": "",
                        "generatedHtml": "",
                    },
                    "patient_symptoms": symptoms_list,
                    "nurse_notes": notes_list,
                    "patient_panel": {
                        "name": clean_name,
                        "patientId": new_id,
                        "wing": clean_wing,
                        "room": room_num,
                        "department": department.strip() or "Internal Medicine",
                        "protocol": protocol.strip() or "Continuous Telemetry & Q4H Vitals",
                        "rhythm": rhythm_short,
                        "telemetryDetail": f"Awake • Baseline {bpm_val} BPM • Signal Locked",
                        "nurseRounds": notes_list[0] if notes_list else "Newly admitted — initial nursing intake complete.",
                        "symptom": symptoms_list[0] if symptoms_list else "No acute discomfort reported upon admission.",
                        "dispatchStatus": "",
                    },
                }
            )
            patients[new_id] = entry
            db["active_patient_id"] = new_id
            self._write_db_unlocked(db)
            return copy.deepcopy(entry)

    def delete_patient(self, patient_id: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Remove/discharge a patient from patients_db.json and return (deleted_entry, active_state)."""
        pid = str(patient_id).strip()
        with self._lock:
            db = self._read_db_unlocked()
            patients = db.get("patients") or {}
            if pid not in patients:
                active = patients.get(db.get("active_patient_id", ""), next(iter(patients.values()), copy.deepcopy(DEFAULT_STATE)))
                return None, copy.deepcopy(active)

            deleted = copy.deepcopy(patients.pop(pid))
            if not patients:
                # If the very last patient was removed, re-seed default 1429 so the ward is never empty/broken
                patients["1429"] = copy.deepcopy(DEFAULT_PATIENTS["1429"])
            if db.get("active_patient_id") not in patients:
                db["active_patient_id"] = sorted(patients.keys())[0]
            active_id = db["active_patient_id"]
            self._write_db_unlocked(db)
            return deleted, copy.deepcopy(patients[active_id])


state_manager = StateManager()


def load_db() -> dict[str, Any]:
    """Module-level helper to load the multi-patient database."""
    return state_manager.load_db()


def list_patients() -> list[dict[str, Any]]:
    """Module-level helper to list compact patient summaries."""
    return state_manager.list_patients()


def set_active_patient(patient_id: str) -> dict[str, Any]:
    """Module-level helper to switch active patient."""
    return state_manager.set_active_patient(patient_id)


def load_state(patient_id: str | None = None) -> dict[str, Any]:
    """Module-level helper to load the active (or specified) patient state."""
    return state_manager.load_state(patient_id=patient_id)


def save_state(state: dict[str, Any], patient_id: str | None = None) -> dict[str, Any]:
    """Module-level helper to save patient state."""
    return state_manager.save_state(state, patient_id=patient_id)


def update_state(
    mutator: Callable[[dict[str, Any]], None] | dict[str, Any],
    patient_id: str | None = None,
    announce: bool = True,
) -> dict[str, Any]:
    """Module-level helper to atomically update patient state."""
    return state_manager.update_state(mutator, patient_id=patient_id, announce=announce)


def append_audit_log(
    actor: str,
    action: str,
    details: str,
    pii_redacted: bool = True,
    engine: str = "Local RAG",
    patient_id: str | None = None,
) -> dict[str, Any]:
    """Module-level helper to append an immutable audit log entry."""
    return state_manager.append_audit_log(
        actor=actor,
        action=action,
        details=details,
        pii_redacted=pii_redacted,
        engine=engine,
        patient_id=patient_id,
    )


def add_bedside_message(
    sender: str,
    text: str,
    patient_id: str | None = None,
) -> dict[str, Any]:
    """Module-level helper to append a bedside message."""
    return state_manager.add_bedside_message(sender=sender, text=text, patient_id=patient_id)


def reset_state() -> dict[str, Any]:
    """Module-level helper to reset the entire patient database back to idle."""
    return state_manager.reset_state()
