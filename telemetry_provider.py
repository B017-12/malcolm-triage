"""Modular ECG telemetry abstraction for Malcolm clinical triage."""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _parse_bool_env(value: str | None, default: bool = False) -> bool:
    """Parse common truthy strings from environment variables."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


USE_HARDWARE_ECG: bool = _parse_bool_env(os.getenv("USE_HARDWARE_ECG"), default=False)


class BaseTelemetryProvider(ABC):
    """Abstract interface for patient ECG and wave-pattern telemetry sources."""

    @abstractmethod
    def get_reading(
        self,
        baseline_bpm: int | None = None,
        wave_state: str | None = None,
    ) -> dict[str, Any]:
        """Return a structured telemetry payload including BPM and wave state."""

    @abstractmethod
    def get_bpm(self) -> int:
        """Return the latest heart rate reading in beats per minute."""

    @abstractmethod
    def get_wave_state(self) -> str:
        """Return the interpreted ECG wave state description."""

    @abstractmethod
    def get_mode_label(self) -> str:
        """Return a human-readable label describing the active telemetry source."""


class SimulatedECGProvider(BaseTelemetryProvider):
    """Active default ECG provider generating realistic telemetry around 90 BPM."""

    def __init__(self, baseline_bpm: int = 90) -> None:
        self.baseline_bpm = baseline_bpm
        self._bpm_distribution = [89, 90, 90, 91, 92]
        self._wave_state = "Awake / Normal Sinus Rhythm"
        self._mode_label = "SIMULATED [Arduino AD8232 Stub Ready]"

    def get_bpm(self) -> int:
        """Return a realistic minor fluctuation between 89 and 92 BPM."""
        return random.choice(self._bpm_distribution)

    def get_wave_state(self) -> str:
        """Return the patient's awake normal sinus rhythm wave classification."""
        return self._wave_state

    def get_mode_label(self) -> str:
        """Return the HUD mode label for simulated telemetry."""
        return self._mode_label

    def get_reading(
        self,
        baseline_bpm: int | None = None,
        wave_state: str | None = None,
    ) -> dict[str, Any]:
        """Return simulated live ECG telemetry centered around the patient's baseline BPM."""
        base = self.baseline_bpm if baseline_bpm is None else int(baseline_bpm)
        bpm = base + random.choice([-1, 0, 0, 1, 2])
        effective_wave = wave_state or self.get_wave_state()
        return {
            "bpm": bpm,
            "baseline_bpm": base,
            "wave_state": effective_wave,
            "status": f"Stats normal, constant rate of {base} BPM",
            "mode": self.get_mode_label(),
            "hardware_connected": False,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


class ArduinoAD8232Provider(BaseTelemetryProvider):
    """
    PLANNED FUNCTIONALITY: Hardware ECG telemetry provider via Arduino + HC-05 + AD8232.

    # TODO: Connect HC-05 Bluetooth Serial & AD8232 pin A0
    Intended wiring:
      - AD8232 OUTPUT -> Arduino Analog Pin A0
      - AD8232 LO- / LO+ -> Arduino Digital Pins D10 / D11 (lead-off detection)
      - HC-05 Bluetooth Module -> Serial COM port (e.g., COM4 @ 9600 baud)
    """

    def __init__(
        self,
        port: str = "COM4",
        baudrate: int = 9600,
        analog_pin: str = "A0",
        lead_off_pins: tuple[str, str] = ("D10", "D11"),
    ) -> None:
        # TODO: Connect HC-05 Bluetooth Serial & AD8232 pin A0
        self.port = port
        self.baudrate = baudrate
        self.analog_pin = analog_pin
        self.lead_off_pins = lead_off_pins
        self._serial_conn: Any = None
        self._connected: bool = False

    def connect(self) -> bool:
        """
        Open the HC-05 Bluetooth serial connection to the Arduino ECG node.

        # TODO: Connect HC-05 Bluetooth Serial & AD8232 pin A0
        Replace stub with `serial.Serial(self.port, self.baudrate, timeout=1)`.
        """
        raise NotImplementedError(
            "ArduinoAD8232Provider is a planned hardware stub. "
            "Set USE_HARDWARE_ECG=False to use SimulatedECGProvider."
        )

    def disconnect(self) -> None:
        """
        Close the HC-05 Bluetooth serial connection safely.

        # TODO: Connect HC-05 Bluetooth Serial & AD8232 pin A0
        """
        if self._serial_conn is not None:
            self._serial_conn = None
        self._connected = False

    def read_raw_sample(self) -> int:
        """
        Read a single raw 10-bit ADC voltage sample (0-1023) from AD8232 pin A0.

        # TODO: Connect HC-05 Bluetooth Serial & AD8232 pin A0
        """
        raise NotImplementedError(
            "Raw ADC sampling from AD8232 pin A0 is not implemented yet."
        )

    def calculate_bpm_from_peaks(self, samples: list[int], sample_rate_hz: int = 250) -> int:
        """
        Detect R-peaks across a sliding window of AD8232 samples and compute BPM.

        # TODO: Connect HC-05 Bluetooth Serial & AD8232 pin A0
        """
        raise NotImplementedError(
            "R-peak BPM calculation from AD8232 stream is not implemented yet."
        )

    def get_bpm(self) -> int:
        """
        Acquire live BPM from the HC-05 + AD8232 hardware stream.

        # TODO: Connect HC-05 Bluetooth Serial & AD8232 pin A0
        """
        raise NotImplementedError(
            "Hardware BPM acquisition via HC-05 is not implemented yet."
        )

    def get_wave_state(self) -> str:
        """
        Classify rhythm/sleep-wake wave state from live AD8232 R-R intervals.

        # TODO: Connect HC-05 Bluetooth Serial & AD8232 pin A0
        """
        raise NotImplementedError(
            "Hardware wave classification via AD8232 is not implemented yet."
        )

    def get_mode_label(self) -> str:
        """Return the HUD mode label for the planned hardware provider."""
        return f"HARDWARE [Arduino AD8232 @ {self.port} {self.analog_pin}]"

    def get_reading(
        self,
        baseline_bpm: int | None = None,
        wave_state: str | None = None,
    ) -> dict[str, Any]:
        """
        Return full telemetry payload from the Arduino AD8232 + HC-05 stream.

        # TODO: Connect HC-05 Bluetooth Serial & AD8232 pin A0
        """
        raise NotImplementedError(
            "ArduinoAD8232Provider.get_reading() is a planned hardware stub."
        )


def get_telemetry_provider(use_hardware: bool | None = None) -> BaseTelemetryProvider:
    """Factory returning SimulatedECGProvider by default or ArduinoAD8232Provider if enabled."""
    use_hw = USE_HARDWARE_ECG if use_hardware is None else bool(use_hardware)
    if use_hw:
        return ArduinoAD8232Provider()
    return SimulatedECGProvider()
