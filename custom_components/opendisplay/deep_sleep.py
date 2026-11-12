"""One-shot deep-sleep override (firmware ``CMD_DEEP_SLEEP``, 0x0053).

The firmware accepts ``[0x00][0x53][seconds:2 big-endian]``: sleep now, and wake
after ``seconds`` for this one sleep only. The override is never saved; the wake
after it goes back to the configured ``deep_sleep_time_seconds`` cadence.

The command gets no reply on success. The device goes to sleep and the link
drops, and that drop *is* the confirmation. A refusal comes back as a NACK frame
``FF 53 <code> 00`` and leaves the device awake:

* ``01`` deep sleep is disabled in the device config
* ``02`` the device is mains powered
* ``00`` the target does not support deep sleep

The opcode is sent by ``OpenDisplayDevice.deep_sleep()`` (py-opendisplay >= 7.17,
the first release on 0x0053). ``0x0052`` is ``CMD_POWER_OFF`` on current firmware,
a hard rail cut that only a button press recovers from; nothing here may send it.

This module holds the pure parts (duration resolution, the pending request) and
the one helper that talks to the device, shared by the live and queued paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import CALLBACK_TYPE
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from opendisplay import (
    OpenDisplayConnectionError,
    OpenDisplayTimeoutError,
    ProtocolError,
)

from .const import DOMAIN

if TYPE_CHECKING:
    from opendisplay import OpenDisplayDevice

_LOGGER = logging.getLogger(__name__)

# The wire opcode this feature must send (CMD_DEEP_SLEEP). Kept here so the
# tests can assert against the library's actual bytes.
CMD_DEEP_SLEEP = 0x0053

# Client-side floor and the 16-bit wire maximum (~18.2 h).
MIN_SLEEP_SECONDS = 60
MAX_SLEEP_SECONDS = 0xFFFF

# NACK sub-codes in ``FF 53 <code> 00`` mapped to exception translation keys.
NACK_TRANSLATION_KEYS: dict[int, str] = {
    0x00: "deep_sleep_unsupported",
    0x01: "deep_sleep_disabled",
    0x02: "deep_sleep_mains_powered",
}


class DeepSleepRejectedError(HomeAssistantError):
    """The device refused the deep-sleep command and is still awake."""


def clamp_sleep_seconds(seconds: float) -> int:
    """Limit a sleep duration to what the command can carry (60..65535 s)."""
    return int(min(max(round(seconds), MIN_SLEEP_SECONDS), MAX_SLEEP_SECONDS))


def seconds_until(until: time, now: datetime | None = None) -> float:
    """Return seconds from ``now`` to the next ``until`` in HA's local timezone.

    The target is the next occurrence strictly after ``now``, so a time that has
    already passed today (or is exactly now) means tomorrow. The date is stepped
    on the local calendar and then localized, so a DST change in between is
    absorbed: 22:00 -> 06:00 across a spring-forward night is 7 h, not 8 h.

    Both ends are converted to UTC before subtracting. Python subtracts two
    aware datetimes that share a tzinfo as naive wall-clock values, which would
    silently ignore the DST shift.
    """
    current = dt_util.as_local(now if now is not None else dt_util.utcnow())
    tz = current.tzinfo
    target = datetime.combine(current.date(), until, tzinfo=tz)
    if target <= current:
        target = datetime.combine(current.date() + timedelta(days=1), until, tzinfo=tz)
    return (dt_util.as_utc(target) - dt_util.as_utc(current)).total_seconds()


@dataclass
class PendingSleep:
    """A deep-sleep request waiting to be sent (queued, or about to go live).

    Exactly one of ``duration`` / ``until`` is set. The seconds are resolved by
    ``resolve_seconds`` at send time, not at call time: an ``until`` request
    queued at 22:00 and delivered at a 22:40 wake must still wake at 06:00.
    """

    duration: timedelta | None
    until: time | None
    device_id: str | None
    queued_at: float
    expires_at: float | None = None
    cancel_deadline: CALLBACK_TYPE | None = None

    def resolve_seconds(self, address: str, now: datetime | None = None) -> int:
        """Return the clamped override for a command sent at ``now``."""
        if self.until is not None:
            raw = seconds_until(self.until, now)
            if raw > MAX_SLEEP_SECONDS:
                _LOGGER.warning(
                    "%s: %s is %.1f h away, beyond the %.1f h a single deep sleep "
                    "can last; sleeping the maximum instead",
                    address,
                    self.until.isoformat(),
                    raw / 3600,
                    MAX_SLEEP_SECONDS / 3600,
                )
        else:
            assert self.duration is not None
            raw = self.duration.total_seconds()
            if not MIN_SLEEP_SECONDS <= raw <= MAX_SLEEP_SECONDS:
                _LOGGER.warning(
                    "%s: deep sleep duration %.0f s is outside %s..%s s; clamping",
                    address,
                    raw,
                    MIN_SLEEP_SECONDS,
                    MAX_SLEEP_SECONDS,
                )
        return clamp_sleep_seconds(raw)


async def async_send_deep_sleep(
    device: OpenDisplayDevice, seconds: int, *, device_id: str | None = None
) -> None:
    """Send the one-shot deep-sleep override over an open connection.

    Returns normally when the command was delivered: the library already treats
    a dropped link or a missing reply as success, and so does this helper for a
    transport-neutral connection error (the WiFi transport raises those rather
    than the BLE subclasses). A NACK raises ``DeepSleepRejectedError``.

    The library folds every ``FF 53 ..`` frame into one ``ProtocolError`` and
    drops the sub-code, so the reply is captured on the way past ``_read`` to
    tell the three refusals apart. Only the read is observed; the command bytes
    are built and written by ``OpenDisplayDevice.deep_sleep()`` itself.
    """
    if not MIN_SLEEP_SECONDS <= seconds <= MAX_SLEEP_SECONDS:
        raise ValueError(f"deep sleep seconds out of range: {seconds}")

    frames: list[bytes] = []
    original_read = getattr(device, "_read", None)
    if original_read is not None:

        async def _capturing_read(*args: Any, **kwargs: Any) -> bytes:
            data = await original_read(*args, **kwargs)
            frames.append(bytes(data))
            return data

        device._read = _capturing_read  # type: ignore[method-assign]  # noqa: SLF001
    try:
        await device.deep_sleep(seconds)
    except ProtocolError as err:
        raise _rejection(frames[-1] if frames else None, err, device_id) from err
    except (OpenDisplayConnectionError, OpenDisplayTimeoutError) as err:
        _LOGGER.debug("Link dropped after deep sleep command (expected): %s", err)
    finally:
        if original_read is not None:
            device._read = original_read  # type: ignore[method-assign]  # noqa: SLF001


def _rejection(
    frame: bytes | None, err: Exception, device_id: str | None
) -> DeepSleepRejectedError:
    """Map a NACK frame to a translated error."""
    placeholders = {"device_id": device_id or "", "error": str(err)}
    if (
        frame is not None
        and len(frame) >= 3
        and frame[0] == 0xFF
        and frame[1] == CMD_DEEP_SLEEP & 0xFF
        and (key := NACK_TRANSLATION_KEYS.get(frame[2])) is not None
    ):
        return DeepSleepRejectedError(
            translation_domain=DOMAIN,
            translation_key=key,
            translation_placeholders=placeholders,
        )
    if frame is not None:
        placeholders["error"] = f"{err} (reply {frame.hex(' ')})"
    return DeepSleepRejectedError(
        translation_domain=DOMAIN,
        translation_key="deep_sleep_rejected",
        translation_placeholders=placeholders,
    )
