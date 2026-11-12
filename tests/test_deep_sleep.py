"""Test the opendisplay.deep_sleep action: the one-shot 0x0053 override.

The wire format is checked against the real py-opendisplay device class with
only its transport read/write replaced, so a library that sent the wrong opcode
(0x0052 is CMD_POWER_OFF, a hard power cut) fails here. The service tests run
the real config entry, delivery manager and coordinator, with the connection
patched in the transport module like the delivery tests.
"""

from datetime import datetime, time, timedelta
import logging
import struct
from time import monotonic
from unittest.mock import AsyncMock, MagicMock

from freezegun.api import FrozenDateTimeFactory
from homeassistant.components.bluetooth import async_get_fallback_availability_interval
from homeassistant.components.bluetooth.const import UNAVAILABLE_TRACK_SECONDS
from homeassistant.const import STATE_UNAVAILABLE, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.util import dt as dt_util
from opendisplay import (
    BLEConnectionError,
    BLETimeoutError,
    OpenDisplayConnectionError,
    ProtocolError,
    RefreshMode,
)
from opendisplay.device import OpenDisplayDevice as LibraryDevice
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)
import voluptuous as vol

from custom_components.opendisplay.const import DOMAIN
from custom_components.opendisplay.deep_sleep import (
    MAX_SLEEP_SECONDS,
    MIN_SLEEP_SECONDS,
    DeepSleepRejectedError,
    PendingSleep,
    async_send_deep_sleep,
    clamp_sleep_seconds,
    seconds_until,
)

from . import (
    TEST_ADDRESS,
    connection_fails,
    connects_to,
    make_sleepy_device_config,
    make_v1_service_info,
)
from .bluetooth import (
    inject_bluetooth_service_info,
    patch_all_discovered_devices,
    patch_bluetooth_time,
)

TZ = "America/Toronto"
NEXT_WAKE = "sensor.opendisplay_1234_next_wake"
SLEEP_EVENT = "event.opendisplay_1234_sleep"
POWER_OFF_OPCODE = b"\x00\x52"
DEEP_SLEEP_OPCODE = b"\x00\x53"

pytestmark = pytest.mark.usefixtures("entity_registry_enabled_by_default")


def _local(*args: int) -> datetime:
    """Return a datetime in the test time zone."""
    return datetime(*args, tzinfo=dt_util.get_time_zone(TZ))


# --- wire format, against the real library ----------------------------------


def _library_device(read: AsyncMock) -> LibraryDevice:
    """Return a real OpenDisplayDevice with only its transport I/O replaced."""
    device = LibraryDevice(mac_address=TEST_ADDRESS)
    device._write = AsyncMock()
    device._read = read
    return device


def _written(device: LibraryDevice) -> list[bytes]:
    return [bytes(call.args[0]) for call in device._write.await_args_list]


@pytest.mark.parametrize(
    "seconds", [MIN_SLEEP_SECONDS, 300, 28_800, 0x5200, MAX_SLEEP_SECONDS]
)
async def test_frame_is_0x0053_with_big_endian_seconds(seconds: int) -> None:
    """The library writes [00 53][seconds:2 BE] and nothing else.

    0x5200 is included so a little-endian encoding would produce a frame that
    starts with the power-off opcode's low byte in the payload.
    """
    device = _library_device(AsyncMock(side_effect=BLETimeoutError("no reply")))

    await async_send_deep_sleep(device, seconds)

    frames = _written(device)
    assert frames == [DEEP_SLEEP_OPCODE + struct.pack(">H", seconds)]
    assert int.from_bytes(frames[0][:2], "big") == 0x0053
    assert not any(frame.startswith(POWER_OFF_OPCODE) for frame in frames)


@pytest.mark.parametrize(
    ("write_error", "read_error"),
    [
        (BLEConnectionError("link lost"), None),
        (None, BLEConnectionError("link lost")),
        (None, BLETimeoutError("no reply")),
    ],
    ids=["drop-on-write", "drop-on-read", "silent"],
)
async def test_link_drop_is_success(
    write_error: Exception | None, read_error: Exception | None
) -> None:
    """No reply and a dropped link both mean the device went to sleep."""
    device = _library_device(AsyncMock(side_effect=read_error))
    if write_error is not None:
        device._write.side_effect = write_error

    await async_send_deep_sleep(device, 3600)


async def test_transport_neutral_drop_is_success() -> None:
    """A WiFi link drop surfaces as the transport-neutral error; also success."""
    device = MagicMock()
    device.deep_sleep = AsyncMock(side_effect=OpenDisplayConnectionError("eof"))

    await async_send_deep_sleep(device, 3600)


@pytest.mark.parametrize(
    ("frame", "translation_key"),
    [
        (b"\xff\x53\x01\x00", "deep_sleep_disabled"),
        (b"\xff\x53\x02\x00", "deep_sleep_mains_powered"),
        (b"\xff\x53\x00\x00", "deep_sleep_unsupported"),
        (b"\xff\x53\x07\x00", "deep_sleep_rejected"),
    ],
    ids=["disabled", "mains", "unsupported", "unknown-code"],
)
async def test_nacks_map_to_clear_errors(frame: bytes, translation_key: str) -> None:
    """Each NACK sub-code becomes its own translated error."""
    read = AsyncMock(return_value=frame)
    device = _library_device(read)

    with pytest.raises(DeepSleepRejectedError) as err:
        await async_send_deep_sleep(device, 3600, device_id="dev1")

    assert err.value.translation_key == translation_key
    assert err.value.translation_placeholders["device_id"] == "dev1"
    # The capture hook is removed again.
    assert device._read is read


@pytest.mark.parametrize("seconds", [0, MIN_SLEEP_SECONDS - 1, MAX_SLEEP_SECONDS + 1])
async def test_out_of_range_seconds_never_reach_the_device(seconds: int) -> None:
    """0 would mean "no override", and above 65535 does not fit the frame."""
    device = _library_device(AsyncMock())

    with pytest.raises(ValueError, match="out of range"):
        await async_send_deep_sleep(device, seconds)

    device._write.assert_not_awaited()


# --- duration resolution -----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1, 60), (59.6, 60), (60, 60), (3600, 3600), (65_535, 65_535), (10**6, 65_535)],
)
def test_clamp(raw: float, expected: int) -> None:
    """Durations are limited to 60..65535 s."""
    assert clamp_sleep_seconds(raw) == expected


def test_duration_is_clamped_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Out-of-range durations are limited, and the clamp is logged."""
    short = PendingSleep(timedelta(seconds=10), None, None, 0.0)
    long = PendingSleep(timedelta(hours=30), None, None, 0.0)

    with caplog.at_level(logging.WARNING):
        assert short.resolve_seconds(TEST_ADDRESS) == MIN_SLEEP_SECONDS
        assert long.resolve_seconds(TEST_ADDRESS) == MAX_SLEEP_SECONDS

    assert caplog.text.count("clamping") == 2


@pytest.mark.parametrize(
    ("now", "until", "hours"),
    [
        (_local(2026, 6, 10, 22, 0), time(6, 0), 8),
        (_local(2026, 6, 10, 23, 30), time(0, 30), 1),
        (_local(2026, 6, 10, 5, 0), time(6, 0), 1),
        # Exactly now means tomorrow, not "sleep zero seconds".
        (_local(2026, 6, 10, 6, 0), time(6, 0), 24),
        # Spring forward (2026-03-08 02:00 EST -> 03:00 EDT): one hour shorter.
        (_local(2026, 3, 7, 22, 0), time(6, 0), 7),
        # Fall back (2026-11-01 02:00 EDT -> 01:00 EST): one hour longer.
        (_local(2026, 10, 31, 22, 0), time(6, 0), 9),
    ],
    ids=["night", "midnight", "same-day", "now", "dst-spring", "dst-fall"],
)
async def test_seconds_until(
    hass: HomeAssistant, now: datetime, until: time, hours: int
) -> None:
    """`until` is the next local wall-clock occurrence, across midnight and DST."""
    await hass.config.async_set_time_zone(TZ)

    assert seconds_until(until, now) == hours * 3600


async def test_until_beyond_one_sleep_sleeps_the_maximum(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A wake time more than ~18.2 h away is capped, with a warning."""
    await hass.config.async_set_time_zone(TZ)
    request = PendingSleep(None, time(6, 0), None, 0.0)

    with caplog.at_level(logging.WARNING):
        seconds = request.resolve_seconds(TEST_ADDRESS, _local(2026, 6, 10, 6, 0, 1))

    assert seconds == MAX_SLEEP_SECONDS
    assert "sleeping the maximum" in caplog.text


# --- the action --------------------------------------------------------------


@pytest.fixture
def platforms() -> list[Platform]:
    """Next wake sensor and Sleep event."""
    return [Platform.SENSOR, Platform.EVENT]


@pytest.fixture
def device_config():
    """Use a battery tag that deep-sleeps (300 s cadence, 10 s wake window)."""
    return make_sleepy_device_config()


@pytest.fixture
async def entry(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, setup_entry
) -> MockConfigEntry:
    """Return the loaded entry, with HA in the test time zone."""
    await hass.config.async_set_time_zone(TZ)
    await setup_entry()
    return mock_config_entry


def _device_id(hass: HomeAssistant, entry: MockConfigEntry) -> str:
    registry = dr.async_get(hass)
    return dr.async_entries_for_config_entry(registry, entry.entry_id)[0].id


def _sleeping_device() -> MagicMock:
    """Return a connected device mock whose deep_sleep succeeds (link drops)."""
    device = MagicMock()
    device.deep_sleep = AsyncMock()
    device.power_off = AsyncMock()
    device.upload_prepared_image = AsyncMock()
    return device


def _nacking_device(frame: bytes) -> MagicMock:
    """Return a device mock that answers the command with ``frame``.

    The library reads the reply through ``_read`` and raises ProtocolError on
    any FF 53 frame; the integration's capture hook sees the frame on the way.
    """
    device = _sleeping_device()
    device._read = AsyncMock(return_value=frame)

    async def _deep_sleep(_seconds):
        await device._read(5.0)
        raise ProtocolError("Device reported deep sleep is not supported")

    device.deep_sleep.side_effect = _deep_sleep
    return device


async def _call(hass: HomeAssistant, device_id: str, **data):
    return await hass.services.async_call(
        DOMAIN,
        "deep_sleep",
        {"device_id": device_id, **data},
        blocking=True,
        return_response=True,
    )


async def _advertise(hass: HomeAssistant, loop_counter: int = 0x11) -> None:
    inject_bluetooth_service_info(hass, make_v1_service_info(loop_counter=loop_counter))
    await hass.async_block_till_done()


async def test_awake_device_sleeps_now(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    """An awake tag gets the command live; the estimate and event follow it."""
    await _advertise(hass)
    device = _sleeping_device()
    before = dt_util.utcnow().timestamp()

    with connects_to(device):
        response = await _call(hass, _device_id(hass, entry), duration="08:00:00")
    await hass.async_block_till_done()

    device.deep_sleep.assert_awaited_once_with(28_800)
    device.power_off.assert_not_awaited()
    assert response["status"] == "sent"
    assert response["sleep_seconds"] == 28_800

    next_wake = dt_util.parse_datetime(hass.states.get(NEXT_WAKE).state)
    assert next_wake is not None
    assert before + 28_800 <= next_wake.timestamp() <= before + 28_800 + 5

    sleep_event = hass.states.get(SLEEP_EVENT)
    assert sleep_event.attributes["event_type"] == "asleep"
    assert sleep_event.attributes["sleep_seconds"] == 28_800


async def test_override_widens_availability_for_one_sleep(
    hass: HomeAssistant, entry: MockConfigEntry, freezer: FrozenDateTimeFactory
) -> None:
    """The window covers the long sleep, then reverts once the device is back."""
    profile = entry.runtime_data.sleep_profile
    await _advertise(hass)
    assert async_get_fallback_availability_interval(hass, TEST_ADDRESS) == (
        profile.availability_interval
    )

    with connects_to(_sleeping_device()):
        await _call(hass, _device_id(hass, entry), duration=7200)

    assert async_get_fallback_availability_interval(hass, TEST_ADDRESS) == (
        7200 + profile.availability_interval
    )

    # A straggler from the wake that just ended keeps the override.
    await _advertise(hass, loop_counter=0x21)
    assert entry.runtime_data.coordinator.sleep_override_active
    assert async_get_fallback_availability_interval(hass, TEST_ADDRESS) == (
        7200 + profile.availability_interval
    )

    # The next real wake ends it.
    freezer.tick(timedelta(seconds=7200))
    await _advertise(hass, loop_counter=0x31)
    assert not entry.runtime_data.coordinator.sleep_override_active
    assert async_get_fallback_availability_interval(hass, TEST_ADDRESS) == (
        profile.availability_interval
    )


@pytest.mark.parametrize("override", [True, False], ids=["override", "control"])
async def test_device_stays_available_through_the_long_sleep(
    hass: HomeAssistant, entry: MockConfigEntry, override: bool
) -> None:
    """Silence longer than the configured window but inside the override is fine.

    The control run shows the same silence does mark it unavailable otherwise.
    """
    start = dt_util.utcnow()
    start_monotonic = monotonic()
    await _advertise(hass)
    if override:
        with connects_to(_sleeping_device()):
            await _call(hass, _device_id(hass, entry), duration="06:00:00")

    profile = entry.runtime_data.sleep_profile
    advance = profile.availability_interval + UNAVAILABLE_TRACK_SECONDS + 600
    assert advance < 6 * 3600
    with (
        patch_bluetooth_time(start_monotonic + advance),
        patch_all_discovered_devices([]),
    ):
        async_fire_time_changed(hass, start + timedelta(seconds=advance))
        await hass.async_block_till_done()

    unavailable = hass.states.get(NEXT_WAKE).state == STATE_UNAVAILABLE
    assert unavailable is not override


async def test_asleep_device_queues_and_until_is_resolved_at_send_time(
    hass: HomeAssistant, entry: MockConfigEntry, freezer: FrozenDateTimeFactory
) -> None:
    """Queued at 22:00, delivered at the 22:40 wake: still wakes at 06:00."""
    freezer.move_to(_local(2026, 6, 10, 22, 0))
    device_id = _device_id(hass, entry)

    with connection_fails(BLEConnectionError("asleep")):
        response = await _call(hass, device_id, until="06:00:00")
    assert response["status"] == "queued"
    assert response["expires_at"] is not None

    freezer.move_to(_local(2026, 6, 10, 22, 40))
    device = _sleeping_device()
    with connects_to(device):
        await _advertise(hass)

    device.deep_sleep.assert_awaited_once_with(7 * 3600 + 20 * 60)
    assert entry.runtime_data.delivery.pending_sleep is None


async def test_probe_reaches_a_tag_that_looks_asleep(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    """A probe that connects sends live instead of queueing."""
    device = _sleeping_device()
    with connects_to(device):
        response = await _call(hass, _device_id(hass, entry), duration=600)

    assert response["status"] == "sent"
    device.deep_sleep.assert_awaited_once_with(600)


async def test_a_new_call_replaces_the_queued_one(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    """Only the latest queued deep sleep is sent."""
    device_id = _device_id(hass, entry)
    with connection_fails(BLEConnectionError("asleep")):
        await _call(hass, device_id, duration=3600)
        await _call(hass, device_id, duration=120)

    device = _sleeping_device()
    with connects_to(device):
        await _advertise(hass)

    device.deep_sleep.assert_awaited_once_with(120)


async def test_queued_deep_sleep_expires(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    """After queue_timeout_hours the command is dropped, not sent late."""
    with connection_fails(BLEConnectionError("asleep")):
        await _call(hass, _device_id(hass, entry), duration=3600)

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(hours=25))
    await hass.async_block_till_done()
    assert entry.runtime_data.delivery.pending_sleep is None

    device = _sleeping_device()
    with connects_to(device):
        await _advertise(hass)
    device.deep_sleep.assert_not_awaited()


async def test_queued_content_is_delivered_before_sleeping(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    """Content waiting for the wake lands first; the sleep ends the session."""
    manager = entry.runtime_data.delivery
    manager.submit_upload(
        prepared=(b"img", None, object()),
        refresh_mode=RefreshMode.FULL,
        partial_state=MagicMock(),
        use_measured_palettes=False,
        preview_jpeg=b"jpeg",
        device_id="dev1",
    )
    response = await _call(hass, _device_id(hass, entry), duration=3600)
    assert response["status"] == "queued"

    order: list[str] = []
    device = _sleeping_device()
    device.upload_prepared_image.side_effect = lambda *a, **k: order.append("upload")
    device.deep_sleep.side_effect = lambda *a, **k: order.append("sleep")
    with connects_to(device):
        await _advertise(hass)

    assert order == ["upload", "sleep"]
    assert manager.state.pending is False
    assert manager.pending_sleep is None


async def test_failed_content_delivery_holds_the_sleep(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    """If the content does not land, the device is not put to sleep yet."""
    manager = entry.runtime_data.delivery
    manager.submit_upload(
        prepared=(b"img", None, object()),
        refresh_mode=RefreshMode.FULL,
        partial_state=MagicMock(),
        use_measured_palettes=False,
        preview_jpeg=b"jpeg",
        device_id="dev1",
    )
    await _call(hass, _device_id(hass, entry), duration=3600)

    device = _sleeping_device()
    device.upload_prepared_image.side_effect = BLEConnectionError("dropped")
    with connects_to(device):
        await _advertise(hass)

    device.deep_sleep.assert_not_awaited()
    assert manager.pending_sleep is not None


@pytest.mark.parametrize(
    ("frame", "translation_key"),
    [
        (b"\xff\x53\x01\x00", "deep_sleep_disabled"),
        (b"\xff\x53\x02\x00", "deep_sleep_mains_powered"),
        (b"\xff\x53\x00\x00", "deep_sleep_unsupported"),
    ],
    ids=["disabled", "mains", "unsupported"],
)
async def test_live_nack_raises_and_changes_nothing(
    hass: HomeAssistant, entry: MockConfigEntry, frame: bytes, translation_key: str
) -> None:
    """A refusal is an error; the device is awake, so no estimate or window change."""
    await _advertise(hass)
    profile = entry.runtime_data.sleep_profile
    next_wake_before = hass.states.get(NEXT_WAKE).state

    with (
        connects_to(_nacking_device(frame)),
        pytest.raises(HomeAssistantError) as err,
    ):
        await _call(hass, _device_id(hass, entry), duration=3600)

    assert err.value.translation_key == translation_key
    assert hass.states.get(NEXT_WAKE).state == next_wake_before
    assert not entry.runtime_data.coordinator.sleep_override_active
    assert async_get_fallback_availability_interval(hass, TEST_ADDRESS) == (
        profile.availability_interval
    )


async def test_queued_nack_is_dropped(
    hass: HomeAssistant, entry: MockConfigEntry, caplog: pytest.LogCaptureFixture
) -> None:
    """A refusal at the wake is permanent: logged once, not retried."""
    with connection_fails(BLEConnectionError("asleep")):
        await _call(hass, _device_id(hass, entry), duration=3600)

    device = _nacking_device(b"\xff\x53\x01\x00")
    with connects_to(device):
        await _advertise(hass)
        await _advertise(hass, loop_counter=0x21)

    device.deep_sleep.assert_awaited_once()
    assert entry.runtime_data.delivery.pending_sleep is None
    assert "queued deep sleep refused" in caplog.text


async def test_queue_false_fails_fast_while_asleep(
    hass: HomeAssistant, entry: MockConfigEntry
) -> None:
    """With queue off, a sleeping tag is an error and nothing is held."""
    with connects_to(_sleeping_device()) as constructor:
        with pytest.raises(HomeAssistantError) as err:
            await _call(hass, _device_id(hass, entry), duration=3600, queue=False)
        constructor.assert_not_called()

    assert err.value.translation_key == "device_sleeping"
    assert entry.runtime_data.delivery.pending_sleep is None


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"duration": 3600, "until": "06:00:00"},
        {"duration": 0},
        {"duration": -5},
        {"until": "25:00"},
    ],
    ids=["neither", "both", "zero", "negative", "bad-time"],
)
async def test_schema_rejects(
    hass: HomeAssistant, entry: MockConfigEntry, data: dict
) -> None:
    """Exactly one of duration/until, and a positive duration."""
    with pytest.raises(vol.Invalid):
        await _call(hass, _device_id(hass, entry), **data)
