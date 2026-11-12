"""Test the wake-time config resync: it must read the device, not the cache.

A sleepy tag is often set up from the cached config while it is dark. The
resync that runs at its next wake is the only thing that brings HA's view of the
device (its sleep timing above all) back in line with what the device holds, so
these tests drive it end to end: a real entry, the real coordinator fed with
advertisements, and only the connection itself mocked.
"""

from collections.abc import Generator
from dataclasses import replace
import time
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from opendisplay import BLETimeoutError, GlobalConfig, ProtocolError
from opendisplay.models.config_json import config_from_json
from opendisplay.models.enums import PowerMode
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.opendisplay import services
from custom_components.opendisplay.ble_lock import _LOCKS
from custom_components.opendisplay.const import CONF_CACHED_STATE, DEFAULT_MISSED_CYCLES
from custom_components.opendisplay.delivery import RESYNC_RETRY_BACKOFF_S

from . import (
    FIRMWARE_VERSION,
    LANDING_URL,
    TEST_ADDRESS,
    connects_to,
    make_cached_state,
    make_sleepy_device_config,
    make_v1_service_info,
)
from .bluetooth import inject_bluetooth_service_info

CACHED_CONFIG = make_sleepy_device_config()  # 300 s deep sleep
FALLBACK = "custom_components.opendisplay.async_set_fallback_availability_interval"

pytestmark = [
    pytest.mark.parametrize("device_config", [CACHED_CONFIG], ids=[""]),
    pytest.mark.parametrize(
        "config_entry_data",
        [{CONF_CACHED_STATE: make_cached_state(CACHED_CONFIG)}],
        ids=[""],
    ),
]


def _with_sleep(seconds: int, config: GlobalConfig = CACHED_CONFIG) -> GlobalConfig:
    """Return ``config`` with a different deep-sleep duration."""
    return replace(config, power=replace(config.power, deep_sleep_time_seconds=seconds))


def _awake_device(reads: GlobalConfig | Exception) -> MagicMock:
    """Return a connected-device mock whose READ_CONFIG returns ``reads``.

    ``config`` deliberately holds the stale cached config: that is what the
    device object reports when it was built with ``config=`` and never asked.
    """
    device = MagicMock()
    device.config = CACHED_CONFIG
    device.is_flex = True
    device.landing_url = MagicMock(return_value=LANDING_URL)
    device.read_firmware_version = AsyncMock(return_value=FIRMWARE_VERSION)
    device.upload_prepared_image = AsyncMock()
    if isinstance(reads, Exception):
        device.interrogate = AsyncMock(side_effect=reads)
    else:
        device.interrogate = AsyncMock(return_value=reads)
    return device


@pytest.fixture(autouse=True)
def _fresh_ble_locks() -> None:
    """Start from an empty per-MAC lock registry.

    Earlier test modules can keep an entry (and so its ``ble_lock``) alive past
    their event loop. These tests wait on that lock from the reload path, and a
    lock bound to a dead loop raises instead of waiting.
    """
    _LOCKS.clear()


@pytest.fixture
def platforms() -> list[Platform]:
    """No platforms are needed; these tests read runtime state."""
    return []


@pytest.fixture
def fallback_interval() -> Generator[MagicMock]:
    """Capture what HA is told about the device's availability horizon."""
    with patch(FALLBACK) as mock:
        yield mock


@pytest.fixture
async def dark_entry(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    fallback_interval: MagicMock,
) -> MockConfigEntry:
    """Set the entry up from cache while the tag is asleep, as after a restart."""
    mock_config_entry.add_to_hass(hass)
    with patch(
        "custom_components.opendisplay.async_ble_device_from_address",
        return_value=None,
    ):
        assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
    assert mock_config_entry.runtime_data.config_resync_pending is True
    return mock_config_entry


async def _wake(hass: HomeAssistant, loop_counter: int = 0x11) -> None:
    """Deliver one advertisement, i.e. the tag is awake right now."""
    inject_bluetooth_service_info(hass, make_v1_service_info(loop_counter=loop_counter))
    await hass.async_block_till_done()


def _cached_sleep(entry: MockConfigEntry) -> int:
    """Return deep_sleep_time_seconds as stored in the dark-start cache."""
    config = config_from_json(entry.data[CONF_CACHED_STATE]["config"])
    return config.power.deep_sleep_time_seconds


async def test_wake_resync_reads_the_device_and_applies_the_new_sleep_time(
    hass: HomeAssistant,
    dark_entry: MockConfigEntry,
    fallback_interval: MagicMock,
) -> None:
    """A changed deep-sleep time reaches every consumer without a reload.

    The scenario from the field: set up from a cache saying 300 s, the device
    really sleeps 3600 s. One wake later the cache, the runtime config, the
    sleep profile and HA's availability horizon must all say 3600 s.
    """
    runtime = dark_entry.runtime_data
    device = _awake_device(_with_sleep(3600))

    with (
        connects_to(device),
        patch.object(hass.config_entries, "async_reload") as reload,
    ):
        await _wake(hass)

    device.interrogate.assert_awaited_once()
    assert runtime.config_resync_pending is False
    assert _cached_sleep(dark_entry) == 3600
    assert runtime.device_config.power.deep_sleep_time_seconds == 3600
    assert runtime.sleep_profile.deep_sleep_time_seconds == 3600
    fallback_interval.assert_called_with(
        hass, TEST_ADDRESS, runtime.sleep_profile.availability_interval
    )
    assert runtime.sleep_profile.availability_interval == (
        3600 * DEFAULT_MISSED_CYCLES + 10 + 60
    )
    # Applied in place: a reload during the wake would fight the BLE link.
    reload.assert_not_called()
    assert dark_entry.state is ConfigEntryState.LOADED


async def test_unchanged_config_does_not_rewrite_the_entry(
    hass: HomeAssistant, dark_entry: MockConfigEntry
) -> None:
    """A resync that finds nothing new leaves the config-entry store alone."""
    device = _awake_device(CACHED_CONFIG)

    with (
        connects_to(device),
        patch.object(
            hass.config_entries,
            "async_update_entry",
            wraps=hass.config_entries.async_update_entry,
        ) as update_entry,
    ):
        await _wake(hass)

    device.interrogate.assert_awaited_once()
    assert dark_entry.runtime_data.config_resync_pending is False
    # Ignore HA's own bluetooth discovery-key bookkeeping; only data writes count.
    assert not [c for c in update_entry.call_args_list if "data" in c.kwargs]


@pytest.mark.parametrize(
    "error",
    [ProtocolError("truncated"), BLETimeoutError("asleep again")],
    ids=["protocol", "timeout"],
)
async def test_failed_read_keeps_the_cache_and_the_resync_pending(
    hass: HomeAssistant,
    dark_entry: MockConfigEntry,
    error: Exception,
) -> None:
    """A failed READ_CONFIG changes nothing and is retried on a later wake.

    It must not reconnect on every remaining advertisement of the same wake.
    """
    runtime = dark_entry.runtime_data
    cached = dict(dark_entry.data[CONF_CACHED_STATE])
    failing = _awake_device(error)

    with connects_to(failing) as connect:
        await _wake(hass, loop_counter=0x11)
        await _wake(hass, loop_counter=0x12)
        await _wake(hass, loop_counter=0x13)

    assert connect.call_count == 1
    assert runtime.config_resync_pending is True
    assert dark_entry.data[CONF_CACHED_STATE] == cached
    assert runtime.sleep_profile.deep_sleep_time_seconds == 300
    manager = runtime.delivery
    assert manager is not None
    assert manager._resync_retry_at > time.monotonic()
    assert manager._resync_retry_at <= time.monotonic() + RESYNC_RETRY_BACKOFF_S

    # A later wake (past the hold-off) tries again and succeeds.
    manager._resync_retry_at = 0.0
    with connects_to(_awake_device(_with_sleep(3600))):
        await _wake(hass, loop_counter=0x14)

    assert runtime.config_resync_pending is False
    assert runtime.sleep_profile.deep_sleep_time_seconds == 3600


async def test_queued_upload_is_not_held_back_by_a_failed_resync(
    hass: HomeAssistant, dark_entry: MockConfigEntry
) -> None:
    """The resync hold-off applies to the resync only, not to queued content."""
    manager = dark_entry.runtime_data.delivery
    assert manager is not None
    with connects_to(_awake_device(ProtocolError("truncated"))):
        await _wake(hass, loop_counter=0x11)

    device = _awake_device(CACHED_CONFIG)
    with connects_to(device):
        manager.submit_upload(
            prepared=(b"img", None, object()),
            refresh_mode=MagicMock(),
            partial_state=MagicMock(),
            use_measured_palettes=False,
            preview_jpeg=b"jpeg",
            device_id=None,
        )
        await _wake(hass, loop_counter=0x12)

    device.upload_prepared_image.assert_awaited_once()
    device.interrogate.assert_not_awaited()
    assert dark_entry.runtime_data.config_resync_pending is True


async def test_resync_that_changes_the_entity_set_reloads_after_the_link(
    hass: HomeAssistant, dark_entry: MockConfigEntry
) -> None:
    """Switching the tag to USB power changes which entities exist: reload.

    Battery sensors and the wake entities are created at setup, so this one
    cannot be applied in place. The reload waits for the BLE lock, i.e. for
    the resync's own connection to be released.
    """
    usb = replace(
        CACHED_CONFIG,
        power=replace(CACHED_CONFIG.power, power_mode=PowerMode.USB),
    )
    lock_held_at_reload: list[bool] = []

    async def _reload(_entry_id: str) -> bool:
        lock_held_at_reload.append(dark_entry.runtime_data.ble_lock.locked())
        return True

    with (
        connects_to(_awake_device(usb)),
        patch.object(hass.config_entries, "async_reload", side_effect=_reload),
    ):
        await _wake(hass)

    assert lock_held_at_reload == [False]
    assert _cached_sleep(dark_entry) == 300
    assert dark_entry.runtime_data.device_config.power.power_mode == PowerMode.USB


async def test_reload_after_resync_sets_up_from_the_corrected_cache(
    hass: HomeAssistant, dark_entry: MockConfigEntry, fallback_interval: MagicMock
) -> None:
    """A later dark reload no longer resurrects the stale sleep time."""
    with connects_to(_awake_device(_with_sleep(3600))):
        await _wake(hass)

    with patch(
        "custom_components.opendisplay.async_ble_device_from_address",
        return_value=None,
    ):
        assert await hass.config_entries.async_reload(dark_entry.entry_id)
        await hass.async_block_till_done()

    runtime = dark_entry.runtime_data
    assert runtime.sleep_profile.deep_sleep_time_seconds == 3600
    fallback_interval.assert_called_with(
        hass, TEST_ADDRESS, runtime.sleep_profile.availability_interval
    )


async def _connect_and_run_now(
    hass: HomeAssistant, entry: MockConfigEntry
) -> MagicMock:
    """Run an immediate (non-queued) service connection with a trivial action."""
    action = AsyncMock()
    await services._async_connect_and_run(hass, entry, action)
    return action


async def test_service_call_piggybacks_a_pending_resync(
    hass: HomeAssistant, dark_entry: MockConfigEntry
) -> None:
    """An immediate (non-queued) connection also clears a pending resync."""
    device = _awake_device(_with_sleep(3600))

    with connects_to(device):
        action = await _connect_and_run_now(hass, dark_entry)

    action.assert_awaited_once_with(device)
    device.interrogate.assert_awaited_once()
    assert dark_entry.runtime_data.config_resync_pending is False
    assert dark_entry.runtime_data.sleep_profile.deep_sleep_time_seconds == 3600


async def test_failed_piggyback_resync_does_not_fail_the_service_call(
    hass: HomeAssistant, dark_entry: MockConfigEntry
) -> None:
    """The requested action succeeded; a failed re-read stays pending quietly."""
    device = _awake_device(ProtocolError("truncated"))

    with connects_to(device):
        action = await _connect_and_run_now(hass, dark_entry)

    action.assert_awaited_once_with(device)
    device.interrogate.assert_awaited_once()
    assert dark_entry.runtime_data.config_resync_pending is True
    assert _cached_sleep(dark_entry) == 300


async def test_no_resync_pending_means_no_extra_read(
    hass: HomeAssistant, dark_entry: MockConfigEntry
) -> None:
    """Once in sync, service connections do not pay for a READ_CONFIG."""
    with connects_to(_awake_device(CACHED_CONFIG)):
        await _wake(hass)
    assert dark_entry.runtime_data.config_resync_pending is False

    device = _awake_device(CACHED_CONFIG)
    with connects_to(device):
        await _connect_and_run_now(hass, dark_entry)

    device.interrogate.assert_not_awaited()
