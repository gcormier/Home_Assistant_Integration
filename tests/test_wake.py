"""Test the wake-cause sensor and wake event (MSD wake-info byte)."""

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import replace
from unittest.mock import MagicMock, patch

from homeassistant.const import Platform
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from opendisplay import GlobalConfig
import pytest

from tests.bluetooth import inject_bluetooth_service_info

from . import DEVICE_CONFIG, TEST_ADDRESS, make_v1_service_info

pytestmark = pytest.mark.usefixtures("entity_registry_enabled_by_default")

WAKE_SLOT = 7  # config value: one-based, so dynamic[6]
WAKE_INDEX = WAKE_SLOT - 1
SENSOR = "sensor.opendisplay_1234_last_wake"
EVENT = "event.opendisplay_1234_wake"

RTC, POWER_ON = 3, 1


def _wake_device_config(slot: int = WAKE_SLOT) -> GlobalConfig:
    config = deepcopy(DEVICE_CONFIG)
    config.power = replace(config.power, reserved=bytes([slot, 0, 0, 0]))
    return config


def _wake_info(cause: int, *, reboot: bool = False, loop_counter: int = 0x10):
    """Build an advert carrying `cause`.

    Vary loop_counter so the bluetooth manager does not drop it as a
    duplicate (the firmware's loop nibble does the same).
    """
    dynamic = bytearray(11)
    dynamic[WAKE_INDEX] = cause
    return make_v1_service_info(
        bytes(dynamic), reboot=reboot, loop_counter=loop_counter
    )


def _record_wake_events(hass: HomeAssistant) -> list[dict]:
    """Collect the attributes of every fired wake event.

    Recorded from state_changed rather than read back afterwards: a power-on
    raises the reboot flag, and the integration reloads the entry on that edge.
    """
    fired: list[dict] = []
    seen: set[str] = set()

    @callback
    def _listener(event: Event) -> None:
        new_state = event.data["new_state"]
        # The state is the firing timestamp. The reboot reload re-adds the
        # entity, which restores (re-writes) the last firing: count it once.
        if (
            event.data["entity_id"] == EVENT
            and new_state is not None
            and new_state.attributes.get("event_type") is not None
            and new_state.state not in seen
        ):
            seen.add(new_state.state)
            fired.append(dict(new_state.attributes))

    hass.bus.async_listen("state_changed", _listener)
    return fired


def _clock(now: float) -> MagicMock:
    """Stand-in for the coordinator module's `time` import."""
    clock = MagicMock()
    clock.time.return_value = now
    return clock


@pytest.fixture
def platforms() -> list[Platform]:
    """Set up the sensor and event platforms."""
    return [Platform.SENSOR, Platform.EVENT]


async def test_no_wake_entities_without_slot(
    hass: HomeAssistant,
    entity_registry: er.EntityRegistry,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """A config with the slot at 0 (every config before the field) gets nothing."""
    await setup_entry()

    assert entity_registry.async_get(SENSOR) is None
    assert entity_registry.async_get(EVENT) is None


@pytest.mark.parametrize("device_config", [_wake_device_config()])
async def test_last_wake_sensor_reports_cause(
    hass: HomeAssistant,
    entity_registry: er.EntityRegistry,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """The enum sensor decodes the configured dynamic byte."""
    await setup_entry()

    entry = entity_registry.async_get(SENSOR)
    assert entry is not None
    assert entry.unique_id == f"{TEST_ADDRESS}-last_wake"

    inject_bluetooth_service_info(hass, _wake_info(RTC))
    await hass.async_block_till_done()

    state = hass.states.get(SENSOR)
    assert state.state == "rtc"
    assert "power_on" in state.attributes["options"]


@pytest.mark.parametrize("device_config", [_wake_device_config()])
async def test_wake_event_on_cause_change(
    hass: HomeAssistant,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """First advert only primes; a different cause fires the event."""
    await setup_entry()
    fired = _record_wake_events(hass)

    inject_bluetooth_service_info(hass, _wake_info(RTC))
    await hass.async_block_till_done()
    assert fired == []

    inject_bluetooth_service_info(hass, _wake_info(POWER_ON, reboot=True))
    await hass.async_block_till_done()

    assert len(fired) == 1
    assert fired[0]["event_type"] == "power_on"
    assert fired[0]["previous_cause"] == "rtc"
    assert fired[0]["detected_by"] == "cause_changed"


@pytest.mark.parametrize("device_config", [_wake_device_config()])
async def test_wake_event_after_sleep_gap(
    hass: HomeAssistant,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """The same cause seen again after a sleep gap is a new wake."""
    await setup_entry()
    fired = _record_wake_events(hass)
    target = "custom_components.opendisplay.coordinator.time"

    with patch(target, _clock(1000.0)):
        inject_bluetooth_service_info(hass, _wake_info(RTC))
        await hass.async_block_till_done()
    with patch(target, _clock(1010.0)):
        inject_bluetooth_service_info(hass, _wake_info(RTC, loop_counter=0x20))
        await hass.async_block_till_done()
    assert fired == []

    with patch(target, _clock(1910.0)):
        inject_bluetooth_service_info(hass, _wake_info(RTC, loop_counter=0x30))
        await hass.async_block_till_done()

    assert len(fired) == 1
    assert fired[0]["event_type"] == "rtc"
    assert fired[0]["detected_by"] == "gap"
    assert hass.states.get(SENSOR) is not None
