"""Test the wake/sleep entities: Last wake, Next wake, Wake and Sleep events."""

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from homeassistant.const import STATE_UNKNOWN, Platform
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from opendisplay import GlobalConfig
from opendisplay.models.enums import PowerMode
import pytest
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.opendisplay.ble_lock import async_get_ble_lock
from tests.bluetooth import inject_bluetooth_service_info

from . import DEVICE_CONFIG, TEST_ADDRESS, make_v1_service_info

pytestmark = pytest.mark.usefixtures("entity_registry_enabled_by_default")

WAKE_SLOT = 7  # config value: one-based, so the block starts at dynamic[6]
WAKE_INDEX = WAKE_SLOT - 1
LAST_WAKE = "sensor.opendisplay_1234_last_wake"
NEXT_WAKE = "sensor.opendisplay_1234_next_wake"
WAKE_EVENT = "event.opendisplay_1234_wake"
SLEEP_EVENT = "event.opendisplay_1234_sleep"

RTC, POWER_ON = 3, 1
CODE_900S = 60  # sleep-duration code: 60 x 15 s
COORDINATOR_TIME = "custom_components.opendisplay.coordinator.time"


def _wake_device_config(slot: int = WAKE_SLOT) -> GlobalConfig:
    """Mains device that publishes the wake-info block (level 2)."""
    config = deepcopy(DEVICE_CONFIG)
    config.power = replace(config.power, reserved=bytes([slot, 0, 0, 0]))
    return config


def _sleepy_device_config() -> GlobalConfig:
    """Deep-sleeping battery device on firmware without wake info (level 1)."""
    config = deepcopy(DEVICE_CONFIG)
    config.power = replace(
        config.power,
        power_mode=PowerMode.BATTERY,
        deep_sleep_time_seconds=900,
        sleep_timeout_ms=20000,
    )
    return config


def _wake_info(
    cause: int | None,
    *,
    until: int = 0xFF,
    sleep_code: int = 0,
    reboot: bool = False,
    loop_counter: int = 0x10,
):
    """Build an advert carrying a wake-info block (None = no block).

    Vary loop_counter so the bluetooth manager does not drop it as a
    duplicate (the firmware's loop nibble does the same).
    """
    dynamic = bytearray(11)
    if cause is not None:
        dynamic[WAKE_INDEX : WAKE_INDEX + 3] = bytes([cause, until, sleep_code])
    return make_v1_service_info(
        bytes(dynamic), reboot=reboot, loop_counter=loop_counter
    )


def _record_events(hass: HomeAssistant, entity_id: str) -> list[dict]:
    """Collect the attributes of every firing of an event entity.

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
            event.data["entity_id"] == entity_id
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


async def test_no_wake_entities_for_mains_device_without_block(
    hass: HomeAssistant,
    entity_registry: er.EntityRegistry,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """No wake info and no deep sleep: nothing to track."""
    await setup_entry()

    for entity_id in (LAST_WAKE, NEXT_WAKE, WAKE_EVENT, SLEEP_EVENT):
        assert entity_registry.async_get(entity_id) is None


# --- Level 2: firmware publishes the wake-info block ------------------------


@pytest.mark.parametrize("device_config", [_wake_device_config()])
async def test_last_and_next_wake_from_block(
    hass: HomeAssistant,
    entity_registry: er.EntityRegistry,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """Last wake decodes the cause; Next wake = now + until-sleep + duration."""
    await setup_entry()

    entry = entity_registry.async_get(LAST_WAKE)
    assert entry is not None
    assert entry.unique_id == f"{TEST_ADDRESS}-last_wake"

    with patch(COORDINATOR_TIME, _clock(1_000_000.0)):
        inject_bluetooth_service_info(
            hass, _wake_info(RTC, until=12, sleep_code=CODE_900S)
        )
        await hass.async_block_till_done()

    state = hass.states.get(LAST_WAKE)
    assert state.state == "rtc"
    assert "scheduled" in state.attributes["options"]
    expected = datetime.fromtimestamp(1_000_000 + 12 + 900, tz=UTC)
    assert hass.states.get(NEXT_WAKE).state == expected.isoformat()


@pytest.mark.parametrize("device_config", [_wake_device_config()])
async def test_wake_event_on_cause_change(
    hass: HomeAssistant,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """First advert only primes; a different cause fires the event."""
    await setup_entry()
    fired = _record_events(hass, WAKE_EVENT)

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
    fired = _record_events(hass, WAKE_EVENT)

    with patch(COORDINATOR_TIME, _clock(1000.0)):
        inject_bluetooth_service_info(hass, _wake_info(RTC))
        await hass.async_block_till_done()
    with patch(COORDINATOR_TIME, _clock(1010.0)):
        inject_bluetooth_service_info(hass, _wake_info(RTC, loop_counter=0x20))
        await hass.async_block_till_done()
    assert fired == []

    with patch(COORDINATOR_TIME, _clock(1910.0)):
        inject_bluetooth_service_info(
            hass, _wake_info(RTC, until=19, sleep_code=CODE_900S, loop_counter=0x30)
        )
        await hass.async_block_till_done()

    assert len(fired) == 1
    assert fired[0]["event_type"] == "rtc"
    assert fired[0]["detected_by"] == "gap"
    assert fired[0]["seconds_until_sleep"] == 19
    assert fired[0]["sleep_seconds"] == 900


@pytest.mark.parametrize("device_config", [_wake_device_config()])
async def test_asleep_event_when_announced_time_passes(
    hass: HomeAssistant,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """Asleep fires after the last announced seconds-until-sleep (+ grace)."""
    await setup_entry()
    fired = _record_events(hass, SLEEP_EVENT)

    inject_bluetooth_service_info(hass, _wake_info(RTC, until=10, sleep_code=8))
    await hass.async_block_till_done()
    # A later advert in the same window re-announces: the timer follows it.
    inject_bluetooth_service_info(
        hass, _wake_info(RTC, until=9, sleep_code=8, loop_counter=0x20)
    )
    await hass.async_block_till_done()

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=8))
    await hass.async_block_till_done()
    assert fired == []

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=13))
    await hass.async_block_till_done()

    assert len(fired) == 1
    assert fired[0]["event_type"] == "asleep"
    assert fired[0]["sleep_seconds"] == 120
    assert fired[0]["next_wake"] is not None
    assert hass.states.get(SLEEP_EVENT).state != "unavailable"


@pytest.mark.parametrize("device_config", [_wake_device_config()])
async def test_no_asleep_event_when_not_sleeping(
    hass: HomeAssistant,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """0xFF seconds-until-sleep (wired / deep sleep off) never fires asleep."""
    await setup_entry()
    fired = _record_events(hass, SLEEP_EVENT)

    inject_bluetooth_service_info(hass, _wake_info(POWER_ON))
    await hass.async_block_till_done()
    assert hass.states.get(NEXT_WAKE).state == STATE_UNKNOWN
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=600))
    await hass.async_block_till_done()

    assert fired == []


@pytest.mark.parametrize("device_config", [_wake_device_config()])
async def test_asleep_waits_while_we_hold_the_link(
    hass: HomeAssistant,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """Our own connection keeps the device awake and silent: not asleep yet."""
    await setup_entry()
    fired = _record_events(hass, SLEEP_EVENT)

    inject_bluetooth_service_info(hass, _wake_info(RTC, until=5, sleep_code=8))
    await hass.async_block_till_done()

    lock = async_get_ble_lock(TEST_ADDRESS)
    await lock.acquire()
    try:
        async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=9))
        await hass.async_block_till_done()
        assert fired == []
    finally:
        lock.release()

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=15))
    await hass.async_block_till_done()
    assert len(fired) == 1


# --- Level 1: reboot flag only (today's firmware) ---------------------------


@pytest.mark.parametrize("device_config", [_sleepy_device_config()])
async def test_level1_scheduled_wake_and_asleep(
    hass: HomeAssistant,
    setup_entry: Callable[[], Awaitable[None]],
) -> None:
    """A deep-sleeping device without the block falls back to the reboot flag."""
    await setup_entry()
    wakes = _record_events(hass, WAKE_EVENT)
    sleeps = _record_events(hass, SLEEP_EVENT)

    with patch(COORDINATOR_TIME, _clock(1000.0)):
        inject_bluetooth_service_info(hass, _wake_info(None, reboot=False))
        await hass.async_block_till_done()
    assert hass.states.get(LAST_WAKE).state == "scheduled"
    expected = datetime.fromtimestamp(1000 + 900, tz=UTC)
    assert hass.states.get(NEXT_WAKE).state == expected.isoformat()

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=31))
    await hass.async_block_till_done()
    assert len(sleeps) == 1
    assert sleeps[0]["sleep_seconds"] == 900

    with patch(COORDINATOR_TIME, _clock(1900.0)):
        inject_bluetooth_service_info(
            hass, _wake_info(None, reboot=True, loop_counter=0x20)
        )
        await hass.async_block_till_done()

    assert len(wakes) == 1
    assert wakes[0]["event_type"] == "restart"
    assert wakes[0]["detected_by"] == "gap"
    assert hass.states.get(LAST_WAKE).state == "restart"
