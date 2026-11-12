"""Passive BLE coordinator for OpenDisplay devices."""

from dataclasses import dataclass, field, replace
from datetime import datetime
import logging
import time

from homeassistant.components.bluetooth import (
    BluetoothChange,
    BluetoothScanningMode,
    BluetoothServiceInfoBleak,
)
from homeassistant.components.bluetooth.passive_update_coordinator import (
    PassiveBluetoothDataUpdateCoordinator,
)
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.event import async_call_later

from opendisplay import MANUFACTURER_ID, AdvertisementTracker, parse_advertisement
from opendisplay.models.advertisement import (
    AdvertisementData,
    ButtonChangeEvent,
    TouchChangeEvent,
    TouchTracker,
    WakeChangeEvent,
    WakeTracker,
)
from opendisplay.models.config import BinaryInputs

from .ble_lock import async_get_ble_lock

_LOGGER: logging.Logger = logging.getLogger(__package__)

# Level 2: the device announces when it will sleep; allow for clock skew and
# the time it takes to stop advertising before declaring it asleep.
ASLEEP_GRACE_SECONDS = 3.0
# Level 1 (no announcement): this much silence after the last advertisement
# means the device went back to sleep.
ASLEEP_SILENCE_SECONDS = 30.0
# While this integration holds the BLE link the device does not advertise;
# re-check this often instead of declaring it asleep.
ASLEEP_RECHECK_SECONDS = 5.0


@dataclass(frozen=True)
class WakeTracking:
    """How wake/sleep is tracked for one device.

    byte_index is PowerOption.published_wake_info_byte_index (level 2: the
    device publishes cause and sleep timing), or None to fall back to the
    reboot flag (level 1). sleep_seconds is the configured
    deep_sleep_time_seconds, used for the level-1 next-wake estimate.
    """

    byte_index: int | None
    sleep_seconds: int | None


@dataclass(frozen=True)
class AsleepEvent:
    """The device went to sleep (announced time passed, or level-1 silence)."""

    next_wake: float | None
    sleep_seconds: int | None


@dataclass
class OpenDisplayUpdate:
    """Parsed advertisement data for one OpenDisplay device."""

    address: str
    advertisement: AdvertisementData
    rssi: int | None = None
    last_seen: float | None = None
    button_events: list[ButtonChangeEvent] = field(default_factory=list)
    touch_events: list[TouchChangeEvent] = field(default_factory=list)
    # Set on the advertisement that revealed a new wake (see WakeTracker).
    wake_event: WakeChangeEvent | None = None
    # Cause of the current/last wake, and the predicted next wake (epoch s).
    wake_cause: str | None = None
    next_wake: float | None = None
    # Set only on the synthetic update pushed when the device fell asleep.
    asleep_event: AsleepEvent | None = None


class OpenDisplayCoordinator(PassiveBluetoothDataUpdateCoordinator):
    """Coordinator for passive BLE advertisement updates from an OpenDisplay device."""

    def __init__(
        self,
        hass: HomeAssistant,
        address: str,
        binary_inputs: list[BinaryInputs] | None = None,
        wake_tracking: WakeTracking | None = None,
    ) -> None:
        """Initialize the coordinator.

        binary_inputs comes from the device config and tells the tracker which
        bytes of the advertisement's dynamic block are really buttons. The rest
        belong to touch controllers and sensors and decode into valid-looking
        button reports, so without it a moving touch coordinate or a refreshed
        SHT40 reading produces phantom button transitions.

        wake_tracking enables the wake/sleep entities (see WakeTracking);
        None for devices that neither publish wake info nor deep sleep.
        """
        super().__init__(
            hass,
            _LOGGER,
            address,
            BluetoothScanningMode.PASSIVE,
            connectable=True,
        )
        self.data: OpenDisplayUpdate | None = None
        self._tracker: AdvertisementTracker = AdvertisementTracker(
            None
            if binary_inputs is None
            else [
                index
                for bi in binary_inputs
                if (index := bi.published_button_byte_index) is not None
            ]
        )
        self.touch_trackers: list[TouchTracker] = []
        self.wake_tracking = wake_tracking
        self._wake_tracker: WakeTracker | None = (
            None if wake_tracking is None else WakeTracker(wake_tracking.byte_index)
        )
        self._cancel_asleep: CALLBACK_TYPE | None = None
        self._pending_asleep: AsleepEvent | None = None
        # When the device is expected to (have) go(ne) to sleep for the pending
        # asleep event; lets a sleep-duration change re-derive its next wake.
        self._pending_sleep_at: float | None = None
        # Subscribers notified once when the advertised reboot flag goes
        # False -> True (the device rebooted since we last talked to it).
        self._reboot_callbacks: set[CALLBACK_TYPE] = set()
        # Subscribers notified on every parsed advertisement (the device is
        # awake and reachable right now). The delivery manager uses this as the
        # wake rendezvous to drain queued work.
        self._device_seen_callbacks: set[CALLBACK_TYPE] = set()
        # Reboot-flag edge detection: the device sets the advertised reboot flag
        # on boot and clears it on first connect. None until the first v1 advert.
        self._last_reboot_flag: bool | None = None

    @callback
    def async_subscribe_reboot(self, callback_: CALLBACK_TYPE) -> CALLBACK_TYPE:
        """Subscribe to device reboots (advertised reboot flag False -> True).

        Returns an unsubscribe callback.
        """
        self._reboot_callbacks.add(callback_)

        @callback
        def _unsubscribe() -> None:
            self._reboot_callbacks.discard(callback_)

        return _unsubscribe

    @callback
    def async_subscribe_device_seen(self, callback_: CALLBACK_TYPE) -> CALLBACK_TYPE:
        """Subscribe to "device seen" events (every parsed advertisement).

        Fired after each successfully parsed advertisement, i.e. whenever the
        device is awake and reachable. Returns an unsubscribe callback.
        """
        self._device_seen_callbacks.add(callback_)

        @callback
        def _unsubscribe() -> None:
            self._device_seen_callbacks.discard(callback_)

        return _unsubscribe

    @callback
    def _async_handle_unavailable(
        self, service_info: BluetoothServiceInfoBleak
    ) -> None:
        """Handle the device going unavailable."""
        if self._available:
            _LOGGER.info("%s: Device is unavailable", service_info.address)
        super()._async_handle_unavailable(service_info)

    @callback
    def _async_handle_bluetooth_event(
        self,
        service_info: BluetoothServiceInfoBleak,
        change: BluetoothChange,
    ) -> None:
        """Handle a Bluetooth advertisement event."""
        if not self._available:
            _LOGGER.info("%s: Device is available again", service_info.address)

        rebooted = False
        if MANUFACTURER_ID not in service_info.manufacturer_data:
            super()._async_handle_bluetooth_event(service_info, change)
            return

        try:
            advertisement = parse_advertisement(
                service_info.manufacturer_data[MANUFACTURER_ID]
            )
        except ValueError as err:
            _LOGGER.debug(
                "%s: Failed to parse advertisement data: %s",
                service_info.address,
                err,
                exc_info=True,
            )
        else:
            rebooted = self._check_reboot_flag(advertisement)
            button_events = self._tracker.update(service_info.address, advertisement)
            touch_events: list[TouchChangeEvent] = []
            for touch_tracker in self.touch_trackers:
                touch_events.extend(
                    touch_tracker.update(service_info.address, advertisement)
                )
            now = time.time()
            wake_event = (
                None
                if self._wake_tracker is None
                else self._wake_tracker.update(
                    service_info.address, advertisement, timestamp=now
                )
            )
            if wake_event is not None:
                _LOGGER.debug(
                    "%s: New wake (%s, detected by %s)",
                    service_info.address,
                    wake_event.cause,
                    wake_event.reason,
                )
            next_wake = self._schedule_asleep(advertisement, now)
            self.data = OpenDisplayUpdate(
                address=service_info.address,
                advertisement=advertisement,
                rssi=service_info.rssi,
                last_seen=now,
                button_events=button_events,
                touch_events=touch_events,
                wake_event=wake_event,
                wake_cause=None
                if self._wake_tracker is None
                else self._wake_tracker.current(service_info.address),
                next_wake=next_wake,
            )
            for device_seen_callback in list(self._device_seen_callbacks):
                device_seen_callback()

        super()._async_handle_bluetooth_event(service_info, change)
        # After the listeners: a reboot callback may reload the entry (and so
        # unload every entity) eagerly, which would drop the events this very
        # advertisement carries -- notably the wake event of the reboot.
        if rebooted:
            self._notify_reboot()

    @callback
    def _schedule_asleep(
        self, advertisement: AdvertisementData, now: float
    ) -> float | None:
        """(Re)arm the asleep timer from this advertisement; return next wake.

        Level 2 uses the announced seconds-until-sleep and sleep duration.
        Level 1 assumes the device sleeps right after its last advertisement
        and declares it asleep after ASLEEP_SILENCE_SECONDS of silence.
        """
        if self._cancel_asleep is not None:
            self._cancel_asleep()
            self._cancel_asleep = None
        self._pending_asleep = None
        self._pending_sleep_at = None
        tracking = self.wake_tracking
        if tracking is None:
            return None
        if tracking.byte_index is not None:
            info = advertisement.wake_info(tracking.byte_index)
            if info is None or info.seconds_until_sleep is None:
                return None
            sleep_at = now + info.seconds_until_sleep
            sleep_seconds = info.sleep_seconds
            delay = info.seconds_until_sleep + ASLEEP_GRACE_SECONDS
        else:
            if tracking.sleep_seconds is None:
                return None
            sleep_at = now
            sleep_seconds = tracking.sleep_seconds
            delay = ASLEEP_SILENCE_SECONDS
        next_wake = None if sleep_seconds is None else sleep_at + sleep_seconds
        self._pending_asleep = AsleepEvent(
            next_wake=next_wake, sleep_seconds=sleep_seconds
        )
        self._pending_sleep_at = sleep_at
        self._cancel_asleep = async_call_later(self.hass, delay, self._async_asleep)
        return next_wake

    @callback
    def _async_asleep(self, _now: datetime) -> None:
        """Declare the device asleep: its announced sleep time passed in silence."""
        self._cancel_asleep = None
        event = self._pending_asleep
        if event is None or self.data is None:
            return
        if async_get_ble_lock(self.address).locked():
            # Our own connection keeps it awake and silent; not asleep yet.
            if self._wake_tracker is not None:
                self._wake_tracker.touch(self.address)
            self._cancel_asleep = async_call_later(
                self.hass, ASLEEP_RECHECK_SECONDS, self._async_asleep
            )
            return
        self._pending_asleep = None
        _LOGGER.debug(
            "%s: Device is asleep (next wake %s)", self.address, event.next_wake
        )
        self.data = replace(
            self.data,
            button_events=[],
            touch_events=[],
            wake_event=None,
            asleep_event=event,
        )
        self.async_update_listeners()

    @callback
    def async_set_wake_tracking(self, tracking: WakeTracking) -> None:
        """Apply re-read wake tracking (e.g. a new deep_sleep_time_seconds).

        Only valid while wake tracking stays enabled: turning it on or off
        changes which entities exist, which needs an entry reload. A level-1
        prediction already made from the old sleep duration is re-derived so the
        next-wake sensor does not wait for another advertisement to correct.
        """
        assert self.wake_tracking is not None
        if tracking == self.wake_tracking:
            return
        if tracking.byte_index != self.wake_tracking.byte_index:
            self._wake_tracker = WakeTracker(tracking.byte_index)
        self.wake_tracking = tracking
        pending = self._pending_asleep
        sleep_at = self._pending_sleep_at
        if (
            tracking.byte_index is not None
            or pending is None
            or sleep_at is None
            or tracking.sleep_seconds is None
        ):
            return
        next_wake = sleep_at + tracking.sleep_seconds
        self._pending_asleep = AsleepEvent(
            next_wake=next_wake, sleep_seconds=tracking.sleep_seconds
        )
        if self.data is not None:
            # Clear the per-advertisement events so listeners do not re-fire them.
            self.data = replace(
                self.data,
                next_wake=next_wake,
                button_events=[],
                touch_events=[],
                wake_event=None,
            )
            self.async_update_listeners()

    @callback
    def async_shutdown_wake_timer(self) -> None:
        """Cancel the pending asleep timer (entry unload)."""
        if self._cancel_asleep is not None:
            self._cancel_asleep()
            self._cancel_asleep = None

    @callback
    def _check_reboot_flag(self, advertisement: AdvertisementData) -> bool:
        """Detect a reboot, i.e. a reboot-flag False -> True edge.

        The device sets the advertised reboot flag on boot and clears it on the
        first BLE connection. We react only to a False -> True transition: the
        initial observation (None -> True) is ignored because setup already
        synced this boot, and a flag that stays True (True -> True, e.g. a device
        that never clears it) is self-guarding and won't fire again.
        """
        reboot_flag = advertisement.reboot_flag
        if reboot_flag is None:
            # Legacy advertisement: no reboot flag, leave previous state intact.
            return False

        previous = self._last_reboot_flag
        self._last_reboot_flag = reboot_flag
        return bool(reboot_flag and previous is False)

    @callback
    def _notify_reboot(self) -> None:
        """Run the reboot subscribers (see _check_reboot_flag)."""
        if not self._reboot_callbacks:
            return
        _LOGGER.info("%s: Device rebooted since last connection", self.address)
        for reboot_callback in list(self._reboot_callbacks):
            reboot_callback()
