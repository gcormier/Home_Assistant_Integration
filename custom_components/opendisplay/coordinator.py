"""Passive BLE coordinator for OpenDisplay devices."""

from dataclasses import dataclass, field
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

_LOGGER: logging.Logger = logging.getLogger(__package__)


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


class OpenDisplayCoordinator(PassiveBluetoothDataUpdateCoordinator):
    """Coordinator for passive BLE advertisement updates from an OpenDisplay device."""

    def __init__(
        self,
        hass: HomeAssistant,
        address: str,
        binary_inputs: list[BinaryInputs] | None = None,
        wake_info_byte_index: int | None = None,
    ) -> None:
        """Initialize the coordinator.

        binary_inputs comes from the device config and tells the tracker which
        bytes of the advertisement's dynamic block are really buttons. The rest
        belong to touch controllers and sensors and decode into valid-looking
        button reports, so without it a moving touch coordinate or a refreshed
        SHT40 reading produces phantom button transitions.

        wake_info_byte_index is PowerOption.published_wake_info_byte_index:
        the dynamic byte carrying the wake cause, or None when the device
        does not publish one.
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
        self.wake_info_byte_index = wake_info_byte_index
        self._wake_tracker: WakeTracker | None = (
            None if wake_info_byte_index is None else WakeTracker(wake_info_byte_index)
        )
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
                    wake_event.cause.name.lower(),
                    wake_event.reason,
                )
            self.data = OpenDisplayUpdate(
                address=service_info.address,
                advertisement=advertisement,
                rssi=service_info.rssi,
                last_seen=now,
                button_events=button_events,
                touch_events=touch_events,
                wake_event=wake_event,
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
