"""NEAT driver adapter for the AI car simulation.

Wraps a ``neat-python`` ``FeedForwardNetwork`` (or any object that exposes
an ``activate(inputs)`` method) so it conforms to :class:`DriverInterface`.

NEAT-specific logic is isolated here; the rest of the application only
sees the :class:`DriverInterface` contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_car_sim.ai.driver_interface import Action, DriverInterface
from ai_car_sim.domain.vehicle_state import VehicleState


# ---------------------------------------------------------------------------
# Minimal network protocol – avoids a hard neat-python import at module level
# ---------------------------------------------------------------------------

@runtime_checkable
class NeatNetwork(Protocol):
    """Subset of ``neat.nn.FeedForwardNetwork`` required by :class:`NeatDriver`."""

    def activate(self, inputs: list[float]) -> list[float]:
        """Run a forward pass and return output activations."""
        ...


# ---------------------------------------------------------------------------
# NeatDriver
# ---------------------------------------------------------------------------

class NeatDriver(DriverInterface):
    """Driver that delegates action selection to a NEAT neural network.

    The network is activated with the normalised sensor inputs and the
    output neuron with the highest activation is mapped to an
    :class:`~ai_car_sim.ai.driver_interface.Action` via ``argmax``.

    This mirrors the original ``newcar.py`` logic::

        output = net.activate(car.get_data())
        choice = output.index(max(output))

    Args:
        network: A NEAT ``FeedForwardNetwork`` (or any object implementing
            :class:`NeatNetwork`) produced by
            ``neat.nn.FeedForwardNetwork.create(genome, config)``.
        expected_outputs: Expected number of output neurons.  When set,
            :meth:`decide_action` raises :exc:`ValueError` if the network
            returns a different number of activations.  Pass ``None`` to
            skip validation (default).
    """

    def __init__(
        self,
        network: NeatNetwork,
        expected_outputs: int | None = None,
    ) -> None:
        self._network = network
        self._expected_outputs = expected_outputs

    # ------------------------------------------------------------------
    # DriverInterface
    # ------------------------------------------------------------------

    def decide_action(
        self,
        sensor_inputs: list[float],
        state: VehicleState,
    ) -> Action:
        """Activate the NEAT network and return the argmax action.

        This is the **neural network forward pass** — the core of the AI
        decision loop.  Each tick, the simulation calls this method with
        the car's 5 normalised radar distances.  The network produces 4
        output activations (one per action) and the action with the highest
        activation is selected.

        Output index → Action mapping (defined in ``driver_interface.py``):
        - index 0 → ``Action.TURN_LEFT``
        - index 1 → ``Action.TURN_RIGHT``
        - index 2 → ``Action.SLOW_DOWN``
        - index 3 → ``Action.SPEED_UP``

        Args:
            sensor_inputs: Normalised radar distances in ``[0.0, 1.0]``.
                5 values, one per radar angle (−90°, −45°, 0°, +45°, +90°).
            state: Current vehicle state (not used by a feed-forward
                network but available for future recurrent drivers).

        Returns:
            The :class:`Action` whose index matches the highest output
            activation.

        Raises:
            ValueError: If *expected_outputs* was set and the network
                returns a different number of activations.
            ValueError: If the network returns an empty output vector.
        """
        # Forward pass: sensor distances → network → 4 activation scores
        outputs = self._network.activate(sensor_inputs)
        self._validate_outputs(outputs)
        # Argmax: the action with the highest score wins this tick
        return Action(outputs.index(max(outputs)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_outputs(self, outputs: list[float]) -> None:
        """Raise :exc:`ValueError` for malformed output vectors.

        Args:
            outputs: Raw activation values from the network.
        """
        if not outputs:
            raise ValueError("NEAT network returned an empty output vector.")
        if self._expected_outputs is not None and len(outputs) != self._expected_outputs:
            raise ValueError(
                f"Expected {self._expected_outputs} network outputs, "
                f"got {len(outputs)}."
            )

    def __repr__(self) -> str:  # pragma: no cover
        return f"NeatDriver(network={self._network!r})"
