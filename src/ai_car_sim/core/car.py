"""Car entity that owns position, angle, speed, and sensor state.

Composes VehicleState (mutable physics state), RadarSensorSystem (distance sensing),
and sprite metadata (image surface, scaled dimensions). Delegates collision detection
to collision_service and radar casting to radar_sensor.

Public interface:
  - Car.reset(): Restore the car to its spawn position and angle.
  - Car.apply_action(action): Update speed and angle based on a driver Action.
  - Car.update(game_map, config): Advance physics, cast radars, check collision.
  - Car.get_sensor_inputs(): Return normalised radar distances for the AI network.
  - Car.get_reward(): Return the reward signal for the current step.
  - Car.draw(screen): Render the rotated car sprite onto the given pygame Surface.
  - Car.is_alive(): Return whether the car has not yet collided with a border.
"""

pass
