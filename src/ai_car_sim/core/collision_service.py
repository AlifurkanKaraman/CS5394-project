"""Border-collision detection logic for the AI car simulation.

Provides functions that determine whether a car has collided with track borders:
  - compute_corners(state, car_size): Compute the four rotated corner positions of the car
    bounding box given its current VehicleState and dimensions.
  - is_collision(corners, game_map, border_color): Check whether any corner pixel on the
    provided pygame Surface matches the border colour, indicating a collision.

Accepts a pygame Surface for pixel lookup but owns no rendering state.
"""

pass
