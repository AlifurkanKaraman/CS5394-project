# Property 1 — Validates: Requirements 4.5, 5.5, 6.5, 7.3, 8.4
import ast
from pathlib import Path
from hypothesis import given, settings
from hypothesis.strategies import sampled_from

STUB_MODULES = [
    # domain
    "src/ai_car_sim/domain/track.py",
    "src/ai_car_sim/domain/vehicle_state.py",
    "src/ai_car_sim/domain/sensor_reading.py",
    "src/ai_car_sim/domain/simulation_config.py",
    # core
    "src/ai_car_sim/core/vector_utils.py",
    "src/ai_car_sim/core/collision_service.py",
    "src/ai_car_sim/core/radar_sensor.py",
    "src/ai_car_sim/core/car.py",
    # ai
    "src/ai_car_sim/ai/driver_interface.py",
    "src/ai_car_sim/ai/neat_driver.py",
    "src/ai_car_sim/ai/training_manager.py",
    "src/ai_car_sim/ai/replay_loader.py",
    # ui
    "src/ai_car_sim/ui/hud_view.py",
    "src/ai_car_sim/ui/menu_controller.py",
    # simulation
    "src/ai_car_sim/simulation/engine.py",
    # analytics
    "src/ai_car_sim/analytics/run_metrics.py",
    # persistence
    "src/ai_car_sim/persistence/save_load.py",
]

PROJECT_ROOT = Path(__file__).parent.parent  # ai-car-simulation/

@given(sampled_from(STUB_MODULES))
def test_stub_modules_have_docstrings(module_path):
    # Property 1 — Validates: Requirements 4.5, 5.5, 6.5, 7.3, 8.4
    path = PROJECT_ROOT / module_path
    assert path.exists(), f"Module file not found: {path}"
    source = path.read_text()
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree)
    assert docstring is not None and len(docstring.strip()) > 0, \
        f"Module {module_path} is missing a module-level docstring"

@given(sampled_from(["app", "ai", "core", "domain", "ui", "simulation", "analytics", "persistence"]))
def test_subpackages_have_init_files(pkg_name):
    # Property 2 — Validates: Requirements 3.2, 3.4
    init_path = PROJECT_ROOT / "src" / "ai_car_sim" / pkg_name / "__init__.py"
    assert init_path.exists(), f"Missing __init__.py in sub-package: {pkg_name}"

import sys
import types
import pytest
import unittest.mock

@given(sampled_from(["neat", "pygame"]))
def test_main_raises_import_error_for_missing_dependency(pkg_name):
    # Property 3 — Validates: Requirements 11.4
    from ai_car_sim.main import main

    all_deps = ["neat", "pygame"]
    other_deps = [d for d in all_deps if d != pkg_name]

    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == pkg_name:
            raise ImportError(f"No module named '{pkg_name}'")
        # Allow other required deps to succeed as stub modules so main()
        # reaches the import we actually want to test.
        if name in other_deps:
            return types.ModuleType(name)
        return original_import(name, *args, **kwargs)

    # Remove all dep modules from sys.modules so imports inside main() are
    # attempted fresh (bypassing the module cache).
    saved = {d: sys.modules.pop(d, None) for d in all_deps}
    try:
        with unittest.mock.patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError) as exc_info:
                main()
        assert pkg_name in str(exc_info.value), (
            f"ImportError message should contain '{pkg_name}', got: {exc_info.value}"
        )
    finally:
        for d, mod in saved.items():
            if mod is not None:
                sys.modules[d] = mod
