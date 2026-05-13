# 2026-05-12 Backend Attack and Controller Unit Test Report

## Overview

This report documents the unit test plan for the backend (Model and Controller)
layers of the adversarial attack visualizer, covering issues ATTACK-01 through
CONTROL-04. Tests follow the project commit convention: `Test ISSUE-XX: ...`.

All tests use **pytest**. NumPy is used only in attack tests that need concrete
array math to exercise the gradient-to-trajectory pipeline. The GUI layer is
excluded — end-to-end tests will be addressed after GUI issues are closed.

---

## Directory Layout

```
tests/
    attacks/
        helpers.py                  # Shared: NumpyLinfConstraint, make_step()
        test_attack_step.py         # Test ATTACK-01
        test_fgsm.py                # Test ATTACK-04
        test_pgd.py                 # Test ATTACK-05
    controller/
        test_attack_controller.py   # Test CONTROL-01 + CONTROL-02
        test_attack_factory.py      # Test CONTROL-03
    utils/
        test_config_loader.py       # Test ATTACK-06
    view/
        test_view_model.py          # Test CONTROL-04
```

`conftest.py` at the repo root adds `src/` to `sys.path` so all test files can
import `attacks`, `controller`, `utils`, and `view` without package installation.

### Why no test file for ATTACK-02 or ATTACK-03?

- **ATTACK-02** (`AdversarialAttack`): the abstract base is fully exercised by the
  FGSM and PGD test files. The one novel assertion (instantiation raises `TypeError`)
  is included in `test_fgsm.py`.
- **ATTACK-03** (constraints): `LinfConstraint._sign/_clip` and
  `L2Constraint._normalize/_project_to_l2_ball` raise `NotImplementedError` — the
  math stubs exist to defer the tensor-library decision. Testing them now would only
  verify that `NotImplementedError` is raised, which is low value. A dedicated
  `test_constraints.py` should be added once a NumPy or PyTorch implementation lands.

---

## File-by-File Test Coverage

### `tests/attacks/helpers.py`
Shared utilities used by `test_fgsm.py` and `test_pgd.py`:
- `NumpyLinfConstraint`: concrete `LinfConstraint` subclass using `np.sign` / `np.clip`
- `make_attack_step(index)`: factory function for lightweight `AttackStep` fixtures

### `tests/attacks/test_attack_step.py` — Test ATTACK-01
| Test | What it checks |
|------|---------------|
| `test_fields_are_set_correctly` | All five fields read back correctly after construction |
| `test_is_immutable` | Assigning any field raises `AttributeError` (frozen dataclass) |
| `test_equal_steps_compare_equal` | Dataclass `__eq__` works correctly |
| `test_unequal_steps_differ` | Different `step_index` produces inequality |

### `tests/attacks/test_fgsm.py` — Test ATTACK-04
Uses `FixedGradientFGSM`, a subclass that returns `np.ones_like(image) * 0.5`
from `_compute_gradient`, to drive the full generate pipeline without a real model.

| Test | What it checks |
|------|---------------|
| `test_generate_returns_one_step` | Trajectory length is exactly 1 |
| `test_step_index_is_zero` | Single step carries `step_index == 0` |
| `test_noise_equals_perturbed_minus_original` | `noise = perturbed_image − input_image` |
| `test_metadata_keys_present` | `epsilon` and `step_size` in metadata |
| `test_returns_attack_step_instances` | All elements are `AttackStep` |
| `test_base_class_cannot_be_instantiated` | `AdversarialAttack` raises `TypeError` |
| `test_compute_gradient_raises_not_implemented` | Plain `FGSMAttack` (no subclass) raises `NotImplementedError` |

### `tests/attacks/test_pgd.py` — Test ATTACK-05
Uses `FixedGradientPGD` with the same fixed-gradient pattern.

| Test | What it checks |
|------|---------------|
| `test_generate_returns_correct_number_of_steps` | `len(steps) == num_steps` |
| `test_step_indices_are_sequential` | Indices are 0, 1, 2, … |
| `test_all_steps_are_attack_step_instances` | Type check on all elements |
| `test_perturbation_bounded_by_epsilon` | `‖noise‖∞ ≤ epsilon` at every step |
| `test_missing_num_steps_raises_key_error` | `config["num_steps"]` fails fast |
| `test_compute_gradient_raises_not_implemented` | Plain `PGDAttack` raises `NotImplementedError` |

### `tests/controller/test_attack_controller.py` — Test CONTROL-01 + CONTROL-02
Uses `StubAttack`: a plain object (no inheritance) with a `generate()` method
returning a fixed list of `AttackStep` objects. No NumPy required.

| Test | What it checks |
|------|---------------|
| `test_get_current_step_before_run_raises` | `AttackNotRunError` before `run_attack` |
| `test_run_attack_stores_steps` | `get_total_steps()` reflects attack output |
| `test_run_attack_resets_index_to_zero` | Second `run_attack` resets index |
| `test_get_current_step_returns_first_step` | Step 0 returned immediately after run |
| `test_next_step_advances_index` | Index increments by 1 |
| `test_previous_step_retreats_index` | Index decrements by 1 |
| `test_next_step_at_last_raises` | `StepNavigationError` at end of trajectory |
| `test_previous_step_at_first_raises` | `StepNavigationError` at start |
| `test_has_next_true_mid_trajectory` | `True` when not at last step |
| `test_has_next_false_at_last` | `False` at last step |
| `test_has_previous_false_at_first` | `False` at first step |
| `test_has_previous_true_after_advance` | `True` after one `next_step` |
| `test_current_step_reflects_navigation` | Step content matches navigation position |
| `test_next_step_before_run_raises` | `AttackNotRunError` when no attack run |
| `test_previous_step_before_run_raises` | `AttackNotRunError` when no attack run |

### `tests/controller/test_attack_factory.py` — Test CONTROL-03

| Test | What it checks |
|------|---------------|
| `test_creates_fgsm` | Returns `FGSMAttack` instance for `"fgsm"` |
| `test_creates_pgd` | Returns `PGDAttack` instance for `"pgd"` |
| `test_result_is_adversarial_attack` | Return type satisfies the abstraction |
| `test_unknown_attack_raises` | `UnknownAttackError` for unregistered name |
| `test_error_message_contains_attack_name` | Error message mentions the bad name |

### `tests/utils/test_config_loader.py` — Test ATTACK-06

| Test | What it checks |
|------|---------------|
| `test_loads_yaml_file` | Returns a dict from `attack_config.yaml` |
| `test_contains_expected_attacks` | `"fgsm"` and `"pgd"` keys present |
| `test_file_not_found_raises` | `FileNotFoundError` for bad path |
| `test_returns_fgsm_config` | `epsilon` and `step_size` present |
| `test_returns_pgd_config` | `epsilon`, `step_size`, `num_steps` present |
| `test_unknown_attack_raises` | `AttackConfigNotFoundError` for unknown name |
| `test_error_message_lists_available_attacks` | Error message references valid attack names |
| `test_fgsm_epsilon_is_positive` | Sanity check on config values |
| `test_pgd_num_steps_is_positive_integer` | Type and value check on `num_steps` |

### `tests/view/test_view_model.py` — Test CONTROL-04

| Test | What it checks |
|------|---------------|
| `test_step_index` | `step_index` property returns constructor argument |
| `test_total_steps` | `total_steps` property returns constructor argument |
| `test_image` | `image` returns `step.perturbed_image` |
| `test_gradient` | `gradient` returns `step.gradient` |
| `test_noise` | `noise` returns `step.noise` |
| `test_metadata_values` | `metadata` returns correct dict |
| `test_metadata_is_a_copy` | Mutating returned dict does not affect the ViewModel |

---

## Deferred

- `test_constraints.py`: add when `LinfConstraint` and `L2Constraint` receive
  real NumPy/PyTorch implementations (post-library selection).
- End-to-end / GUI tests: after GUI-01 through GUI-07 are closed.
