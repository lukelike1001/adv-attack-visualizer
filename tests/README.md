# Tests

## Running tests

```bash
make test
```

To run a specific layer:

```bash
make test-attacks
make test-controller
make test-utils
make test-view
```

## Structure

```
tests/
    attacks/
        helpers.py                  # Shared: NumpyLinfConstraint, make_attack_step
        test_attack_step.py         # AttackStep immutability and equality
        test_fgsm.py                # FGSM trajectory shape and noise correctness
        test_pgd.py                 # PGD step count, indices, epsilon bound
    controller/
        test_attack_controller.py   # State transitions, navigation, boundary errors
        test_attack_factory.py      # Attack instantiation by name, error handling
    utils/
        test_config_loader.py       # YAML loading, missing key errors
    view/
        test_view_model.py          # ViewModel properties, metadata copy isolation
```

## Dependencies

```bash
pip install pytest numpy pyyaml
```

## Notes

- Constraint math (`LinfConstraint`, `L2Constraint`) is not yet tested — those classes defer to a tensor library not yet chosen. Tests will be added once an implementation lands.
- `helpers.py` provides a `NumpyLinfConstraint` used across attack tests. Update it there when real constraint implementations arrive.
