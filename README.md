# Samurai Real-Time

## TransReID setup note

This repository uses the original TransReID repository as a Git submodule under:

```text
external/reid/TransReID
```

After cloning and initializing the submodules, one manual compatibility change is required because recent PyTorch versions no longer provide `torch._six`.

Open:

```text
external/reid/TransReID/model/backbones/vit_pytorch.py
```

Replace:

```python
from torch._six import container_abcs
```

with:

```python
import collections.abc as container_abcs
```

This change is intentionally not committed inside the TransReID submodule, because the submodule points to the original upstream repository.