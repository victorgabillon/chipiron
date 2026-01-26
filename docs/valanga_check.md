## Valanga presence check

Command:

```bash
python - <<'PY'
import importlib.util
print(importlib.util.find_spec('valanga'))
PY
```

Output:

```
None
```
