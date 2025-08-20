# Centralized Path Management

This project now uses a centralized path management system to ensure consistency across Python, Makefile, and Dockerfile.

## How It Works

### 1. Central Configuration (`.env`)
All paths are defined in the `.env` file at project root:
```bash
EXTERNAL_DATA_DIR=external_data
SYZYGY_TABLES_DIR=external_data/syzygy-tables
STOCKFISH_DIR=external_data/stockfish
GUI_DIR=external_data/gui
# ... etc
```

### 2. Python Integration
Python automatically loads `.env` variables in `chipiron/utils/path_variables.py`:
```python
from chipiron.utils.path_variables import GUI_DIR, STOCKFISH_DIR
# These now use .env values with fallbacks
```

### 3. Makefile Integration
Makefile includes the `.env` file:
```makefile
include .env
export
STOCKFISH_DESTINATION?=${ROOT_DIR}/$(STOCKFISH_DIR)
```

### 4. Dockerfile Integration
Dockerfile sets ENV variables that match `.env`:
```dockerfile
ENV EXTERNAL_DATA_DIR=external_data
ENV STOCKFISH_DIR=external_data/stockfish
```

## Benefits

✅ **Single Source of Truth**: All paths defined once in `.env`
✅ **Consistency**: Impossible for Python/Make/Docker to use different paths
✅ **Environment Flexibility**: Different `.env` files for dev/prod/docker
✅ **Easy Maintenance**: Change a path once, works everywhere
✅ **Backward Compatible**: Existing code continues to work

## Usage

### Viewing Current Paths
```bash
# Python paths with environment integration
python chipiron/utils/path_variables.py

# Validate consistency across all systems
python validate_paths.py

# Check Makefile variables
make -p | grep "STOCKFISH_DESTINATION\|EXTERNAL_DATA_DIR"
```

### Customizing Paths
1. **For development**: Modify `.env` directly
2. **For specific environments**: Create `.env.local` (gitignored)
3. **For Docker**: Override ENV variables in Dockerfile
4. **For CI/CD**: Set environment variables in pipeline

### Testing the System
```bash
# Test Python path loading
python -c "from chipiron.utils.path_variables import GUI_DIR; print(GUI_DIR)"

# View all paths with debugging info
python chipiron/utils/path_variables.py

# Test Makefile variable expansion
make -n stockfish

# Run automated path consistency tests
pytest tests/test_path_consistency.py -v
```

## Migration Notes

- ✅ **No breaking changes**: Existing imports continue to work
- ✅ **Gradual adoption**: Can migrate components one by one
- ✅ **Default values**: System works even without `.env` file
- ✅ **Path objects**: Python still returns `pathlib.Path` objects

## Files Modified

- **`.env`**: New central configuration file
- **`Makefile`**: Now includes `.env` and uses variables
- **`Dockerfile`**: Uses ENV variables matching `.env`
- **`path_variables.py`**: Enhanced to load from environment
- **`validate_paths.py`**: New validation script

## Environment Variables

All path-related environment variables:
- `EXTERNAL_DATA_DIR`
- `LICHESS_PGN_DIR`
- `SYZYGY_TABLES_DIR`
- `STOCKFISH_DIR`
- `GUI_DIR`
- `LICHESS_PGN_FILE`
- `STOCKFISH_BINARY_PATH`
- `ML_FLOW_URI_PATH`
- `ML_FLOW_URI_PATH_TEST`
