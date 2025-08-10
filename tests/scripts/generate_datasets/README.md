# Board Generation Tests

This directory contains comprehensive tests for the board generation functionality in `chipiron.scripts.generate_datasets.generate_boards`.

## Test Coverage

### TestProcessGame
Tests the core `process_game` function that extracts chess positions from PGN games:
- **Basic functionality**: Verifies correct move processing and position extraction
- **Sampling frequency**: Tests that positions are sampled at correct intervals
- **Offset boundaries**: Tests edge cases with offset_min parameter
- **Error handling**: Tests robustness with invalid inputs

### TestSaveDatasetProgress
Tests the dataset persistence functionality:
- **Basic saving**: Verifies correct pickle file creation and metadata
- **Final saves**: Tests additional metadata for completed datasets
- **Progress tracking**: Validates progress reporting and statistics

### TestIterateMonths
Tests the month iteration utility for dynamic downloads:
- **Basic iteration**: Verifies correct month sequence generation
- **Year boundaries**: Tests December→January transitions
- **Date arithmetic**: Validates month/year calculations

### TestDatasetGeneration
Tests the complete dataset generation workflow:
- **Multi-month generation**: Tests end-to-end dataset creation
- **Quality checks**: Validates that generated positions are legal chess positions
- **Mocking**: Uses mocks to avoid actual file downloads during testing

### TestErrorHandling
Tests error handling and edge cases:
- **Empty games**: Verifies graceful handling of empty PGN files
- **Invalid parameters**: Tests robustness with edge case parameters
- **Malformed input**: Tests behavior with corrupted data

### TestIntegration
End-to-end integration tests:
- **Complete workflow**: Tests full pipeline from PGN to dataset
- **Metadata validation**: Verifies all required metadata is present
- **Data quality**: Validates output dataset meets quality standards

## Running Tests

Run all board generation tests:
```bash
python -m pytest tests/scripts/generate_datasets/test_generate_boards.py -v
```

Run with coverage:
```bash
python -m pytest tests/scripts/generate_datasets/test_generate_boards.py --cov=chipiron.scripts.generate_datasets.generate_boards
```

Run specific test class:
```bash
python -m pytest tests/scripts/generate_datasets/test_generate_boards.py::TestProcessGame -v
```

## Test Features

- **Isolated testing**: Uses temporary directories for file operations
- **Mocked dependencies**: Avoids network calls and large file operations
- **Deterministic**: Uses fixed random seeds for reproducible results
- **Quality validation**: Verifies chess positions are legal and properly formatted
- **Error scenarios**: Tests edge cases and error conditions
- **Performance**: Fast execution suitable for CI/CD pipelines

## Expected Behavior

These tests verify that the board generation system:
1. Correctly parses PGN files and extracts chess positions
2. Applies sampling strategies (frequency, offset) as specified
3. Generates valid chess positions in FEN format
4. Saves datasets with proper metadata and progress tracking
5. Handles edge cases and errors gracefully
6. Maintains reproducible results with fixed random seeds
