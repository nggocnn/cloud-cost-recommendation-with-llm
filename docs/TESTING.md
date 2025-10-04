# Testing Framework Documentation

## Overview

The LLM Cost Recommendation System includes a comprehensive testing framework with 68+ tests covering all aspects of the system from API security to performance optimization. The test suite ensures reliability, security, and performance across all components.

## Test Structure

### Test Categories

```text
tests/
├── api/                   # API Security Tests (16 tests)
│   └── test_api_security.py
├── config/                # Configuration Tests (15 tests)  
│   └── test_configuration_issues.py
├── performance/           # Performance Tests (9 tests)
│   └── test_performance_issues.py
├── edge_cases/            # Edge Case Tests (15 tests)
│   └── test_error_handling.py
├── validation/            # Data Validation Tests (14 tests)
│   └── test_data_validation.py
├── security/              # Security Tests (11 tests)
│   └── test_security_vulnerabilities.py
├── e2e/                   # End-to-End Tests
├── integration/           # Integration Tests
├── functional/            # Functional Tests
├── unit/                  # Unit Tests
└── conftest.py           # Test Configuration
```

## Running Tests

### Complete Test Suite

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run tests with coverage report
python -m pytest tests/ --cov=llm_cost_recommendation --cov-report=html

# Run tests with API server integration
python -m pytest tests/ -v --tb=short
```

### Category-Specific Testing

```bash
# API Security Tests
python -m pytest tests/api/ -v

# Performance Tests
python -m pytest tests/performance/ -v

# Configuration Tests
python -m pytest tests/config/ -v

# Edge Case Tests
python -m pytest tests/edge_cases/ -v

# Data Validation Tests
python -m pytest tests/validation/ -v

# Security Tests
python -m pytest tests/security/ -v
```

## Test Categories Explained

### API Security Tests (16 tests)

**Purpose**: Validates API security mechanisms and protections

**Key Test Areas**:
- Rate limiting enforcement
- Input validation and sanitization
- Request size limits
- Authentication mechanisms
- CORS policy validation
- SQL injection protection
- XSS prevention
- Error message security

**Example Tests**:
```python
def test_rate_limiting():
    """Test API rate limiting works correctly"""
    
def test_input_validation():
    """Test input validation prevents malicious payloads"""
    
def test_request_size_limits():
    """Test request size limits are enforced"""
```

### Performance Tests (9 tests)

**Purpose**: Identifies performance bottlenecks and validates system efficiency

**Key Test Areas**:
- Memory leak detection
- Batch processing efficiency
- Concurrent request handling
- Large dataset processing
- JSON parsing performance
- Async bottleneck detection
- Configuration loading performance
- Resource cleanup validation
- CPU intensive operations

**Example Tests**:
```python
def test_memory_leak_detection():
    """Test system doesn't have memory leaks during analysis"""
    
def test_concurrent_request_handling():
    """Test system handles concurrent requests efficiently"""
    
def test_batch_processing_efficiency():
    """Test batch processing performs within acceptable limits"""
```

### Configuration Tests (15 tests)

**Purpose**: Validates configuration loading, agent discovery, and YAML processing

**Key Test Areas**:
- YAML configuration validation
- Agent configuration loading
- Service discovery mechanisms
- Configuration error handling
- Large configuration file processing
- Invalid configuration detection
- Default value fallbacks
- Configuration caching

**Example Tests**:
```python
def test_agent_configuration_loading():
    """Test agent configurations load correctly"""
    
def test_invalid_yaml_handling():
    """Test system handles invalid YAML gracefully"""
    
def test_large_configuration_files():
    """Test performance with large configuration files"""
```

### Edge Case Tests (15 tests)

**Purpose**: Validates system behavior under unusual or error conditions

**Key Test Areas**:
- Network timeout handling
- Invalid LLM responses
- Missing configuration files
- Resource validation edge cases
- Extremely large values
- Concurrent access conflicts
- Disk space simulation
- Memory pressure simulation
- Permission denied scenarios
- Circular reference handling
- Graceful shutdown simulation

**Example Tests**:
```python
def test_network_timeout_simulation():
    """Test system handles network timeouts gracefully"""
    
def test_invalid_llm_responses():
    """Test handling of malformed LLM responses"""
    
def test_concurrent_access_conflicts():
    """Test system handles concurrent access safely"""
```

### Data Validation Tests (14 tests)

**Purpose**: Ensures data integrity and format validation across all inputs

**Key Test Areas**:
- Schema validation for all data types
- CSV/JSON format validation
- Required field validation
- Data type enforcement
- Range validation
- Format consistency checks
- Error message clarity
- Validation performance

**Example Tests**:
```python
def test_billing_data_validation():
    """Test billing data validation rules"""
    
def test_inventory_schema_validation():
    """Test inventory data schema enforcement"""
    
def test_metrics_data_validation():
    """Test metrics data validation and ranges"""
```

### Security Tests (11 tests)

**Purpose**: Validates security controls and vulnerability protections

**Key Test Areas**:
- Input sanitization
- Path traversal prevention
- Command injection protection
- File upload security
- Error information leakage
- Secure defaults validation
- Authentication bypass attempts
- Authorization checks

## Test Infrastructure

### Test Fixtures and Configuration

The `conftest.py` file provides shared test fixtures:

```python
@pytest.fixture
def sample_resources():
    """Provides sample resource data for testing"""
    
@pytest.fixture
def mock_llm_service():
    """Provides mocked LLM service for testing"""
    
@pytest.fixture
def test_configuration():
    """Provides test-specific configuration"""
```

### Mocking Strategy

The test suite uses strategic mocking to:
- Avoid API costs during testing
- Ensure predictable test results
- Test error conditions safely
- Isolate components for unit testing

**Key Mocking Areas**:
- OpenAI API calls
- File system operations
- Network requests
- External service dependencies

### Test Data Management

**Sample Data Generation**:
```python
# Generate test data for different scenarios
def generate_test_billing_data(scenario="normal"):
    """Generate billing data for specific test scenarios"""
    
def generate_test_resources(count=10, service="AWS.EC2"):
    """Generate resource data for testing"""
```

## Continuous Integration

### GitHub Actions Integration

The test suite is designed for CI/CD integration:

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest tests/ -v --cov=llm_cost_recommendation
```

### Quality Gates

**Test Coverage Requirements**:
- Minimum 80% code coverage
- All critical paths must be tested
- Security tests must pass 100%
- Performance tests within acceptable limits

**Performance Thresholds**:
- Memory usage < 500MB for standard datasets
- API response time < 2 seconds for typical requests
- Concurrent request handling > 10 requests/second

## Test Development Guidelines

### Writing New Tests

**Test Naming Convention**:
```python
def test_[component]_[scenario]_[expected_outcome]():
    """Clear description of what the test validates"""
```

**Test Structure**:
```python
def test_example():
    # Arrange: Set up test data and conditions
    
    # Act: Execute the functionality being tested
    
    # Assert: Verify the expected outcomes
```

### Mock Usage Guidelines

**When to Mock**:
- External API calls
- File system operations
- Network dependencies
- Expensive computations

**When NOT to Mock**:
- Core business logic
- Data transformations
- Internal service calls
- Configuration loading

### Test Data Guidelines

**Data Principles**:
- Use realistic but anonymized data
- Cover edge cases and boundary conditions
- Include both valid and invalid scenarios
- Maintain data consistency across tests

## Performance Testing Details

### Memory Leak Detection

The memory leak tests monitor memory usage over multiple analysis cycles:

```python
def test_memory_leak_detection():
    """Monitor memory usage during repeated analyses"""
    initial_memory = get_memory_usage()
    
    for i in range(10):
        # Run analysis cycle
        coordinator.analyze_resources(sample_resources)
        
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < ACCEPTABLE_MEMORY_INCREASE
```

### Concurrent Processing Tests

Validates system behavior under concurrent load:

```python
def test_concurrent_request_handling():
    """Test concurrent request processing"""
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(10):
            future = executor.submit(coordinator.analyze_resources, resources)
            futures.append(future)
        
        results = [future.result() for future in futures]
        assert all(result.success for result in results)
```

## Security Testing Details

### Input Validation Tests

Comprehensive input validation across all endpoints:

```python
def test_malicious_input_handling():
    """Test system handles malicious inputs safely"""
    malicious_inputs = [
        "../../../etc/passwd",
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "{{7*7}}",  # Template injection
    ]
    
    for malicious_input in malicious_inputs:
        response = client.post("/analyze", data={"input": malicious_input})
        assert response.status_code in [400, 422]  # Rejected
```

### Authentication and Authorization

Tests for proper access controls:

```python
def test_unauthorized_access():
    """Test system properly rejects unauthorized requests"""
    
def test_privilege_escalation():
    """Test system prevents privilege escalation"""
```

## Troubleshooting Test Issues

### Common Test Failures

**Memory-Related Failures**:
- Check for resource cleanup in tearDown methods
- Verify mocks are properly reset between tests
- Monitor test execution order dependencies

**Timing-Related Failures**:
- Use appropriate timeouts for async operations
- Add retry logic for network-dependent tests
- Consider test execution environment differences

**Configuration Failures**:
- Verify test configuration files exist
- Check environment variable setup
- Validate test data paths

### Debugging Test Failures

```bash
# Run specific failing test with maximum verbosity
python -m pytest tests/path/to/test.py::test_name -v -s

# Run with debugger
python -m pytest tests/path/to/test.py::test_name --pdb

# Run with detailed coverage
python -m pytest tests/ --cov=llm_cost_recommendation --cov-report=term-missing
```

## Test Metrics and Reporting

The test framework generates comprehensive metrics:

- **Test execution time** per category
- **Code coverage** by module and function
- **Performance benchmarks** for critical operations
- **Security scan results** and vulnerability assessments
- **Flaky test detection** and reliability metrics

These metrics help maintain high code quality and system reliability across all components.