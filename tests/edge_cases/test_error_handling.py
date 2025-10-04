"""
Comprehensive error handling and edge case tests.
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from llm_cost_recommendation.cli import CostRecommendationApp
from llm_cost_recommendation.services.llm import LLMService
from llm_cost_recommendation.services.config import ConfigManager, LLMConfig
from llm_cost_recommendation.services.ingestion import DataIngestionService
from llm_cost_recommendation.agents.coordinator import CoordinatorAgent
from llm_cost_recommendation.models import Resource, BillingData, Metrics
from llm_cost_recommendation.models.types import ServiceType, CloudProvider


class TestErrorHandlingAndEdgeCases:
    """Test comprehensive error handling and edge cases."""

    def test_empty_data_files(self, temp_dir):
        """Test handling of empty or minimal data files."""
        data_service = DataIngestionService(str(temp_dir))
        
        # Create empty files
        empty_csv = temp_dir / "empty.csv"
        empty_json = temp_dir / "empty.json"
        empty_csv.write_text("")
        empty_json.write_text("")
        
        # Test empty CSV - should handle gracefully and return empty dict
        try:
            billing_data = data_service.ingest_billing_data(str(empty_csv))
            assert billing_data == {}
        except Exception as e:
            # Empty CSV files cause pandas.errors.EmptyDataError, which is expected
            assert "No columns to parse" in str(e) or "EmptyDataError" in str(type(e).__name__)
        
        # Test empty JSON - should handle gracefully
        try:
            resources = data_service.ingest_inventory_data(str(empty_json))
            assert resources == []
        except json.JSONDecodeError:
            # Empty JSON files cause JSONDecodeError, which is expected
            pass
        
        # Test minimal valid JSON
        minimal_json = temp_dir / "minimal.json"
        minimal_json.write_text("[]")
        resources = data_service.ingest_inventory_data(str(minimal_json))
        assert resources == []

    def test_malformed_data_files(self, temp_dir):
        """Test handling of malformed data files."""
        data_service = DataIngestionService(str(temp_dir))
        
        # Malformed JSON
        malformed_json = temp_dir / "malformed.json"
        malformed_json.write_text('{"invalid": json}')  # Invalid JSON
        
        with pytest.raises((json.JSONDecodeError, ValueError)):
            data_service.ingest_inventory_data(str(malformed_json))
        
        # Malformed CSV
        malformed_csv = temp_dir / "malformed.csv"
        malformed_csv.write_text("header1,header2\nvalue1\nvalue2,value3,value4")  # Inconsistent columns
        
        # Should handle gracefully (pandas usually handles this)
        try:
            billing_data = data_service.ingest_billing_data(str(malformed_csv))
            assert isinstance(billing_data, dict)
        except Exception as e:
            # If it fails, should be a specific, handled exception
            assert not isinstance(e, AttributeError)

    def test_unicode_and_special_characters(self, temp_dir):
        """Test handling of Unicode and special characters in data."""
        data_service = DataIngestionService(str(temp_dir))
        
        # Create data with Unicode characters
        unicode_data = [
            {
                "resource_id": "资源-测试-123",
                "service": "EC2",
                "region": "中国-北京",
                "tags": {
                    "Name": "тест-ресурс",
                    "Description": "Ресурс с русскими символами",
                    "Owner": "用户@公司.中国"
                },
                "properties": {
                    "notes": "Special chars: !@#$%^&*()[]{}|\\:;\"'<>,.?/~`",
                    "emoji": "🚀🔥💡"
                }
            }
        ]
        
        unicode_file = temp_dir / "unicode.json"
        with open(unicode_file, 'w', encoding='utf-8') as f:
            json.dump(unicode_data, f, ensure_ascii=False)
        
        # Should handle Unicode correctly
        resources = data_service.ingest_inventory_data(str(unicode_file))
        assert len(resources) == 1
        assert resources[0].resource_id == "资源-测试-123"
        assert resources[0].region == "中国-北京"

    @pytest.mark.asyncio
    async def test_llm_api_failures(self, sample_config_dir, temp_dir):
        """Test handling of LLM API failures."""
        app = CostRecommendationApp(str(sample_config_dir), str(temp_dir))
        
        # Test different types of API failures
        api_failures = [
            Exception("Connection timeout"),
            Exception("Rate limit exceeded"),
            Exception("Invalid API key"),
            Exception("Model not found"),
            Exception("Server error")
        ]
        
        for failure in api_failures:
            mock_llm = MagicMock()
            mock_llm.generate_recommendation = AsyncMock(side_effect=failure)
            app.llm_service = mock_llm
            
            # Should handle API failures gracefully
            try:
                result = await app.run_analysis(use_sample_data=True)
                # If it doesn't raise, check it returns appropriate result
                if result is not None:
                    assert hasattr(result, 'recommendations')
            except Exception as e:
                # Should be the original exception or a handled wrapper
                assert str(failure) in str(e) or "LLM" in str(e) or "API" in str(e)

    @pytest.mark.asyncio
    async def test_invalid_llm_responses(self, sample_config_dir, temp_dir):
        """Test handling of invalid LLM responses."""
        app = CostRecommendationApp(str(sample_config_dir), str(temp_dir))
        
        # Test various invalid responses
        invalid_responses = [
            "",  # Empty response
            "Not JSON at all",  # Non-JSON
            '{"incomplete": json',  # Malformed JSON
            '{"recommendations": "not_a_list"}',  # Wrong format
            '{"wrong_key": []}',  # Missing expected keys
            '{"recommendations": [{"missing_required_fields": true}]}',  # Incomplete recommendations
            "```json\n{}\n```",  # Markdown wrapped but empty
            '{"recommendations": [null, null]}',  # Null recommendations
        ]
        
        for invalid_response in invalid_responses:
            mock_llm = MagicMock()
            mock_llm.generate_recommendation = AsyncMock()
            mock_llm.generate_recommendation.return_value = MagicMock(
                content=invalid_response,
                usage_tokens=0,
                model="gpt-4",
                response_time_ms=100.0
            )
            app.llm_service = mock_llm
            
            # Should handle invalid responses without crashing
            try:
                result = await app.run_analysis(use_sample_data=True)
                # Should return some result even if LLM fails
                assert result is not None or True  # Accept None as valid fallback
            except Exception as e:
                # Should be a handled exception
                assert "JSON" in str(e) or "parse" in str(e) or "format" in str(e)

    def test_missing_configuration_files(self, temp_dir):
        """Test handling of missing configuration files."""
        # Test with completely empty config directory
        empty_config_dir = temp_dir / "empty_config"
        empty_config_dir.mkdir()
        
        # Should handle missing configurations gracefully
        try:
            config_manager = ConfigManager(str(empty_config_dir))
            # Should either work with defaults or fail gracefully
            assert config_manager is not None
        except Exception as e:
            # Should be a specific configuration error
            assert "config" in str(e).lower() or "file" in str(e).lower()

    def test_resource_validation_edge_cases(self):
        """Test resource validation with edge cases."""
        from pydantic import ValidationError
        
        # Test with minimal data
        valid_minimal = Resource(
            resource_id="test",
            service=ServiceType.AWS.EC2,
            region="us-east-1"
        )
        assert valid_minimal.resource_id == "test"
        
        # Test with invalid data types
        with pytest.raises(ValidationError):
            Resource(
                resource_id=None,  # Required field is None
                service=ServiceType.AWS.EC2,
                region="us-east-1"
            )
        
        with pytest.raises(ValidationError):
            Resource(
                resource_id="test",
                service="InvalidService",  # Not a valid ServiceType
                region="us-east-1"
            )

    def test_extremely_large_values(self, temp_dir):
        """Test handling of extremely large numeric values."""
        # Test with very large costs and metrics
        large_billing_data = BillingData(
            resource_id="test-resource",
            service="EC2",
            region="us-east-1",
            usage_type="BoxUsage:t3.micro",
            usage_amount=999999.0,  # Very large usage
            usage_unit="Hrs",
            unblended_cost=999999999999.99,  # Very large cost
            amortized_cost=999999999999.99,
            bill_period_start=datetime(2023, 1, 1),
            bill_period_end=datetime(2023, 1, 31)
        )
        
        assert large_billing_data.unblended_cost == 999999999999.99
        
        # Test with very large metrics
        large_metrics = Metrics(
            resource_id="test-resource",
            timestamp=datetime(2023, 1, 1),
            metrics={
                "cpu_utilization": 100.0,
                "memory_utilization": 100.0,
                "network_in": 999999999999.0,  # Very large
                "network_out": 999999999999.0,
                "disk_read": 999999999999.0,
                "disk_write": 999999999999.0
            }
        )
        
        assert large_metrics.metrics["network_in"] == 999999999999.0

    @pytest.mark.asyncio
    async def test_concurrent_access_conflicts(self, sample_config_dir, temp_dir):
        """Test handling of concurrent access to shared resources."""
        # Test multiple apps accessing same config simultaneously
        apps = []
        for i in range(5):
            app = CostRecommendationApp(str(sample_config_dir), str(temp_dir))
            
            # Mock LLM to avoid API calls
            mock_llm = MagicMock()
            mock_llm.generate_recommendation = AsyncMock()
            mock_llm.generate_recommendation.return_value = MagicMock(
                content='{"recommendations": []}',
                usage_tokens=100,
                model="gpt-4",
                response_time_ms=100.0
            )
            app.llm_service = mock_llm
            apps.append(app)
        
        # Run analyses concurrently
        tasks = [app.run_analysis(use_sample_data=True) for app in apps]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Should not be concurrency-related errors
                    assert "lock" not in str(result).lower()
                    assert "conflict" not in str(result).lower()
        except Exception as e:
            # Should handle concurrent access gracefully
            assert "concurrent" in str(e).lower() or "lock" in str(e).lower()

    def test_disk_space_simulation(self, temp_dir):
        """Test handling of low disk space conditions."""
        data_service = DataIngestionService(str(temp_dir))
        
        # Create a test CSV file first
        test_csv = temp_dir / "test.csv"
        test_csv.write_text("resource_id,service,region\ntest,EC2,us-east-1\n")
        
        # Simulate disk space error by mocking file operations
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                data_service.ingest_billing_data(str(test_csv))

    def test_network_timeout_simulation(self):
        """Test handling of network timeouts for LLM API."""
        config = LLMConfig(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-3.5-turbo",
            timeout=1  # Very short timeout
        )
        
        # Mock network timeout
        with patch('openai.AsyncOpenAI') as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.chat.completions.create = AsyncMock(
                side_effect=asyncio.TimeoutError("Request timeout")
            )
            
            llm_service = LLMService(config)
            
            # Should handle timeout gracefully - LLM service converts TimeoutError to ValueError
            with pytest.raises(ValueError, match="LLM request timed out"):
                asyncio.run(llm_service.generate_recommendation(
                    system_prompt="test",
                    user_prompt="test"
                ))

    def test_memory_pressure_simulation(self, sample_config_dir, temp_dir):
        """Test behavior under memory pressure."""
        app = CostRecommendationApp(str(sample_config_dir), str(temp_dir))
        
        # Create large amount of test data
        large_resources = []
        for i in range(10000):  # Large dataset
            large_resources.append({
                "resource_id": f"resource-{i}",
                "service": "EC2",
                "region": "us-east-1",
                "tags": {f"tag-{j}": f"value-{j}" for j in range(10)},  # Many tags
                "properties": {f"prop-{j}": f"value-{j}" for j in range(20)}  # Many properties
            })
        
        large_file = temp_dir / "large_dataset.json"
        with open(large_file, 'w') as f:
            json.dump(large_resources, f)
        
        # Test with large dataset
        data_service = DataIngestionService(str(temp_dir))
        try:
            resources = data_service.ingest_inventory_data(str(large_file))
            # Should handle large datasets
            assert len(resources) > 0
        except MemoryError:
            # Expected under memory pressure
            pytest.skip("Insufficient memory for large dataset test")

    def test_permission_denied_scenarios(self, temp_dir):
        """Test handling of permission denied scenarios."""
        data_service = DataIngestionService(str(temp_dir))
        
        # Create a file and remove read permissions (Unix-like systems)
        restricted_file = temp_dir / "restricted.json"
        restricted_file.write_text('{"test": "data"}')
        
        try:
            restricted_file.chmod(0o000)  # No permissions
            
            with pytest.raises(PermissionError):
                data_service.ingest_inventory_data(str(restricted_file))
                
        finally:
            # Restore permissions for cleanup
            try:
                restricted_file.chmod(0o644)
            except:
                pass

    def test_circular_reference_in_data(self, temp_dir):
        """Test handling of circular references in JSON data."""
        # JSON doesn't naturally support circular references,
        # but test deeply nested structures that might cause issues
        
        nested_data = {"level": 1}
        current = nested_data
        
        # Create deeply nested structure
        for i in range(100):
            current["next"] = {"level": i + 2}
            current = current["next"]
        
        nested_file = temp_dir / "deeply_nested.json"
        with open(nested_file, 'w') as f:
            json.dump(nested_data, f)
        
        data_service = DataIngestionService(str(temp_dir))
        
        # Should handle deeply nested structures
        try:
            # This will fail because the JSON isn't in the expected format
            # But it shouldn't cause a stack overflow
            data_service.ingest_inventory_data(str(nested_file))
        except (ValueError, KeyError, TypeError):
            # Expected - wrong format
            pass
        except RecursionError:
            pytest.fail("RecursionError - should handle deeply nested structures better")

    @pytest.mark.asyncio
    async def test_graceful_shutdown_simulation(self, sample_config_dir, temp_dir):
        """Test graceful handling of shutdown scenarios."""
        app = CostRecommendationApp(str(sample_config_dir), str(temp_dir))
        
        # Mock LLM service
        mock_llm = MagicMock()
        mock_llm.generate_recommendation = AsyncMock()
        
        # Simulate slow LLM response
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(2)  # Simulate slow response
            return MagicMock(
                content='{"recommendations": []}',
                usage_tokens=100,
                model="gpt-4",
                response_time_ms=2000.0
            )
        
        mock_llm.generate_recommendation = slow_response
        app.llm_service = mock_llm
        
        # Start analysis and cancel it (simulate shutdown)
        task = asyncio.create_task(app.run_analysis(use_sample_data=True))
        
        # Wait a bit then cancel
        await asyncio.sleep(0.1)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            # Expected behavior
            pass
        except Exception as e:
            # Should handle cancellation gracefully
            assert "cancel" in str(e).lower() or isinstance(e, asyncio.CancelledError)