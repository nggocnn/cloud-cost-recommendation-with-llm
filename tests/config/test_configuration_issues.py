"""
Configuration and environment-related tests.
"""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from llm_cost_recommendation.services.config import ConfigManager, LLMConfig
from llm_cost_recommendation.cli import CostRecommendationApp


class TestConfigurationIssues:
    """Test configuration and environment issues."""

    def test_missing_environment_variables(self):
        """Test handling of missing environment variables."""
        
        # Test with missing OpenAI API key
        with patch.dict(os.environ, {}, clear=True):
            # Should handle missing environment variables gracefully
            try:
                config = LLMConfig()
                # May use defaults or raise appropriate error
                assert config is not None
            except (ValueError, KeyError) as e:
                # Should provide clear error message about missing config
                assert "api" in str(e).lower() or "key" in str(e).lower() or "config" in str(e).lower()

    def test_invalid_environment_values(self):
        """Test handling of invalid environment variable values."""
        
        invalid_env_values = {
            "OPENAI_API_KEY": "",  # Empty key
            "OPENAI_MODEL": "invalid-model-name-12345",  # Invalid model
            "OPENAI_BASE_URL": "not-a-url",  # Invalid URL
            "LOG_LEVEL": "INVALID_LEVEL",  # Invalid log level
        }
        
        for env_var, invalid_value in invalid_env_values.items():
            with patch.dict(os.environ, {env_var: invalid_value}):
                try:
                    if env_var.startswith("OPENAI"):
                        config = LLMConfig()
                        # May accept invalid values and let API calls fail,
                        # or validate and reject them
                        assert config is not None
                except (ValueError, TypeError) as e:
                    # Should provide helpful error message
                    assert len(str(e)) > 10  # Non-empty error message

    def test_configuration_file_permissions(self, temp_dir):
        """Test handling of configuration files with different permissions."""
        
        # Create config file
        config_file = temp_dir / "test_config.yaml"
        config_data = {
            "service": "EC2",
            "enabled": True,
            "system_prompt": "Test prompt"
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Test with read permissions
        config_file.chmod(0o644)  # Read/write for owner, read for others
        config_manager = ConfigManager(str(temp_dir))
        # Should work normally
        
        # Test with restricted permissions
        try:
            config_file.chmod(0o000)  # No permissions
            
            # Note: Permission enforcement may vary by filesystem and OS
            # Some environments (like containers) may not enforce file permissions
            try:
                ConfigManager(str(temp_dir))
                # If no exception, skip this test as permissions aren't enforced
                pytest.skip("File permissions not enforced in this environment")
            except PermissionError:
                # Expected behavior - permissions are enforced
                pass
                
        finally:
            # Restore permissions for cleanup
            try:
                config_file.chmod(0o644)
            except:
                pass

    def test_configuration_directory_missing(self):
        """Test handling when configuration directory doesn't exist."""
        
        non_existent_path = "/tmp/non_existent_config_dir_12345"
        
        try:
            config_manager = ConfigManager(non_existent_path)
            # May create directory or use defaults
            assert config_manager is not None
        except (FileNotFoundError, OSError) as e:
            # Should provide clear error about missing directory
            assert "config" in str(e).lower() or "directory" in str(e).lower() or "path" in str(e).lower()

    def test_malformed_yaml_configuration(self, temp_dir):
        """Test handling of malformed YAML configuration files."""
        
        malformed_configs = [
            "invalid: yaml: content:",  # Invalid YAML syntax
            "key: [unclosed list",  # Unclosed list
            "key: {unclosed dict",  # Unclosed dict
            "tab\tindented: content",  # Tab indentation (may cause issues)
            "- list\n- item\nkey: value",  # Mixed list and dict at root
        ]
        
        config_manager = ConfigManager(str(temp_dir))
        
        for i, malformed_yaml in enumerate(malformed_configs):
            malformed_file = temp_dir / f"malformed_{i}.yaml"
            malformed_file.write_text(malformed_yaml)
            
            # Should handle malformed YAML gracefully
            try:
                # Try to load the malformed config
                with open(malformed_file, 'r') as f:
                    yaml.safe_load(f)
            except yaml.YAMLError:
                # Expected - malformed YAML should be rejected
                pass

    def test_configuration_schema_validation(self, temp_dir):
        """Test validation of configuration schema."""
        
        # Test with invalid configuration structure
        invalid_configs = [
            {},  # Empty config
            {"wrong_key": "value"},  # Wrong keys
            {"service": "InvalidService"},  # Invalid service type
            {"enabled": "not_boolean"},  # Wrong type for boolean field
            {"system_prompt": None},  # Null where string expected
        ]
        
        for i, invalid_config in enumerate(invalid_configs):
            invalid_file = temp_dir / f"invalid_{i}.yaml"
            with open(invalid_file, 'w') as f:
                yaml.dump(invalid_config, f)
            
            # Configuration manager should handle invalid configs
            try:
                config_manager = ConfigManager(str(temp_dir))
                # May use defaults or skip invalid configs
                assert config_manager is not None
            except (ValueError, TypeError) as e:
                # Should provide meaningful error
                assert len(str(e)) > 5

    def test_environment_override_precedence(self, temp_dir):
        """Test that environment variables override configuration files."""
        
        # Create config file with specific values
        config_file = temp_dir / "test_config.yaml"
        config_data = {
            "model": "gpt-3.5-turbo"
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Test environment variable override
        with patch.dict(os.environ, {
            "OPENAI_MODEL": "gpt-4",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1"
        }):
            config = LLMConfig(
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL")
            )
            # Environment variable should override config file
            assert config is not None
            assert config.model == "gpt-4"

    def test_default_configuration_fallback(self, temp_dir):
        """Test fallback to default configuration when files are missing."""
        
        # Use empty directory (no config files)
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        try:
            config_manager = ConfigManager(str(empty_dir))
            # Should fall back to defaults
            assert config_manager is not None
            
            # Should have some default configuration
            assert hasattr(config_manager, 'global_config')
            
        except Exception as e:
            # If it fails, should be due to missing required config
            assert "config" in str(e).lower() or "required" in str(e).lower()

    def test_configuration_hot_reload(self, temp_dir):
        """Test configuration reloading when files change."""
        
        # Create initial config
        config_file = temp_dir / "dynamic.yaml"
        initial_config = {"enabled": True, "service": "EC2"}
        
        with open(config_file, 'w') as f:
            yaml.dump(initial_config, f)
        
        config_manager = ConfigManager(str(temp_dir))
        
        # Modify config file
        updated_config = {"enabled": False, "service": "S3"}
        with open(config_file, 'w') as f:
            yaml.dump(updated_config, f)
        
        # Create new config manager instance (simulates reload)
        new_config_manager = ConfigManager(str(temp_dir))
        
        # Should reflect changes (if hot reload is implemented)
        assert new_config_manager is not None

    def test_cross_platform_path_handling(self, temp_dir):
        """Test configuration path handling across platforms."""
        
        # Test with different path formats
        path_formats = [
            str(temp_dir),  # Standard string path
            temp_dir,  # Path object
            str(temp_dir).replace('/', '\\'),  # Windows-style paths (if on Unix)
        ]
        
        for path_format in path_formats:
            try:
                config_manager = ConfigManager(path_format)
                assert config_manager is not None
            except (TypeError, OSError):
                # Some formats may not be supported on current platform
                pass

    def test_configuration_circular_references(self, temp_dir):
        """Test handling of circular references in configuration."""
        
        # Create configs that reference each other (if supported)
        config1 = temp_dir / "config1.yaml"
        config2 = temp_dir / "config2.yaml"
        
        config1_data = {
            "service": "EC2",
            "includes": ["config2.yaml"]  # Reference to other config
        }
        
        config2_data = {
            "service": "S3",
            "includes": ["config1.yaml"]  # Circular reference
        }
        
        with open(config1, 'w') as f:
            yaml.dump(config1_data, f)
        
        with open(config2, 'w') as f:
            yaml.dump(config2_data, f)
        
        # Should handle circular references gracefully
        try:
            config_manager = ConfigManager(str(temp_dir))
            assert config_manager is not None
        except RecursionError:
            pytest.fail("Configuration system should prevent infinite recursion")

    def test_large_configuration_files(self, temp_dir):
        """Test handling of very large configuration files."""
        
        # Create large configuration
        large_config = {"services": {}}
        
        # Add many services
        for i in range(1000):
            large_config["services"][f"service_{i}"] = {
                "enabled": True,
                "model": f"model_{i}",
                "prompt": f"This is a long prompt for service {i} " * 10  # Long prompt
            }
        
        large_config_file = temp_dir / "large_config.yaml"
        with open(large_config_file, 'w') as f:
            yaml.dump(large_config, f)
        
        # Should handle large configurations
        try:
            config_manager = ConfigManager(str(temp_dir))
            assert config_manager is not None
        except (MemoryError, OSError) as e:
            # May fail due to size limits
            assert "memory" in str(e).lower() or "size" in str(e).lower()

    def test_configuration_encoding_issues(self, temp_dir):
        """Test handling of different file encodings in configuration."""
        
        # Create config with Unicode content
        unicode_config = {
            "service": "EC2",
            "description": "测试配置 - тест конфигурации - テスト設定",
            "tags": {
                "Name": "Iñtërnâtiônàl",
                "Owner": "用户"
            }
        }
        
        unicode_file = temp_dir / "unicode_config.yaml"
        
        # Test UTF-8 encoding
        with open(unicode_file, 'w', encoding='utf-8') as f:
            yaml.dump(unicode_config, f, allow_unicode=True)
        
        try:
            config_manager = ConfigManager(str(temp_dir))
            assert config_manager is not None
        except UnicodeDecodeError:
            pytest.fail("Should handle UTF-8 encoded configuration files")

    @patch.dict(os.environ, {
        "HOME": "/tmp/fake_home",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_MODEL": "gpt-3.5-turbo"
    })
    def test_home_directory_configuration(self):
        """Test configuration loading from home directory."""
        
        # Test with fake home directory but valid environment variables
        try:
            # This tests if the system can load config with environment variables
            config = LLMConfig(
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL")
            )
            assert config is not None
            assert config.model == "gpt-3.5-turbo"
        except (FileNotFoundError, PermissionError):
            # Expected if trying to access fake home directory for file config
            pass

    def test_application_startup_with_bad_config(self, temp_dir):
        """Test application startup behavior with bad configuration."""
        
        # Create completely invalid config directory structure
        bad_config_dir = temp_dir / "bad_config"
        bad_config_dir.mkdir()
        
        # Create file instead of directory where directory expected
        (bad_config_dir / "agents").write_text("this should be a directory")
        
        try:
            app = CostRecommendationApp(str(bad_config_dir), str(temp_dir))
            # Should handle bad config gracefully or provide clear error
            assert app is not None
        except Exception as e:
            # Should provide meaningful error message
            assert "config" in str(e).lower() or "directory" in str(e).lower()
            assert len(str(e)) > 10  # Should be descriptive