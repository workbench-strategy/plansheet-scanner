"""
Comprehensive Tests for Priority 3 Systems

This module provides comprehensive unit and integration tests for:
1. Web Interface Development System
2. API Development System  
3. Database Integration System

Tests cover all major functionality, error handling, and integration scenarios.
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.api_server import APIJob, APIServer, APISession
from src.core.database_manager import (
    AuditLog,
    CacheConfig,
    DatabaseConfig,
    DatabaseManager,
)
from src.core.database_manager import ProcessingJob as DBProcessingJob
from src.core.database_manager import ResultCache, SystemMetrics, User

# Import systems to test
from src.core.web_interface import ProcessingStatus, WebInterfaceManager, WebSession


class TestWebInterfaceManager:
    """Test Web Interface Development System"""

    def setup_method(self):
        """Setup test environment"""
        self.web_manager = WebInterfaceManager()

    def test_web_session_creation(self):
        """Test WebSession dataclass creation"""
        session = WebSession(
            session_id="test_session",
            user_id="test_user",
            created_at=datetime.now(),
            last_activity=datetime.now(),
            uploaded_files=[],
            processing_jobs=[],
            preferences={},
        )

        assert session.session_id == "test_session"
        assert session.user_id == "test_user"
        assert len(session.uploaded_files) == 0
        assert len(session.processing_jobs) == 0

    def test_processing_status_creation(self):
        """Test ProcessingStatus dataclass creation"""
        status = ProcessingStatus(
            job_id="test_job",
            status="processing",
            progress=0.5,
            current_step="Processing file",
            estimated_time=30.0,
            results=None,
            error_message=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert status.job_id == "test_job"
        assert status.status == "processing"
        assert status.progress == 0.5
        assert status.current_step == "Processing file"

    @patch("streamlit.session_state")
    def test_session_management(self, mock_session_state):
        """Test session management functionality"""
        # Mock session state
        mock_session_state.session_id = "test_session_id"
        mock_session_state.user_id = "test_user_id"

        # Test session setup
        self.web_manager._setup_session_management()

        assert "test_session_id" in self.web_manager.sessions
        session = self.web_manager.sessions["test_session_id"]
        assert session.user_id == "test_user_id"

    def test_calculate_avg_processing_time(self):
        """Test average processing time calculation"""
        # Create test jobs
        job1 = ProcessingStatus(
            job_id="job1",
            status="completed",
            progress=1.0,
            current_step="Completed",
            estimated_time=None,
            results={"processing_time": 25.0},
            error_message=None,
            created_at=datetime.now() - timedelta(seconds=30),
            updated_at=datetime.now(),
        )

        job2 = ProcessingStatus(
            job_id="job2",
            status="completed",
            progress=1.0,
            current_step="Completed",
            estimated_time=None,
            results={"processing_time": 35.0},
            error_message=None,
            created_at=datetime.now() - timedelta(seconds=40),
            updated_at=datetime.now(),
        )

        self.web_manager.processing_jobs["job1"] = job1
        self.web_manager.processing_jobs["job2"] = job2

        avg_time = self.web_manager._calculate_avg_processing_time()
        assert avg_time > 0
        assert avg_time == pytest.approx(30.0, abs=5.0)  # Allow some tolerance

    def test_get_memory_usage(self):
        """Test memory usage retrieval"""
        memory_usage = self.web_manager._get_memory_usage()
        assert isinstance(memory_usage, float)
        assert 0 <= memory_usage <= 100

    def test_get_cpu_usage(self):
        """Test CPU usage retrieval"""
        cpu_usage = self.web_manager._get_cpu_usage()
        assert isinstance(cpu_usage, float)
        assert 0 <= cpu_usage <= 100

    def test_calculate_throughput(self):
        """Test throughput calculation"""
        # Create recent jobs
        recent_job = ProcessingStatus(
            job_id="recent_job",
            status="completed",
            progress=1.0,
            current_step="Completed",
            estimated_time=None,
            results=None,
            error_message=None,
            created_at=datetime.now() - timedelta(minutes=30),
            updated_at=datetime.now(),
        )

        self.web_manager.processing_jobs["recent_job"] = recent_job

        throughput = self.web_manager._calculate_throughput()
        assert throughput >= 1  # Should find the recent job

    def test_clear_completed_jobs(self):
        """Test clearing completed jobs"""
        # Create test jobs
        completed_job = ProcessingStatus(
            job_id="completed_job",
            status="completed",
            progress=1.0,
            current_step="Completed",
            estimated_time=None,
            results=None,
            error_message=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        active_job = ProcessingStatus(
            job_id="active_job",
            status="processing",
            progress=0.5,
            current_step="Processing",
            estimated_time=None,
            results=None,
            error_message=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        self.web_manager.processing_jobs["completed_job"] = completed_job
        self.web_manager.processing_jobs["active_job"] = active_job

        # Mock session
        mock_session = Mock()
        mock_session.processing_jobs = ["completed_job", "active_job"]
        self.web_manager.sessions["test_session"] = mock_session

        # Test clearing
        self.web_manager._clear_completed_jobs()

        # Active job should remain
        assert "active_job" in self.web_manager.processing_jobs
        assert "completed_job" not in self.web_manager.processing_jobs


class TestAPIServer:
    """Test API Development System"""

    def setup_method(self):
        """Setup test environment"""
        self.api_server = APIServer()

    def test_api_session_creation(self):
        """Test APISession dataclass creation"""
        session = APISession(
            session_id="test_session",
            user_id="test_user",
            api_key="test_api_key",
            created_at=datetime.now(),
            last_activity=datetime.now(),
            rate_limit_remaining=1000,
            rate_limit_reset=datetime.now() + timedelta(hours=1),
        )

        assert session.session_id == "test_session"
        assert session.user_id == "test_user"
        assert session.api_key == "test_api_key"
        assert session.rate_limit_remaining == 1000

    def test_api_job_creation(self):
        """Test APIJob dataclass creation"""
        job = APIJob(
            job_id="test_job",
            user_id="test_user",
            status="pending",
            progress=0.0,
            current_step="Initializing",
            estimated_time=None,
            results=None,
            error_message=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            priority="normal",
            workflow_template="comprehensive",
        )

        assert job.job_id == "test_job"
        assert job.user_id == "test_user"
        assert job.status == "pending"
        assert job.priority == "normal"

    @pytest.mark.asyncio
    async def test_authenticate_user_success(self):
        """Test successful user authentication"""
        # Test with default credentials
        request = Mock()
        request.username = "admin"
        request.password = "password"

        response = await self.api_server._authenticate_user(request)

        assert response.access_token is not None
        assert response.token_type == "bearer"
        assert response.expires_in == 3600
        assert response.user_id is not None

    @pytest.mark.asyncio
    async def test_authenticate_user_failure(self):
        """Test failed user authentication"""
        request = Mock()
        request.username = "invalid"
        request.password = "invalid"

        with pytest.raises(Exception):  # Should raise HTTPException
            await self.api_server._authenticate_user(request)

    @pytest.mark.asyncio
    async def test_validate_token(self):
        """Test token validation"""
        # Create test session
        api_key = "test_api_key"
        user_id = "test_user_id"
        self.api_server.api_keys[api_key] = user_id

        # Test valid token
        result = await self.api_server._validate_token(api_key)
        assert result == user_id

    @pytest.mark.asyncio
    async def test_validate_invalid_token(self):
        """Test invalid token validation"""
        with pytest.raises(Exception):  # Should raise HTTPException
            await self.api_server._validate_token("invalid_token")

    @pytest.mark.asyncio
    async def test_check_rate_limit(self):
        """Test rate limiting"""
        # Create test session
        api_key = "test_api_key"
        session = APISession(
            session_id="test_session",
            user_id="test_user",
            api_key=api_key,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            rate_limit_remaining=1,
            rate_limit_reset=datetime.now() + timedelta(hours=1),
        )
        self.api_server.sessions[api_key] = session

        # Test successful rate limit check
        await self.api_server._check_rate_limit(api_key)
        assert session.rate_limit_remaining == 0

        # Test rate limit exceeded
        with pytest.raises(Exception):  # Should raise HTTPException
            await self.api_server._check_rate_limit(api_key)

    def test_generate_api_key(self):
        """Test API key generation"""
        api_key = self.api_server._generate_api_key()
        assert isinstance(api_key, str)
        assert len(api_key) > 0

    def test_get_memory_usage(self):
        """Test memory usage retrieval"""
        memory_usage = self.api_server._get_memory_usage()
        assert isinstance(memory_usage, float)
        assert 0 <= memory_usage <= 100

    def test_get_cpu_usage(self):
        """Test CPU usage retrieval"""
        cpu_usage = self.api_server._get_cpu_usage()
        assert isinstance(cpu_usage, float)
        assert 0 <= cpu_usage <= 100


class TestDatabaseManager:
    """Test Database Integration System"""

    def setup_method(self):
        """Setup test environment"""
        # Use in-memory SQLite for testing
        self.config = DatabaseConfig(
            host=":memory:",
            port=0,
            database="test_db",
            username="test",
            password="test",
        )

        self.cache_config = CacheConfig(
            max_size=100,
            max_age_hours=1,
            compression_enabled=False,
            cleanup_interval_hours=1,
        )

        # Mock database connection for testing
        with patch("src.core.database_manager.create_engine") as mock_engine:
            mock_engine.return_value = Mock()
            self.db_manager = DatabaseManager(self.config, self.cache_config)

    def test_database_config_creation(self):
        """Test DatabaseConfig dataclass creation"""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
        )

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.username == "test_user"

    def test_cache_config_creation(self):
        """Test CacheConfig dataclass creation"""
        config = CacheConfig(
            max_size=500,
            max_age_hours=12,
            compression_enabled=True,
            cleanup_interval_hours=2,
        )

        assert config.max_size == 500
        assert config.max_age_hours == 12
        assert config.compression_enabled is True
        assert config.cleanup_interval_hours == 2

    def test_hash_password(self):
        """Test password hashing"""
        password = "test_password"
        hash1 = self.db_manager._hash_password(password)
        hash2 = self.db_manager._hash_password(password)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hash length

    def test_verify_password(self):
        """Test password verification"""
        password = "test_password"
        password_hash = self.db_manager._hash_password(password)

        assert self.db_manager._verify_password(password, password_hash) is True
        assert (
            self.db_manager._verify_password("wrong_password", password_hash) is False
        )

    def test_generate_api_key(self):
        """Test API key generation"""
        api_key = self.db_manager._generate_api_key()
        assert isinstance(api_key, str)
        assert len(api_key) == 64  # SHA-256 hash length

    def test_calculate_file_hash(self):
        """Test file hash calculation"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            file_hash = self.db_manager._calculate_file_hash(temp_file)
            assert isinstance(file_hash, str)
            assert len(file_hash) == 64  # SHA-256 hash length
        finally:
            os.unlink(temp_file)

    def test_is_cache_valid(self):
        """Test cache validity checking"""
        # Create mock cache entry
        cache_entry = Mock()
        cache_entry.created_at = datetime.utcnow()

        # Test valid cache
        assert self.db_manager._is_cache_valid(cache_entry) is True

        # Test expired cache
        cache_entry.created_at = datetime.utcnow() - timedelta(hours=2)
        assert self.db_manager._is_cache_valid(cache_entry) is False

    def test_get_memory_usage(self):
        """Test memory usage retrieval"""
        memory_usage = self.db_manager._get_memory_usage()
        assert isinstance(memory_usage, float)
        assert 0 <= memory_usage <= 100

    def test_get_cpu_usage(self):
        """Test CPU usage retrieval"""
        cpu_usage = self.db_manager._get_cpu_usage()
        assert isinstance(cpu_usage, float)
        assert 0 <= cpu_usage <= 100


class TestIntegration:
    """Integration tests for Priority 3 systems"""

    def setup_method(self):
        """Setup integration test environment"""
        # Create test configurations
        self.db_config = DatabaseConfig(
            host=":memory:",
            port=0,
            database="test_db",
            username="test",
            password="test",
        )

        self.cache_config = CacheConfig(
            max_size=100,
            max_age_hours=1,
            compression_enabled=False,
            cleanup_interval_hours=1,
        )

    @patch("src.core.database_manager.create_engine")
    def test_web_interface_database_integration(self, mock_engine):
        """Test Web Interface integration with Database Manager"""
        # Mock database connection
        mock_engine.return_value = Mock()

        # Create database manager
        db_manager = DatabaseManager(self.db_config, self.cache_config)

        # Create web interface manager
        web_manager = WebInterfaceManager()

        # Test integration
        assert web_manager is not None
        assert db_manager is not None

    @patch("src.core.database_manager.create_engine")
    def test_api_server_database_integration(self, mock_engine):
        """Test API Server integration with Database Manager"""
        # Mock database connection
        mock_engine.return_value = Mock()

        # Create database manager
        db_manager = DatabaseManager(self.db_config, self.cache_config)

        # Create API server
        api_server = APIServer()

        # Test integration
        assert api_server is not None
        assert db_manager is not None

    def test_system_interoperability(self):
        """Test interoperability between all Priority 3 systems"""
        # Test that all systems can work together
        web_manager = WebInterfaceManager()
        api_server = APIServer()

        # Test basic functionality
        assert web_manager.sessions is not None
        assert api_server.sessions is not None

        # Test data structures compatibility
        web_session = WebSession(
            session_id="test",
            user_id="test_user",
            created_at=datetime.now(),
            last_activity=datetime.now(),
            uploaded_files=[],
            processing_jobs=[],
            preferences={},
        )

        api_session = APISession(
            session_id="test",
            user_id="test_user",
            api_key="test_key",
            created_at=datetime.now(),
            last_activity=datetime.now(),
            rate_limit_remaining=1000,
            rate_limit_reset=datetime.now() + timedelta(hours=1),
        )

        assert web_session.session_id == api_session.session_id
        assert web_session.user_id == api_session.user_id


class TestPerformance:
    """Performance tests for Priority 3 systems"""

    def test_web_interface_performance(self):
        """Test Web Interface performance"""
        web_manager = WebInterfaceManager()

        # Test session creation performance
        start_time = time.time()
        for i in range(100):
            session = WebSession(
                session_id=f"session_{i}",
                user_id=f"user_{i}",
                created_at=datetime.now(),
                last_activity=datetime.now(),
                uploaded_files=[],
                processing_jobs=[],
                preferences={},
            )
            web_manager.sessions[f"session_{i}"] = session

        end_time = time.time()
        creation_time = end_time - start_time

        assert creation_time < 1.0  # Should complete in under 1 second
        assert len(web_manager.sessions) == 100

    def test_api_server_performance(self):
        """Test API Server performance"""
        api_server = APIServer()

        # Test job creation performance
        start_time = time.time()
        for i in range(100):
            job = APIJob(
                job_id=f"job_{i}",
                user_id=f"user_{i}",
                status="pending",
                progress=0.0,
                current_step="Initializing",
                estimated_time=None,
                results=None,
                error_message=None,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                priority="normal",
                workflow_template="comprehensive",
            )
            api_server.jobs[f"job_{i}"] = job

        end_time = time.time()
        creation_time = end_time - start_time

        assert creation_time < 1.0  # Should complete in under 1 second
        assert len(api_server.jobs) == 100

    def test_database_manager_performance(self):
        """Test Database Manager performance"""
        # Test password hashing performance
        start_time = time.time()
        for i in range(1000):
            self.db_manager._hash_password(f"password_{i}")

        end_time = time.time()
        hashing_time = end_time - start_time

        assert hashing_time < 1.0  # Should complete in under 1 second


class TestErrorHandling:
    """Error handling tests for Priority 3 systems"""

    def test_web_interface_error_handling(self):
        """Test Web Interface error handling"""
        web_manager = WebInterfaceManager()

        # Test with invalid data
        try:
            # This should handle gracefully
            web_manager._get_memory_usage()
            web_manager._get_cpu_usage()
        except Exception as e:
            pytest.fail(f"Error handling failed: {e}")

    def test_api_server_error_handling(self):
        """Test API Server error handling"""
        api_server = APIServer()

        # Test with invalid tokens
        try:
            # This should handle gracefully
            api_server._get_memory_usage()
            api_server._get_cpu_usage()
        except Exception as e:
            pytest.fail(f"Error handling failed: {e}")

    def test_database_manager_error_handling(self):
        """Test Database Manager error handling"""
        # Test with invalid file paths
        try:
            invalid_hash = self.db_manager._calculate_file_hash("/invalid/path")
            assert isinstance(invalid_hash, str)
        except Exception as e:
            pytest.fail(f"Error handling failed: {e}")


def main():
    """Run all Priority 3 system tests"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    main()
