"""
Database Integration System for Plansheet Scanner

This module provides comprehensive database integration for the plansheet scanner,
enabling persistent storage, result caching, user management, and analytics.

Features:
- PostgreSQL integration with SQLAlchemy ORM
- Result caching and persistence
- User preferences and configurations
- Comprehensive audit logging
- Data analytics and insights
- Backup and recovery capabilities
"""

import asyncio
import gzip
import hashlib
import json
import logging
import os
import pickle
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psycopg2
from psycopg2.extras import RealDictCursor

# Database imports
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from .advanced_geospatial import AdvancedGeospatialProcessor, GeospatialVisualizer
from .ml_enhanced_symbol_recognition import MLSymbolRecognizer
from .performance_optimizer import PerformanceOptimizer
from .quality_assurance import QualityAssurance
from .real_time_processor import CollaborationSession, ProcessingJob, RealTimeProcessor

# Import core systems
from .unified_workflow import WorkflowOrchestrator, WorkflowResult, WorkflowStep

# SQLAlchemy Base
Base = declarative_base()


# Database Models
class User(Base):
    """User model for authentication and preferences"""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    api_key = Column(String(255), unique=True, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    preferences = Column(JSON, default={})


class ProcessingJob(Base):
    """Processing job model for tracking and persistence"""

    __tablename__ = "processing_jobs"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False)
    file_path = Column(String(500), nullable=False)
    workflow_template = Column(String(100), nullable=False)
    status = Column(
        String(50), nullable=False
    )  # pending, processing, completed, failed, cancelled
    progress = Column(Float, default=0.0)
    current_step = Column(String(200))
    estimated_time = Column(Float)
    priority = Column(String(50), default="normal")
    memory_limit = Column(Float, default=2.0)
    num_workers = Column(Integer, default=4)
    enable_ml = Column(Boolean, default=True)
    results = Column(JSON)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)


class ResultCache(Base):
    """Result cache model for storing processed results"""

    __tablename__ = "result_cache"

    id = Column(String(36), primary_key=True)
    file_hash = Column(String(64), unique=True, nullable=False)
    file_path = Column(String(500), nullable=False)
    workflow_template = Column(String(100), nullable=False)
    results = Column(JSON, nullable=False)
    processing_time = Column(Float, nullable=False)
    accuracy = Column(Float)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    accessed_at = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=1)


class AuditLog(Base):
    """Audit log model for tracking system activity"""

    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(String(36))
    details = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    timestamp = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)
    error_message = Column(Text)


class SystemMetrics(Base):
    """System metrics model for performance tracking"""

    __tablename__ = "system_metrics"

    id = Column(String(36), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_type = Column(String(100), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    tags = Column(JSON, default={})


@dataclass
class DatabaseConfig:
    """Database configuration"""

    host: str = "localhost"
    port: int = 5432
    database: str = "plansheet_scanner"
    username: str = "postgres"
    password: str = "password"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class CacheConfig:
    """Cache configuration"""

    max_size: int = 1000
    max_age_hours: int = 24
    compression_enabled: bool = True
    cleanup_interval_hours: int = 1


class DatabaseManager:
    """Manages database operations and caching for the plansheet scanner"""

    def __init__(self, config: DatabaseConfig, cache_config: CacheConfig):
        self.config = config
        self.cache_config = cache_config
        self.logger = logging.getLogger(__name__)

        # Initialize database connection
        self.engine = None
        self.SessionLocal = None
        self._setup_database()

        # Initialize core systems
        self.workflow_orchestrator = WorkflowOrchestrator()
        self.performance_optimizer = PerformanceOptimizer()
        self.ml_recognizer = MLSymbolRecognizer()
        self.real_time_processor = RealTimeProcessor()
        self.geospatial_processor = AdvancedGeospatialProcessor()
        self.quality_assurance = QualityAssurance()

        # Setup periodic tasks
        self._setup_periodic_tasks()

    def _setup_database(self):
        """Setup database connection and create tables"""
        try:
            # Create database URL
            database_url = (
                f"postgresql://{self.config.username}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}"
            )

            # Create engine
            self.engine = create_engine(
                database_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=False,
            )

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )

            # Create tables
            Base.metadata.create_all(bind=self.engine)

            self.logger.info("Database connection established successfully")

        except Exception as e:
            self.logger.error(f"Failed to setup database: {e}")
            raise

    def _setup_periodic_tasks(self):
        """Setup periodic database maintenance tasks"""
        # Start cache cleanup task
        asyncio.create_task(self._periodic_cache_cleanup())

        # Start metrics collection task
        asyncio.create_task(self._periodic_metrics_collection())

        # Start audit log cleanup task
        asyncio.create_task(self._periodic_audit_cleanup())

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    # User Management
    def create_user(
        self, username: str, email: str, password: str, is_admin: bool = False
    ) -> str:
        """Create a new user"""
        try:
            with self.get_session() as session:
                # Check if user already exists
                existing_user = (
                    session.query(User)
                    .filter((User.username == username) | (User.email == email))
                    .first()
                )

                if existing_user:
                    raise ValueError("User already exists")

                # Create user
                user_id = str(uuid.uuid4())
                password_hash = self._hash_password(password)
                api_key = self._generate_api_key()

                user = User(
                    id=user_id,
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    api_key=api_key,
                    is_admin=is_admin,
                    created_at=datetime.utcnow(),
                )

                session.add(user)
                session.commit()

                self.logger.info(f"Created user: {username}")
                return user_id

        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            raise

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user"""
        try:
            with self.get_session() as session:
                user = session.query(User).filter(User.username == username).first()

                if user and self._verify_password(password, user.password_hash):
                    # Update last login
                    user.last_login = datetime.utcnow()
                    session.commit()

                    self.logger.info(f"User authenticated: {username}")
                    return user

                return None

        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return None

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key"""
        try:
            with self.get_session() as session:
                return session.query(User).filter(User.api_key == api_key).first()

        except Exception as e:
            self.logger.error(f"Failed to get user by API key: {e}")
            return None

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences"""
        try:
            with self.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()

                if user:
                    user.preferences.update(preferences)
                    session.commit()

                    self.logger.info(f"Updated preferences for user: {user_id}")

        except Exception as e:
            self.logger.error(f"Failed to update user preferences: {e}")
            raise

    # Job Management
    def create_job(
        self,
        user_id: str,
        file_path: str,
        workflow_template: str,
        priority: str = "normal",
        memory_limit: float = 2.0,
        num_workers: int = 4,
        enable_ml: bool = True,
    ) -> str:
        """Create a new processing job"""
        try:
            with self.get_session() as session:
                job_id = str(uuid.uuid4())

                job = ProcessingJob(
                    id=job_id,
                    user_id=user_id,
                    file_path=file_path,
                    workflow_template=workflow_template,
                    status="pending",
                    progress=0.0,
                    current_step="Initializing",
                    priority=priority,
                    memory_limit=memory_limit,
                    num_workers=num_workers,
                    enable_ml=enable_ml,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )

                session.add(job)
                session.commit()

                self.logger.info(f"Created job: {job_id}")
                return job_id

        except Exception as e:
            self.logger.error(f"Failed to create job: {e}")
            raise

    def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: float = None,
        current_step: str = None,
        results: Dict[str, Any] = None,
        error_message: str = None,
    ):
        """Update job status"""
        try:
            with self.get_session() as session:
                job = (
                    session.query(ProcessingJob)
                    .filter(ProcessingJob.id == job_id)
                    .first()
                )

                if job:
                    job.status = status
                    job.updated_at = datetime.utcnow()

                    if progress is not None:
                        job.progress = progress

                    if current_step is not None:
                        job.current_step = current_step

                    if results is not None:
                        job.results = results

                    if error_message is not None:
                        job.error_message = error_message

                    if status == "completed":
                        job.completed_at = datetime.utcnow()

                    session.commit()

                    self.logger.info(f"Updated job {job_id} status to {status}")

        except Exception as e:
            self.logger.error(f"Failed to update job status: {e}")
            raise

    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job by ID"""
        try:
            with self.get_session() as session:
                return (
                    session.query(ProcessingJob)
                    .filter(ProcessingJob.id == job_id)
                    .first()
                )

        except Exception as e:
            self.logger.error(f"Failed to get job: {e}")
            return None

    def get_user_jobs(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ProcessingJob]:
        """Get user's jobs with optional filtering"""
        try:
            with self.get_session() as session:
                query = session.query(ProcessingJob).filter(
                    ProcessingJob.user_id == user_id
                )

                if status:
                    query = query.filter(ProcessingJob.status == status)

                return (
                    query.order_by(ProcessingJob.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                    .all()
                )

        except Exception as e:
            self.logger.error(f"Failed to get user jobs: {e}")
            return []

    def get_job_statistics(self) -> Dict[str, Any]:
        """Get job statistics"""
        try:
            with self.get_session() as session:
                total_jobs = session.query(ProcessingJob).count()
                completed_jobs = (
                    session.query(ProcessingJob)
                    .filter(ProcessingJob.status == "completed")
                    .count()
                )
                failed_jobs = (
                    session.query(ProcessingJob)
                    .filter(ProcessingJob.status == "failed")
                    .count()
                )
                active_jobs = (
                    session.query(ProcessingJob)
                    .filter(ProcessingJob.status.in_(["pending", "processing"]))
                    .count()
                )

                return {
                    "total_jobs": total_jobs,
                    "completed_jobs": completed_jobs,
                    "failed_jobs": failed_jobs,
                    "active_jobs": active_jobs,
                    "success_rate": (completed_jobs / total_jobs * 100)
                    if total_jobs > 0
                    else 0,
                }

        except Exception as e:
            self.logger.error(f"Failed to get job statistics: {e}")
            return {}

    # Result Caching
    def get_cached_result(
        self, file_path: str, workflow_template: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached result for file and workflow"""
        try:
            file_hash = self._calculate_file_hash(file_path)

            with self.get_session() as session:
                cache_entry = (
                    session.query(ResultCache)
                    .filter(
                        ResultCache.file_hash == file_hash,
                        ResultCache.workflow_template == workflow_template,
                    )
                    .first()
                )

                if cache_entry:
                    # Check if cache is still valid
                    if self._is_cache_valid(cache_entry):
                        # Update access info
                        cache_entry.accessed_at = datetime.utcnow()
                        cache_entry.access_count += 1
                        session.commit()

                        self.logger.info(f"Cache hit for file: {file_path}")
                        return cache_entry.results
                    else:
                        # Remove expired cache entry
                        session.delete(cache_entry)
                        session.commit()

                return None

        except Exception as e:
            self.logger.error(f"Failed to get cached result: {e}")
            return None

    def cache_result(
        self,
        file_path: str,
        workflow_template: str,
        results: Dict[str, Any],
        processing_time: float,
        accuracy: Optional[float] = None,
        confidence: Optional[float] = None,
    ):
        """Cache processing result"""
        try:
            file_hash = self._calculate_file_hash(file_path)

            with self.get_session() as session:
                # Check if cache entry already exists
                existing_entry = (
                    session.query(ResultCache)
                    .filter(
                        ResultCache.file_hash == file_hash,
                        ResultCache.workflow_template == workflow_template,
                    )
                    .first()
                )

                if existing_entry:
                    # Update existing entry
                    existing_entry.results = results
                    existing_entry.processing_time = processing_time
                    existing_entry.accuracy = accuracy
                    existing_entry.confidence = confidence
                    existing_entry.created_at = datetime.utcnow()
                    existing_entry.accessed_at = datetime.utcnow()
                    existing_entry.access_count = 1
                else:
                    # Create new cache entry
                    cache_entry = ResultCache(
                        id=str(uuid.uuid4()),
                        file_hash=file_hash,
                        file_path=file_path,
                        workflow_template=workflow_template,
                        results=results,
                        processing_time=processing_time,
                        accuracy=accuracy,
                        confidence=confidence,
                        created_at=datetime.utcnow(),
                        accessed_at=datetime.utcnow(),
                        access_count=1,
                    )
                    session.add(cache_entry)

                session.commit()
                self.logger.info(f"Cached result for file: {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to cache result: {e}")
            raise

    def clear_expired_cache(self):
        """Clear expired cache entries"""
        try:
            with self.get_session() as session:
                cutoff_time = datetime.utcnow() - timedelta(
                    hours=self.cache_config.max_age_hours
                )

                expired_entries = (
                    session.query(ResultCache)
                    .filter(ResultCache.created_at < cutoff_time)
                    .all()
                )

                for entry in expired_entries:
                    session.delete(entry)

                session.commit()

                if expired_entries:
                    self.logger.info(
                        f"Cleared {len(expired_entries)} expired cache entries"
                    )

        except Exception as e:
            self.logger.error(f"Failed to clear expired cache: {e}")

    # Audit Logging
    def log_audit_event(
        self,
        user_id: Optional[str],
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """Log audit event"""
        try:
            with self.get_session() as session:
                audit_entry = AuditLog(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details=details,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    timestamp=datetime.utcnow(),
                    success=success,
                    error_message=error_message,
                )

                session.add(audit_entry)
                session.commit()

        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")

    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLog]:
        """Get audit logs with filtering"""
        try:
            with self.get_session() as session:
                query = session.query(AuditLog)

                if user_id:
                    query = query.filter(AuditLog.user_id == user_id)

                if action:
                    query = query.filter(AuditLog.action == action)

                if start_time:
                    query = query.filter(AuditLog.timestamp >= start_time)

                if end_time:
                    query = query.filter(AuditLog.timestamp <= end_time)

                return (
                    query.order_by(AuditLog.timestamp.desc())
                    .offset(offset)
                    .limit(limit)
                    .all()
                )

        except Exception as e:
            self.logger.error(f"Failed to get audit logs: {e}")
            return []

    # System Metrics
    def record_metric(
        self,
        metric_type: str,
        metric_name: str,
        metric_value: float,
        metric_unit: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """Record system metric"""
        try:
            with self.get_session() as session:
                metric = SystemMetrics(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    metric_type=metric_type,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    metric_unit=metric_unit,
                    tags=tags or {},
                )

                session.add(metric)
                session.commit()

        except Exception as e:
            self.logger.error(f"Failed to record metric: {e}")

    def get_metrics(
        self,
        metric_type: Optional[str] = None,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[SystemMetrics]:
        """Get system metrics with filtering"""
        try:
            with self.get_session() as session:
                query = session.query(SystemMetrics)

                if metric_type:
                    query = query.filter(SystemMetrics.metric_type == metric_type)

                if metric_name:
                    query = query.filter(SystemMetrics.metric_name == metric_name)

                if start_time:
                    query = query.filter(SystemMetrics.timestamp >= start_time)

                if end_time:
                    query = query.filter(SystemMetrics.timestamp <= end_time)

                return query.order_by(SystemMetrics.timestamp.desc()).limit(limit).all()

        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            return []

    def get_metrics_summary(
        self, metric_type: str, metric_name: str, hours: int = 24
    ) -> Dict[str, float]:
        """Get metrics summary for specified time period"""
        try:
            with self.get_session() as session:
                start_time = datetime.utcnow() - timedelta(hours=hours)

                metrics = (
                    session.query(SystemMetrics)
                    .filter(
                        SystemMetrics.metric_type == metric_type,
                        SystemMetrics.metric_name == metric_name,
                        SystemMetrics.timestamp >= start_time,
                    )
                    .all()
                )

                if not metrics:
                    return {}

                values = [m.metric_value for m in metrics]

                return {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": values[0] if values else 0,
                }

        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            return {}

    # Analytics
    def get_analytics_report(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics report"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            with self.get_session() as session:
                # Job statistics
                total_jobs = (
                    session.query(ProcessingJob)
                    .filter(ProcessingJob.created_at >= start_date)
                    .count()
                )

                completed_jobs = (
                    session.query(ProcessingJob)
                    .filter(
                        ProcessingJob.status == "completed",
                        ProcessingJob.created_at >= start_date,
                    )
                    .count()
                )

                failed_jobs = (
                    session.query(ProcessingJob)
                    .filter(
                        ProcessingJob.status == "failed",
                        ProcessingJob.created_at >= start_date,
                    )
                    .count()
                )

                # User statistics
                active_users = (
                    session.query(ProcessingJob.user_id)
                    .filter(ProcessingJob.created_at >= start_date)
                    .distinct()
                    .count()
                )

                # Cache statistics
                cache_hits = (
                    session.query(ResultCache)
                    .filter(ResultCache.accessed_at >= start_date)
                    .count()
                )

                total_cache_entries = session.query(ResultCache).count()

                # Average processing time
                completed_jobs_data = (
                    session.query(ProcessingJob)
                    .filter(
                        ProcessingJob.status == "completed",
                        ProcessingJob.created_at >= start_date,
                        ProcessingJob.completed_at.isnot(None),
                    )
                    .all()
                )

                avg_processing_time = 0
                if completed_jobs_data:
                    total_time = sum(
                        (job.completed_at - job.created_at).total_seconds()
                        for job in completed_jobs_data
                    )
                    avg_processing_time = total_time / len(completed_jobs_data)

                return {
                    "period_days": days,
                    "total_jobs": total_jobs,
                    "completed_jobs": completed_jobs,
                    "failed_jobs": failed_jobs,
                    "success_rate": (completed_jobs / total_jobs * 100)
                    if total_jobs > 0
                    else 0,
                    "active_users": active_users,
                    "cache_hits": cache_hits,
                    "cache_entries": total_cache_entries,
                    "avg_processing_time_seconds": avg_processing_time,
                }

        except Exception as e:
            self.logger.error(f"Failed to get analytics report: {e}")
            return {}

    # Periodic Tasks
    async def _periodic_cache_cleanup(self):
        """Periodic cache cleanup task"""
        while True:
            try:
                self.clear_expired_cache()
                await asyncio.sleep(self.cache_config.cleanup_interval_hours * 3600)

            except Exception as e:
                self.logger.error(f"Cache cleanup task failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def _periodic_metrics_collection(self):
        """Periodic metrics collection task"""
        while True:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                await asyncio.sleep(300)  # Collect every 5 minutes

            except Exception as e:
                self.logger.error(f"Metrics collection task failed: {e}")
                await asyncio.sleep(300)

    async def _periodic_audit_cleanup(self):
        """Periodic audit log cleanup task"""
        while True:
            try:
                # Keep audit logs for 90 days
                cutoff_date = datetime.utcnow() - timedelta(days=90)

                with self.get_session() as session:
                    old_logs = (
                        session.query(AuditLog)
                        .filter(AuditLog.timestamp < cutoff_date)
                        .all()
                    )

                    for log in old_logs:
                        session.delete(log)

                    session.commit()

                    if old_logs:
                        self.logger.info(f"Cleaned up {len(old_logs)} old audit logs")

                await asyncio.sleep(86400)  # Run daily

            except Exception as e:
                self.logger.error(f"Audit cleanup task failed: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying

    def _collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # Memory usage
            memory_usage = self._get_memory_usage()
            self.record_metric(
                "system", "memory_usage_percent", memory_usage, "percent"
            )

            # CPU usage
            cpu_usage = self._get_cpu_usage()
            self.record_metric("system", "cpu_usage_percent", cpu_usage, "percent")

            # Active jobs
            job_stats = self.get_job_statistics()
            self.record_metric(
                "jobs", "active_jobs", job_stats.get("active_jobs", 0), "count"
            )
            self.record_metric(
                "jobs", "total_jobs", job_stats.get("total_jobs", 0), "count"
            )
            self.record_metric(
                "jobs", "success_rate", job_stats.get("success_rate", 0), "percent"
            )

        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")

    # Utility Methods
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return self._hash_password(password) == password_hash

    def _generate_api_key(self) -> str:
        """Generate secure API key"""
        return hashlib.sha256(os.urandom(32)).hexdigest()

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return hashlib.sha256(file_path.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: ResultCache) -> bool:
        """Check if cache entry is still valid"""
        age = datetime.utcnow() - cache_entry.created_at
        return age.total_seconds() < (self.cache_config.max_age_hours * 3600)

    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            import psutil

            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil

            return psutil.cpu_percent()
        except ImportError:
            return 0.0

    def backup_database(self, backup_path: str):
        """Create database backup"""
        try:
            # Use pg_dump for PostgreSQL backup
            backup_command = (
                f"pg_dump -h {self.config.host} -p {self.config.port} "
                f"-U {self.config.username} -d {self.config.database} "
                f"-f {backup_path}"
            )

            # Set password environment variable
            env = os.environ.copy()
            env["PGPASSWORD"] = self.config.password

            import subprocess

            result = subprocess.run(
                backup_command, shell=True, env=env, capture_output=True
            )

            if result.returncode == 0:
                self.logger.info(f"Database backup created: {backup_path}")
            else:
                raise Exception(f"Backup failed: {result.stderr.decode()}")

        except Exception as e:
            self.logger.error(f"Failed to create database backup: {e}")
            raise

    def restore_database(self, backup_path: str):
        """Restore database from backup"""
        try:
            # Use psql for PostgreSQL restore
            restore_command = (
                f"psql -h {self.config.host} -p {self.config.port} "
                f"-U {self.config.username} -d {self.config.database} "
                f"-f {backup_path}"
            )

            # Set password environment variable
            env = os.environ.copy()
            env["PGPASSWORD"] = self.config.password

            import subprocess

            result = subprocess.run(
                restore_command, shell=True, env=env, capture_output=True
            )

            if result.returncode == 0:
                self.logger.info(f"Database restored from: {backup_path}")
            else:
                raise Exception(f"Restore failed: {result.stderr.decode()}")

        except Exception as e:
            self.logger.error(f"Failed to restore database: {e}")
            raise


def main():
    """Main function to test database manager"""
    # Create database configuration
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="plansheet_scanner",
        username="postgres",
        password="password",
    )

    cache_config = CacheConfig(
        max_size=1000,
        max_age_hours=24,
        compression_enabled=True,
        cleanup_interval_hours=1,
    )

    # Create database manager
    db_manager = DatabaseManager(config, cache_config)

    # Test basic operations
    try:
        # Create test user
        user_id = db_manager.create_user("testuser", "test@example.com", "password123")
        print(f"Created user: {user_id}")

        # Authenticate user
        user = db_manager.authenticate_user("testuser", "password123")
        if user:
            print(f"Authenticated user: {user.username}")

        # Get analytics report
        analytics = db_manager.get_analytics_report(days=7)
        print(f"Analytics report: {analytics}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
