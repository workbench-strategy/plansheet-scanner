"""
API Development System for Plansheet Scanner

This module provides a comprehensive FastAPI-based REST API for the plansheet scanner,
enabling external integration with all core systems and agents.

Features:
- Modern REST API with FastAPI
- Authentication and authorization system
- Rate limiting and abuse prevention
- Comprehensive API documentation
- SDK development capabilities
- Real-time WebSocket support
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn

# FastAPI imports
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .advanced_geospatial import AdvancedGeospatialProcessor, GeospatialVisualizer

# Import agents
from .interdisciplinary_reviewer import InterdisciplinaryReviewer
from .ml_enhanced_symbol_recognition import MLSymbolRecognizer
from .performance_optimizer import PerformanceOptimizer
from .quality_assurance import QualityAssurance
from .real_time_processor import CollaborationSession, ProcessingJob, RealTimeProcessor
from .traffic_plan_reviewer import TrafficPlanReviewer

# Import core systems
from .unified_workflow import WorkflowOrchestrator, WorkflowResult, WorkflowStep

# from .legend_extractor import LegendExtractor  # Function-based module
# from .overlay_generator import OverlayGenerator  # Function-based module
# from .line_matcher import LineMatcher  # Function-based module
# from .cable_entity_pipeline import CableEntityPipeline  # Function-based module


# Pydantic models for API requests/responses
class ProcessingRequest(BaseModel):
    """Request model for processing operations"""

    file_path: str = Field(..., description="Path to the file to process")
    workflow_template: str = Field(
        default="comprehensive", description="Workflow template to use"
    )
    memory_limit: float = Field(default=2.0, description="Memory limit in GB")
    num_workers: int = Field(default=4, description="Number of parallel workers")
    enable_ml: bool = Field(default=True, description="Enable ML processing")
    priority: str = Field(default="normal", description="Processing priority")

    class Config:
        schema_extra = {
            "example": {
                "file_path": "/path/to/plan_sheet.pdf",
                "workflow_template": "comprehensive",
                "memory_limit": 2.0,
                "num_workers": 4,
                "enable_ml": True,
                "priority": "normal",
            }
        }


class ProcessingResponse(BaseModel):
    """Response model for processing operations"""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Response message")
    estimated_time: Optional[float] = Field(
        None, description="Estimated processing time"
    )
    created_at: datetime = Field(..., description="Job creation timestamp")


class JobStatusResponse(BaseModel):
    """Response model for job status"""

    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current status")
    progress: float = Field(..., description="Progress percentage (0-100)")
    current_step: str = Field(..., description="Current processing step")
    estimated_time: Optional[float] = Field(
        None, description="Estimated time remaining"
    )
    results: Optional[Dict[str, Any]] = Field(None, description="Processing results")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class BatchProcessingRequest(BaseModel):
    """Request model for batch processing"""

    file_paths: List[str] = Field(..., description="List of file paths to process")
    workflow_template: str = Field(
        default="comprehensive", description="Workflow template"
    )
    memory_limit: float = Field(default=2.0, description="Memory limit per job")
    num_workers: int = Field(default=4, description="Number of workers per job")
    enable_ml: bool = Field(default=True, description="Enable ML processing")
    priority: str = Field(default="normal", description="Processing priority")

    class Config:
        schema_extra = {
            "example": {
                "file_paths": ["/path/to/file1.pdf", "/path/to/file2.pdf"],
                "workflow_template": "comprehensive",
                "memory_limit": 2.0,
                "num_workers": 4,
                "enable_ml": True,
                "priority": "normal",
            }
        }


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing"""

    batch_id: str = Field(..., description="Batch processing identifier")
    job_ids: List[str] = Field(..., description="List of created job IDs")
    total_files: int = Field(..., description="Total number of files")
    status: str = Field(..., description="Batch status")
    message: str = Field(..., description="Response message")
    created_at: datetime = Field(..., description="Batch creation timestamp")


class SystemStatusResponse(BaseModel):
    """Response model for system status"""

    status: str = Field(..., description="System status")
    active_jobs: int = Field(..., description="Number of active jobs")
    completed_jobs: int = Field(..., description="Number of completed jobs")
    failed_jobs: int = Field(..., description="Number of failed jobs")
    memory_usage: float = Field(..., description="Memory usage percentage")
    cpu_usage: float = Field(..., description="CPU usage percentage")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="API version")


class AuthenticationRequest(BaseModel):
    """Request model for authentication"""

    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class AuthenticationResponse(BaseModel):
    """Response model for authentication"""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user_id: str = Field(..., description="User identifier")


@dataclass
class APISession:
    """API session data"""

    session_id: str
    user_id: str
    api_key: str
    created_at: datetime
    last_activity: datetime
    rate_limit_remaining: int
    rate_limit_reset: datetime


@dataclass
class APIJob:
    """API job data"""

    job_id: str
    user_id: str
    status: str
    progress: float
    current_step: str
    estimated_time: Optional[float]
    results: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    priority: str
    workflow_template: str


class APIServer:
    """FastAPI server for plansheet scanner API"""

    def __init__(self):
        self.app = FastAPI(
            title="Plansheet Scanner API",
            description="AI-Powered Engineering Plan Analysis & Processing API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
        )

        # Initialize core systems
        self.workflow_orchestrator = WorkflowOrchestrator()
        self.performance_optimizer = PerformanceOptimizer()
        self.ml_recognizer = MLSymbolRecognizer()
        self.real_time_processor = RealTimeProcessor()
        self.geospatial_processor = AdvancedGeospatialProcessor()
        self.quality_assurance = QualityAssurance()

        # Initialize agents
        self.interdisciplinary_reviewer = InterdisciplinaryReviewer()
        self.traffic_plan_reviewer = TrafficPlanReviewer()
        # self.legend_extractor = LegendExtractor()  # Function-based module
        # self.overlay_generator = OverlayGenerator()  # Function-based module
        # self.line_matcher = LineMatcher()  # Function-based module
        # self.cable_entity_pipeline = CableEntityPipeline()  # Function-based module

        # Session and job management
        self.sessions: Dict[str, APISession] = {}
        self.jobs: Dict[str, APIJob] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}

        # Authentication
        self.security = HTTPBearer()
        self.api_keys: Dict[str, str] = {}  # In production, use database

        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_websockets()

    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes"""

        # Health and status endpoints
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint"""
            return {
                "message": "Plansheet Scanner API",
                "version": "1.0.0",
                "status": "operational",
            }

        @self.app.get("/health", response_model=Dict[str, str])
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy"}

        @self.app.get("/status", response_model=SystemStatusResponse)
        async def system_status():
            """Get system status"""
            return await self._get_system_status()

        # Authentication endpoints
        @self.app.post("/auth/login", response_model=AuthenticationResponse)
        async def login(request: AuthenticationRequest):
            """Authenticate user and get access token"""
            return await self._authenticate_user(request)

        @self.app.post("/auth/logout")
        async def logout(
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """Logout user and invalidate token"""
            return await self._logout_user(credentials.credentials)

        # Processing endpoints
        @self.app.post("/process", response_model=ProcessingResponse)
        async def process_file(
            request: ProcessingRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """Process a single file"""
            return await self._process_file(
                request, background_tasks, credentials.credentials
            )

        @self.app.post("/process/batch", response_model=BatchProcessingResponse)
        async def process_batch(
            request: BatchProcessingRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """Process multiple files in batch"""
            return await self._process_batch(
                request, background_tasks, credentials.credentials
            )

        @self.app.get("/jobs/{job_id}", response_model=JobStatusResponse)
        async def get_job_status(
            job_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """Get job status and results"""
            return await self._get_job_status(job_id, credentials.credentials)

        @self.app.get("/jobs", response_model=List[JobStatusResponse])
        async def list_jobs(
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
            status: Optional[str] = None,
            limit: int = 50,
            offset: int = 0,
        ):
            """List user's jobs with optional filtering"""
            return await self._list_jobs(credentials.credentials, status, limit, offset)

        @self.app.delete("/jobs/{job_id}")
        async def cancel_job(
            job_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """Cancel a processing job"""
            return await self._cancel_job(job_id, credentials.credentials)

        # Agent-specific endpoints
        @self.app.post("/agents/symbol-recognition")
        async def symbol_recognition(
            file_path: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """Use ML symbol recognition agent"""
            return await self._symbol_recognition(file_path, credentials.credentials)

        @self.app.post("/agents/geospatial-analysis")
        async def geospatial_analysis(
            file_path: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """Use geospatial analysis agent"""
            return await self._geospatial_analysis(file_path, credentials.credentials)

        @self.app.post("/agents/performance-review")
        async def performance_review(
            code: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """Use performance review agent"""
            return await self._performance_review(code, credentials.credentials)

        # Analytics endpoints
        @self.app.get("/analytics/performance")
        async def performance_analytics(
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """Get performance analytics"""
            return await self._get_performance_analytics(credentials.credentials)

        @self.app.get("/analytics/usage")
        async def usage_analytics(
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """Get usage analytics"""
            return await self._get_usage_analytics(credentials.credentials)

        # File management endpoints
        @self.app.post("/files/upload")
        async def upload_file(
            file_path: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """Upload file for processing"""
            return await self._upload_file(file_path, credentials.credentials)

        @self.app.get("/files/{file_id}/download")
        async def download_file(
            file_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security),
        ):
            """Download processed file"""
            return await self._download_file(file_id, credentials.credentials)

    def _setup_websockets(self):
        """Setup WebSocket endpoints"""

        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """WebSocket endpoint for real-time updates"""
            await self._handle_websocket_connection(websocket, client_id)

    async def _authenticate_user(
        self, request: AuthenticationRequest
    ) -> AuthenticationResponse:
        """Authenticate user and generate access token"""
        # In production, validate against database
        if request.username == "admin" and request.password == "password":
            user_id = str(uuid.uuid4())
            api_key = self._generate_api_key()

            # Create session
            session = APISession(
                session_id=str(uuid.uuid4()),
                user_id=user_id,
                api_key=api_key,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                rate_limit_remaining=1000,
                rate_limit_reset=datetime.now() + timedelta(hours=1),
            )

            self.sessions[api_key] = session
            self.api_keys[api_key] = user_id

            return AuthenticationResponse(
                access_token=api_key,
                token_type="bearer",
                expires_in=3600,
                user_id=user_id,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
            )

    async def _logout_user(self, token: str) -> Dict[str, str]:
        """Logout user and invalidate token"""
        if token in self.sessions:
            del self.sessions[token]
            if token in self.api_keys:
                del self.api_keys[token]

        return {"message": "Logged out successfully"}

    async def _process_file(
        self, request: ProcessingRequest, background_tasks: BackgroundTasks, token: str
    ) -> ProcessingResponse:
        """Process a single file"""
        # Validate token and rate limits
        user_id = await self._validate_token(token)
        await self._check_rate_limit(token)

        # Create job
        job_id = str(uuid.uuid4())
        job = APIJob(
            job_id=job_id,
            user_id=user_id,
            status="pending",
            progress=0.0,
            current_step="Initializing",
            estimated_time=None,
            results=None,
            error_message=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            priority=request.priority,
            workflow_template=request.workflow_template,
        )

        self.jobs[job_id] = job

        # Start processing in background
        background_tasks.add_task(self._process_file_background, job_id, request)

        return ProcessingResponse(
            job_id=job_id,
            status="pending",
            message="Processing job created successfully",
            estimated_time=None,
            created_at=job.created_at,
        )

    async def _process_batch(
        self,
        request: BatchProcessingRequest,
        background_tasks: BackgroundTasks,
        token: str,
    ) -> BatchProcessingResponse:
        """Process multiple files in batch"""
        # Validate token and rate limits
        user_id = await self._validate_token(token)
        await self._check_rate_limit(token)

        # Create batch
        batch_id = str(uuid.uuid4())
        job_ids = []

        for file_path in request.file_paths:
            job_id = str(uuid.uuid4())
            job = APIJob(
                job_id=job_id,
                user_id=user_id,
                status="pending",
                progress=0.0,
                current_step="Initializing",
                estimated_time=None,
                results=None,
                error_message=None,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                priority=request.priority,
                workflow_template=request.workflow_template,
            )

            self.jobs[job_id] = job
            job_ids.append(job_id)

            # Start processing in background
            background_tasks.add_task(
                self._process_file_background,
                job_id,
                ProcessingRequest(
                    file_path=file_path,
                    workflow_template=request.workflow_template,
                    memory_limit=request.memory_limit,
                    num_workers=request.num_workers,
                    enable_ml=request.enable_ml,
                    priority=request.priority,
                ),
            )

        return BatchProcessingResponse(
            batch_id=batch_id,
            job_ids=job_ids,
            total_files=len(request.file_paths),
            status="processing",
            message=f"Batch processing started for {len(request.file_paths)} files",
            created_at=datetime.now(),
        )

    async def _get_job_status(self, job_id: str, token: str) -> JobStatusResponse:
        """Get job status and results"""
        # Validate token
        user_id = await self._validate_token(token)

        if job_id not in self.jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
            )

        job = self.jobs[job_id]

        # Check if user owns this job
        if job.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
            )

        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress * 100,
            current_step=job.current_step,
            estimated_time=job.estimated_time,
            results=job.results,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    async def _list_jobs(
        self, token: str, status: Optional[str], limit: int, offset: int
    ) -> List[JobStatusResponse]:
        """List user's jobs with optional filtering"""
        # Validate token
        user_id = await self._validate_token(token)

        # Filter jobs by user
        user_jobs = [job for job in self.jobs.values() if job.user_id == user_id]

        # Apply status filter
        if status:
            user_jobs = [job for job in user_jobs if job.status == status]

        # Apply pagination
        user_jobs = user_jobs[offset : offset + limit]

        return [
            JobStatusResponse(
                job_id=job.job_id,
                status=job.status,
                progress=job.progress * 100,
                current_step=job.current_step,
                estimated_time=job.estimated_time,
                results=job.results,
                error_message=job.error_message,
                created_at=job.created_at,
                updated_at=job.updated_at,
            )
            for job in user_jobs
        ]

    async def _cancel_job(self, job_id: str, token: str) -> Dict[str, str]:
        """Cancel a processing job"""
        # Validate token
        user_id = await self._validate_token(token)

        if job_id not in self.jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
            )

        job = self.jobs[job_id]

        # Check if user owns this job
        if job.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
            )

        # Cancel job
        job.status = "cancelled"
        job.updated_at = datetime.now()

        return {"message": "Job cancelled successfully"}

    async def _symbol_recognition(self, file_path: str, token: str) -> Dict[str, Any]:
        """Use ML symbol recognition agent"""
        # Validate token
        await self._validate_token(token)
        await self._check_rate_limit(token)

        try:
            # Load model if available
            model_path = "models/symbol_recognition_model.pth"
            if os.path.exists(model_path):
                self.ml_recognizer.load_model(model_path)

            # Process file
            result = self.ml_recognizer.recognize_symbols(file_path)

            return {
                "status": "success",
                "detections": len(result.detections),
                "classifications": len(result.classifications),
                "confidence": result.overall_confidence,
                "processing_time": result.processing_time,
            }

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Symbol recognition failed: {str(e)}",
            )

    async def _geospatial_analysis(self, file_path: str, token: str) -> Dict[str, Any]:
        """Use geospatial analysis agent"""
        # Validate token
        await self._validate_token(token)
        await self._check_rate_limit(token)

        try:
            # Process file
            gdf = self.geospatial_processor.read_file(file_path)

            # Perform analysis
            analysis_results = {
                "total_features": len(gdf),
                "geometry_types": gdf.geometry.geom_type.value_counts().to_dict(),
                "crs": str(gdf.crs),
                "bounds": gdf.total_bounds.tolist(),
            }

            return {"status": "success", "analysis": analysis_results}

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Geospatial analysis failed: {str(e)}",
            )

    async def _performance_review(self, code: str, token: str) -> Dict[str, Any]:
        """Use performance review agent"""
        # Validate token
        await self._validate_token(token)
        await self._check_rate_limit(token)

        try:
            # Analyze code
            result = self.interdisciplinary_reviewer.analyze_code(code)

            return {
                "status": "success",
                "performance_analysis": result.get("performance", {}),
                "complexity_score": result.get("complexity_score", 0),
                "recommendations": result.get("recommendations", []),
            }

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Performance review failed: {str(e)}",
            )

    async def _get_performance_analytics(self, token: str) -> Dict[str, Any]:
        """Get performance analytics"""
        # Validate token
        await self._validate_token(token)

        # Get performance report
        report = self.performance_optimizer.get_optimization_report()

        return {"status": "success", "analytics": report}

    async def _get_usage_analytics(self, token: str) -> Dict[str, Any]:
        """Get usage analytics"""
        # Validate token
        user_id = await self._validate_token(token)

        # Get user's jobs
        user_jobs = [job for job in self.jobs.values() if job.user_id == user_id]

        analytics = {
            "total_jobs": len(user_jobs),
            "completed_jobs": len([j for j in user_jobs if j.status == "completed"]),
            "failed_jobs": len([j for j in user_jobs if j.status == "failed"]),
            "active_jobs": len(
                [j for j in user_jobs if j.status in ["pending", "processing"]]
            ),
            "average_processing_time": 0.0,
        }

        # Calculate average processing time
        completed_jobs = [j for j in user_jobs if j.status == "completed"]
        if completed_jobs:
            total_time = sum(
                (j.updated_at - j.created_at).total_seconds() for j in completed_jobs
            )
            analytics["average_processing_time"] = total_time / len(completed_jobs)

        return {"status": "success", "analytics": analytics}

    async def _upload_file(self, file_path: str, token: str) -> Dict[str, Any]:
        """Upload file for processing"""
        # Validate token
        await self._validate_token(token)
        await self._check_rate_limit(token)

        # Validate file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
            )

        # Generate file ID
        file_id = str(uuid.uuid4())

        return {
            "status": "success",
            "file_id": file_id,
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "message": "File uploaded successfully",
        }

    async def _download_file(self, file_id: str, token: str) -> FileResponse:
        """Download processed file"""
        # Validate token
        await self._validate_token(token)

        # In production, implement file storage and retrieval
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="File download not implemented",
        )

    async def _get_system_status(self) -> SystemStatusResponse:
        """Get system status"""
        active_jobs = len(
            [j for j in self.jobs.values() if j.status in ["pending", "processing"]]
        )
        completed_jobs = len([j for j in self.jobs.values() if j.status == "completed"])
        failed_jobs = len([j for j in self.jobs.values() if j.status == "failed"])

        # Get system metrics
        memory_usage = self._get_memory_usage()
        cpu_usage = self._get_cpu_usage()
        uptime = time.time()  # In production, track actual uptime

        return SystemStatusResponse(
            status="operational",
            active_jobs=active_jobs,
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            uptime=uptime,
            version="1.0.0",
        )

    async def _handle_websocket_connection(self, websocket: WebSocket, client_id: str):
        """Handle WebSocket connection for real-time updates"""
        await websocket.accept()
        self.websocket_connections[client_id] = websocket

        try:
            while True:
                # Send periodic updates
                await asyncio.sleep(5)

                # Get system status
                status = await self._get_system_status()

                # Send update
                await websocket.send_text(json.dumps(asdict(status)))

        except WebSocketDisconnect:
            if client_id in self.websocket_connections:
                del self.websocket_connections[client_id]

    async def _process_file_background(self, job_id: str, request: ProcessingRequest):
        """Background processing for uploaded files"""
        job = self.jobs[job_id]

        try:
            # Update status
            job.status = "processing"
            job.current_step = "File uploaded"
            job.progress = 0.1
            job.updated_at = datetime.now()

            # Process with unified workflow
            def progress_callback(step: str, progress: float):
                job.current_step = step
                job.progress = progress
                job.updated_at = datetime.now()

            self.workflow_orchestrator.set_progress_callback(progress_callback)

            result = self.workflow_orchestrator.process_plan_sheet(
                request.file_path, template=request.workflow_template
            )

            # Update final status
            job.status = "completed"
            job.progress = 1.0
            job.current_step = "Completed"
            job.results = asdict(result)
            job.updated_at = datetime.now()

        except Exception as e:
            # Update error status
            job.status = "failed"
            job.error_message = str(e)
            job.updated_at = datetime.now()

    async def _validate_token(self, token: str) -> str:
        """Validate API token and return user ID"""
        if token not in self.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

        # Update session activity
        if token in self.sessions:
            self.sessions[token].last_activity = datetime.now()

        return self.api_keys[token]

    async def _check_rate_limit(self, token: str):
        """Check rate limiting for API calls"""
        if token not in self.sessions:
            return

        session = self.sessions[token]

        # Check if rate limit reset is needed
        if datetime.now() > session.rate_limit_reset:
            session.rate_limit_remaining = 1000
            session.rate_limit_reset = datetime.now() + timedelta(hours=1)

        # Check if rate limit exceeded
        if session.rate_limit_remaining <= 0:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

        # Decrement rate limit
        session.rate_limit_remaining -= 1

    def _generate_api_key(self) -> str:
        """Generate a secure API key"""
        return base64.b64encode(os.urandom(32)).decode("utf-8")

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

    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the API server"""
        uvicorn.run(self.app, host=host, port=port, debug=debug, log_level="info")


def main():
    """Main function to run the API server"""
    # Create API server
    api_server = APIServer()

    # Run server
    api_server.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    main()
