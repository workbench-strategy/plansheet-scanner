"""
Real-Time Processing System

This module provides real-time plan sheet processing capabilities including:
- Streaming PDF processing
- WebSocket-based progress updates
- Live collaboration features
- Instant feedback and analysis

Author: Plansheet Scanner Team
Date: 2024
"""

import asyncio
import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol, serve

from src.core.ml_enhanced_symbol_recognition import MLSymbolRecognizer
from src.core.performance_optimizer import PerformanceOptimizer

# Import our existing systems
from src.core.unified_workflow import (
    ProgressUpdate,
    WorkflowOrchestrator,
    WorkflowResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingJob:
    """Represents a real-time processing job."""

    job_id: str
    file_path: str
    file_hash: str
    user_id: str
    session_id: str
    status: str = "pending"  # pending, processing, completed, failed
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[WorkflowResult] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationSession:
    """Represents a collaborative processing session."""

    session_id: str
    owner_id: str
    participants: Set[str] = field(default_factory=set)
    jobs: Dict[str, ProcessingJob] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealTimeEvent:
    """Represents a real-time event for WebSocket communication."""

    event_type: str  # job_started, job_progress, job_completed, job_failed, user_joined, etc.
    timestamp: datetime
    data: Dict[str, Any]
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class StreamingProcessor:
    """Handles streaming PDF processing with real-time updates."""

    def __init__(self, max_workers: int = 4, memory_limit_gb: float = 2.0):
        self.max_workers = max_workers
        self.memory_limit_gb = memory_limit_gb
        self.workflow_orchestrator = WorkflowOrchestrator(max_workers, memory_limit_gb)
        self.performance_optimizer = PerformanceOptimizer()
        self.symbol_recognizer = MLSymbolRecognizer()

        # Job management
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.completed_jobs: Dict[str, ProcessingJob] = {}
        self.job_queue = queue.Queue()

        # Threading
        self.processing_thread = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Event callbacks
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        self.completion_callbacks: Dict[str, List[Callable]] = {}

        logger.info(
            f"StreamingProcessor initialized with {max_workers} workers and {memory_limit_gb}GB memory limit"
        )

    def start(self):
        """Start the streaming processor."""
        if self.running:
            logger.warning("StreamingProcessor is already running")
            return

        self.running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self.processing_thread.start()
        logger.info("StreamingProcessor started")

    def stop(self):
        """Stop the streaming processor."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)
        logger.info("StreamingProcessor stopped")

    def submit_job(
        self,
        file_path: str,
        user_id: str,
        session_id: str,
        workflow_template: str = "comprehensive",
    ) -> str:
        """
        Submit a new processing job.

        Args:
            file_path: Path to the PDF file
            user_id: ID of the user submitting the job
            session_id: Session ID for collaboration
            workflow_template: Workflow template to use

        Returns:
            Job ID for tracking
        """
        # Generate job ID and file hash
        job_id = str(uuid.uuid4())
        file_hash = self._calculate_file_hash(file_path)

        # Create job
        job = ProcessingJob(
            job_id=job_id,
            file_path=file_path,
            file_hash=file_hash,
            user_id=user_id,
            session_id=session_id,
            metadata={"workflow_template": workflow_template},
        )

        # Add to active jobs
        self.active_jobs[job_id] = job

        # Add to processing queue
        self.job_queue.put((job_id, workflow_template))

        logger.info(f"Job {job_id} submitted for file {file_path}")
        return job_id

    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get the status of a processing job."""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        return None

    def add_progress_callback(
        self, job_id: str, callback: Callable[[ProgressUpdate], None]
    ):
        """Add a progress callback for a job."""
        if job_id not in self.progress_callbacks:
            self.progress_callbacks[job_id] = []
        self.progress_callbacks[job_id].append(callback)

    def add_completion_callback(
        self, job_id: str, callback: Callable[[ProcessingJob], None]
    ):
        """Add a completion callback for a job."""
        if job_id not in self.completion_callbacks:
            self.completion_callbacks[job_id] = []
        self.completion_callbacks[job_id].append(callback)

    def _processing_loop(self):
        """Main processing loop for handling jobs."""
        while self.running:
            try:
                # Get next job from queue
                job_id, workflow_template = self.job_queue.get(timeout=1.0)

                # Process the job
                self._process_job(job_id, workflow_template)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")

    def _process_job(self, job_id: str, workflow_template: str):
        """Process a single job."""
        job = self.active_jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        try:
            # Update job status
            job.status = "processing"
            job.start_time = datetime.now()

            # Set up progress callback
            def progress_callback(update: ProgressUpdate):
                job.progress = update.progress
                self._notify_progress_callbacks(job_id, update)

            self.workflow_orchestrator.set_progress_callback(progress_callback)

            # Process the job
            result = self.workflow_orchestrator.process_plan_sheet(
                job.file_path, workflow_template, f"output/job_{job_id}"
            )

            # Update job with results
            job.status = "completed"
            job.end_time = datetime.now()
            job.result = result
            job.progress = 100.0

            # Move to completed jobs
            self.completed_jobs[job_id] = job
            del self.active_jobs[job_id]

            # Notify completion callbacks
            self._notify_completion_callbacks(job_id, job)

            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            # Handle job failure
            job.status = "failed"
            job.end_time = datetime.now()
            job.error_message = str(e)

            # Move to completed jobs (failed)
            self.completed_jobs[job_id] = job
            del self.active_jobs[job_id]

            # Notify completion callbacks
            self._notify_completion_callbacks(job_id, job)

            logger.error(f"Job {job_id} failed: {e}")

    def _notify_progress_callbacks(self, job_id: str, update: ProgressUpdate):
        """Notify all progress callbacks for a job."""
        if job_id in self.progress_callbacks:
            for callback in self.progress_callbacks[job_id]:
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")

    def _notify_completion_callbacks(self, job_id: str, job: ProcessingJob):
        """Notify all completion callbacks for a job."""
        if job_id in self.completion_callbacks:
            for callback in self.completion_callbacks[job_id]:
                try:
                    callback(job)
                except Exception as e:
                    logger.error(f"Error in completion callback: {e}")

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class CollaborationManager:
    """Manages collaborative processing sessions."""

    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids
        self.session_timeout = timedelta(hours=24)  # 24 hour timeout

    def create_session(
        self, owner_id: str, session_name: str = "Collaboration Session"
    ) -> str:
        """Create a new collaboration session."""
        session_id = str(uuid.uuid4())

        session = CollaborationSession(
            session_id=session_id, owner_id=owner_id, settings={"name": session_name}
        )

        self.sessions[session_id] = session
        self._add_user_to_session(owner_id, session_id)

        logger.info(f"Created collaboration session {session_id} by user {owner_id}")
        return session_id

    def join_session(self, user_id: str, session_id: str) -> bool:
        """Join an existing collaboration session."""
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return False

        session = self.sessions[session_id]
        session.participants.add(user_id)
        session.last_activity = datetime.now()

        self._add_user_to_session(user_id, session_id)

        logger.info(f"User {user_id} joined session {session_id}")
        return True

    def leave_session(self, user_id: str, session_id: str):
        """Leave a collaboration session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.participants.discard(user_id)
            session.last_activity = datetime.now()

            # Remove user from user_sessions mapping
            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]

            logger.info(f"User {user_id} left session {session_id}")

    def add_job_to_session(self, session_id: str, job: ProcessingJob):
        """Add a job to a collaboration session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.jobs[job.job_id] = job
            session.last_activity = datetime.now()

            logger.info(f"Job {job.job_id} added to session {session_id}")

    def get_session_jobs(self, session_id: str) -> List[ProcessingJob]:
        """Get all jobs in a session."""
        if session_id in self.sessions:
            return list(self.sessions[session_id].jobs.values())
        return []

    def get_user_sessions(self, user_id: str) -> List[CollaborationSession]:
        """Get all sessions a user is part of."""
        if user_id not in self.user_sessions:
            return []

        sessions = []
        for session_id in self.user_sessions[user_id]:
            if session_id in self.sessions:
                sessions.append(self.sessions[session_id])
        return sessions

    def cleanup_expired_sessions(self):
        """Remove sessions that have expired."""
        now = datetime.now()
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if now - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Removed expired session {session_id}")

    def _add_user_to_session(self, user_id: str, session_id: str):
        """Add user to session mapping."""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        self.user_sessions[user_id].add(session_id)


class WebSocketServer:
    """WebSocket server for real-time communication."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, WebSocketServerProtocol] = {}
        self.client_sessions: Dict[str, str] = {}  # client_id -> session_id
        self.session_clients: Dict[
            str, Set[str]
        ] = {}  # session_id -> set of client_ids

        # Dependencies
        self.streaming_processor = StreamingProcessor()
        self.collaboration_manager = CollaborationManager()

        # Start processing
        self.streaming_processor.start()

        logger.info(f"WebSocket server initialized on {host}:{port}")

    async def start(self):
        """Start the WebSocket server."""
        async with serve(self._handle_client, self.host, self.port):
            logger.info(f"WebSocket server started on {self.host}:{self.port}")
            await asyncio.Future()  # run forever

    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket client connection."""
        client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket

        try:
            async for message in websocket:
                await self._handle_message(client_id, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        finally:
            # Clean up client
            await self._cleanup_client(client_id)

    async def _handle_message(self, client_id: str, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "join_session":
                await self._handle_join_session(client_id, data)
            elif message_type == "leave_session":
                await self._handle_leave_session(client_id, data)
            elif message_type == "submit_job":
                await self._handle_submit_job(client_id, data)
            elif message_type == "get_job_status":
                await self._handle_get_job_status(client_id, data)
            elif message_type == "get_session_info":
                await self._handle_get_session_info(client_id, data)
            else:
                await self._send_error(
                    client_id, f"Unknown message type: {message_type}"
                )

        except json.JSONDecodeError:
            await self._send_error(client_id, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling message from client {client_id}: {e}")
            await self._send_error(client_id, f"Internal server error: {str(e)}")

    async def _handle_join_session(self, client_id: str, data: Dict[str, Any]):
        """Handle join session request."""
        session_id = data.get("session_id")
        user_id = data.get("user_id")

        if not session_id or not user_id:
            await self._send_error(client_id, "Missing session_id or user_id")
            return

        # Join session
        success = self.collaboration_manager.join_session(user_id, session_id)
        if success:
            # Track client in session
            self.client_sessions[client_id] = session_id
            if session_id not in self.session_clients:
                self.session_clients[session_id] = set()
            self.session_clients[session_id].add(client_id)

            # Send confirmation
            await self._send_message(
                client_id,
                {
                    "type": "session_joined",
                    "session_id": session_id,
                    "user_id": user_id,
                },
            )

            # Notify other clients in session
            await self._broadcast_to_session(
                session_id,
                {"type": "user_joined", "session_id": session_id, "user_id": user_id},
                exclude_client=client_id,
            )
        else:
            await self._send_error(client_id, "Failed to join session")

    async def _handle_leave_session(self, client_id: str, data: Dict[str, Any]):
        """Handle leave session request."""
        session_id = data.get("session_id")
        user_id = data.get("user_id")

        if session_id and user_id:
            self.collaboration_manager.leave_session(user_id, session_id)

            # Remove client from session tracking
            if client_id in self.client_sessions:
                del self.client_sessions[client_id]
            if session_id in self.session_clients:
                self.session_clients[session_id].discard(client_id)

            # Notify other clients
            await self._broadcast_to_session(
                session_id,
                {"type": "user_left", "session_id": session_id, "user_id": user_id},
                exclude_client=client_id,
            )

    async def _handle_submit_job(self, client_id: str, data: Dict[str, Any]):
        """Handle job submission request."""
        file_path = data.get("file_path")
        user_id = data.get("user_id")
        session_id = data.get("session_id")
        workflow_template = data.get("workflow_template", "comprehensive")

        if not all([file_path, user_id, session_id]):
            await self._send_error(client_id, "Missing required fields")
            return

        try:
            # Submit job
            job_id = self.streaming_processor.submit_job(
                file_path, user_id, session_id, workflow_template
            )

            # Get job object
            job = self.streaming_processor.get_job_status(job_id)
            if job:
                # Add to session
                self.collaboration_manager.add_job_to_session(session_id, job)

                # Set up progress callback
                def progress_callback(update: ProgressUpdate):
                    asyncio.create_task(
                        self._broadcast_job_progress(session_id, job_id, update)
                    )

                self.streaming_processor.add_progress_callback(
                    job_id, progress_callback
                )

                # Set up completion callback
                def completion_callback(job: ProcessingJob):
                    asyncio.create_task(self._broadcast_job_completion(session_id, job))

                self.streaming_processor.add_completion_callback(
                    job_id, completion_callback
                )

            # Send confirmation
            await self._send_message(
                client_id,
                {"type": "job_submitted", "job_id": job_id, "session_id": session_id},
            )

        except Exception as e:
            await self._send_error(client_id, f"Failed to submit job: {str(e)}")

    async def _handle_get_job_status(self, client_id: str, data: Dict[str, Any]):
        """Handle job status request."""
        job_id = data.get("job_id")

        if not job_id:
            await self._send_error(client_id, "Missing job_id")
            return

        job = self.streaming_processor.get_job_status(job_id)
        if job:
            await self._send_message(
                client_id,
                {
                    "type": "job_status",
                    "job_id": job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "error_message": job.error_message,
                },
            )
        else:
            await self._send_error(client_id, "Job not found")

    async def _handle_get_session_info(self, client_id: str, data: Dict[str, Any]):
        """Handle session info request."""
        session_id = data.get("session_id")

        if not session_id:
            await self._send_error(client_id, "Missing session_id")
            return

        if session_id in self.collaboration_manager.sessions:
            session = self.collaboration_manager.sessions[session_id]
            jobs = self.collaboration_manager.get_session_jobs(session_id)

            await self._send_message(
                client_id,
                {
                    "type": "session_info",
                    "session_id": session_id,
                    "participants": list(session.participants),
                    "jobs": [
                        {
                            "job_id": job.job_id,
                            "status": job.status,
                            "progress": job.progress,
                            "user_id": job.user_id,
                        }
                        for job in jobs
                    ],
                    "settings": session.settings,
                },
            )
        else:
            await self._send_error(client_id, "Session not found")

    async def _broadcast_job_progress(
        self, session_id: str, job_id: str, update: ProgressUpdate
    ):
        """Broadcast job progress to session participants."""
        message = {
            "type": "job_progress",
            "session_id": session_id,
            "job_id": job_id,
            "progress": update.progress,
            "message": update.message,
            "timestamp": update.timestamp.isoformat(),
        }

        await self._broadcast_to_session(session_id, message)

    async def _broadcast_job_completion(self, session_id: str, job: ProcessingJob):
        """Broadcast job completion to session participants."""
        message = {
            "type": "job_completed",
            "session_id": session_id,
            "job_id": job.job_id,
            "status": job.status,
            "error_message": job.error_message,
            "timestamp": datetime.now().isoformat(),
        }

        await self._broadcast_to_session(session_id, message)

    async def _broadcast_to_session(
        self, session_id: str, message: Dict[str, Any], exclude_client: str = None
    ):
        """Broadcast message to all clients in a session."""
        if session_id not in self.session_clients:
            return

        for client_id in self.session_clients[session_id]:
            if client_id != exclude_client and client_id in self.clients:
                try:
                    await self.clients[client_id].send(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending message to client {client_id}: {e}")

    async def _send_message(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client."""
        if client_id in self.clients:
            try:
                await self.clients[client_id].send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")

    async def _send_error(self, client_id: str, error_message: str):
        """Send error message to client."""
        await self._send_message(client_id, {"type": "error", "message": error_message})

    async def _cleanup_client(self, client_id: str):
        """Clean up client resources."""
        # Remove from session tracking
        if client_id in self.client_sessions:
            session_id = self.client_sessions[client_id]
            if session_id in self.session_clients:
                self.session_clients[session_id].discard(client_id)
            del self.client_sessions[client_id]

        # Remove from clients
        if client_id in self.clients:
            del self.clients[client_id]


class RealTimeProcessor:
    """Main real-time processing system that coordinates all components."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.websocket_server = WebSocketServer(host, port)
        self.streaming_processor = self.websocket_server.streaming_processor
        self.collaboration_manager = self.websocket_server.collaboration_manager

        # Cleanup thread
        self.cleanup_thread = None
        self.running = False

        logger.info("RealTimeProcessor initialized")

    def start(self):
        """Start the real-time processing system."""
        if self.running:
            logger.warning("RealTimeProcessor is already running")
            return

        self.running = True

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

        # Start WebSocket server
        asyncio.run(self.websocket_server.start())

        logger.info("RealTimeProcessor started")

    def stop(self):
        """Stop the real-time processing system."""
        self.running = False

        # Stop streaming processor
        self.streaming_processor.stop()

        # Wait for cleanup thread
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)

        logger.info("RealTimeProcessor stopped")

    def _cleanup_loop(self):
        """Periodic cleanup loop."""
        while self.running:
            try:
                # Clean up expired sessions
                self.collaboration_manager.cleanup_expired_sessions()

                # Sleep for 1 hour
                time.sleep(3600)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Real-Time Plan Sheet Processor")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of processing workers"
    )
    parser.add_argument(
        "--memory-limit", type=float, default=2.0, help="Memory limit in GB"
    )

    args = parser.parse_args()

    # Create and start real-time processor
    processor = RealTimeProcessor(args.host, args.port)

    try:
        processor.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        processor.stop()


if __name__ == "__main__":
    main()
