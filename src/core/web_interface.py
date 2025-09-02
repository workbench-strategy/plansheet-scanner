"""
Web Interface Development System for Plansheet Scanner

This module provides a comprehensive Streamlit-based web interface for the plansheet scanner,
enabling user-friendly interaction with all 10 specialized agents and the unified workflow system.

Features:
- Interactive file upload with drag-and-drop support
- Real-time processing visualization
- Results browser with interactive exploration
- Multi-user session management
- Integration with all core systems
"""

import asyncio
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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


@dataclass
class WebSession:
    """Web session data for user management"""

    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    uploaded_files: List[str]
    processing_jobs: List[str]
    preferences: Dict[str, Any]


@dataclass
class ProcessingStatus:
    """Real-time processing status"""

    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: float  # 0.0 to 1.0
    current_step: str
    estimated_time: Optional[float]
    results: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime


class WebInterfaceManager:
    """Manages the Streamlit web interface and user sessions"""

    def __init__(self):
        self.sessions: Dict[str, WebSession] = {}
        self.processing_jobs: Dict[str, ProcessingStatus] = {}
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

        # Setup session management
        self._setup_session_management()

    def _setup_session_management(self):
        """Initialize session management"""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        if "user_id" not in st.session_state:
            st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"

        # Create or retrieve session
        session_id = st.session_state.session_id
        if session_id not in self.sessions:
            self.sessions[session_id] = WebSession(
                session_id=session_id,
                user_id=st.session_state.user_id,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                uploaded_files=[],
                processing_jobs=[],
                preferences={},
            )

        # Update last activity
        self.sessions[session_id].last_activity = datetime.now()

    def render_header(self):
        """Render the main header"""
        st.set_page_config(
            page_title="Plansheet Scanner - AI-Powered Analysis",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.title("🔍 Plansheet Scanner")
            st.markdown("**AI-Powered Engineering Plan Analysis & Processing**")

        with col2:
            session = self.sessions[st.session_state.session_id]
            st.metric("Active Session", f"User {session.user_id[:8]}")

        with col3:
            st.metric("Processing Jobs", len(session.processing_jobs))

    def render_sidebar(self):
        """Render the sidebar with navigation and options"""
        st.sidebar.title("🎛️ Navigation")

        # Main navigation
        page = st.sidebar.selectbox(
            "Select Page",
            [
                "🏠 Dashboard",
                "📁 File Upload",
                "⚙️ Processing",
                "📊 Results",
                "🔧 Settings",
                "📈 Analytics",
            ],
        )

        # User session info
        st.sidebar.markdown("---")
        st.sidebar.subheader("👤 Session Info")
        session = self.sessions[st.session_state.session_id]
        st.sidebar.text(f"User: {session.user_id[:8]}")
        st.sidebar.text(f"Files: {len(session.uploaded_files)}")
        st.sidebar.text(f"Jobs: {len(session.processing_jobs)}")

        # Quick actions
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Quick Actions")

        if st.sidebar.button("🔄 Refresh Status"):
            st.rerun()

        if st.sidebar.button("🧹 Clear Completed"):
            self._clear_completed_jobs()

        if st.sidebar.button("📊 Performance Report"):
            self._show_performance_report()

        return page

    def render_dashboard(self):
        """Render the main dashboard"""
        st.header("🏠 Dashboard")

        # System status
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Active Sessions", len(self.sessions))

        with col2:
            active_jobs = sum(
                1
                for job in self.processing_jobs.values()
                if job.status in ["pending", "processing"]
            )
            st.metric("Active Jobs", active_jobs)

        with col3:
            completed_today = sum(
                1
                for job in self.processing_jobs.values()
                if job.status == "completed"
                and job.updated_at.date() == datetime.now().date()
            )
            st.metric("Completed Today", completed_today)

        with col4:
            avg_processing_time = self._calculate_avg_processing_time()
            st.metric("Avg Processing Time", f"{avg_processing_time:.1f}s")

        # Recent activity
        st.subheader("📈 Recent Activity")
        self._render_recent_activity()

        # System performance
        st.subheader("⚡ System Performance")
        self._render_performance_metrics()

        # Quick start
        st.subheader("🚀 Quick Start")
        self._render_quick_start()

    def render_file_upload(self):
        """Render the file upload interface"""
        st.header("📁 File Upload")

        # Upload area
        uploaded_files = st.file_uploader(
            "Upload Plan Sheets (PDF, PNG, JPG)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Drag and drop files or click to browse",
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")

            # File preview
            st.subheader("📋 File Preview")
            for i, file in enumerate(uploaded_files):
                col1, col2, col3 = st.columns([1, 2, 1])

                with col1:
                    st.text(f"File {i+1}: {file.name}")
                    st.text(f"Size: {file.size / 1024:.1f} KB")

                with col2:
                    if file.type.startswith("image/"):
                        st.image(file, width=200)
                    else:
                        st.text("PDF file - preview not available")

                with col3:
                    if st.button(f"Process {file.name}", key=f"process_{i}"):
                        self._process_file(file)

            # Batch processing options
            st.subheader("⚙️ Processing Options")

            col1, col2 = st.columns(2)

            with col1:
                workflow_template = st.selectbox(
                    "Workflow Template",
                    ["comprehensive", "quick", "geospatial", "analysis"],
                    help="Select the processing workflow template",
                )

                memory_limit = st.slider(
                    "Memory Limit (GB)",
                    min_value=0.5,
                    max_value=4.0,
                    value=2.0,
                    step=0.5,
                    help="Maximum memory usage per job",
                )

            with col2:
                num_workers = st.slider(
                    "Number of Workers",
                    min_value=1,
                    max_value=8,
                    value=4,
                    help="Number of parallel processing workers",
                )

                enable_ml = st.checkbox(
                    "Enable ML Processing",
                    value=True,
                    help="Use ML-enhanced symbol recognition",
                )

            if st.button("🚀 Start Batch Processing", type="primary"):
                self._start_batch_processing(
                    uploaded_files,
                    workflow_template,
                    memory_limit,
                    num_workers,
                    enable_ml,
                )

    def render_processing(self):
        """Render the processing status interface"""
        st.header("⚙️ Processing Status")

        session = self.sessions[st.session_state.session_id]

        if not session.processing_jobs:
            st.info("📭 No processing jobs found. Upload files to get started!")
            return

        # Job status overview
        st.subheader("📊 Job Overview")
        self._render_job_overview()

        # Active jobs
        st.subheader("🔄 Active Jobs")
        active_jobs = [
            job_id
            for job_id in session.processing_jobs
            if self.processing_jobs[job_id].status in ["pending", "processing"]
        ]

        if active_jobs:
            for job_id in active_jobs:
                self._render_job_status(job_id)
        else:
            st.info("✅ No active jobs")

        # Completed jobs
        st.subheader("✅ Completed Jobs")
        completed_jobs = [
            job_id
            for job_id in session.processing_jobs
            if self.processing_jobs[job_id].status == "completed"
        ]

        if completed_jobs:
            for job_id in completed_jobs:
                self._render_completed_job(job_id)
        else:
            st.info("📭 No completed jobs")

        # Failed jobs
        failed_jobs = [
            job_id
            for job_id in session.processing_jobs
            if self.processing_jobs[job_id].status == "failed"
        ]

        if failed_jobs:
            st.subheader("❌ Failed Jobs")
            for job_id in failed_jobs:
                self._render_failed_job(job_id)

    def render_results(self):
        """Render the results browser interface"""
        st.header("📊 Results Browser")

        session = self.sessions[st.session_state.session_id]
        completed_jobs = [
            job_id
            for job_id in session.processing_jobs
            if self.processing_jobs[job_id].status == "completed"
        ]

        if not completed_jobs:
            st.info("📭 No completed jobs to display results for")
            return

        # Job selection
        selected_job_id = st.selectbox(
            "Select Job to View Results",
            completed_jobs,
            format_func=lambda x: f"{self.processing_jobs[x].current_step} - {x[:8]}",
        )

        if selected_job_id:
            self._render_job_results(selected_job_id)

    def render_settings(self):
        """Render the settings interface"""
        st.header("🔧 Settings")

        session = self.sessions[st.session_state.session_id]

        # User preferences
        st.subheader("👤 User Preferences")

        col1, col2 = st.columns(2)

        with col1:
            default_workflow = st.selectbox(
                "Default Workflow Template",
                ["comprehensive", "quick", "geospatial", "analysis"],
                index=0
                if "default_workflow" not in session.preferences
                else ["comprehensive", "quick", "geospatial", "analysis"].index(
                    session.preferences.get("default_workflow", "comprehensive")
                ),
            )

            default_memory_limit = st.slider(
                "Default Memory Limit (GB)",
                min_value=0.5,
                max_value=4.0,
                value=session.preferences.get("default_memory_limit", 2.0),
                step=0.5,
            )

        with col2:
            default_workers = st.slider(
                "Default Number of Workers",
                min_value=1,
                max_value=8,
                value=session.preferences.get("default_workers", 4),
            )

            auto_refresh = st.checkbox(
                "Auto-refresh Status",
                value=session.preferences.get("auto_refresh", True),
            )

        # System settings
        st.subheader("⚙️ System Settings")

        col1, col2 = st.columns(2)

        with col1:
            enable_notifications = st.checkbox(
                "Enable Notifications",
                value=session.preferences.get("enable_notifications", True),
            )

            enable_analytics = st.checkbox(
                "Enable Analytics",
                value=session.preferences.get("enable_analytics", True),
            )

        with col2:
            max_file_size = st.number_input(
                "Max File Size (MB)",
                min_value=1,
                max_value=100,
                value=session.preferences.get("max_file_size", 50),
            )

            session_timeout = st.number_input(
                "Session Timeout (hours)",
                min_value=1,
                max_value=24,
                value=session.preferences.get("session_timeout", 8),
            )

        # Save settings
        if st.button("💾 Save Settings"):
            session.preferences.update(
                {
                    "default_workflow": default_workflow,
                    "default_memory_limit": default_memory_limit,
                    "default_workers": default_workers,
                    "auto_refresh": auto_refresh,
                    "enable_notifications": enable_notifications,
                    "enable_analytics": enable_analytics,
                    "max_file_size": max_file_size,
                    "session_timeout": session_timeout,
                }
            )
            st.success("✅ Settings saved successfully!")

    def render_analytics(self):
        """Render the analytics interface"""
        st.header("📈 Analytics")

        # Processing statistics
        st.subheader("📊 Processing Statistics")
        self._render_processing_statistics()

        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        self._render_performance_analytics()

        # User activity
        st.subheader("👥 User Activity")
        self._render_user_analytics()

        # System health
        st.subheader("🏥 System Health")
        self._render_system_health()

    def _process_file(self, file):
        """Process a single uploaded file"""
        job_id = str(uuid.uuid4())

        # Create processing status
        self.processing_jobs[job_id] = ProcessingStatus(
            job_id=job_id,
            status="pending",
            progress=0.0,
            current_step="Initializing",
            estimated_time=None,
            results=None,
            error_message=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Add to session
        session = self.sessions[st.session_state.session_id]
        session.processing_jobs.append(job_id)
        session.uploaded_files.append(file.name)

        # Start processing in background
        threading.Thread(
            target=self._process_file_background, args=(job_id, file), daemon=True
        ).start()

        st.success(f"🚀 Started processing {file.name} (Job ID: {job_id[:8]})")

    def _process_file_background(self, job_id: str, file):
        """Background processing for uploaded files"""
        try:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f"_{file.name}"
            ) as tmp_file:
                tmp_file.write(file.getvalue())
                file_path = tmp_file.name

            # Update status
            self.processing_jobs[job_id].status = "processing"
            self.processing_jobs[job_id].current_step = "File uploaded"
            self.processing_jobs[job_id].progress = 0.1
            self.processing_jobs[job_id].updated_at = datetime.now()

            # Process with unified workflow
            session = self.sessions[st.session_state.session_id]
            workflow_template = session.preferences.get(
                "default_workflow", "comprehensive"
            )

            def progress_callback(step: str, progress: float):
                self.processing_jobs[job_id].current_step = step
                self.processing_jobs[job_id].progress = progress
                self.processing_jobs[job_id].updated_at = datetime.now()

            self.workflow_orchestrator.set_progress_callback(progress_callback)

            result = self.workflow_orchestrator.process_plan_sheet(
                file_path, template=workflow_template
            )

            # Update final status
            self.processing_jobs[job_id].status = "completed"
            self.processing_jobs[job_id].progress = 1.0
            self.processing_jobs[job_id].current_step = "Completed"
            self.processing_jobs[job_id].results = asdict(result)
            self.processing_jobs[job_id].updated_at = datetime.now()

            # Cleanup
            os.unlink(file_path)

        except Exception as e:
            # Update error status
            self.processing_jobs[job_id].status = "failed"
            self.processing_jobs[job_id].error_message = str(e)
            self.processing_jobs[job_id].updated_at = datetime.now()

    def _start_batch_processing(
        self,
        files,
        workflow_template: str,
        memory_limit: float,
        num_workers: int,
        enable_ml: bool,
    ):
        """Start batch processing for multiple files"""
        st.info(f"🚀 Starting batch processing for {len(files)} files...")

        for file in files:
            self._process_file(file)

        st.success(f"✅ Batch processing initiated for {len(files)} files!")

    def _render_recent_activity(self):
        """Render recent activity chart"""
        # Get recent jobs
        recent_jobs = sorted(
            self.processing_jobs.values(), key=lambda x: x.created_at, reverse=True
        )[:10]

        if not recent_jobs:
            st.info("📭 No recent activity")
            return

        # Create activity data
        activity_data = []
        for job in recent_jobs:
            activity_data.append(
                {
                    "Time": job.created_at,
                    "Status": job.status,
                    "Step": job.current_step,
                    "Progress": job.progress * 100,
                }
            )

        df = pd.DataFrame(activity_data)

        # Activity timeline
        fig = px.timeline(
            df,
            x_start="Time",
            y="Step",
            color="Status",
            title="Recent Processing Activity",
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_performance_metrics(self):
        """Render system performance metrics"""
        col1, col2 = st.columns(2)

        with col1:
            # Memory usage
            memory_usage = self._get_memory_usage()
            st.metric("Memory Usage", f"{memory_usage:.1f}%")

            # CPU usage
            cpu_usage = self._get_cpu_usage()
            st.metric("CPU Usage", f"{cpu_usage:.1f}%")

        with col2:
            # Active sessions
            active_sessions = len(
                [
                    s
                    for s in self.sessions.values()
                    if (datetime.now() - s.last_activity).seconds < 300
                ]
            )
            st.metric("Active Sessions", active_sessions)

            # Processing throughput
            throughput = self._calculate_throughput()
            st.metric("Throughput (jobs/hour)", f"{throughput:.1f}")

    def _render_quick_start(self):
        """Render quick start guide"""
        st.markdown(
            """
        ### 🚀 Quick Start Guide
        
        1. **📁 Upload Files**: Go to File Upload page and drag & drop your plan sheets
        2. **⚙️ Configure Processing**: Select workflow template and processing options
        3. **🚀 Start Processing**: Click "Start Batch Processing" to begin analysis
        4. **📊 View Results**: Monitor progress and explore results in real-time
        5. **🔧 Customize**: Adjust settings and preferences as needed
        
        ### 📋 Supported Formats
        - **PDF**: Plan sheets, drawings, documents
        - **PNG/JPG**: Images, scans, photos
        
        ### 🎯 Workflow Templates
        - **Comprehensive**: Full analysis with all agents
        - **Quick**: Fast processing for basic analysis
        - **Geospatial**: Focus on coordinate and spatial analysis
        - **Analysis**: Detailed performance and quality analysis
        """
        )

    def _render_job_overview(self):
        """Render job overview statistics"""
        session = self.sessions[st.session_state.session_id]

        if not session.processing_jobs:
            return

        # Calculate statistics
        total_jobs = len(session.processing_jobs)
        completed_jobs = sum(
            1
            for job_id in session.processing_jobs
            if self.processing_jobs[job_id].status == "completed"
        )
        failed_jobs = sum(
            1
            for job_id in session.processing_jobs
            if self.processing_jobs[job_id].status == "failed"
        )
        active_jobs = total_jobs - completed_jobs - failed_jobs

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Jobs", total_jobs)

        with col2:
            st.metric(
                "Completed",
                completed_jobs,
                delta=f"{(completed_jobs/total_jobs*100):.1f}%"
                if total_jobs > 0
                else "0%",
            )

        with col3:
            st.metric("Active", active_jobs)

        with col4:
            st.metric(
                "Failed",
                failed_jobs,
                delta=f"{(failed_jobs/total_jobs*100):.1f}%"
                if total_jobs > 0
                else "0%",
            )

    def _render_job_status(self, job_id: str):
        """Render individual job status"""
        job = self.processing_jobs[job_id]

        with st.expander(f"Job {job_id[:8]} - {job.current_step}"):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.text(f"Status: {job.status}")
                st.text(f"Current Step: {job.current_step}")
                st.progress(job.progress)

            with col2:
                st.text(f"Created: {job.created_at.strftime('%H:%M:%S')}")
                st.text(f"Updated: {job.updated_at.strftime('%H:%M:%S')}")

            with col3:
                if job.estimated_time:
                    st.text(f"ETA: {job.estimated_time:.1f}s")

                if st.button("Cancel", key=f"cancel_{job_id}"):
                    job.status = "cancelled"
                    job.updated_at = datetime.now()
                    st.rerun()

    def _render_completed_job(self, job_id: str):
        """Render completed job summary"""
        job = self.processing_jobs[job_id]

        with st.expander(f"✅ Completed - {job.current_step}"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.text(f"Job ID: {job_id}")
                st.text(f"Completed: {job.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")

                if job.results:
                    st.text(f"Results: {len(job.results)} items processed")

            with col2:
                if st.button("View Results", key=f"view_{job_id}"):
                    st.session_state.selected_job = job_id
                    st.rerun()

    def _render_failed_job(self, job_id: str):
        """Render failed job details"""
        job = self.processing_jobs[job_id]

        with st.expander(f"❌ Failed - {job.current_step}"):
            st.error(f"Error: {job.error_message}")
            st.text(f"Failed at: {job.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")

            if st.button("Retry", key=f"retry_{job_id}"):
                # Reset job status
                job.status = "pending"
                job.progress = 0.0
                job.error_message = None
                job.updated_at = datetime.now()
                st.rerun()

    def _render_job_results(self, job_id: str):
        """Render detailed job results"""
        job = self.processing_jobs[job_id]

        if not job.results:
            st.warning("No results available for this job")
            return

        # Results overview
        st.subheader("📊 Results Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            if "total_elements" in job.results:
                st.metric("Total Elements", job.results["total_elements"])

        with col2:
            if "processing_time" in job.results:
                st.metric("Processing Time", f"{job.results['processing_time']:.2f}s")

        with col3:
            if "accuracy" in job.results:
                st.metric("Accuracy", f"{job.results['accuracy']:.1f}%")

        # Detailed results
        st.subheader("🔍 Detailed Results")

        # Create tabs for different result types
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 Summary", "🎯 Elements", "📈 Performance", "🗺️ Geospatial"]
        )

        with tab1:
            self._render_summary_results(job.results)

        with tab2:
            self._render_element_results(job.results)

        with tab3:
            self._render_performance_results(job.results)

        with tab4:
            self._render_geospatial_results(job.results)

    def _render_summary_results(self, results: Dict[str, Any]):
        """Render summary results"""
        if "summary" in results:
            st.json(results["summary"])
        else:
            st.info("No summary data available")

    def _render_element_results(self, results: Dict[str, Any]):
        """Render element detection results"""
        if "elements" in results and results["elements"]:
            # Create DataFrame for elements
            elements_df = pd.DataFrame(results["elements"])
            st.dataframe(elements_df)

            # Element type distribution
            if "type" in elements_df.columns:
                fig = px.pie(
                    elements_df, names="type", title="Element Type Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No element data available")

    def _render_performance_results(self, results: Dict[str, Any]):
        """Render performance analysis results"""
        if "performance" in results:
            perf_data = results["performance"]

            col1, col2 = st.columns(2)

            with col1:
                if "complexity_analysis" in perf_data:
                    st.subheader("Complexity Analysis")
                    st.json(perf_data["complexity_analysis"])

            with col2:
                if "optimization_recommendations" in perf_data:
                    st.subheader("Optimization Recommendations")
                    for rec in perf_data["optimization_recommendations"]:
                        st.text(f"• {rec}")
        else:
            st.info("No performance data available")

    def _render_geospatial_results(self, results: Dict[str, Any]):
        """Render geospatial analysis results"""
        if "geospatial" in results:
            geo_data = results["geospatial"]

            if "coordinates" in geo_data:
                # Create map visualization
                coords_df = pd.DataFrame(geo_data["coordinates"])

                if (
                    len(coords_df) > 0
                    and "lat" in coords_df.columns
                    and "lon" in coords_df.columns
                ):
                    fig = px.scatter_mapbox(
                        coords_df,
                        lat="lat",
                        lon="lon",
                        title="Geospatial Elements",
                        mapbox_style="open-street-map",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No geospatial data available")

    def _clear_completed_jobs(self):
        """Clear completed jobs from session"""
        session = self.sessions[st.session_state.session_id]
        session.processing_jobs = [
            job_id
            for job_id in session.processing_jobs
            if self.processing_jobs[job_id].status not in ["completed", "failed"]
        ]
        st.success("✅ Cleared completed jobs")

    def _show_performance_report(self):
        """Show system performance report"""
        st.subheader("📊 Performance Report")

        # Get performance data
        report = self.performance_optimizer.get_optimization_report()

        if report:
            st.json(report)
        else:
            st.info("No performance data available")

    def _calculate_avg_processing_time(self) -> float:
        """Calculate average processing time"""
        completed_jobs = [
            job
            for job in self.processing_jobs.values()
            if job.status == "completed" and job.results
        ]

        if not completed_jobs:
            return 0.0

        total_time = sum(
            (job.updated_at - job.created_at).total_seconds() for job in completed_jobs
        )

        return total_time / len(completed_jobs)

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

    def _calculate_throughput(self) -> float:
        """Calculate jobs per hour throughput"""
        # Get jobs from last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_jobs = [
            job
            for job in self.processing_jobs.values()
            if job.created_at > one_hour_ago
        ]

        return len(recent_jobs)

    def _render_processing_statistics(self):
        """Render processing statistics"""
        # Calculate statistics
        total_jobs = len(self.processing_jobs)
        completed_jobs = sum(
            1 for job in self.processing_jobs.values() if job.status == "completed"
        )
        failed_jobs = sum(
            1 for job in self.processing_jobs.values() if job.status == "failed"
        )

        success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Jobs", total_jobs)

        with col2:
            st.metric("Success Rate", f"{success_rate:.1f}%")

        with col3:
            st.metric("Failed Jobs", failed_jobs)

        with col4:
            avg_time = self._calculate_avg_processing_time()
            st.metric("Avg Time", f"{avg_time:.1f}s")

    def _render_performance_analytics(self):
        """Render performance analytics"""
        # Processing time distribution
        completed_jobs = [
            job for job in self.processing_jobs.values() if job.status == "completed"
        ]

        if completed_jobs:
            processing_times = [
                (job.updated_at - job.created_at).total_seconds()
                for job in completed_jobs
            ]

            fig = px.histogram(
                x=processing_times,
                title="Processing Time Distribution",
                labels={"x": "Time (seconds)", "y": "Number of Jobs"},
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_user_analytics(self):
        """Render user activity analytics"""
        # Session activity
        session_activity = []
        for session in self.sessions.values():
            session_activity.append(
                {
                    "User": session.user_id[:8],
                    "Files": len(session.uploaded_files),
                    "Jobs": len(session.processing_jobs),
                    "Last Activity": session.last_activity,
                }
            )

        if session_activity:
            df = pd.DataFrame(session_activity)
            st.dataframe(df)

    def _render_system_health(self):
        """Render system health metrics"""
        col1, col2 = st.columns(2)

        with col1:
            memory_usage = self._get_memory_usage()
            st.metric("Memory Usage", f"{memory_usage:.1f}%")

            if memory_usage > 80:
                st.warning("⚠️ High memory usage detected")

        with col2:
            cpu_usage = self._get_cpu_usage()
            st.metric("CPU Usage", f"{cpu_usage:.1f}%")

            if cpu_usage > 80:
                st.warning("⚠️ High CPU usage detected")


def main():
    """Main function to run the web interface"""
    # Initialize web interface manager
    web_manager = WebInterfaceManager()

    # Render header
    web_manager.render_header()

    # Render sidebar and get page selection
    page = web_manager.render_sidebar()

    # Render selected page
    if page == "🏠 Dashboard":
        web_manager.render_dashboard()
    elif page == "📁 File Upload":
        web_manager.render_file_upload()
    elif page == "⚙️ Processing":
        web_manager.render_processing()
    elif page == "📊 Results":
        web_manager.render_results()
    elif page == "🔧 Settings":
        web_manager.render_settings()
    elif page == "📈 Analytics":
        web_manager.render_analytics()


if __name__ == "__main__":
    main()
