"""
Traffic Plan Reviewer - Streamlit Web Interface
Specialized web interface for traffic signal, ITS, and MUTCD signing plan review.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.traffic_plan_reviewer import TrafficPlanReviewer, PlanReviewResult

# Page configuration
st.set_page_config(
    page_title="Traffic Plan Reviewer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'reviewer' not in st.session_state:
    st.session_state.reviewer = TrafficPlanReviewer()
if 'review_history' not in st.session_state:
    st.session_state.review_history = []

def main():
    """Main Streamlit application."""
    
    # Sidebar
    st.sidebar.title("🚦 Traffic Plan Reviewer")
    st.sidebar.markdown("Specialized review for traffic signals, ITS, and MUTCD signing")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["📋 Plan Review", "📊 Batch Review", "📚 Standards Reference", "📈 Review History", "⚙️ Settings"]
    )
    
    if page == "📋 Plan Review":
        show_plan_review()
    elif page == "📊 Batch Review":
        show_batch_review()
    elif page == "📚 Standards Reference":
        show_standards_reference()
    elif page == "📈 Review History":
        show_review_history()
    elif page == "⚙️ Settings":
        show_settings()

def show_plan_review():
    """Show single plan review interface."""
    st.title("📋 Traffic Plan Review")
    st.markdown("Review traffic signal, ITS, and MUTCD signing plans for compliance")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a traffic plan file",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload a traffic plan file (PDF or image) for review"
    )
    
    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            plan_type = st.selectbox(
                "Plan Type:",
                ['auto', 'traffic_signal', 'its', 'mutcd_signing'],
                help="Select the type of plan or let the system auto-detect"
            )
        
        with col2:
            if st.button("🔍 Review Plan", type="primary"):
                with st.spinner("Reviewing plan..."):
                    try:
                        result = st.session_state.reviewer.review_plan(temp_path, plan_type)
                        
                        # Add to history
                        st.session_state.review_history.append({
                            'file': uploaded_file.name,
                            'plan_type': result.plan_type,
                            'compliance_score': result.compliance_score,
                            'timestamp': pd.Timestamp.now()
                        })
                        
                        # Clean up
                        os.remove(temp_path)
                        
                        display_plan_review_results(result, uploaded_file.name)
                        
                    except Exception as e:
                        st.error(f"Review failed: {e}")
                        os.remove(temp_path)

def display_plan_review_results(result: PlanReviewResult, filename: str):
    """Display plan review results."""
    
    # Header with compliance score
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Plan Type", result.plan_type.replace('_', ' ').title())
    
    with col2:
        st.metric("Compliance Score", f"{result.compliance_score:.2f}")
    
    with col3:
        st.metric("Issues Found", len(result.issues))
    
    with col4:
        st.metric("Elements Detected", len(result.elements_found))
    
    # Compliance status indicator
    if result.compliance_score >= 0.9:
        status_color = "green"
        status_text = "✅ EXCELLENT"
    elif result.compliance_score >= 0.7:
        status_color = "orange"
        status_text = "🟡 GOOD"
    elif result.compliance_score >= 0.5:
        status_color = "red"
        status_text = "🟠 FAIR"
    else:
        status_color = "darkred"
        status_text = "🔴 POOR"
    
    st.markdown(f"""
    <div style="background-color: {status_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0;">
        <h3>{status_text}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Standards checked
    st.subheader("📚 Standards Checked")
    for standard in result.standards_checked:
        st.write(f"• {standard}")
    
    # Issues analysis
    if result.issues:
        st.subheader("⚠️ Issues Analysis")
        
        # Create issues dataframe
        issues_df = pd.DataFrame(result.issues)
        
        # Severity distribution
        col1, col2 = st.columns(2)
        
        with col1:
            severity_counts = issues_df['severity'].value_counts()
            fig = px.pie(values=severity_counts.values, names=severity_counts.index, 
                        title="Issues by Severity")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            issue_types = issues_df['type'].value_counts()
            fig = px.bar(x=issue_types.index, y=issue_types.values, 
                        title="Issues by Type")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed issues
        st.subheader("📋 Detailed Issues")
        
        # Group by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            severity_issues = [i for i in result.issues if i['severity'] == severity]
            if severity_issues:
                severity_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                
                with st.expander(f"{severity_emoji[severity]} {severity.title()} Issues ({len(severity_issues)})"):
                    for i, issue in enumerate(severity_issues, 1):
                        st.write(f"**{i}. {issue['issue']}**")
                        st.write(f"   Element: {issue['element']}")
                        st.write(f"   Standard: {issue['standard']}")
                        st.write(f"   Recommendation: {issue['recommendation']}")
                        st.divider()
    else:
        st.success("✅ No issues found! Plan appears to meet all standards.")
    
    # Recommendations
    if result.recommendations:
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(result.recommendations, 1):
            st.info(f"{i}. {rec}")
    
    # Elements detected
    if result.elements_found:
        st.subheader("🔍 Elements Detected")
        
        # Create elements dataframe
        elements_data = []
        for element in result.elements_found:
            elements_data.append({
                'Type': element.element_type,
                'X': element.location[0],
                'Y': element.location[1],
                'Confidence': element.confidence
            })
        
        elements_df = pd.DataFrame(elements_data)
        
        # Elements by type
        col1, col2 = st.columns(2)
        
        with col1:
            element_counts = elements_df['Type'].value_counts()
            fig = px.pie(values=element_counts.values, names=element_counts.index, 
                        title="Elements by Type")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig = px.histogram(elements_df, x='Confidence', title="Confidence Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Elements table
        st.dataframe(elements_df, use_container_width=True)

def show_batch_review():
    """Show batch review interface."""
    st.title("📊 Batch Plan Review")
    st.markdown("Review multiple traffic plans at once")
    
    # Directory input
    directory = st.text_input(
        "Directory path:",
        placeholder="/path/to/plans",
        help="Enter the directory path containing plan files"
    )
    
    if directory and os.path.exists(directory):
        # Find plan files
        plan_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        plan_files = []
        
        for ext in plan_extensions:
            plan_files.extend(Path(directory).glob(f"*{ext}"))
            plan_files.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        if plan_files:
            st.write(f"Found {len(plan_files)} plan files")
            
            # Configuration
            col1, col2 = st.columns(2)
            
            with col1:
                plan_type = st.selectbox(
                    "Plan Type:",
                    ['auto', 'traffic_signal', 'its', 'mutcd_signing'],
                    help="Select the type of plans to review"
                )
            
            with col2:
                if st.button("🚀 Start Batch Review", type="primary"):
                    batch_review_results = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, plan_file in enumerate(plan_files):
                        status_text.text(f"Reviewing {plan_file.name}...")
                        
                        try:
                            result = st.session_state.reviewer.review_plan(str(plan_file), plan_type)
                            batch_review_results.append({
                                'file': plan_file.name,
                                'plan_type': result.plan_type,
                                'compliance_score': result.compliance_score,
                                'issues_count': len(result.issues),
                                'elements_count': len(result.elements_found),
                                'standards_checked': result.standards_checked
                            })
                        except Exception as e:
                            batch_review_results.append({
                                'file': plan_file.name,
                                'error': str(e)
                            })
                        
                        progress_bar.progress((i + 1) / len(plan_files))
                    
                    status_text.text("Batch review complete!")
                    
                    # Display batch results
                    display_batch_results(batch_review_results)
        else:
            st.warning("No plan files found in the specified directory")
    elif directory:
        st.error("Directory not found")

def display_batch_results(results: List[Dict[str, Any]]):
    """Display batch review results."""
    
    # Convert to dataframe
    df = pd.DataFrame(results)
    
    # Summary metrics
    successful_reviews = df[df['compliance_score'].notna()]
    failed_reviews = df[df['error'].notna()]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Plans", len(df))
    
    with col2:
        st.metric("Successful Reviews", len(successful_reviews))
    
    with col3:
        st.metric("Failed Reviews", len(failed_reviews))
    
    with col4:
        if len(successful_reviews) > 0:
            avg_compliance = successful_reviews['compliance_score'].mean()
            st.metric("Average Compliance", f"{avg_compliance:.2f}")
    
    # Results visualization
    if len(successful_reviews) > 0:
        st.subheader("📊 Batch Review Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Compliance score distribution
            fig = px.histogram(successful_reviews, x='compliance_score', 
                             title="Compliance Score Distribution",
                             nbins=10)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Issues count distribution
            fig = px.scatter(successful_reviews, x='compliance_score', y='issues_count',
                           title="Compliance vs Issues",
                           hover_data=['file'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Top and bottom performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performers")
            top_plans = successful_reviews.nlargest(5, 'compliance_score')
            for _, plan in top_plans.iterrows():
                st.write(f"• {plan['file']}: {plan['compliance_score']:.2f}")
        
        with col2:
            st.subheader("⚠️ Needs Attention")
            bottom_plans = successful_reviews.nsmallest(5, 'compliance_score')
            for _, plan in bottom_plans.iterrows():
                st.write(f"• {plan['file']}: {plan['compliance_score']:.2f}")
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    st.dataframe(df, use_container_width=True)

def show_standards_reference():
    """Show traffic engineering standards reference."""
    st.title("📚 Traffic Engineering Standards Reference")
    
    standards = {
        'MUTCD (Manual on Uniform Traffic Control Devices)': {
            'description': 'Federal standards for traffic control devices',
            'sections': ['Part 1 - General', 'Part 2 - Signs', 'Part 3 - Markings', 'Part 4 - Signals'],
            'applicable_to': ['mutcd_signing', 'traffic_signal'],
            'url': 'https://mutcd.fhwa.dot.gov/'
        },
        'ITE Signal Timing Manual': {
            'description': 'Institute of Transportation Engineers signal timing guidelines',
            'sections': ['Signal Design', 'Timing Parameters', 'Coordination', 'Pedestrian Features'],
            'applicable_to': ['traffic_signal'],
            'url': 'https://www.ite.org/technical-resources/signal-timing-manual/'
        },
        'AASHTO Green Book': {
            'description': 'American Association of State Highway and Transportation Officials geometric design',
            'sections': ['Geometric Design', 'Intersection Design', 'Safety Considerations'],
            'applicable_to': ['traffic_signal', 'mutcd_signing'],
            'url': 'https://www.aashto.org/'
        },
        'NTCIP (National Transportation Communications for ITS Protocol)': {
            'description': 'Standards for ITS communications and protocols',
            'sections': ['Device Communications', 'Data Exchange', 'System Integration'],
            'applicable_to': ['its'],
            'url': 'https://www.ntcip.org/'
        },
        'ADA Standards': {
            'description': 'Americans with Disabilities Act accessibility requirements',
            'sections': ['Pedestrian Access', 'Signal Accessibility', 'Crossing Design'],
            'applicable_to': ['traffic_signal', 'mutcd_signing'],
            'url': 'https://www.ada.gov/'
        }
    }
    
    for standard, info in standards.items():
        with st.expander(f"📖 {standard}"):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Applicable to:** {', '.join(info['applicable_to'])}")
            st.write(f"**Key Sections:** {', '.join(info['sections'])}")
            st.write(f"**Reference:** [{standard}]({info['url']})")

def show_review_history():
    """Show review history."""
    st.title("📈 Review History")
    
    if st.session_state.review_history:
        # Convert to dataframe
        history_df = pd.DataFrame(st.session_state.review_history)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Reviews", len(history_df))
        
        with col2:
            avg_score = history_df['compliance_score'].mean()
            st.metric("Average Compliance", f"{avg_score:.2f}")
        
        with col3:
            plan_types = history_df['plan_type'].value_counts()
            most_common = plan_types.index[0] if len(plan_types) > 0 else "None"
            st.metric("Most Common Type", most_common)
        
        # History visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Compliance over time
            fig = px.line(history_df, x='timestamp', y='compliance_score',
                         title="Compliance Score Over Time",
                         hover_data=['file', 'plan_type'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Plan type distribution
            fig = px.pie(history_df, names='plan_type', title="Reviews by Plan Type")
            st.plotly_chart(fig, use_container_width=True)
        
        # History table
        st.subheader("📋 Review History")
        st.dataframe(history_df, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.review_history = []
            st.rerun()
    else:
        st.info("No review history yet. Start reviewing plans to build history!")

def show_settings():
    """Show settings interface."""
    st.title("⚙️ Settings")
    st.markdown("Configure Traffic Plan Reviewer settings")
    
    st.subheader("Review Configuration")
    
    # Plan type preferences
    st.write("**Default Plan Type Detection:**")
    auto_detect = st.checkbox("Enable auto-detection", value=True)
    
    if not auto_detect:
        default_type = st.selectbox(
            "Default Plan Type:",
            ['traffic_signal', 'its', 'mutcd_signing']
        )
    
    st.subheader("Compliance Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        excellent_threshold = st.slider("Excellent Threshold", 0.0, 1.0, 0.9)
        good_threshold = st.slider("Good Threshold", 0.0, 1.0, 0.7)
    
    with col2:
        fair_threshold = st.slider("Fair Threshold", 0.0, 1.0, 0.5)
        poor_threshold = st.slider("Poor Threshold", 0.0, 1.0, 0.3)
    
    st.subheader("About")
    st.markdown("""
    **Traffic Plan Reviewer** is specialized for:
    - **Traffic Signal Plans**: Signal head placement, detector placement, pedestrian features
    - **ITS Plans**: Camera coverage, sensor placement, communication infrastructure
    - **MUTCD Signing Plans**: Sign placement, spacing, pavement markings
    
    Built with ❤️ for transportation engineering professionals.
    """)

if __name__ == "__main__":
    main()