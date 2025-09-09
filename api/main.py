from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
import threading
import queue
import time
import json
import base64

# Azure AI components
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    DeepResearchTool,  # still used elsewhere
    SubmitToolOutputsAction,
    ToolOutput,
    RunStepStatus,
    ListSortOrder,
    AgentEventHandler,
    MessageRole
)
from azure.identity import DefaultAzureCredential
from reportlab.lib.pagesizes import A4  # Ensure A4 is imported for PDF generation

app = FastAPI(
    title="Deep Research API",
    description="API for deep research using Azure AI Foundry",
    version="1.0.0"
)

load_dotenv(override=True) # Included ONLY for testing - REMOVE FOR PRODUCTION DEPLOYMENT


# Request/Response Models (trimmed to only currently used models)
class DeepResearchRequest(BaseModel):
    query: str

class ResearchResponse(BaseModel):
    thread_id: str
    run_id: str
    status: str
    created_at: datetime

class ResearchStatusResponse(BaseModel):
    thread_id: str
    status: str
    progress: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class StreamAgentRequest(BaseModel):
    """Request body for the /agent/stream endpoint.

    Attributes:
        thread_id: Existing thread identifier (create one via /thread/create).
        message: User's input message to send to the agent.
        agent_id: Optional override; defaults to AGENT_ID env var if not provided.
    """
    thread_id: str
    message: str
    agent_id: Optional[str] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health Check Endpoint
    
    Check the health status of the API service.
    
    Returns:
        dict: A dictionary containing the service status and current timestamp
        
    Example:
        ```json
        {
            "status": "healthy",
            "timestamp": "2025-08-03T10:30:00.123456"
        }
        ```
    """
    return {"status": "healthy", "timestamp": datetime.now()}

def create_thread() -> str:
    """
    Create Azure AI Thread
    
    Initializes an Azure AI Project client and creates a new conversation thread
    for research sessions.
    
    Returns:
        str: The unique thread identifier that can be used for subsequent operations
        
    Raises:
        Exception: If Azure AI Project client initialization or thread creation fails
        
    Note:
        Requires PROJECT_ENDPOINT environment variable and proper Azure credentials
    """
    # Initialize project client
    project_client = AIProjectClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )
    
    # Create thread
    thread = project_client.agents.threads.create()
    
    return thread.id

@app.post("/thread/create")
async def create_thread_endpoint():
    """
    Create New Thread Endpoint
    
    Creates a new Azure AI conversation thread for research sessions.
    
    Returns:
        dict: A dictionary containing the newly created thread ID
        
    Raises:
        HTTPException: 500 error if thread creation fails
        
    Example Response:
        ```json
        {
            "thread_id": "thread_abc123def456"
        }
        ```
    """
    try:
        thread_id = create_thread()
        return {"thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create thread: {str(e)}")

# Deep Research Endpoints
@app.post("/deep_research/submit", response_model=ResearchResponse)
async def submit_deep_research(request: DeepResearchRequest, thread_id: str = Query(..., description="Thread ID for the research session")):
    """
    Start or Continue a Deep Research Job
    
    Initiates a new deep research job using Azure AI agents with the specified query.
    The research job runs asynchronously in the background with automatic status polling.
    
    Args:
        request (DeepResearchRequest): The research request containing the query
        thread_id (str): The thread ID for the research session (required query parameter)
        
    Returns:
        ResearchResponse: Research job details including thread_id, run_id, status, and creation time
        
    Raises:
        HTTPException: 500 error if research job creation fails
        
    Example Request Body:
        ```json
        {
            "query": "Latest developments in quantum computing"
        }
        ```
    """
    try:
        # Initialize project client
        project_client = AIProjectClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential(),
        )
      
        # Create agent
        agents_client = project_client.agents
        agent = agents_client.get_agent(os.environ['AGENT_ID'])
        
        # Create research prompt
        research_prompt = request.query
        
        # Create message and run
        message = project_client.agents.messages.create(
            thread_id=thread_id,
            role="user",
            content=research_prompt,
        )
        
        run = project_client.agents.runs.create(thread_id=thread_id, agent_id=agent.id)
        
        return ResearchResponse(
            thread_id=thread_id,
            run_id=run.id,
            status=run.status,
            created_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start research: {str(e)}")

# Alias endpoint for backward compatibility
@app.get("/deep_research/{thread_id}/{run_id}/status", response_model=ResearchStatusResponse)
async def get_research_status(thread_id: str, run_id: str):
    """
    Get Research Job Status
    
    Retrieves the current status and progress of a research job, including all messages
    exchanged between user and AI agent, with properly formatted citations and references.
    
    Args:
        thread_id (str): The unique thread identifier for the research session
        run_id (str): The unique run identifier for the specific research job
        
    Returns:
        ResearchStatusResponse: Complete status information including:
            - Current job status (running, completed, failed, error)
            - Progress information
            - All conversation messages with citations
            - Download availability for completed research
            
    Example Response:
        ```json
        {
            "thread_id": "thread_123",
            "status": "completed",
            "progress": "Azure status: completed",
            "messages": [...],
            "result": {
                "report_available": true,
                "formats": ["md", "pdf"]
            }
        }
        ```
    """
    
    try:
        # Get current run status from Azure
        # Initialize project client
        project_client = AIProjectClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential(),
        )

        run = project_client.agents.runs.get(
            thread_id=thread_id, 
            run_id=run_id
        )
        
        
        # Get messages from the thread
        from azure.ai.agents.models import MessageRole, ListSortOrder
        all_messages = project_client.agents.messages.list(
            thread_id=thread_id, 
            order=ListSortOrder.ASCENDING
        )
        
        # Convert messages to a readable format with citation processing
        import re
        messages_data = []
        for msg in all_messages:
            if msg.role == MessageRole.AGENT:
                message_content = ""
                
                # First, create citation numbering map for the entire message (outside the text_messages loop)
                used_citations = {}  # {start_index: citation_number}
                if msg.url_citation_annotations:
                    sorted_annotations_for_numbering = sorted(msg.url_citation_annotations, key=lambda x: x.start_index)
                    for idx, ann in enumerate(sorted_annotations_for_numbering):
                        used_citations[ann.start_index] = idx + 1
                
                # Now process each text message with the pre-computed citation numbers
                if msg.text_messages:
                    for text_msg in msg.text_messages:
                        text_content = text_msg.text.value
                        
                        # Replace citation text with markdown links using index-based replacement
                        if msg.url_citation_annotations and used_citations:
                            # Replace text in reverse order to avoid index shifting
                            sorted_annotations_for_replacement = sorted(msg.url_citation_annotations, key=lambda x: x.start_index, reverse=True)
                            
                            for ann in sorted_annotations_for_replacement:
                                citation_url = ann.url_citation.url
                                start_index = ann.start_index
                                end_index = ann.end_index
                                citation_number = used_citations[start_index]
                                
                                markdown_citation = f"[{citation_number}]({citation_url})"
                                # Replace text using indices instead of string replacement
                                text_content = text_content[:start_index] + markdown_citation + text_content[end_index:]
                        
                        message_content += text_content + "\n"
                
                # Process URL citations for references - only for citations that were actually used
                references = []
                if msg.url_citation_annotations and 'used_citations' in locals():
                    citations_for_references = []
                    
                    # Create references only for used citations
                    for ann in msg.url_citation_annotations:
                        start_index = ann.start_index
                        if start_index in used_citations:
                            url = ann.url_citation.url
                            title = ann.url_citation.title or url
                            citation_number = used_citations[start_index]
                            
                            citations_for_references.append({
                                "number": citation_number,
                                "citation_number": str(citation_number),
                                "title": title,
                                "url": url
                            })
                    
                    # Sort citations numerically by citation number
                    citations_for_references.sort(key=lambda x: x["number"])
                    references = citations_for_references
                
                messages_data.append({
                    "role": "agent",
                    "content": message_content.strip().replace("cot_summary: ", "").replace("cot_summary:", ""),
                    "references": references,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None
                })
            elif msg.role == MessageRole.USER:
                message_content = ""
                if msg.text_messages:
                    for text_msg in msg.text_messages:
                        message_content += text_msg.text.value + "\n"
                
                messages_data.append({
                    "role": "user", 
                    "content": message_content.strip().replace("cot_summary: ", "").replace("cot_summary:", ""),
                    "references": [],
                    "created_at": msg.created_at.isoformat() if msg.created_at else None
                })
        
        # If completed, indicate report is available for download
        result = None
        if run.status == "completed":
            result = {
                "report_available": True,
                "formats": ["md", "pdf"],
                "summary": "Research completed - use download endpoint to get report"
            }
        
        return ResearchStatusResponse(
            thread_id=thread_id,
            status=run.status,
            progress=f"Azure status: {run.status}",
            messages=messages_data,
            result=result,
            error=None
        )
        
    except Exception as e:
        return ResearchStatusResponse(
            thread_id=thread_id,
            status="error",
            progress=None,
            messages=None,
            result=None,
            error=str(e)
        )

def create_markdown_content_in_memory(message) -> str:
    """
    Create Markdown Content In Memory

    Converts an Azure AI research message into properly formatted Markdown content
    with inline citations and a references section. Citations are automatically
    numbered and linked to their corresponding references.

    Args:
        message: Azure AI message object containing text and citation annotations

    Returns:
        str: Complete Markdown content with inline citations and references section

    Features:
        - Automatic citation numbering based on appearance order
        - Inline citations as [[1]](url) format
        - References section with numbered list
        - Handles multiple text messages within a single response
    """
    from io import StringIO

    content_buffer = StringIO()

    used_citations = {}
    if message.url_citation_annotations:
        sorted_annotations_for_numbering = sorted(message.url_citation_annotations, key=lambda x: x.start_index)
        for idx, ann in enumerate(sorted_annotations_for_numbering):
            used_citations[ann.start_index] = idx + 1

    text_parts = []
    for t in message.text_messages:
        text_content = t.text.value.strip()
        if message.url_citation_annotations and used_citations:
            sorted_annotations_for_replacement = sorted(message.url_citation_annotations, key=lambda x: x.start_index, reverse=True)
            for ann in sorted_annotations_for_replacement:
                citation_url = ann.url_citation.url
                start_index = ann.start_index
                end_index = ann.end_index
                citation_number = used_citations[start_index]
                markdown_citation = f" [[{citation_number}]]({citation_url})"
                text_content = text_content[:start_index] + markdown_citation + text_content[end_index:]
        text_parts.append(text_content)

    text_summary = "\n\n".join(text_parts)
    content_buffer.write(text_summary)

    if used_citations and message.url_citation_annotations:
        content_buffer.write("\n\n## References\n")
        citations_for_references = []
        for ann in message.url_citation_annotations:
            start_index = ann.start_index
            if start_index in used_citations:
                url = ann.url_citation.url
                title = ann.url_citation.title or url
                citation_number = used_citations[start_index]
                citations_for_references.append((citation_number, str(citation_number), title, url))
        citations_for_references.sort(key=lambda x: x[0])
        for _, citation_number, title, url in citations_for_references:
            content_buffer.write(f"{citation_number}. [{title}]({url})\n")

    return content_buffer.getvalue()


@app.post("/agent/stream")
async def stream_agent(req: StreamAgentRequest):
    """Stream agent response in real-time.

    Initiates a run for the specified agent + thread and streams incremental
    message deltas (including code interpreter outputs or citations) back to the caller.

    This endpoint adapts the provided Azure Functions style sample to FastAPI.

    Stream Protocol:
        Plain text chunks are yielded as they arrive. A terminating token '<<DONE>>' is internal only
        and not sent to the client (filtered before yielding). Client should treat connection close as completion.

    Request Body:
        thread_id (str): Existing thread ID.
        message (str): User message to append to thread.
        agent_id (str, optional): Override environment AGENT_ID.

    Returns:
        StreamingResponse: text/plain stream of incremental output.
    """

    # Initialize client
    try:
        project_client = AIProjectClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to init project client: {e}")

    agent_id = req.agent_id or os.environ.get("WEB_SEARCH_AGENT_ID")
    if not agent_id:
        raise HTTPException(status_code=400, detail="WEB_SEARCH_AGENT_ID not provided and not set in environment")

    try:
        agent = project_client.agents.get_agent(agent_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Agent not found: {e}")

    q: "queue.Queue[str]" = queue.Queue()
    SENTINEL = "<<DONE>>"

    class StreamingEventHandler(AgentEventHandler):
        def __init__(self):
            super().__init__()
            self.code_interpreter_active = False

        def on_message_delta(self, delta):
            # delta.text may be None for non-text events
            text = getattr(delta, "text", None)
            if text:
                q.put(text)

        def on_thread_run(self, run):
            # Handle tool calls if required (no custom functions implemented here)
            if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for tc in tool_calls:
                    # Provide placeholder output so run can continue
                    if tc.type == "function":
                        tool_outputs.append(
                            ToolOutput(tool_call_id=tc.id, output="Function execution not implemented in /agent/stream.")
                        )
                if tool_outputs:
                    try:
                        with project_client.agents.runs.submit_tool_outputs_stream(
                            thread_id=run.thread_id,
                            run_id=run.id,
                            tool_outputs=tool_outputs,
                            event_handler=self,
                        ) as stream:
                            stream.until_done()
                    except Exception as e:
                        logging.error(f"Failed submitting tool outputs: {e}")

        def on_run_step(self, step):
            try:
                if step.status == RunStepStatus.COMPLETED and hasattr(step, "step_details"):
                    if "tool_calls" in step.step_details:
                        for tc in step.step_details.tool_calls:
                            if tc.type == "code_interpreter":
                                # Close code block if previously opened
                                if self.code_interpreter_active:
                                    q.put("\n")
                                self.code_interpreter_active = False
                                # Stream any image outputs (as base64 <img>)
                                for output in tc.code_interpreter.outputs:
                                    if output.type == "image":
                                        try:
                                            file_id = output.image.file_id
                                            data_iter = project_client.agents.files.get_content(file_id)
                                            chunks = []
                                            for chunk in data_iter:
                                                if isinstance(chunk, (bytes, bytearray)):
                                                    chunks.append(chunk)
                                            combined = b"".join(chunks)
                                            encoded = base64.b64encode(combined).decode("utf-8")
                                            q.put(f'<img width="750px" src="data:image/png;base64,{encoded}"/>\n')
                                        except Exception as e:
                                            logging.error(f"Error streaming image output: {e}")
            except Exception as e:
                logging.error(f"on_run_step error: {e}")

        def on_run_step_delta(self, delta):
            # Stream code interpreter input if present
            try:
                details = getattr(delta, "delta", None)
                if not details:
                    return
                tool_calls = getattr(details, "step_details", {}).get("tool_calls", [])
                for tc in tool_calls:
                    if tc.type == "code_interpreter":
                        code_input = tc.code_interpreter.input
                        if code_input:
                            if not self.code_interpreter_active:
                                self.code_interpreter_active = True
                                q.put("\n```python\n")
                            q.put(code_input + "\n")
            except Exception as e:
                logging.error(f"on_run_step_delta error: {e}")

        def on_error(self, data):
            logging.error(f"Stream error: {data}")
            q.put(f"\n[ERROR] {data}\n")

        def on_done(self):
            # Close any open code fence
            if self.code_interpreter_active:
                q.put("```\n")
            q.put(SENTINEL)

    def worker():
        try:
            # Create user message
            project_client.agents.messages.create(
                thread_id=req.thread_id,
                role="user",
                content=req.message,
            )
            handler = StreamingEventHandler()
            with project_client.agents.runs.stream(
                thread_id=req.thread_id,
                agent_id=agent.id,
                event_handler=handler,
            ) as stream:
                stream.until_done()
        except Exception as e:
            logging.error(f"Streaming failure: {e}")
            q.put(f"[FATAL] {e}\n")
            q.put(SENTINEL)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    cutoff = datetime.utcnow() + timedelta(seconds=90)

    def iterator():
        while True:
            try:
                remaining = (cutoff - datetime.utcnow()).total_seconds()
                if remaining <= 0:
                    q.put(SENTINEL)
                item = q.get(timeout=max(0.1, min(5, remaining)))
            except Exception:
                if datetime.utcnow() > cutoff:
                    break
                continue
            if item == SENTINEL:
                last_message = project_client.agents.messages.get_last_message_by_role(thread_id=req.thread_id, role=MessageRole.AGENT)
                citations = last_message.url_citation_annotations
                if len(citations)>0:
                    formatted_citations = [x.as_dict() for x in citations]
                    yield '<br/>##CITATIONS:' + json.dumps(formatted_citations)
                break
            if item is None:
                continue
            yield item

    return StreamingResponse(iterator(), media_type="text/plain")


def create_pdf_content_in_memory(message) -> bytes:
    """
    Create PDF Content In Memory

    Converts an Azure AI research message into a PDF document with proper formatting,
    clickable citations, and references. Uses multiple PDF generation approaches with
    fallback support for different library availability.

    Args:
        message: Azure AI message object containing text and citation annotations

    Returns:
        bytes: Complete PDF document as binary data

    PDF Generation Methods (in order of preference):
        1. ReportLab
        2. FPDF2
        3. Simple PDF fallback
    """
    markdown_content = create_markdown_content_in_memory(message)
    try:
        return _create_pdf_reportlab_in_memory(markdown_content)
    except ImportError:
        try:
            return _create_pdf_fpdf_in_memory(markdown_content)
        except ImportError:
            return _create_pdf_simple_in_memory(markdown_content)

def _create_pdf_reportlab_in_memory(markdown_content: str) -> bytes:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from io import BytesIO
    import re

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        spaceBefore=10,
    )
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=15,
        spaceBefore=15,
    )
    reference_style = ParagraphStyle(
        'Reference',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=5,
    )

    story = []
    lines = markdown_content.split('\n')
    current_paragraph = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if current_paragraph:
                para_text = ' '.join(current_paragraph)
                para_text = re.sub(r'\[\[(\d+)\]\]\(([^)]+)\)', r'<link href="\2" color="blue"><u>[\1]</u></link>', para_text)
                para_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', para_text)
                para_text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', para_text)
                story.append(Paragraph(para_text, styles['Normal']))
                current_paragraph = []
            story.append(Spacer(1, 12))
            continue
        if line_stripped.startswith('## '):
            if current_paragraph:
                para_text = ' '.join(current_paragraph)
                para_text = re.sub(r'\[\[(\d+)\]\]\(([^)]+)\)', r'<link href="\2" color="blue"><u>[\1]</u></link>', para_text)
                para_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', para_text)
                para_text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', para_text)
                story.append(Paragraph(para_text, styles['Normal']))
                current_paragraph = []
            header_text = line_stripped[3:]
            story.append(Paragraph(header_text, header_style))
            continue
        elif line_stripped.startswith('# '):
            if current_paragraph:
                para_text = ' '.join(current_paragraph)
                para_text = re.sub(r'\[\[(\d+)\]\]\(([^)]+)\)', r'<link href="\2" color="blue"><u>[\1]</u></link>', para_text)
                para_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', para_text)
                para_text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', para_text)
                story.append(Paragraph(para_text, styles['Normal']))
                current_paragraph = []
            title_text = line_stripped[2:]
            story.append(Paragraph(title_text, title_style))
            continue
        elif re.match(r'^\d+\.\s', line_stripped):
            if current_paragraph:
                para_text = ' '.join(current_paragraph)
                para_text = re.sub(r'\[\[(\d+)\]\]\(([^)]+)\)', r'<link href="\2" color="blue"><u>[\1]</u></link>', para_text)
                para_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', para_text)
                para_text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', para_text)
                story.append(Paragraph(para_text, styles['Normal']))
                current_paragraph = []
            ref_text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<link href="\2" color="blue"><u>\1</u></link>', line_stripped)
            story.append(Paragraph(ref_text, reference_style))
            continue
        else:
            current_paragraph.append(line_stripped)
    if current_paragraph:
        para_text = ' '.join(current_paragraph)
        para_text = re.sub(r'\[\[(\d+)\]\]\(([^)]+)\)', r'<link href="\2" color="blue"><u>[\1]</u></link>', para_text)
        para_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', para_text)
        para_text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', para_text)
        story.append(Paragraph(para_text, styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

def _create_pdf_fpdf_in_memory(markdown_content: str) -> bytes:
    """
    Create PDF Using FPDF2
    
    Creates a PDF document using FPDF2 library with basic markdown parsing
    and text formatting. This is the fallback option when ReportLab is unavailable.
    
    Args:
        markdown_content (str): Markdown content with citations and references
        
    Returns:
        bytes: PDF document binary data
        
    Features:
        - Basic markdown parsing (headers, bold text)
        - Text citations (non-clickable)
        - Multi-cell text wrapping
        - Different font sizes for headers and content
        - Auto page breaks
        
    Raises:
        ImportError: If FPDF2 library is not available
    """
    from fpdf import FPDF
    from io import BytesIO
    import re
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Process content line by line
    lines = markdown_content.split('\n')
    current_paragraph = []
    
    for line in lines:
        line_stripped = line.strip()
        
        if not line_stripped:
            # Empty line - finish current paragraph and add space
            if current_paragraph:
                para_text = ' '.join(current_paragraph)
                # Convert markdown citations to simple text
                para_text = re.sub(r'\[\[(\d+)\]\]\(([^)]+)\)', r'[\1]', para_text)
                
                pdf.set_font('Arial', size=11)
                # Handle text wrapping
                pdf.multi_cell(0, 6, para_text)
                current_paragraph = []
            pdf.ln(3)
            continue
            
        if line_stripped.startswith('## '):
            # Header - finish current paragraph first
            if current_paragraph:
                para_text = ' '.join(current_paragraph)
                para_text = re.sub(r'\[\[(\d+)\]\]\(([^)]+)\)', r'[\1]', para_text)
                pdf.set_font('Arial', size=11)
                pdf.multi_cell(0, 6, para_text)
                current_paragraph = []
            
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 8, line_stripped[3:], ln=True)
            pdf.ln(3)
            continue
            
        elif line_stripped.startswith('# '):
            # Main title - finish current paragraph first
            if current_paragraph:
                para_text = ' '.join(current_paragraph)
                para_text = re.sub(r'\[\[(\d+)\]\]\(([^)]+)\)', r'[\1]', para_text)
                pdf.set_font('Arial', size=11)
                pdf.multi_cell(0, 6, para_text)
                current_paragraph = []
            
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, line_stripped[2:], ln=True)
            pdf.ln(5)
            continue
            
        elif re.match(r'^\d+\.\s', line_stripped):
            # Reference list item - finish current paragraph first
            if current_paragraph:
                para_text = ' '.join(current_paragraph)
                para_text = re.sub(r'\[\[(\d+)\]\]\(([^)]+)\)', r'[\1]', para_text)
                pdf.set_font('Arial', size=11)
                pdf.multi_cell(0, 6, para_text)
                current_paragraph = []
            
            # Convert markdown links in references to readable format
            ref_text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', line_stripped)
            pdf.set_font('Arial', size=10)
            pdf.multi_cell(0, 5, ref_text)
            continue
        
        else:
            # Regular text - add to current paragraph
            current_paragraph.append(line_stripped)
    
    # Don't forget the last paragraph
    if current_paragraph:
        para_text = ' '.join(current_paragraph)
        para_text = re.sub(r'\[\[(\d+)\]\]\(([^)]+)\)', r'[\1]', para_text)
        pdf.set_font('Arial', size=11)
        pdf.multi_cell(0, 6, para_text)
    
    # Get PDF bytes
    buffer = BytesIO()
    pdf_string = pdf.output(dest='S').encode('latin1')
    buffer.write(pdf_string)
    buffer.seek(0)
    return buffer.read()

def _create_pdf_simple_in_memory(markdown_content: str) -> bytes:
    """
    Create Simple PDF In Memory
    
    Creates a minimal PDF document using basic PDF structure when no PDF libraries
    are available. This is the final fallback option with very limited formatting.
    
    Args:
        markdown_content (str): Markdown content to include in PDF
        
    Returns:
        bytes: Minimal PDF document binary data
        
    Features:
        - Basic PDF 1.4 structure
        - Plain text content only
        - No formatting or clickable links
        - Single page layout
        
    Note:
        This is a last resort fallback and provides minimal PDF functionality.
        Content may not display properly in all PDF viewers.
    """
    from io import BytesIO
    
    # Very simple PDF creation (minimal PDF structure)
    buffer = BytesIO()
    
    # Simple PDF header and content (this is a minimal approach)
    simple_pdf = f"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length {len(markdown_content) + 50}
>>
stream
BT
/F1 12 Tf
50 750 Td
({markdown_content.replace(chr(10), ') Tj T* (')}) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000199 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
{300 + len(markdown_content)}
%%EOF"""
    
    buffer.write(simple_pdf.encode('utf-8'))
    buffer.seek(0)
    return buffer.read()

@app.get("/deep_research/{thread_id}/{run_id}/download")
async def download_research_report(thread_id: str, run_id: str, format: str = "md"):
    """
    Download Research Report
    
    Downloads the completed research report in either Markdown or PDF format.
    The report includes all research findings with properly formatted citations
    and references.
    
    Args:
        thread_id (str): The unique thread identifier for the research session
        run_id (str): The unique run identifier for the specific research job
        format (str): Output format - either "md" for Markdown or "pdf" for PDF (default: "md")
        
    Returns:
        dict: Download response containing:
            - filename: Suggested filename for the report
            - format: The requested format
            - content: Report content (text for MD, base64 for PDF)
            - thread_id: The source thread ID
            - encoding: "base64" for PDF files
            
    Raises:
        HTTPException: 
            - 400 if format is not "md" or "pdf"
            - 404 if no agent message found in thread
            - 500 if report generation fails
        
    Example Response (MD):
        ```json
        {
            "filename": "research_report_thread_123.md",
            "format": "md",
            "content": "# Research Report\n\nContent here...",
            "thread_id": "thread_123"
        }
        ```
    """
    
    if format not in ["md", "pdf"]:
        raise HTTPException(status_code=400, detail="Format must be 'md' or 'pdf'")
    
    try:
        project_client = AIProjectClient(
            endpoint=os.environ["PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential(),
        )
        
        from azure.ai.agents.models import MessageRole, ListSortOrder
        all_messages = project_client.agents.messages.list(
            thread_id=thread_id,
            order=ListSortOrder.DESCENDING
        )
        last_agent_message = next((m for m in all_messages if m.role == MessageRole.AGENT), None)
        if not last_agent_message:
            raise HTTPException(status_code=404, detail="No agent message found in thread")
        
        if format == "md":
            markdown_content = create_markdown_content_in_memory(last_agent_message)
            return {
                "filename": f"research_report_{thread_id}.md",
                "format": format,
                "content": markdown_content,
                "thread_id": thread_id
            }
        else:
            import base64
            pdf_content = create_pdf_content_in_memory(last_agent_message)
            pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
            return {
                "filename": f"research_report_{thread_id}.pdf",
                "format": format,
                "content": pdf_base64,
                "thread_id": thread_id,
                "encoding": "base64"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")