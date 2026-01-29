#!/usr/bin/env python3
"""
FastAPI Server for RKLLM Vision Language Model Service
Provides OpenAI-compatible API for multimodal inference on RK3588 platform
Supports complete OpenAI API parameters including temperature, top_p, max_tokens, etc.
"""

import ctypes
import os
import sys
import time
import uuid
import json
import asyncio
import logging
import threading
import argparse
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Union, Generator
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==================== System Library Preloading ====================
def preload_libraries():
    """Preload necessary system libraries to fix OpenCL issues"""
    try:
        # Set environment variables
        os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libmali.so.1:' + os.environ.get('LD_PRELOAD', '')
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/aarch64-linux-gnu:/usr/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        
        # Preload libraries
        libs = [
            'libmali.so.1',
            'libOpenCL.so',
            'librknnrt.so',
            '/usr/lib/librkllmrt.so'
        ]
        
        for lib in libs:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                print(f"✓ Preloaded: {lib}")
            except Exception as e:
                print(f"⚠ Failed to preload {lib}: {e}")
    except Exception as e:
        print(f"⚠ Error during library preloading: {e}")

print("Preloading system libraries...")
preload_libraries()

# ==================== RKLLM Service Wrapper ====================
class RKLLMService:
    """Wrapper for RKLLM Vision Language Model service"""
    
    def __init__(self, library_path: str = "/usr/lib/librkllm_service.so"):
        """
        Initialize RKLLM service wrapper.
        
        Args:
            library_path: Path to the shared library
        """
        try:
            self.lib = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
            print(f"✓ Successfully loaded {library_path}")
        except Exception as e:
            print(f"✗ Failed to load {library_path}: {e}")
            print("Please ensure RKLLM service library is built and available")
            sys.exit(1)
        
        # Define function signatures
        self.lib.create_service.restype = ctypes.c_void_p
        self.lib.create_service.argtypes = []
        
        self.lib.initialize_service.restype = ctypes.c_int
        self.lib.initialize_service.argtypes = [
            ctypes.c_void_p,  # ctx
            ctypes.c_char_p,  # image_path
            ctypes.c_char_p,  # encoder_model_path
            ctypes.c_char_p,  # llm_model_path
            ctypes.c_int,     # max_new_tokens
            ctypes.c_int,     # max_context_len
            ctypes.c_int,     # rknn_core_num
            ctypes.c_char_p,  # img_start
            ctypes.c_char_p,  # img_end
            ctypes.c_char_p,  # img_content
        ]
        
        self.lib.generate_response.restype = ctypes.c_char_p
        self.lib.generate_response.argtypes = [
            ctypes.c_void_p,  # ctx
            ctypes.c_char_p,  # prompt
        ]

        self.lib.generate_response_with_params.restype = ctypes.c_char_p
        self.lib.generate_response_with_params.argtypes = [
            ctypes.c_void_p,  # ctx
            ctypes.c_char_p,  # prompt
            ctypes.c_int,     # top_k
            ctypes.c_float,   # top_p
            ctypes.c_float,   # temperature
        ]

        self.lib.cleanup_service.argtypes = [ctypes.c_void_p]
        self.lib.destroy_service.argtypes = [ctypes.c_void_p]

        self.lib.simple_inference.restype = ctypes.c_char_p
        self.lib.simple_inference.argtypes = [
            ctypes.c_char_p,  # image_path
            ctypes.c_char_p,  # encoder_model_path
            ctypes.c_char_p,  # llm_model_path
            ctypes.c_char_p,  # prompt
            ctypes.c_int,     # max_new_tokens
            ctypes.c_int,     # max_context_len
            ctypes.c_int,     # rknn_core_num
        ]

        # Add function signatures for runtime parameter updates
        self.lib.update_runtime_params.restype = ctypes.c_int
        self.lib.update_runtime_params.argtypes = [
            ctypes.c_void_p,  # ctx
            ctypes.c_int,     # max_new_tokens
            ctypes.c_int,     # max_context_len
            ctypes.c_int,     # rknn_core_num
        ]

        self.lib.get_runtime_params.argtypes = [
            ctypes.c_void_p,  # ctx
            ctypes.POINTER(ctypes.c_int),  # max_new_tokens
            ctypes.POINTER(ctypes.c_int),  # max_context_len
            ctypes.POINTER(ctypes.c_int),  # rknn_core_num
        ]

        self.ctx = None
        self.lock = threading.Lock()
        
    def initialize(self, 
                   image_path: str,
                   encoder_model_path: str,
                   llm_model_path: str,
                   max_new_tokens: int = 128,
                   max_context_len: int = 2048,
                   rknn_core_num: int = 1,
                   img_start: Optional[str] = None,
                   img_end: Optional[str] = None,
                   img_content: Optional[str] = None) -> bool:
        """
        Initialize the service with models.
        """
        with self.lock:
            if self.ctx is None:
                self.ctx = self.lib.create_service()
                if not self.ctx:
                    return False
            
            # Validate file existence
            for path, name in [(image_path, "image"), 
                              (encoder_model_path, "encoder model"),
                              (llm_model_path, "llm model")]:
                if not Path(path).exists():
                    print(f"❌ {name} file not found: {path}")
                    return False
            
            img_start_bytes = img_start.encode() if img_start else None
            img_end_bytes = img_end.encode() if img_end else None
            img_content_bytes = img_content.encode() if img_content else None
            
            ret = self.lib.initialize_service(
                self.ctx,
                image_path.encode(),
                encoder_model_path.encode(),
                llm_model_path.encode(),
                max_new_tokens,
                max_context_len,
                rknn_core_num,
                img_start_bytes,
                img_end_bytes,
                img_content_bytes
            )
            
            return ret == 0
    
    def generate(self, prompt: str, top_k: int = None, top_p: float = None, temperature: float = None) -> str:
        """
        Generate response for a prompt.
        """
        with self.lock:
            if self.ctx is None:
                return "Error: Service not initialized"

            # If any parameter is specified, use the parameterized version
            if top_k is not None or top_p is not None or temperature is not None:
                # Use default values if not specified
                if top_k is None: top_k = 1
                if top_p is None: top_p = 1.0
                if temperature is None: temperature = 0.7

                result = self.lib.generate_response_with_params(
                    self.ctx, prompt.encode(), top_k, top_p, temperature
                )
            else:
                result = self.lib.generate_response(self.ctx, prompt.encode())

            if result:
                return result.decode('utf-8', errors='ignore')
            return ""
    
    def simple_inference(self,
                        image_path: str,
                        encoder_model_path: str,
                        llm_model_path: str,
                        prompt: str,
                        max_new_tokens: int = 128,
                        max_context_len: int = 2048,
                        rknn_core_num: int = 1) -> str:
        """
        One-time inference without maintaining context.
        """
        result = self.lib.simple_inference(
            image_path.encode(),
            encoder_model_path.encode(),
            llm_model_path.encode(),
            prompt.encode(),
            max_new_tokens,
            max_context_len,
            rknn_core_num
        )
        
        if result:
            return result.decode('utf-8', errors='ignore')
        return ""
    
    def cleanup(self):
        """Clean up resources."""
        with self.lock:
            if self.ctx:
                self.lib.cleanup_service(self.ctx)
    
    def update_runtime_params(self, max_new_tokens: int = None, max_context_len: int = None, rknn_core_num: int = None) -> bool:
        """
        Update runtime parameters.
        """
        with self.lock:
            if self.ctx is None:
                return False

            # Use current values if not specified
            if max_new_tokens is None:
                max_new_tokens = 128
            if max_context_len is None:
                max_context_len = 2048
            if rknn_core_num is None:
                rknn_core_num = 1

            ret = self.lib.update_runtime_params(
                self.ctx,
                max_new_tokens,
                max_context_len,
                rknn_core_num
            )

            return ret == 0

    def get_runtime_params(self) -> Dict[str, int]:
        """
        Get current runtime parameters.
        """
        with self.lock:
            if self.ctx is None:
                return {"max_new_tokens": 128, "max_context_len": 2048, "rknn_core_num": 1}

            # Create ctypes variables to receive the values
            max_new_tokens = ctypes.c_int()
            max_context_len = ctypes.c_int()
            rknn_core_num = ctypes.c_int()

            self.lib.get_runtime_params(
                self.ctx,
                ctypes.byref(max_new_tokens),
                ctypes.byref(max_context_len),
                ctypes.byref(rknn_core_num)
            )

            return {
                "max_new_tokens": max_new_tokens.value,
                "max_context_len": max_context_len.value,
                "rknn_core_num": rknn_core_num.value
            }

    def __del__(self):
        """Destructor to clean up resources."""
        if self.ctx:
            self.lib.destroy_service(self.ctx)
            self.ctx = None

# ==================== Pydantic Model Definitions ====================
class ChatMessageContentItem(BaseModel):
    type: str = Field(..., description="Content type: text or image_url")
    text: Optional[str] = Field(None, description="Text content")
    image_url: Optional[str] = Field(None, description="Image URL")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, assistant")
    content: Union[str, List[ChatMessageContentItem]] = Field(..., description="Message content")

class Function(BaseModel):
    name: str = Field(..., description="Function name")
    description: Optional[str] = Field(None, description="Function description")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Function parameters")

class Tool(BaseModel):
    type: str = Field(default="function", description="Tool type")
    function: Optional[Function] = Field(None, description="Function definition")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="rkllm-vision", description="Model name")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Temperature parameter (0.0-2.0)")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter (0.0-1.0)")
    top_k: Optional[int] = Field(default=1, ge=1, le=100, description="Top-k sampling parameter (1-100)")
    n: Optional[int] = Field(default=1, ge=1, le=10, description="Number of completions to generate")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(default=128, ge=1, le=4096, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    user: Optional[str] = Field(None, description="User identifier")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    tools: Optional[List[Tool]] = Field(None, description="List of tools")
    tool_choice: Optional[str] = Field(None, description="Tool choice")
    max_context_len: Optional[int] = Field(default=2048, ge=512, le=8192, description="Maximum context length")
    rknn_core_num: Optional[int] = Field(default=1, ge=1, le=4, description="Number of RKNN cores to use")

class UsageInfo(BaseModel):
    prompt_tokens: int = Field(default=0, description="Prompt tokens")
    completion_tokens: int = Field(default=0, description="Completion tokens")
    total_tokens: int = Field(default=0, description="Total tokens")

class ChatCompletionResponseChoice(BaseModel):
    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Message")
    finish_reason: Optional[str] = Field(default="stop", description="Finish reason")

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Request ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[ChatCompletionResponseChoice] = Field(..., description="List of choices")
    usage: UsageInfo = Field(..., description="Usage information")
    system_fingerprint: Optional[str] = Field(default="fp_rkllm_vision", description="System fingerprint")

class DeltaMessage(BaseModel):
    role: Optional[str] = Field(None, description="Role")
    content: Optional[str] = Field(None, description="Content")

class ChatCompletionStreamResponseChoice(BaseModel):
    index: int = Field(..., description="Choice index")
    delta: DeltaMessage = Field(..., description="Delta message")
    finish_reason: Optional[str] = Field(None, description="Finish reason")

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(..., description="Request ID")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[ChatCompletionStreamResponseChoice] = Field(..., description="List of choices")
    system_fingerprint: Optional[str] = Field(default="fp_rkllm_vision", description="System fingerprint")

class ModelInfo(BaseModel):
    id: str = Field(..., description="Model ID")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Creation time")
    owned_by: str = Field(default="rockchip", description="Owner")

class ModelsListResponse(BaseModel):
    object: str = Field(default="list", description="Object type")
    data: List[ModelInfo] = Field(..., description="List of models")

class RuntimeParams(BaseModel):
    max_new_tokens: int = Field(default=128, description="Maximum new tokens")
    max_context_len: int = Field(default=2048, description="Maximum context length")
    rknn_core_num: int = Field(default=1, description="Number of RKNN cores")

# ==================== Global State Management ====================
class RequestState:
    """State management for individual requests"""
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.response_text = ""
        self.completed = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.start_time = time.time()

# Server configuration
class ServerConfig:
    def __init__(self):
        self.max_context_len = 2048
        self.default_temperature = 0.7
        self.default_top_p = 1.0
        self.default_top_k = 1
        self.default_max_tokens = 128
        self.max_concurrent_requests = 2
        self.timeout_seconds = 120
        self.rknn_core_num = 1
        self.image_path = ""
        self.encoder_model_path = ""
        self.llm_model_path = ""
        self.img_start = "<|image_start|>"
        self.img_end = "<|image_end|>"
        self.img_content = "<|image_content|>"

config = ServerConfig()

# Global variables
request_lock = threading.Lock()
active_requests = 0
request_states: Dict[str, RequestState] = {}
rkllm_service = None
executor = None

# ==================== Helper Functions ====================
def extract_user_prompt(messages: List[ChatMessage]) -> str:
    """Extract user prompt from messages"""
    for msg in reversed(messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                return msg.content
            elif isinstance(msg.content, list):
                # Handle multimodal content
                for item in msg.content:
                    if item.type == "text" and item.text:
                        return item.text
    return ""

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)"""
    if not text:
        return 0
    
    # Simple estimation: Chinese characters ~1.5 tokens, others ~0.3 tokens
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    
    return int(chinese_chars * 1.5 + other_chars * 0.3)

def process_chat_completion(request: ChatCompletionRequest, request_id: str) -> RequestState:
    """Process chat completion request"""
    global rkllm_service
    
    # Create request state
    req_state = RequestState(request_id)
    request_states[request_id] = req_state
    
    try:
        # Extract user prompt
        prompt = extract_user_prompt(request.messages)
        if not prompt:
            req_state.error = "No user message found"
            req_state.completed.set()
            return req_state
        
        # Update runtime parameters if needed
        if (request.max_tokens != config.default_max_tokens or 
            request.max_context_len != config.max_context_len or
            request.rknn_core_num != config.rknn_core_num):
            
            success = rkllm_service.update_runtime_params(
                max_new_tokens=request.max_tokens,
                max_context_len=request.max_context_len,
                rknn_core_num=request.rknn_core_num
            )
            if not success:
                print(f"[{request_id}] Warning: Failed to update runtime parameters")
        
        # Print debug information
        print(f"[{request_id}] Processing request:")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Temperature: {request.temperature}")
        print(f"  Top-p: {request.top_p}")
        print(f"  Top-k: {request.top_k}")
        print(f"  Max tokens: {request.max_tokens}")
        
        # Generate response
        response_text = rkllm_service.generate(
            prompt=prompt,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature
        )
        
        req_state.response_text = response_text
        req_state.completed.set()
        
        elapsed = time.time() - req_state.start_time
        print(f"✅ [{request_id}] Inference completed in {elapsed:.2f}s")
        
        return req_state
        
    except Exception as e:
        error_msg = f"Error processing request {request_id}: {str(e)}"
        print(f"✗ {error_msg}")
        req_state.error = error_msg
        req_state.completed.set()
        return req_state

# ==================== Application Lifecycle Management ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    global rkllm_service, executor
    
    # Startup
    print("=" * 60)
    print("Starting RKLLM Vision API Server")
    print("=" * 60)
    
    # Initialize thread pool
    executor = ThreadPoolExecutor(
        max_workers=config.max_concurrent_requests + 2,
        thread_name_prefix="rkllm_vision_worker"
    )
    print("✅ Thread pool initialized")
    
    # Initialize RKLLM service
    try:
        # Load RKLLM service library
        lib_path = os.getenv("RKLLM_SERVICE_LIB", "/usr/lib/librkllm_service.so")
        
        # Initialize service
        rkllm_service = RKLLMService(lib_path)
        
        # Apply configuration
        success = rkllm_service.initialize(
            image_path=config.image_path,
            encoder_model_path=config.encoder_model_path,
            llm_model_path=config.llm_model_path,
            max_new_tokens=config.default_max_tokens,
            max_context_len=config.max_context_len,
            rknn_core_num=config.rknn_core_num,
            img_start=config.img_start,
            img_end=config.img_end,
            img_content=config.img_content
        )
        
        if not success:
            raise RuntimeError("Failed to initialize RKLLM vision service")
        
        print("✅ RKLLM vision service initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        print("Please check:")
        print("1. Model files exist and are accessible")
        print("2. RKLLM service library is built")
        print("3. OpenCL drivers are installed")
        raise
    
    yield
    
    # Shutdown
    print("\nShutting down server...")
    
    # Clean up request states
    request_states.clear()
    
    # Shutdown thread pool
    if executor:
        executor.shutdown(wait=False)
        print("✅ Thread pool shut down")
    
    # Release service
    if rkllm_service:
        rkllm_service.cleanup()
        print("✅ Service resources released")

# ==================== FastAPI Application ====================
app = FastAPI(
    title="RKLLM Vision API Server",
    version="1.0.0",
    description="OpenAI API compatible server for RKLLM Vision Language Models",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== API Endpoints ====================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RKLLM Vision API Server",
        "status": "running",
        "vision_model": config.encoder_model_path.split('/')[-1],
        "llm_model": config.llm_model_path.split('/')[-1],
        "image": config.image_path.split('/')[-1],
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Server information",
            "GET /health": "Health check",
            "GET /v1/models": "List models",
            "POST /v1/chat/completions": "Chat completion",
            "GET /v1/runtime_params": "Get runtime parameters",
            "POST /v1/runtime_params": "Update runtime parameters"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if rkllm_service else "unhealthy",
        "service_initialized": rkllm_service is not None,
        "active_requests": active_requests,
        "max_concurrent": config.max_concurrent_requests,
        "timestamp": int(time.time())
    }

@app.get("/v1/models", response_model=ModelsListResponse)
async def list_models():
    """List available models"""
    return ModelsListResponse(
        data=[
            ModelInfo(
                id="rkllm-vision",
                created=int(time.time()),
                owned_by="rockchip"
            ),
            ModelInfo(
                id="rkllm-vision-2b",
                created=int(time.time()),
                owned_by="rockchip"
            )
        ]
    )

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion - Fully OpenAI API compatible"""
    global active_requests
    
    # Check concurrent request limit
    with request_lock:
        if active_requests >= config.max_concurrent_requests:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": "Too many requests, please try again later",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                }
            )
        active_requests += 1
    
    try:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        
        print(f"[{request_id}] New request: stream={request.stream}, messages={len(request.messages)}")
        
        if request.stream:
            # Streaming response (simulated since RKLLM service doesn't support true streaming)
            async def generate_stream():
                nonlocal request_id
                
                try:
                    # Submit task to thread pool
                    future = executor.submit(process_chat_completion, request, request_id)
                    
                    # Send initial message
                    initial_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=DeltaMessage(role="assistant"),
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {initial_chunk.model_dump_json(exclude_unset=True, ensure_ascii=False)}\n\n"
                    
                    # Wait for completion
                    req_state = None
                    start_time = time.time()
                    
                    while True:
                        if request_id in request_states:
                            req_state = request_states[request_id]
                            
                            if req_state.completed.is_set():
                                break
                        
                        # Check timeout
                        if time.time() - start_time > 30:  # 30 seconds timeout
                            print(f"[{request_id}] Streaming response timeout")
                            break
                        
                        # Brief wait
                        await asyncio.sleep(0.1)
                    
                    if req_state and req_state.completed.is_set():
                        if req_state.error:
                            error_data = {
                                "error": {
                                    "message": req_state.error,
                                    "type": "server_error"
                                }
                            }
                            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                        else:
                            # Send content in chunks (simulated streaming)
                            content = req_state.response_text
                            chunk_size = 20
                            for i in range(0, len(content), chunk_size):
                                chunk = content[i:i+chunk_size]
                                data_chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    created=created,
                                    model=request.model,
                                    choices=[
                                        ChatCompletionStreamResponseChoice(
                                            index=0,
                                            delta=DeltaMessage(content=chunk),
                                            finish_reason=None
                                        )
                                    ]
                                )
                                yield f"data: {data_chunk.model_dump_json(exclude_unset=True, ensure_ascii=False)}\n\n"
                                await asyncio.sleep(0.05)  # Simulate streaming delay
                    
                    # Send completion marker
                    done_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=DeltaMessage(),
                                finish_reason="stop"
                            )
                        ]
                    )
                    yield f"data: {done_chunk.model_dump_json(exclude_unset=True, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    print(f"[{request_id}] Stream generation error: {e}")
                    error_data = {
                        "error": {
                            "message": str(e),
                            "type": "server_error"
                        }
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                finally:
                    # Clean up request state
                    if request_id in request_states:
                        del request_states[request_id]
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
            
        else:
            # Non-streaming response
            def process_non_stream():
                nonlocal request_id
                
                try:
                    req_state = process_chat_completion(request, request_id)
                    
                    if req_state.error:
                        raise HTTPException(status_code=500, detail=req_state.error)
                    
                    # Estimate token usage
                    prompt = extract_user_prompt(request.messages)
                    prompt_tokens = estimate_tokens(prompt)
                    completion_tokens = estimate_tokens(req_state.response_text)
                    
                    # Build response
                    response = ChatCompletionResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionResponseChoice(
                                index=0,
                                message=ChatMessage(
                                    role="assistant",
                                    content=req_state.response_text
                                ),
                                finish_reason="stop"
                            )
                        ],
                        usage=UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens
                        )
                    )
                    
                    return response
                finally:
                    # Clean up request state
                    if request_id in request_states:
                        del request_states[request_id]
            
            # Execute in thread pool
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    executor, process_non_stream
                )
                return response
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        with request_lock:
            active_requests -= 1

@app.get("/v1/runtime_params")
async def get_runtime_params():
    """Get current runtime parameters."""
    if not rkllm_service:
        raise HTTPException(status_code=500, detail="RKLLM service not initialized")
    
    try:
        params = rkllm_service.get_runtime_params()
        return {"runtime_params": params}
    except Exception as e:
        print(f"Error getting runtime parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/runtime_params")
async def update_runtime_params(
    max_new_tokens: Optional[int] = None,
    max_context_len: Optional[int] = None,
    rknn_core_num: Optional[int] = None
):
    """Update runtime parameters."""
    if not rkllm_service:
        raise HTTPException(status_code=500, detail="RKLLM service not initialized")
    
    try:
        success = rkllm_service.update_runtime_params(
            max_new_tokens=max_new_tokens,
            max_context_len=max_context_len,
            rknn_core_num=rknn_core_num
        )

        if success:
            params = rkllm_service.get_runtime_params()
            return {"success": True, "updated_params": params}
        else:
            raise HTTPException(status_code=500, detail="Failed to update runtime parameters")
    except Exception as e:
        print(f"Error updating runtime parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Main Program ====================
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="RKLLM Vision API Compatible Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rkllm_vision_server.py --image_path ../data/demo.jpg \\
                                --encoder_model ../model/Qwen2-VL-2B_vision_rk3588.rknn \\
                                --llm_model ../model/Qwen2-VL-2B_llm_w8a8_rk3588.rkllm \\
                                --target_platform rk3588
  
  python rkllm_vision_server.py --image_path ../data/demo.jpg \\
                                --encoder_model ../model/vision.rknn \\
                                --llm_model ../model/llm.rkllm \\
                                --port 8080 --max_concurrent 2 --default_temperature 0.7
        """
    )
    
    # Required arguments
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image file')
    parser.add_argument('--encoder_model', type=str, required=True,
                       help='Path to vision encoder model (.rknn)')
    parser.add_argument('--llm_model', type=str, required=True,
                       help='Path to LLM model (.rkllm)')
    
    # Server parameters
    parser.add_argument('--port', type=int, default=8001,
                       help='Server port (default: 8001)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Server host (default: 0.0.0.0)')
    
    # Model parameters
    parser.add_argument('--max_context_len', type=int, default=2048,
                       help='Maximum context length (default: 2048)')
    parser.add_argument('--default_temperature', type=float, default=0.7,
                       help='Default temperature parameter (default: 0.7)')
    parser.add_argument('--default_top_p', type=float, default=1.0,
                       help='Default top_p parameter (default: 1.0)')
    parser.add_argument('--default_top_k', type=int, default=1,
                       help='Default top_k parameter (default: 1, range: 1-100)')
    parser.add_argument('--default_max_tokens', type=int, default=128,
                       help='Default maximum tokens to generate (default: 128)')
    
    # Performance parameters
    parser.add_argument('--max_concurrent', type=int, default=2,
                       help='Maximum concurrent requests (default: 2)')
    parser.add_argument('--timeout', type=int, default=120,
                       help='Request timeout in seconds (default: 120)')
    parser.add_argument('--rknn_core_num', type=int, default=1,
                       help='Number of RKNN cores to use (default: 1)')
    
    # Image token parameters
    parser.add_argument('--img_start', type=str, default='<|image_start|>',
                       help='Image start token (default: <|image_start|>)')
    parser.add_argument('--img_end', type=str, default='<|image_end|>',
                       help='Image end token (default: <|image_end|>)')
    parser.add_argument('--img_content', type=str, default='<|image_content|>',
                       help='Image content token (default: <|image_content|>)')
    
    # Debug parameters
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Validate model files
    for path, name in [(args.image_path, "image"), 
                      (args.encoder_model, "encoder model"),
                      (args.llm_model, "LLM model")]:
        if not os.path.exists(path):
            print(f"❌ Error: {name} file not found: {path}")
            sys.exit(1)
    
    # Validate top_k range
    if args.default_top_k < 1 or args.default_top_k > 100:
        print(f"⚠ Warning: default_top_k should be between 1 and 100, got {args.default_top_k}")
        args.default_top_k = max(1, min(100, args.default_top_k))
        print(f"  Adjusted to: {args.default_top_k}")
    
    # Apply configuration
    config.image_path = os.path.abspath(args.image_path)
    config.encoder_model_path = os.path.abspath(args.encoder_model)
    config.llm_model_path = os.path.abspath(args.llm_model)
    config.max_context_len = args.max_context_len
    config.default_temperature = args.default_temperature
    config.default_top_p = args.default_top_p
    config.default_top_k = args.default_top_k
    config.default_max_tokens = args.default_max_tokens
    config.max_concurrent_requests = args.max_concurrent
    config.timeout_seconds = args.timeout
    config.rknn_core_num = args.rknn_core_num
    config.img_start = args.img_start
    config.img_end = args.img_end
    config.img_content = args.img_content
    
    # Print configuration
    print("=" * 60)
    print("Configuration:")
    print(f"  Image: {args.image_path}")
    print(f"  Vision encoder: {args.encoder_model}")
    print(f"  LLM model: {args.llm_model}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Max context length: {config.max_context_len}")
    print(f"  Default temperature: {config.default_temperature}")
    print(f"  Default Top-p: {config.default_top_p}")
    print(f"  Default Top-k: {config.default_top_k}")
    print(f"  Default max tokens: {config.default_max_tokens}")
    print(f"  RKNN cores: {config.rknn_core_num}")
    print(f"  Max concurrent requests: {config.max_concurrent_requests}")
    print(f"  Request timeout: {config.timeout_seconds}s")
    print("=" * 60)
    
    # Start server
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info" if not args.debug else "debug",
            access_log=True,
            timeout_keep_alive=30,
            server_header=False
        )
    except KeyboardInterrupt:
        print("\n👋 Server interrupted by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)