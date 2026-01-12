import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import argparse
import json
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# --- RKLLM 结构体与常量定义 ---
rkllm_lib = ctypes.CDLL('/usr/lib/librkllmrt.so')
RKLLM_Handle_t = ctypes.c_void_p

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL  = 0
LLMCallState.RKLLM_RUN_WAITING  = 1
LLMCallState.RKLLM_RUN_FINISH  = 2
LLMCallState.RKLLM_RUN_ERROR   = 3

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT = 0

class RKLLMExtendParam(ctypes.Structure):
    _pack_ = 1 # 强制字节对齐，防止 NPU 读取错位
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p)]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("reserved", ctypes.c_uint8 * 128) # 简化处理，只保留核心text
    ]

# --- 全局状态 ---
lock = threading.Lock()
global_text = []
global_state = -1

def callback_impl(result, userdata, state):
    global global_text, global_state
    if state == LLMCallState.RKLLM_RUN_NORMAL:
        global_text.append(result.contents.text.decode('utf-8'))
    global_state = state
    return 0

callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback_func = callback_type(callback_impl)

class RKLLMModel:
    def __init__(self, model_path, platform="rk3588"):
        param = RKLLMParam()
        # ！！！非常重要：先清空整个结构体的内存 ！！！
        ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(param))
        
        param.model_path = bytes(model_path, 'utf-8')
        param.max_context_len = 4096
        param.max_new_tokens = 4096
        param.top_k = 1
        param.top_p = 0.9
        param.temperature = 0.8
        
        # ！！！显式设置 extend_param ！！！
        param.extend_param.n_batch = 1
        param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1
        param.extend_param.enabled_cpus_num = 4
        param.extend_param.enabled_cpus_mask = 0xf0 
        
        self.handle = RKLLM_Handle_t()
        print(f"DEBUG: n_batch is set to {param.extend_param.n_batch}") # 添加调试打印
        
        ret = rkllm_lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(param), callback_func)
        if ret != 0: 
            raise Exception(f"Init Failed with error code: {ret}")

    def run(self, prompt):
        rk_input = RKLLMInput()
        rk_input.role = b"user"
        rk_input.enable_thinking = True
        rk_input.input_type = 0
        rk_input.input_data.prompt_input = prompt.encode('utf-8')
        rkllm_lib.rkllm_run(self.handle, ctypes.byref(rk_input), ctypes.byref(self.infer_param), None)

# --- Flask API 路由 ---
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    global global_text, global_state
    if not lock.acquire(blocking=False):
        return jsonify({"error": "Server busy"}), 503
    
    try:
        data = request.json
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        prompt = messages[-1]['content'] if messages else ""

        global_text = []
        global_state = -1
        
        run_thread = threading.Thread(target=rkllm_model.run, args=(prompt,))
        run_thread.start()

        if not stream:
            while global_state != LLMCallState.RKLLM_RUN_FINISH:
                time.sleep(0.1)
            full_content = "".join(global_text)
            return jsonify({"choices": [{"message": {"content": full_content}}]})
        else:
            def generate():
                while global_state != LLMCallState.RKLLM_RUN_FINISH:
                    if global_text:
                        token = global_text.pop(0)
                        chunk = {"choices": [{"delta": {"content": token}}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                    time.sleep(0.02)
                yield "data: [DONE]\n\n"
            return Response(generate(), content_type='text/event-stream')
    finally:
        lock.release()

# --- 交互式命令行模式 ---
def interactive_shell():
    print("\n\033[93m[System] 交互模式已就绪。输入 'exit' 退出。\033[0m")
    while True:
        user_input = input("\n\033[94mUser >>> \033[0m")
        if user_input.lower() in ['exit', 'quit']: break
        
        if lock.acquire(blocking=False):
            try:
                global global_text, global_state
                global_text = []
                global_state = -1
                print("\033[92mAssistant: \033[0m", end="", flush=True)
                
                run_thread = threading.Thread(target=rkllm_model.run, args=(user_input,))
                run_thread.start()
                
                while global_state != LLMCallState.RKLLM_RUN_FINISH:
                    if global_text:
                        print(global_text.pop(0), end="", flush=True)
                    time.sleep(0.01)
                print("")
            finally:
                lock.release()
        else:
            print("\033[91m[Busy] NPU 正在处理 API 请求，请稍后再试...\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, required=True)
    parser.add_argument('--target_platform', type=str, default="rk3588")
    args = parser.parse_args()

    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))
    
    print("正在初始化 RKLLM 模型...")
    rkllm_model = RKLLMModel(args.rkllm_model_path, args.target_platform)

    # 启动 Flask 服务线程
    api_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8080), daemon=True)
    api_thread.start()

    # 进入交互式 Shell
    interactive_shell()