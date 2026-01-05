from flask import Flask, request, jsonify
import os
import requests
import json

app = Flask(__name__)

# 从环境变量获取模型路径
model_path = os.environ.get('MODEL_PATH', '/app/models/model.rkllm')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_path': model_path})

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        # 转发请求到后端的RKLLM服务
        backend_url = os.environ.get('BACKEND_URL', 'http://localhost:8080/v1/chat/completions')
        
        # 获取请求数据
        data = request.json
        
        # 设置默认模型名
        if 'model' not in data:
            data['model'] = os.path.basename(model_path).replace('.rkllm', '') if model_path else 'default-model'
        
        # 发送请求到后端服务
        response = requests.post(
            backend_url,
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            # 对于流式响应，需要特殊处理
            if request.json.get('stream', False):
                # 对于流式响应，直接返回后端的响应
                return response.content, 200, {'Content-Type': 'text/event-stream'}
            else:
                # 对于非流式响应，返回JSON
                return jsonify(response.json())
        else:
            return jsonify({'error': 'Backend service error', 'details': response.text}), response.status_code
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    try:
        # 转发请求到后端的RKLLM服务
        backend_url = os.environ.get('BACKEND_URL', 'http://localhost:8080/v1/models')
        
        response = requests.get(backend_url)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Backend service error', 'details': response.text}), response.status_code
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)