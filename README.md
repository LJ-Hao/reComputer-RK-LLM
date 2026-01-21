使用下面命令下载并运行推理服务

```
docker pull ghcr.io/lj-hao/rk3588-deepseek-r1-distill-qwen:1.5b-w8a8-latest

docker run -it --name deepseek-r1-1.5b    --privileged    --net=host    --device /dev/dri    --device /dev/dma_heap    --device /dev/rknpu    --device /dev/mali0    -v /dev:/dev      ghcr.io/lj-hao/rk3588-deepseek-r1-distill-qwen:1.5b-w8a8-latest

```

使用下面命令来测试：

```
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
"messages": [
{"role": "user", "content": "请解释一下相对论的基本概念。"}
],
"n_keep": 0,
"cache_prompt": false,
"id_slot": 0,
"n_predict": 512,
"stream": true
}'
```

或者使用OpenAI Python包进行测试：

```python
import openai

# Configure the OpenAI client to use your local server
client = openai.OpenAI(
    base_url="http://localhost:8080/v1",  # Point to your local server
    api_key="dummy-key"  # The API key can be anything for this local server
)

# Test the API
response = client.chat.completions.create(
    model="rkllm-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请解释一下相对论的基本概念。"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```