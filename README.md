# Start inference

```
docker pull ghcr.io/lj-hao/rk3588-deepseek-r1-distill-qwen:1.5b-w8a8-latest

docker run -it --name deepseek-r1-1.5b    --privileged    --net=host    --device /dev/dri    --device /dev/dma_heap    --device /dev/rknpu    --device /dev/mali0    -v /dev:/dev      ghcr.io/lj-hao/rk3588-deepseek-r1-distill-qwen:1.5b-w8a8-latest

```

# Test API：

## Non-streaming response：
```
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-model",
    "messages": [
      {"role": "user", "content": "Where is the capital of China？"}
    ],
    "temperature": 2,
    "max_tokens": 512,
    ""
    "stream": false
  }'
```

## Streaming response:

```
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-model",
    "messages": [
      {"role": "user", "content": "Where is the capital of China？"}
    ],
    "temperature": 2,
    "max_tokens": 512,
    "stream": true
  }'
```

# Use OpenAI API to test

## Non-streaming response：

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
        {"role": "user", "content": "Where is the capital of China？"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

## Streaming response:

```python
import openai

# Configure the OpenAI client to use your local server
client = openai.OpenAI(
    base_url="http://localhost:8080/v1",  # Point to your local server
    api_key="dummy-key"  # The API key can be anything for this local server
)

# Test the API with streaming
response_stream = client.chat.completions.create(
    model="rkllm-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Where is the capital of China？"}
    ],
    temperature=0.7,
    max_tokens=512,
    stream=True  # Enable streaming
)

# Process the streaming response
for chunk in response_stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```
