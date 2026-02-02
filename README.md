[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Huggingface][huggingface-shield]][huggingface-url]

# Install Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh 
```

# Start inference

### LLM

| Device | Model |
|--------|-------|
| **RK3588** | [rk3588-deepseek-r1-distill-qwen:7b-w8a8-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3588-deepseek-r1-distill-qwen/662247747?tag=7b-w8a8-latest)<br>[rk3588-deepseek-r1-distill-qwen:1.5b-fp16-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3588-deepseek-r1-distill-qwen/662243577?tag=1.5b-fp16-latest)<br>[rk3588-deepseek-r1-distill-qwen:1.5b-w8a8-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3588-deepseek-r1-distill-qwen/662236226?tag=1.5b-w8a8-latest) | 
| **RK3576** | [rk3576-deepseek-r1-distill-qwen:7b-w4a16-g128-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/662240247?tag=7b-w4a16-g128-latest)<br>[rk3576-deepseek-r1-distill-qwen:7b-w4a16-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/662239597?tag=7b-w4a16-latest)<br>[rk3576-deepseek-r1-distill-qwen:1.5b-fp16-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/662236690?tag=1.5b-fp16-latest)<br>[rk3576-deepseek-r1-distill-qwen:1.5b-w4a16-g128-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/662235949?tag=1.5b-w4a16-g128-latest)<br>[rk3576-deepseek-r1-distill-qwen:1.5b-w4a16-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3576-deepseek-r1-distill-qwen/662234478?tag=1.5b-w4a16-latest) | 

For example:

```bash
docker run -it --name deepseek-r1-1.5b-fp16 \
  --privileged \
  --net=host \
  --device /dev/dri \
  --device /dev/dma_heap \
  --device /dev/rknpu \
  --device /dev/mali0 \
  -v /dev:/dev \
  ghcr.io/lj-hao/rk3588-deepseek-r1-distill-qwen:1.5b-fp16-latest
```

>Note: When you start the service, you can access `http://localhost:8001/docs` and `http://localhost:8001/redoc` to view the documentation.

### VLM

| Device | Model |
|--------|-------|
| **RK3588** | [rk3588-qwen2-vl:7b-w8a8-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3588-qwen2-vl/666595093?tag=7b-w8a8-latest)<br>[rk3588-qwen2-vl:2b-w8a8-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3588-qwen2-vl/666591327?tag=2b-w8a8-latest)<br> | 
| **RK3576** | [rk3576-qwen2.5-vl:3b-w4a16-latest](https://github.com/LJ-Hao/reComputer-RK-LLM/pkgs/container/rk3576-qwen2.5-vl)<br>| 


For example:
```bash
sudo docker run -it --name qwen2.5-3b-w4a16-vl \
  --privileged \
  --net=host \
  --device /dev/dri \
  --device /dev/dma_heap \
  --device /dev/rknpu \
  --device /dev/mali0 \
  -v /dev:/dev \
  ghcr.io/lj-hao/rk3576-qwen2.5-vl:3b-w4a16-latest
```

>Note: When you start the service, you can access `http://localhost:8002/docs` and `http://localhost:8002/redoc` to view the documentation.

# API Test
## LLM
### Commandline
#### Non-streaming response：

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-model",
    "messages": [
      {"role": "user", "content": "Where is the capital of China？"}
    ],
    "temperature": 1,
    "max_tokens": 512,
    "top_k": 1,
    "stream": false
  }'
```

#### Streaming response:

```bash
curl -N http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-model",
    "messages": [
      {"role": "user", "content": "Where is the capital of China？"}
    ],
    "temperature": 1,
    "max_tokens": 512,
    "top_k": 1,
    "stream": true
  }'
```

### Use OpenAI API

#### Non-streaming response：

```python
import openai

# Configure the OpenAI client to use your local server
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",  # Point to your local server
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

#### Streaming response:

```python
import openai

# Configure the OpenAI client to use your local server
client = openai.OpenAI(
    base_url="http://localhost:8001/v1",  # Point to your local server
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

## VLM
### Commandline
#### Non-streaming response：

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-vision",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the image"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://github.com/LJ-Hao/reComputer-RK-LLM/blob/main/img/test.jpeg"
            }
          }
        ]
      }
    ],
    "stream": false,
    "max_tokens": 50
  }'

```

#### Streaming response:

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-vision",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the image"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://github.com/LJ-Hao/reComputer-RK-LLM/blob/main/img/test.jpeg"
            }
          }
        ]
      }
    ],
    "stream": true,
    "max_tokens": 50
  }'
```
### Use OpenAI API

#### Non-streaming response：

```python

```


#### Streaming response:

```python

```
# Speed test

> Note: A rough estimate of a model's inference speed includes both TTFT and TPOT.
> Note: You can use `python test_inference_speed.py --help` to view the help function.

```bash
python -m venv .env && source .env/bin/activate
pip install requests
python test_inference_speed.py
```

# 💞 Top contributors:

<a href="https://github.com/Seeed-Projects/reComputer-RK-LLM/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Seeed-Projects/reComputer-RK-LLM" alt="contrib.rocks image" />
</a>

# 🌟 Star History

![Star History Chart](https://api.star-history.com/svg?repos=Seeed-Projects/reComputer-RK-LLM&type=Date)

Reference: [rknn-llm](https://github.com/airockchip/rknn-llm/tree/main)


[contributors-shield]: https://img.shields.io/github/contributors/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[contributors-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[forks-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/network/members
[stars-shield]: https://img.shields.io/github/stars/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[stars-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/stargazers
[issues-shield]: https://img.shields.io/github/issues/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[issues-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/issues
[license-shield]: https://img.shields.io/github/license/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[license-url]: https://github.com/Seeed-Projects/reComputer-RK-LLM/blob/master/LICENSE.txt
[huggingface-shield]: https://img.shields.io/github/license/Seeed-Projects/reComputer-RK-LLM.svg?style=for-the-badge
[huggingface-url]: https://huggingface.co/JiahaoLi