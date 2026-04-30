# llm_proxy

大模型API调用代理 （openai_compatible）

## 配置

需要在 config.yaml 中配置 provider
可以在 provider 中配置上游的api_key，并面向下游分发新的 key


## 调用

在端侧，通过 http://{base_url}/{provider} 来调用