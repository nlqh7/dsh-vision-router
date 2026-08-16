# @deepseek-ai/dsh-vision-router

[English](README.md) | 中文

让纯文本主模型（如 `deepseek-v4-pro`）在聊天框接收图片：图片在前置步骤被路由到用户配置的视觉模型转成文字描述，再回填给主模型。主模型全程只看到文字，无需切换模型。

## 工作原理

插件做三件事：

1. **能力声明改写** — 把主模型的 `inputModalities` 报告为 `['text', 'image']`，使前端放行聊天框图片。通过替换 `ctx.llm.resolveModelInfo` 实现（cordis 无方法装饰 API，采用 monkey-patch 并在插件卸载时恢复原方法，避免 HMR 泄漏）。
2. **自动发现视觉模型** — 扫描 `ctx.llm.listProviders()` 与 `listModels(provider)`，找到第一个 `inputModalities` 含 `image` 的模型作为视觉模型；也可用 `visionProvider`/`visionModel` 显式指定。发现结果按 `llm/adapters-updated` 失效重扫。
3. **pre-step 分流转文字** — 监听 `agent/pre-step`，扫描进入步骤的消息里的 `ImageBlock`，对每张图构造一条 `[{type:'image', attachment}, {type:'text', text: 描述指令}]` 的 user 消息经 `ctx.llm.stream()` 发给视觉模型，把返回文字作为 `【图片内容】…` 文本块替换原图片块。视觉模型适配器（pi-ai）负责读附件字节与 base64 序列化，插件零视觉 API 代码。

## 配置

```ts
interface Config {
  /** 主模型 provider 路由，默认 `deepseek-official`。 */
  provider?: string
  /** 主模型 id，默认 `deepseek-v4-pro`。 */
  model?: string
  /** 优先视觉 provider；省略则自动发现。 */
  visionProvider?: string
  /** 优先视觉模型 id；省略则自动发现。 */
  visionModel?: string
  /** 发给视觉模型的描述指令。 */
  imagePrompt?: string
  /** 每次视觉调用的输出 token 上限，默认 1024。 */
  maxTokens?: number
}
```

## 挂载

插件加载前，用户需在 DSH「设置 → 模型 → 添加自定义提供方」里配置一个 OpenAI 兼容视觉模型，并给该模型声明 `input: [text, image]`（见 DSH 官方文档《配置模型》「图片输入」一节）。插件不内置任何视觉 provider 的地址或密钥。

### 安装

```sh
# 方式一：从 GitHub 仓库安装
dsh plugin --profile web add "github:<owner>/dsh-vision-router"
# 方式二：npm 安装后手动挂载
npm i @deepseek-ai/dsh-vision-router
```

### 挂载（cordis.patch.yml）

```yaml
- insert:
    - id: vision-router
      name: '@deepseek-ai/dsh-vision-router'
      config:
        provider: deepseek-official
        model: deepseek-v4-pro
```

装好并重启后，用纯文本主模型在聊天框拖图，图片会自动转成文字描述。

## Model Experience

### 主模型请求（图片被替换）

#### What the model sees

主模型收到的消息里，每个 `ImageBlock` 被替换成一个文本块。视觉调用成功时文本为 `【图片内容】` 前缀 + 描述正文；未配置视觉模型时为 `[图片：未配置视觉模型，无法识别]`；视觉调用失败或返回空文本时为 `[图片：识别失败]`。

#### Token effect

每张图贡献一次替换文本，长度由视觉模型描述决定；图片字节从不进入主模型上下文。每张不同的图各触发一次视觉调用，其 token 计费发生在视觉模型侧，与主模型请求分离。

#### KV Cache effect

替换发生在 `agent/pre-step`，先于主模型请求，因此主模型请求的前缀稳定（图片被替换成确定性文本后按普通文本参与前缀）。视觉调用是独立请求，不影响主模型前缀复用。

### 视觉模型请求（辅助调用）

#### What the model sees

视觉模型收到一条用户消息，含图片引用块与描述指令文本。指令默认要求输出纯描述、不加前缀标题。

#### Token effect

每次辅助调用为一条带图消息；输出上限由 `maxTokens`（默认 1024）约束。

#### KV Cache effect

独立请求；与主模型请求前缀无关，不共享、不失效主模型缓存。

## Known Limitations and Deferred Work

- **前端硬拦截仍按主模型能力判断** — 插件改写的是 `resolveModelInfo` 的返回值，所以需要该调用路径真正经过 `ctx.llm.resolveModelInfo`（前端发送图片前的检查与主请求都走这里）。若某个未来前端改走别的能力查询，图片仍会被拦。
- **图片不进主模型上下文** — 图片块留在会话日志里（界面渲染缩略图），但发给主模型的请求被替换成文字描述，所以主模型永远拿不到原图字节；若后续主模型自身支持视觉、需要原图，需停用本插件或调整 `provider`/`model`。
- **视觉调用缓存在插件生命周期内** — 同一 `attachmentId` 只调用一次视觉模型（进程内缓存 + in-flight 去重）；跨会话重启后缓存清空，相同图片会重新描述。
- **发现依赖 `llm/adapters-updated`** — 视觉模型在会话中途新增时，下一个请求会重扫；但已缓存的失败发现不会主动重试，除非拓扑变化或新会话。
