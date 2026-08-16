import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, {
  LlmAdapter,
  createUserMessage,
  type GenerateOptions,
  type LlmModelInfo,
  type LlmResolvedModelInfo,
  type StreamChunk,
} from '@deepseek-ai/dsh-llm'
import SessionStore, { SessionId, type SessionEvent } from '@deepseek-ai/dsh-session'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import AgentRegistry, { type Agent } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import LocalAttachmentStore from '@deepseek-ai/dsh-attachment-local'
import * as visionRouter from '../src/index.ts'

/** A minimal valid 1x1 transparent PNG, accepted by the attachment store. */
const ONE_PX_PNG = Uint8Array.from(Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
  'base64',
))

/** Scripted text response, mirroring `packages/core/agent-loop/tests/mock-adapter.ts`. */
function textChunks(text: string): StreamChunk[] {
  return [
    { type: 'block-start', index: 0, blockType: 'text' },
    ...Array.from(text, (char): StreamChunk => ({ type: 'text-delta', index: 0, text: char })),
    { type: 'block-end', index: 0, block: { type: 'text', text } },
    { type: 'usage', usage: { inputTokens: 1, outputTokens: text.length } },
    { type: 'finish', reason: { kind: 'stop' } },
  ]
}

/** A text-only adapter whose `resolveModel` reports `inputModalities: ['text']`. */
class TextOnlyAdapter extends LlmAdapter {
  requests: GenerateOptions[] = []
  constructor(private readonly inputModalities: readonly string[] = ['text']) {
    super()
  }
  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    return Promise.resolve({
      provider,
      id: model,
      name: model,
      inputModalities: this.inputModalities as LlmResolvedModelInfo['inputModalities'],
    })
  }
  async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.requests.push(options)
    for (const chunk of textChunks('ok')) yield chunk
  }
}

/** A vision adapter that advertises one image-capable model and answers descriptions. */
class VisionAdapter extends LlmAdapter {
  requests: GenerateOptions[] = []
  constructor(private readonly description: string) {
    super()
  }
  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    return Promise.resolve({
      provider,
      id: model,
      name: model,
      inputModalities: ['text', 'image'],
    })
  }
  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    return Promise.resolve([{
      provider,
      id: 'vision-model',
      name: 'vision-model',
      inputModalities: ['text', 'image'],
    }])
  }
  async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.requests.push(options)
    for (const chunk of textChunks(this.description)) yield chunk
  }
}

function waitForIdle(ctx: Context, agent: Agent): Promise<void> {
  return new Promise((resolve) => {
    const dispose = ctx.on('agent/status', ({ agent: subject, status }) => {
      if (subject === agent && status === 'idle') {
        dispose()
        resolve()
      }
    })
  })
}

function events(agent: Agent): SessionEvent[] {
  return [...agent.session.events]
}

function messageHasImage(message: { content: readonly { type: string }[] }): boolean {
  return message.content.some(block => block.type === 'image')
}

describe('vision-router', () => {
  it('rewrites the primary model inputModalities so the frontend admits images', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    const main = new TextOnlyAdapter()
    ctx.llm.registerAdapter(['main'], main)
    await ctx.plugin(visionRouter, { provider: 'main', model: 'main-model' })

    const rewritten = await ctx.llm.resolveModelInfo('main', 'main-model')
    expect(rewritten.inputModalities).toEqual(['text', 'image'])

    // Other models on the same route are left untouched.
    const untouched = await ctx.llm.resolveModelInfo('main', 'other-model')
    expect(untouched.inputModalities).toEqual(['text'])
  })

  it('replaces image blocks with the vision model description before the primary request', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(SessionStore)
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    await ctx.plugin(AgentRegistry)
    await ctx.plugin(LocalAttachmentStore, { dshHome: undefined })

    const main = new TextOnlyAdapter()
    const vision = new VisionAdapter('图中是一只猫。')
    ctx.llm.registerAdapter(['main'], main)
    ctx.llm.registerAdapter(['vision'], vision)

    await ctx.plugin(visionRouter, {
      provider: 'main',
      model: 'main-model',
      visionProvider: 'vision',
      visionModel: 'vision-model',
    })
    await ctx.plugin(AgentLoop, { agents: [] })

    const ref = await ctx.attachments.saveImage({ data: ONE_PX_PNG, mediaType: 'image/png' })
    const agent = ctx.agentLoop.create(SessionId('img-1'), { provider: 'main', model: 'main-model' })

    agent.followup(createUserMessage({
      content: [
        { type: 'image', attachment: ref },
        { type: 'text', text: '看下这张图' },
      ],
      source: { kind: 'user' },
    }))
    await waitForIdle(ctx, agent)

    // The vision model received the image; the primary model received only text.
    expect(vision.requests.length).toBeGreaterThan(0)
    expect(vision.requests[0]!.messages[0]!.content.some(block => block.type === 'image')).toBe(true)

    const primaryRequest = main.requests[0]
    expect(primaryRequest).toBeDefined()
    expect(primaryRequest!.messages.some(messageHasImage)).toBe(false)
    const flattened = JSON.stringify(primaryRequest!.messages)
    expect(flattened).toContain('【图片内容】图中是一只猫。')
    expect(flattened).not.toContain('"image"')

    // The durable surface records the rewritten text (the image block never enters the log).
    const userMessage = events(agent).find(e => e.type === 'user/message' && e.data.source.kind === 'user')
    expect(userMessage?.type === 'user/message'
      && userMessage.data.content.some(block => block.type === 'text' && block.text.includes('【图片内容】'))).toBe(true)
  })

  it('degrades to a placeholder when no vision model is discoverable', async () => {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(SessionStore)
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    await ctx.plugin(AgentRegistry)
    await ctx.plugin(LocalAttachmentStore, { dshHome: undefined })

    const main = new TextOnlyAdapter()
    ctx.llm.registerAdapter(['main'], main)

    // No vision provider configured and none advertised -> discovery fails.
    await ctx.plugin(visionRouter, { provider: 'main', model: 'main-model' })
    await ctx.plugin(AgentLoop, { agents: [] })

    const ref = await ctx.attachments.saveImage({ data: ONE_PX_PNG, mediaType: 'image/png' })
    const agent = ctx.agentLoop.create(SessionId('img-2'), { provider: 'main', model: 'main-model' })

    agent.followup(createUserMessage({
      content: [{ type: 'image', attachment: ref }],
      source: { kind: 'user' },
    }))
    await waitForIdle(ctx, agent)

    const primaryRequest = main.requests[0]
    expect(primaryRequest).toBeDefined()
    expect(primaryRequest!.messages.some(messageHasImage)).toBe(false)
    expect(JSON.stringify(primaryRequest!.messages)).toContain('未配置视觉模型')
  })
})
