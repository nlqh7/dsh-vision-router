/**
 * Image-to-text routing for text-only primary models.
 *
 * Lets a text-only primary model (e.g. `deepseek-v4-pro`) accept chat-box
 * images: the plugin rewrites the primary model's reported input modalities so
 * the frontend admits images, then — before the primary request — routes every
 * image block to a user-configured vision model through the shared `ctx.llm`
 * seam and replaces the image block with the returned text description. The
 * primary model never sees the image, so a text-only adapter never rejects it.
 *
 * @module @deepseek-ai/dsh-vision-router
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { PreStepDecision } from '@deepseek-ai/dsh-agent'
import {
  BlockAssembler,
  contentHasImage,
  createUserMessage,
} from '@deepseek-ai/dsh-llm'
import type {
  ContentBlock,
  GenerateOptions,
  LlmModelInfo,
  LlmResolvedModelInfo,
  UserMessage,
} from '@deepseek-ai/dsh-llm'
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'

/** Cordis plugin name used by loader diagnostics. */
export const name = 'vision-router'

/** The LLM seam this plugin wraps and reuses for vision calls. */
export const inject = ['llm']

/**
 * The default instruction a vision model receives for one image. The model is
 * asked to output a plain description so the text can be spliced back into the
 * primary request.
 */
const DEFAULT_IMAGE_PROMPT =
  '请详细描述这张图片的内容，包括所有可见文字、数字、UI 元素、图表、品牌标识和布局结构。'
  + '直接输出描述正文，不要加任何前缀、标题或解释。'

/** Text inserted in place of an image when no vision model is discoverable. */
const NO_VISION_TEXT = '[图片：未配置视觉模型，无法识别]'
/** Text inserted when the vision call fails or returns no text. */
const VISION_FAILED_TEXT = '[图片：识别失败]'
/** Prefix for a successful description so the primary model knows the text replaced an image. */
const IMAGE_DESCRIPTION_PREFIX = '【图片内容】'

/**
 * Configuration for the vision router.
 *
 * Every field is optional. `provider`/`model` name the text-only primary route
 * whose reported modalities are rewritten to admit images; they default to the
 * DeepSeek official route. `visionProvider`/`visionModel` pin a preferred
 * vision route; when omitted the plugin discovers the first registered model
 * whose `inputModalities` includes `image`.
 */
export interface Config {
  /** Primary provider route whose modalities are rewritten. Defaults to `deepseek-official`. */
  provider?: string
  /** Primary model id whose modalities are rewritten. Defaults to `deepseek-v4-pro`. */
  model?: string
  /** Preferred vision provider route; omitted means auto-discovery. */
  visionProvider?: string
  /** Preferred vision model id; omitted means auto-discovery. */
  visionModel?: string
  /** Prompt sent to the vision model with each image. */
  imagePrompt?: string
  /** Output-token cap for each vision call. Defaults to 1024. */
  maxTokens?: number
}

/** Schemastery validation for {@link Config}. */
export const Config: z<Config> = z.object({
  provider: z.string(),
  model: z.string(),
  visionProvider: z.string(),
  visionModel: z.string(),
  imagePrompt: z.string(),
  maxTokens: z.number(),
})

/** One discovered vision route: the provider/model pair used for image calls. */
interface VisionTarget {
  provider: string
  model: string
}

/** Report the primary model as image-capable so the frontend admits images. */
function rewritePrimaryModel(config: Config, provider: string, model: string): boolean {
  const target = config.provider ?? 'deepseek-official'
  const targetModel = config.model ?? 'deepseek-v4-pro'
  return provider === target && model === targetModel
}

/**
 * Discover a vision model: the configured pin when both fields are present,
 * otherwise the first registered model whose `inputModalities` includes
 * `image`. Returns `undefined` when none is available.
 * @param ctx - context providing the LLM service.
 * @param config - resolved plugin configuration.
 * @returns the vision provider/model pair, or `undefined`.
 */
async function discoverVisionModel(ctx: Context, config: Config): Promise<VisionTarget | undefined> {
  if (config.visionProvider !== undefined && config.visionProvider.length > 0
    && config.visionModel !== undefined && config.visionModel.length > 0) {
    return { provider: config.visionProvider, model: config.visionModel }
  }
  for (const provider of ctx.llm.listProviders()) {
    let models: LlmModelInfo[]
    try {
      models = await ctx.llm.listModels(provider.id)
    } catch {
      continue
    }
    const vision = models.find(model => model.inputModalities?.includes('image'))
    if (vision !== undefined) return { provider: provider.id, model: vision.id }
  }
  return undefined
}

/**
 * Send one image to the vision model through the shared LLM seam and collect
 * its text description. The image block is passed by attachment reference, so
 * the vision adapter owns byte reading and base64 serialization.
 * @param ctx - context providing the LLM service.
 * @param attachment - the durable image reference to describe.
 * @param target - the vision provider/model route.
 * @param prompt - the description instruction.
 * @param maxTokens - output-token cap for the vision call.
 * @param signal - live turn cancellation forwarded to the vision call.
 * @returns the description text, or `undefined` when the call fails.
 */
async function describeImage(
  ctx: Context,
  attachment: ImageAttachmentRef,
  target: VisionTarget,
  prompt: string,
  maxTokens: number,
  signal: AbortSignal,
): Promise<string | undefined> {
  const options: GenerateOptions = {
    provider: target.provider,
    model: target.model,
    messages: [createUserMessage({
      content: [
        { type: 'image', attachment },
        { type: 'text', text: prompt },
      ],
      source: { kind: 'plugin', plugin: name },
    })],
    maxTokens,
    signal,
  }
  try {
    const assembler = new BlockAssembler()
    for await (const chunk of ctx.llm.stream(options)) assembler.push(chunk)
    const finish = assembler.finish
    if (finish.kind === 'error' || finish.kind === 'aborted') return undefined
    const text = assembler.blocks()
      .filter((block): block is Extract<ContentBlock, { type: 'text' }> => block.type === 'text')
      .map(block => block.text)
      .join('')
      .trim()
    return text.length > 0 ? text : undefined
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error)
    ctx.logger.warn(`vision-router: image description failed: ${message}`)
    return undefined
  }
}

/** Replace the image blocks of one message using the pre-resolved descriptions. */
function rewriteMessage(
  message: UserMessage,
  descriptions: ReadonlyMap<string, string>,
): UserMessage {
  const content = message.content.map((block) => {
    if (block.type !== 'image') return block
    const description = descriptions.get(String(block.attachment.attachmentId))
    return { type: 'text', text: description ?? VISION_FAILED_TEXT } as ContentBlock
  })
  return { ...message, content }
}

/**
 * Register modality rewriting and the pre-step image-to-text router.
 * @param ctx - plugin context; listeners and the wrapped method are disposed with it.
 * @param config - resolved configuration.
 */
export function apply(ctx: Context, config: Config): void {
  const prompt = config.imagePrompt ?? DEFAULT_IMAGE_PROMPT
  const maxTokens = config.maxTokens ?? 1024

  // Rewrite the primary model's reported input modalities so the frontend
  // admits chat-box images. Monkey-patched because cordis exposes no method
  // decoration; restored on disposal so HMR cannot leak the wrapper.
  ctx.effect(() => {
    const llm = ctx.llm
    const original = llm.resolveModelInfo.bind(llm)
    const wrapped = async (
      provider: string,
      model: string,
      signal?: AbortSignal,
    ): Promise<LlmResolvedModelInfo> => {
      const info = await original(provider, model, signal)
      if (rewritePrimaryModel(config, provider, model)) {
        return { ...info, inputModalities: ['text', 'image'] }
      }
      return info
    }
    ;(llm as { resolveModelInfo: typeof llm.resolveModelInfo }).resolveModelInfo = wrapped
    return () => {
      ;(llm as { resolveModelInfo: typeof llm.resolveModelInfo }).resolveModelInfo = original
    }
  })

  // Cache the discovered vision model; invalidated when the adapter topology
  // changes so a user adding a vision provider is picked up on the next step.
  let visionCache: { target: VisionTarget | undefined } | undefined
  const invalidateVision = (): void => { visionCache = undefined }
  ctx.on('llm/adapters-updated', invalidateVision, { global: true })

  ctx.on('agent/pre-step', async (
    { signal },
    next,
  ): Promise<PreStepDecision> => {
    const decision = await next()
    if (decision.kind === 'reject' || signal.aborted) return decision
    if (!decision.messages.some(message => contentHasImage(message.content))) return decision

    if (visionCache === undefined) {
      visionCache = { target: await discoverVisionModel(ctx, config) }
    }
    const vision = visionCache.target

    const rewritten: UserMessage[] = []
    for (const message of decision.messages) {
      if (!contentHasImage(message.content)) {
        rewritten.push(message)
        continue
      }
      // Resolve each distinct image once; identical references share one call.
      const descriptions = new Map<string, string>()
      for (const block of message.content) {
        if (block.type !== 'image') continue
        const key = String(block.attachment.attachmentId)
        if (descriptions.has(key)) continue
        let text: string | undefined
        if (vision !== undefined) {
          text = await describeImage(ctx, block.attachment, vision, prompt, maxTokens, signal)
        }
        if (text !== undefined && text.length > 0) {
          descriptions.set(key, `${IMAGE_DESCRIPTION_PREFIX}${text}`)
        } else {
          descriptions.set(key, vision === undefined ? NO_VISION_TEXT : VISION_FAILED_TEXT)
        }
      }
      rewritten.push(rewriteMessage(message, descriptions))
    }
    return { kind: 'enter', messages: rewritten }
  })
}
