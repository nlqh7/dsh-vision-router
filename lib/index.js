import z from "@deepseek-ai/schemastery";
import { BlockAssembler, contentHasImage, createUserMessage } from "@deepseek-ai/dsh-llm";
//#region lib/types/index.js
/**
* Image-to-text routing for text-only primary models.
*
* Lets a text-only primary model (e.g. `deepseek-v4-pro`) accept chat-box
* images: the plugin rewrites the primary model's reported input modalities so
* the frontend admits images, then — right before the primary request — routes
* every image block to a user-configured vision model through the shared
* `ctx.llm` seam and replaces the image block with the returned text
* description. Images stay in the session log (so the UI renders thumbnails);
* only the outgoing primary-model request is rewritten, so a text-only adapter
* never rejects it.
*
* @module @deepseek-ai/dsh-vision-router
*/
/** Cordis plugin name used by loader diagnostics. */
const name = "vision-router";
/** The LLM seam this plugin wraps and reuses for vision calls. */
const inject = ["llm"];
/**
* The default instruction a vision model receives for one image. The model is
* asked to output a plain description so the text can be spliced back into the
* primary request.
*/
const DEFAULT_IMAGE_PROMPT = "请详细描述这张图片的内容，包括所有可见文字、数字、UI 元素、图表、品牌标识和布局结构。直接输出描述正文，不要加任何前缀、标题或解释。";
/** Text inserted in place of an image when no vision model is discoverable. */
const NO_VISION_TEXT = "[图片：未配置视觉模型，无法识别]";
/** Text inserted when the vision call fails or returns no text. */
const VISION_FAILED_TEXT = "[图片：识别失败]";
/** Prefix for a successful description so the primary model knows the text replaced an image. */
const IMAGE_DESCRIPTION_PREFIX = "【图片内容】";
/** Schemastery validation for {@link Config}. */
const Config = z.object({
	provider: z.string(),
	model: z.string(),
	visionProvider: z.string(),
	visionModel: z.string(),
	imagePrompt: z.string(),
	maxTokens: z.number()
});
/** Report every text-only model under the primary provider as image-capable. */
function rewritePrimaryModel(config, provider) {
	return provider === (config.provider ?? "deepseek-official");
}
/**
* Whether a request is the primary-model request that needs image rewriting —
* as opposed to the plugin's own vision-call (which must pass through
* untouched to avoid infinite recursion).
*
* The primary route is matched on `provider`+`model`; the vision-call is
* additionally excluded by its `plugin` source, so even a misconfigured
* vision route pointing at the primary model cannot recurse.
* @param config - resolved plugin configuration.
* @param options - the request being dispatched.
* @returns true when the request targets the primary model and is not the
*   plugin's own vision call.
*/
function isPrimaryRequest(config, options) {
	const targetProvider = config.provider ?? "deepseek-official";
	if (options.provider !== targetProvider) return false;
	const first = options.messages[0];
	const source = first !== void 0 && "source" in first ? first.source : void 0;
	if (source !== void 0 && source.kind === "plugin" && source.plugin === "vision-router") return false;
	return true;
}
/**
* Discover a vision model: the configured pin when both fields are present,
* otherwise the first registered model whose `inputModalities` includes
* `image`. Returns `undefined` when none is available.
* @param ctx - context providing the LLM service.
* @param config - resolved plugin configuration.
* @returns the vision provider/model pair, or `undefined`.
*/
async function discoverVisionModel(ctx, config) {
	if (config.visionProvider !== void 0 && config.visionProvider.length > 0 && config.visionModel !== void 0 && config.visionModel.length > 0) return {
		provider: config.visionProvider,
		model: config.visionModel
	};
	for (const provider of ctx.llm.listProviders()) {
		let models;
		try {
			models = await ctx.llm.listModels(provider.id);
		} catch {
			continue;
		}
		const vision = models.find((model) => model.inputModalities?.includes("image"));
		if (vision !== void 0) return {
			provider: provider.id,
			model: vision.id
		};
	}
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
async function describeImage(ctx, attachment, target, prompt, maxTokens, signal) {
	const options = {
		provider: target.provider,
		model: target.model,
		messages: [createUserMessage({
			content: [{
				type: "image",
				attachment
			}, {
				type: "text",
				text: prompt
			}],
			source: {
				kind: "plugin",
				plugin: name
			}
		})],
		maxTokens,
		signal
	};
	try {
		const assembler = new BlockAssembler();
		for await (const chunk of ctx.llm.stream(options)) assembler.push(chunk);
		const finish = assembler.finish;
		if (finish.kind === "error" || finish.kind === "aborted") return void 0;
		const text = assembler.blocks().filter((block) => block.type === "text").map((block) => block.text).join("").trim();
		return text.length > 0 ? text : void 0;
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		ctx.logger.warn(`vision-router: image description failed: ${message}`);
		return;
	}
}
/**
* Recursively replace image blocks inside content with text descriptions,
* descending into nested tool-result content so images surfaced by tools (e.g.
* `read_image`) are also rewritten before reaching a text-only adapter.
* @param blocks - content blocks to rewrite.
* @param resolve - maps one image attachment to its replacement text.
* @returns the rewritten block list.
*/
async function rewriteBlocks(blocks, resolve) {
	const out = [];
	for (const block of blocks) if (block.type === "image") out.push({
		type: "text",
		text: await resolve(block.attachment)
	});
	else if (block.type === "tool-result") out.push({
		...block,
		content: await rewriteBlocks(block.content, resolve)
	});
	else out.push(block);
	return out;
}
/**
* Register modality rewriting and the request-time image-to-text router.
* @param ctx - plugin context; listeners and the wrapped method are disposed with it.
* @param config - resolved configuration.
*/
function apply(ctx, config) {
	const prompt = config.imagePrompt ?? DEFAULT_IMAGE_PROMPT;
	const maxTokens = config.maxTokens ?? 1024;
	ctx.effect(() => {
		const llm = ctx.llm;
		const original = llm.resolveModelInfo.bind(llm);
		const wrapped = async (provider, model, signal) => {
			const info = await original(provider, model, signal);
			if (rewritePrimaryModel(config, provider) && !info.inputModalities?.includes("image")) return {
				...info,
				inputModalities: ["text", "image"]
			};
			return info;
		};
		llm.resolveModelInfo = wrapped;
		return () => {
			llm.resolveModelInfo = original;
		};
	});
	let visionCache;
	ctx.on("llm/adapters-updated", () => {
		visionCache = void 0;
	}, { global: true });
	const descriptions = /* @__PURE__ */ new Map();
	const inFlight = /* @__PURE__ */ new Map();
	const ensureVision = async () => {
		if (visionCache === void 0) visionCache = { target: await discoverVisionModel(ctx, config) };
		return visionCache.target;
	};
	const resolveDescription = (attachment, signal) => {
		const key = String(attachment.attachmentId);
		const cached = descriptions.get(key);
		if (cached !== void 0) return Promise.resolve(cached);
		const pending = inFlight.get(key);
		if (pending !== void 0) return pending;
		const promise = (async () => {
			const vision = await ensureVision();
			let text;
			if (vision !== void 0) text = await describeImage(ctx, attachment, vision, prompt, maxTokens, signal ?? new AbortController().signal);
			const result = text !== void 0 && text.length > 0 ? `${IMAGE_DESCRIPTION_PREFIX}${text}` : vision === void 0 ? NO_VISION_TEXT : VISION_FAILED_TEXT;
			descriptions.set(key, result);
			return result;
		})();
		inFlight.set(key, promise);
		return promise.finally(() => {
			inFlight.delete(key);
		});
	};
	const rewritePrimaryStream = async function* (options) {
		const rewrittenMessages = [];
		for (const message of options.messages) if (contentHasImage(message.content)) rewrittenMessages.push({
			...message,
			content: await rewriteBlocks(message.content, (attachment) => resolveDescription(attachment, options.signal))
		});
		else rewrittenMessages.push(message);
		yield* ctx.llm.stream({
			...options,
			messages: rewrittenMessages
		});
	};
	ctx.on("llm/stream", (options, next) => {
		if (!isPrimaryRequest(config, options)) return next();
		if (!options.messages.some((message) => contentHasImage(message.content))) return next();
		return rewritePrimaryStream(options);
	}, {
		global: true,
		prepend: true
	});
}
//#endregion
export { Config, apply, inject, name };
