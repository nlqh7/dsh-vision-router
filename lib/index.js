import z from "@deepseek-ai/schemastery";
import { BlockAssembler, contentHasImage, createUserMessage } from "@deepseek-ai/dsh-llm";
//#region lib/types/index.js
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
/** Report the primary model as image-capable so the frontend admits images. */
function rewritePrimaryModel(config, provider, model) {
	const target = config.provider ?? "deepseek-official";
	const targetModel = config.model ?? "deepseek-v4-pro";
	return provider === target && model === targetModel;
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
/** Replace the image blocks of one message using the pre-resolved descriptions. */
function rewriteMessage(message, descriptions) {
	const content = message.content.map((block) => {
		if (block.type !== "image") return block;
		return {
			type: "text",
			text: descriptions.get(String(block.attachment.attachmentId)) ?? VISION_FAILED_TEXT
		};
	});
	return {
		...message,
		content
	};
}
/**
* Register modality rewriting and the pre-step image-to-text router.
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
			if (rewritePrimaryModel(config, provider, model)) return {
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
	const invalidateVision = () => {
		visionCache = void 0;
	};
	ctx.on("llm/adapters-updated", invalidateVision, { global: true });
	ctx.on("agent/pre-step", async ({ signal }, next) => {
		const decision = await next();
		if (decision.kind === "reject" || signal.aborted) return decision;
		if (!decision.messages.some((message) => contentHasImage(message.content))) return decision;
		if (visionCache === void 0) visionCache = { target: await discoverVisionModel(ctx, config) };
		const vision = visionCache.target;
		const rewritten = [];
		for (const message of decision.messages) {
			if (!contentHasImage(message.content)) {
				rewritten.push(message);
				continue;
			}
			const descriptions = /* @__PURE__ */ new Map();
			for (const block of message.content) {
				if (block.type !== "image") continue;
				const key = String(block.attachment.attachmentId);
				if (descriptions.has(key)) continue;
				let text;
				if (vision !== void 0) text = await describeImage(ctx, block.attachment, vision, prompt, maxTokens, signal);
				if (text !== void 0 && text.length > 0) descriptions.set(key, `${IMAGE_DESCRIPTION_PREFIX}${text}`);
				else descriptions.set(key, vision === void 0 ? NO_VISION_TEXT : VISION_FAILED_TEXT);
			}
			rewritten.push(rewriteMessage(message, descriptions));
		}
		return {
			kind: "enter",
			messages: rewritten
		};
	});
}
//#endregion
export { Config, apply, inject, name };
