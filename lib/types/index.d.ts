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
import type { Context } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
/** Cordis plugin name used by loader diagnostics. */
export declare const name = "vision-router";
/** The LLM seam this plugin wraps and reuses for vision calls. */
export declare const inject: string[];
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
    provider?: string;
    /** Primary model id whose modalities are rewritten. Defaults to `deepseek-v4-pro`. */
    model?: string;
    /** Preferred vision provider route; omitted means auto-discovery. */
    visionProvider?: string;
    /** Preferred vision model id; omitted means auto-discovery. */
    visionModel?: string;
    /** Prompt sent to the vision model with each image. */
    imagePrompt?: string;
    /** Output-token cap for each vision call. Defaults to 1024. */
    maxTokens?: number;
}
/** Schemastery validation for {@link Config}. */
export declare const Config: z<Config>;
/**
 * Register modality rewriting and the pre-step image-to-text router.
 * @param ctx - plugin context; listeners and the wrapped method are disposed with it.
 * @param config - resolved configuration.
 */
export declare function apply(ctx: Context, config: Config): void;
//# sourceMappingURL=index.d.ts.map