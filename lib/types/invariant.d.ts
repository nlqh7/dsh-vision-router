/** Package-owned invariant companion for `@deepseek-ai/dsh-vision-router`. @module @deepseek-ai/dsh-vision-router/invariant */
import type { Context } from '@deepseek-ai/cordis';
/** Cordis companion plugin name. */
export declare const name = "vision-router-invariant";
/** Services required before package ownership can be reserved. */
export declare const inject: string[];
/**
 * Register the package invariant companion.
 * @param ctx - Cordis context carrying the invariant service.
 * @returns the registration disposer.
 */
export declare const apply: (ctx: Context) => Promise<() => void>;
//# sourceMappingURL=invariant.d.ts.map