//#region lib/types/invariant.js
/** Package-owned invariant companion for `@deepseek-ai/dsh-vision-router`. @module @deepseek-ai/dsh-vision-router/invariant */
const PACKAGE_NAME = "@deepseek-ai/dsh-vision-router";
/** Cordis companion plugin name. */
const name = "vision-router-invariant";
/** Services required before package ownership can be reserved. */
const inject = ["invariants"];
/** No runtime invariant: the plugin only rewrites in-flight request messages and reported capabilities; it persists no owned state. */
const install = () => {};
/**
* Register the package invariant companion.
* @param ctx - Cordis context carrying the invariant service.
* @returns the registration disposer.
*/
const apply = (ctx) => Promise.resolve(ctx.invariants.register(PACKAGE_NAME, install));
//#endregion
export { apply, inject, name };
