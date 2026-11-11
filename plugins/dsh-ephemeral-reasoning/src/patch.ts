/** Reversible instance patch. A later wrapper can keep this one as an inert delegate. */
export function patchMethod<T extends object, K extends keyof T>(target: T, key: K, replacement: T[K]): () => void {
  const descriptor = Object.getOwnPropertyDescriptor(target, key)
  Object.defineProperty(target, key, { configurable: true, writable: true, value: replacement })
  return () => {
    // Cordis returns bound wrappers on reads; compare the stored property instead.
    if (Object.getOwnPropertyDescriptor(target, key)?.value !== replacement) return
    if (descriptor) Object.defineProperty(target, key, descriptor)
    else Reflect.deleteProperty(target, key)
  }
}
