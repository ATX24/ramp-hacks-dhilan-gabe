const SHA256_PATTERN = /^[0-9a-f]{64}$/;

export function isSha256(value: unknown): value is string {
  return typeof value === "string" && SHA256_PATTERN.test(value);
}

export function assertSha256(value: string, label = "digest"): void {
  if (!isSha256(value)) {
    throw new Error(`${label} must be a 64-character lowercase hexadecimal SHA-256 digest`);
  }
}

function checkedSha256(value: string, label: string): string {
  assertSha256(value, label);
  return value;
}

/** SHA-256 digests of deterministic `distillery-fixture-v1:<name>` fixture inputs. */
export const HASH = {
  dataset: checkedSha256(
    "e769a8eb883e43d1655e7cb117705e56c3a5be26f4e9db126c146a540c0da3db",
    "dataset",
  ),
  train: checkedSha256(
    "6cb76a9b4bd058946f90e4cad005a5f8e23ab7d0d4e4917e0d90e817acef49a5",
    "train split",
  ),
  validation: checkedSha256(
    "38e3861972fd1920f335b58ecbb33b77c6627e7c00d54b900ac808fd49cfdc09",
    "validation split",
  ),
  iid: checkedSha256(
    "b605749fe1284483353891bd04bca594607bf3826ce08e538de5a09365b31c26",
    "IID split",
  ),
  ood: checkedSha256(
    "5e3169d9a5949562735b6f34260dee460072b677cab773276f8e1facf1d2bf2f",
    "OOD split",
  ),
  world: checkedSha256(
    "64365bf6af44a6097136714a702fdcf19b798636a9f3e57891914befc97cf4ee",
    "world",
  ),
  protocol: checkedSha256(
    "e552a72fbddb2b36ce780498fc4d95bc91b741a4a1bf8766e910becd665af002",
    "proof protocol",
  ),
  adapter: checkedSha256(
    "a7724e673ceaee8aeb779dd19afc80f7512953e0bbd8d905c0ca6da2ee74b8a1",
    "adapter artifact",
  ),
  manifest: checkedSha256(
    "33b4467857c07968add8ab00c0334a21cb90a433acf54fc3e7731feec2d3df09",
    "manifest",
  ),
  prediction: checkedSha256(
    "a8d21e15c3491667eeb3a84b0dcda48486e0fef18bba815e65e9e30636c883bc",
    "predictions",
  ),
} as const;
