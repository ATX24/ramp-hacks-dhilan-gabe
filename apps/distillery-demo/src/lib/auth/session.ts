export const SESSION_COOKIE_NAME = "distillery_demo_session";
export const SESSION_TTL_SECONDS = 8 * 60 * 60;

const encoder = new TextEncoder();

function assertSigningSecret(secret: string): void {
  if (secret.length < 32) {
    throw new Error("DISTILLERY_AUTH_SECRET must contain at least 32 characters");
  }
}

function bytesToBase64Url(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary)
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replaceAll("=", "");
}

function base64UrlToBytes(value: string): Uint8Array {
  const base64 = value.replaceAll("-", "+").replaceAll("_", "/");
  const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, "=");
  const binary = atob(padded);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

async function hmacKey(
  secret: string,
  usage: KeyUsage,
): Promise<CryptoKey> {
  assertSigningSecret(secret);
  return crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    [usage],
  );
}

export async function createSessionToken(
  secret: string,
  nowMs = Date.now(),
): Promise<{ token: string; expiresAt: Date }> {
  const expiresAt = new Date(nowMs + SESSION_TTL_SECONDS * 1000);
  const nonce = bytesToBase64Url(crypto.getRandomValues(new Uint8Array(18)));
  const payload = `v1.${Math.floor(expiresAt.getTime() / 1000)}.${nonce}`;
  const key = await hmacKey(secret, "sign");
  const signature = new Uint8Array(
    await crypto.subtle.sign("HMAC", key, encoder.encode(payload)),
  );
  return {
    token: `${payload}.${bytesToBase64Url(signature)}`,
    expiresAt,
  };
}

export async function verifySessionToken(
  token: string,
  secret: string,
  nowMs = Date.now(),
): Promise<boolean> {
  try {
    if (token.length > 512) return false;
    const parts = token.split(".");
    if (parts.length !== 4) return false;
    const version = parts[0]!;
    const expiresRaw = parts[1]!;
    const nonce = parts[2]!;
    const signatureRaw = parts[3]!;
    if (version !== "v1" || !/^\d{10}$/.test(expiresRaw) || nonce.length < 16) {
      return false;
    }

    const expiresAtSeconds = Number(expiresRaw);
    const nowSeconds = Math.floor(nowMs / 1000);
    if (
      !Number.isSafeInteger(expiresAtSeconds) ||
      expiresAtSeconds <= nowSeconds ||
      expiresAtSeconds > nowSeconds + SESSION_TTL_SECONDS + 60
    ) {
      return false;
    }

    const payload = `${version}.${expiresRaw}.${nonce}`;
    const signature = base64UrlToBytes(signatureRaw);
    const key = await hmacKey(secret, "verify");
    return crypto.subtle.verify(
      "HMAC",
      key,
      signature as BufferSource,
      encoder.encode(payload),
    );
  } catch {
    return false;
  }
}
