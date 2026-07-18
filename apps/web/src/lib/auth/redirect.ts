export function safeNextPath(value: FormDataEntryValue | string | null | undefined): string {
  if (
    typeof value !== "string" ||
    !value.startsWith("/") ||
    value.startsWith("//") ||
    value.startsWith("/login") ||
    value.startsWith("/api/auth/")
  ) {
    return "/";
  }
  return value;
}

export function requestOrigin(request: Request): string {
  const forwardedHost = request.headers
    .get("x-forwarded-host")
    ?.split(",")[0]
    ?.trim();
  const host = forwardedHost || request.headers.get("host");
  const forwardedProtocol = request.headers
    .get("x-forwarded-proto")
    ?.split(",")[0]
    ?.trim();
  const protocol =
    forwardedProtocol === "http" || forwardedProtocol === "https"
      ? forwardedProtocol
      : new URL(request.url).protocol.replace(":", "");

  if (host && /^[a-z0-9.:[\]-]+$/i.test(host)) {
    return `${protocol}://${host}`;
  }
  return new URL(request.url).origin;
}
