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
