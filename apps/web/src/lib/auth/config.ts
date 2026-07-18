function requiredSecret(
  name: "DISTILLERY_DEMO_PASSWORD" | "DISTILLERY_AUTH_SECRET",
  minimumLength: number,
): string {
  const value = process.env[name];
  if (!value || value.length < minimumLength) {
    throw new Error(`${name} is required and must be at least ${minimumLength} characters`);
  }
  return value;
}

export function getDemoPassword(): string {
  return requiredSecret("DISTILLERY_DEMO_PASSWORD", 12);
}

export function getAuthSecret(): string {
  return requiredSecret("DISTILLERY_AUTH_SECRET", 32);
}
