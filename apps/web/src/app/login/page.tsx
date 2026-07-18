import { LockKeyhole } from "lucide-react";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { getAuthSecret } from "@/lib/auth/config";
import { safeNextPath } from "@/lib/auth/redirect";
import {
  SESSION_COOKIE_NAME,
  verifySessionToken,
} from "@/lib/auth/session";

type LoginSearchParams = Record<string, string | string[] | undefined>;

function singleValue(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<LoginSearchParams>;
}) {
  const params = await searchParams;
  const nextPath = safeNextPath(singleValue(params.next));
  const token = (await cookies()).get(SESSION_COOKIE_NAME)?.value;

  if (token && (await verifySessionToken(token, getAuthSecret()))) {
    redirect(nextPath);
  }

  return (
    <main className="grid min-h-svh place-items-center bg-[#f1eee6] px-4 py-10">
      <div className="w-full max-w-md">
        <div className="mb-8">
          <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-black/50">
            Distillation workspace
          </p>
          <p className="mt-2 font-serif text-4xl leading-none tracking-[-0.05em]">
            Distillery
          </p>
        </div>

        <Card className="rounded-[20px] border-0 bg-white py-0 shadow-2xl shadow-black/10 ring-1 ring-black/10">
          <CardHeader className="border-b border-black/10 p-6">
            <div className="mb-4 grid size-10 place-items-center rounded-xl bg-[#141413] text-white">
              <LockKeyhole className="size-4" aria-hidden />
            </div>
            <h1 className="font-serif text-3xl font-normal tracking-[-0.035em]">
              Sign in to Distillery
            </h1>
            <CardDescription>
              Enter the shared demo password to continue.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-6">
            <form action="/api/auth/login" method="post" className="grid gap-4">
              <input type="hidden" name="next" value={nextPath} />
              <div className="grid gap-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="current-password"
                  autoFocus
                  required
                />
              </div>
              {params.error !== undefined ? (
                <p
                  className="text-sm text-destructive"
                  role="alert"
                  data-testid="login-error"
                >
                  Sign in failed. Check the password and try again.
                </p>
              ) : null}
              <Button type="submit" size="lg" className="mt-1 w-full">
                Sign in
              </Button>
            </form>
          </CardContent>
        </Card>
        <p className="mt-5 text-center text-xs text-black/45">
          This sign-in protects the shared demo.
        </p>
      </div>
    </main>
  );
}
