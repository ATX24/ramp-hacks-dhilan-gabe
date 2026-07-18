import { redirect } from "next/navigation";
import { buildCentralHref } from "@/lib/navigation";

export default async function DemoPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  redirect(buildCentralHref("demo", await searchParams));
}
