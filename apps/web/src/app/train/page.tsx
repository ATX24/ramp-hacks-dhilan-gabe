import { redirect } from "next/navigation";
import { buildCentralHref } from "@/lib/navigation";

export default async function TrainPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  redirect(buildCentralHref("train", await searchParams));
}
