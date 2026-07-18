import { redirect } from "next/navigation";
import { buildRootRedirect, type SearchParams } from "@/lib/navigation";

export default async function HomePage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  redirect(buildRootRedirect(await searchParams));
}
