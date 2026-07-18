import { ProjectPageClient } from "@/components/ProjectPageClient";
import { loadStageRequest } from "@/lib/loadStage";
import type { SearchParams } from "@/lib/navigation";

export default async function HomePage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const params = await searchParams;
  const request = await loadStageRequest({
    ...params,
    mode: params.mode ?? "proved",
  });
  return (
    <ProjectPageClient
      initialBundle={request.bundle}
      runSelection={request.runSelection}
    />
  );
}
