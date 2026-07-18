import { StagePageClient } from "@/components/StagePageClient";
import { loadStageRequest } from "@/lib/loadStage";
import type { SearchParams } from "@/lib/navigation";

export default async function HomePage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const params = await searchParams;
  const request = await loadStageRequest(params);
  return (
    <StagePageClient
      stage="central"
      initialStage={params.stage === "demo" ? "demo" : "train"}
      initialBundle={request.bundle}
      runSelection={request.runSelection}
    />
  );
}
