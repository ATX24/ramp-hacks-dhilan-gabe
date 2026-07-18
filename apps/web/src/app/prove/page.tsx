import { StagePageClient } from "@/components/StagePageClient";
import { loadStageRequest } from "@/lib/loadStage";

export default async function ProvePage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const request = await loadStageRequest(params);
  return (
    <StagePageClient
      stage="prove"
      initialBundle={request.bundle}
      runSelection={request.runSelection}
    />
  );
}
