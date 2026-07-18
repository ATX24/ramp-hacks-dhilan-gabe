import { StagePageClient } from "@/components/StagePageClient";
import { loadStageRequest } from "@/lib/loadStage";

export default async function TrainPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const request = await loadStageRequest(params);
  return (
    <StagePageClient
      stage="train"
      initialBundle={request.bundle}
      runSelection={request.runSelection}
    />
  );
}
