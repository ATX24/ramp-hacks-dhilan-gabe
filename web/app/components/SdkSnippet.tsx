export function SdkSnippet({ compact = false }: { compact?: boolean }) {
  return (
    <div className={compact ? "sdk-card sdk-card-compact" : "sdk-card"}>
      <div className="sdk-card-header">
        <span>distill.py</span>
        <span className="sdk-live">ready</span>
      </div>
      <pre>
        <code>
          <span className="code-muted">distillery</span> = Distillery(api_key=os.environ[<span className="code-string">&quot;DISTILLERY_API_KEY&quot;</span>]){"\n"}
          <span className="code-muted">dataset</span> = distillery.datasets.create(<span className="code-string">&quot;./finance_world.jsonl&quot;</span>){"\n"}
          <span className="code-muted">run</span> = distillery.distill(dataset, recipe=<span className="code-string">&quot;auto&quot;</span>).wait()
        </code>
      </pre>
      <div className="sdk-card-footer">
        <span>one dataset</span>
        <span>one resolved recipe</span>
        <span>one portable model</span>
      </div>
    </div>
  );
}
