import { Button } from "@/components/ui/button";
import {
  DEMO_EXAMPLE_CATALOG,
  type DemoExamplePresetId,
} from "@/lib/demo/exampleCatalog";

export function DemoExamplePicker({
  selectedId,
  onSelect,
}: {
  selectedId: DemoExamplePresetId;
  onSelect: (id: DemoExamplePresetId) => void;
}) {
  return (
    <fieldset>
      <legend className="mb-2 text-sm font-medium">Choose an example</legend>
      <div className="flex flex-wrap gap-2">
        {DEMO_EXAMPLE_CATALOG.map((example) => {
          const selected = example.id === selectedId;
          return (
            <Button
              key={example.id}
              type="button"
              size="sm"
              variant={selected ? "default" : "outline"}
              aria-pressed={selected}
              data-testid={`example-preset-${example.id}`}
              onClick={() => onSelect(example.id)}
            >
              {example.label}
            </Button>
          );
        })}
      </div>
    </fieldset>
  );
}
