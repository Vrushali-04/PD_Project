import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface VoiceFeatureInputsProps {
  formData: {
    mdvpFo: string;
    mdvpJitter: string;
    mdvpShimmer: string;
    hnr: string;
    rpde: string;
    dfa: string;
    spread1: string;
    spread2: string;
    ppe: string;
  };
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const VoiceFeatureInputs = ({ formData, onChange }: VoiceFeatureInputsProps) => {
  const fields = [
    {
      id: "mdvpFo",
      label: "MDVP:Fo(Hz) - Frequency",
      placeholder: "e.g., 119.992",
      step: "0.001",
      tooltip: "Average vocal fundamental frequency in Hz",
    },
    {
      id: "mdvpJitter",
      label: "MDVP:Jitter(%)",
      placeholder: "e.g., 0.00784",
      step: "0.00001",
      tooltip: "Percentage variation in frequency from cycle to cycle",
    },
    {
      id: "mdvpShimmer",
      label: "MDVP:Shimmer",
      placeholder: "e.g., 0.0554",
      step: "0.0001",
      tooltip: "Variation in amplitude",
    },
    {
      id: "hnr",
      label: "HNR - Harmonic to Noise Ratio",
      placeholder: "e.g., 21.033",
      step: "0.001",
      tooltip: "Ratio of harmonic sound to noise in voice",
    },
    {
      id: "rpde",
      label: "RPDE",
      placeholder: "e.g., 0.498536",
      step: "0.00001",
      tooltip: "Recurrence Period Density Entropy measure",
    },
    {
      id: "dfa",
      label: "DFA - Detrended Fluctuation",
      placeholder: "e.g., 0.718099",
      step: "0.00001",
      tooltip: "Signal fractal scaling exponent",
    },
    {
      id: "spread1",
      label: "Spread1",
      placeholder: "e.g., -4.813031",
      step: "0.00001",
      tooltip: "Nonlinear measure of fundamental frequency variation",
    },
    {
      id: "spread2",
      label: "Spread2",
      placeholder: "e.g., 0.266482",
      step: "0.00001",
      tooltip: "Nonlinear measure of fundamental frequency variation",
    },
    {
      id: "ppe",
      label: "PPE - Pitch Period Entropy",
      placeholder: "e.g., 0.301442",
      step: "0.00001",
      tooltip: "Measure of impaired vocal control",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <TooltipProvider>
        {fields.map((field) => (
          <div key={field.id} className="space-y-2 group">
            <div className="flex items-center gap-2">
              <Label htmlFor={field.id} className="text-sm font-medium">
                {field.label}
              </Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="h-4 w-4 text-muted-foreground cursor-help hover:text-primary transition-colors" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">{field.tooltip}</p>
                </TooltipContent>
              </Tooltip>
            </div>
            <Input
              id={field.id}
              name={field.id}
              type="number"
              step={field.step}
              value={formData[field.id as keyof typeof formData]}
              onChange={onChange}
              placeholder={field.placeholder}
              className="bg-background border-border/50 focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all duration-200 hover:border-primary/50"
            />
          </div>
        ))}
      </TooltipProvider>
    </div>
  );
};

export default VoiceFeatureInputs;
