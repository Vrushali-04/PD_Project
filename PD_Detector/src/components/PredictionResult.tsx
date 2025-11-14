import { CheckCircle2, AlertTriangle } from "lucide-react";

interface PredictionResultProps {
  result: "healthy" | "detected";
  confidence?: number;
  explanation?: string;
}

const PredictionResult = ({ result, confidence = 85, explanation }: PredictionResultProps) => {
  const isHealthy = result === "healthy";

  const defaultExplanation = isHealthy
    ? "Voice parameters are within normal ranges. No significant irregularities detected in frequency, jitter, or shimmer features."
    : "Voice irregularities detected in frequency and shimmer features. RPDE and DFA values indicate potential neurological patterns consistent with Parkinson's disease.";

  return (
    <div
      className={`mt-8 p-6 rounded-lg animate-scale-in ${
        isHealthy
          ? "bg-success/10 border-2 border-success"
          : "bg-destructive/10 border-2 border-destructive"
      }`}
    >
      <div className="flex items-center gap-3 mb-3">
        {isHealthy ? (
          <CheckCircle2 className="h-8 w-8 text-success flex-shrink-0" />
        ) : (
          <AlertTriangle className="h-8 w-8 text-destructive flex-shrink-0" />
        )}
        <div className="flex-1">
          <h3
            className={`text-2xl font-bold ${
              isHealthy ? "text-success" : "text-destructive"
            }`}
          >
            {isHealthy ? "No Parkinson's Detected" : "High Probability of Parkinson's Disease"}
          </h3>
          <p className="text-sm font-medium mt-1 opacity-80">
            Confidence: {confidence}%
          </p>
        </div>
      </div>

      <div className="space-y-3">
        <p className="text-foreground/80 leading-relaxed">
          {isHealthy
            ? "Based on the analysis, you appear healthy. No signs of Parkinson's disease were detected in the measurements."
            : "The analysis indicates potential signs of Parkinson's disease. Please consult with a medical professional for proper diagnosis and treatment."}
        </p>

        <div className="p-4 bg-background/50 rounded-md border border-border/30">
          <h4 className="font-semibold text-sm mb-2 text-primary">
            AI Explanation Summary
          </h4>
          <p className="text-sm text-foreground/70">
            {explanation || defaultExplanation}
          </p>
        </div>

        {!isHealthy && (
          <div className="mt-4 p-3 bg-destructive/5 rounded border border-destructive/20">
            <p className="text-sm font-medium text-destructive">
              ⚠️ Important: This is a screening tool, not a diagnostic test. Consult a
              healthcare professional for proper medical evaluation.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionResult;
