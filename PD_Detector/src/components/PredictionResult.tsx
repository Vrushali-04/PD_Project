import React from "react";
import { CheckCircle2, AlertTriangle, ShieldAlert } from "lucide-react";

export interface PredictionResultProps {
  result: "healthy" | "detected" | "error"; 
  confidence?: number;
  explanation?: string;
}

const PredictionResult: React.FC<PredictionResultProps> = ({
  result,
  confidence = 0,
  explanation,
}) => {
  // If the Gatekeeper rejected the image
  if (result === "error") {
    return (
      <div className="mt-8 p-6 rounded-lg bg-amber-50 border-2 border-amber-500 animate-in fade-in zoom-in duration-300">
        <div className="flex items-center gap-3">
          <ShieldAlert className="h-8 w-8 text-amber-600 flex-shrink-0" />
          <div>
            <h3 className="text-2xl font-bold text-amber-700">Invalid Input Detected</h3>
            <p className="text-amber-800 font-medium mt-1">{explanation}</p>
          </div>
        </div>
      </div>
    );
  }

  const isHealthy = result === "healthy";
  const defaultExplanation = isHealthy
    ? "AI analysis shows parameters within normal ranges. No significant neurological irregularities detected."
    : "AI analysis detected irregularities. These patterns are statistically consistent with Parkinson's disease indicators.";

  return (
    <div
      className={`mt-8 p-6 rounded-lg animate-scale-in border-2 ${
        isHealthy ? "bg-green-50 border-green-500" : "bg-red-50 border-red-500"
      }`}
    >
      <div className="flex items-center gap-3 mb-3">
        {isHealthy ? (
          <CheckCircle2 className="h-8 w-8 text-green-600 flex-shrink-0" />
        ) : (
          <AlertTriangle className="h-8 w-8 text-red-600 flex-shrink-0" />
        )}
        <div className="flex-1">
          <h3 className={`text-2xl font-bold ${isHealthy ? "text-green-700" : "text-red-700"}`}>
            {isHealthy ? "No Parkinson's Detected" : "High Probability of Parkinson's Disease"}
          </h3>
          <p className="text-sm font-medium mt-1 opacity-80">
           AI Confidence: {confidence.toFixed(2)}%
          </p>
        </div>
      </div>

      <div className="space-y-3">
        <p className="text-gray-700 leading-relaxed">
          {isHealthy
            ? "Based on the image features, you appear healthy. The Gatekeeper validated the scan/drawing as authentic medical data."
            : "The analysis indicates potential signs of Parkinson's. The model has identified micro-tremors or structural changes."}
        </p>

        <div className="p-4 bg-white/50 rounded-md border border-gray-200">
          <h4 className="font-semibold text-sm mb-2 text-blue-600">Gatekeeper Verification & AI Summary</h4>
          <p className="text-sm text-gray-600 italic">
            {explanation || defaultExplanation}
          </p>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;