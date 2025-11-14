import VoicePrediction from "@/components/VoicePrediction";
import ImageUpload from "@/components/ImageUpload";
import { useState } from "react";

export default function Prediction() {
  const [imageResult, setImageResult] = useState<string | null>(null);

  return (
    <div className="container mx-auto py-10 space-y-12">
      <h1 className="text-4xl font-bold text-center mb-8">
        🧠 Parkinson's Disease Detection
      </h1>

      <section className="space-y-8">
        <h2 className="text-2xl font-semibold text-center">
          🎤 Voice-Based Prediction
        </h2>
        <VoicePrediction />
      </section>

      <section className="space-y-8">
        <h2 className="text-2xl font-semibold text-center">
          🖼️ Image-Based Prediction
        </h2>
        <ImageUpload onImageAnalyzed={(result) => setImageResult(result)} />

        {imageResult && (
          <div
            className={`text-center font-semibold text-lg ${
              imageResult.includes("healthy")
                ? "text-green-600"
                : "text-red-600"
            }`}
          >
            Prediction Result: {imageResult}
          </div>
        )}
      </section>
    </div>
  );
}
