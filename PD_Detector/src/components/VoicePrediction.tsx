import { useState } from "react";
import { Button } from "@/components/ui/button";
import VoiceFeatureInputs from "@/components/VoiceFeatureInputs";
import { toast } from "sonner";

interface VoiceFormData {
  mdvpFo: string;
  mdvpJitter: string;
  mdvpShimmer: string;
  hnr: string;
  rpde: string;
  dfa: string;
  spread1: string;
  spread2: string;
  ppe: string;
}

const VoicePrediction = () => {
  console.log("🎤 VoicePrediction component loaded");

  const [formData, setFormData] = useState<VoiceFormData>({
    mdvpFo: "",
    mdvpJitter: "",
    mdvpShimmer: "",
    hnr: "",
    rpde: "",
    dfa: "",
    spread1: "",
    spread2: "",
    ppe: "",
  });

  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Handle input change
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  // Convert and validate numeric input
  const parseNumber = (value: string) => {
    const num = parseFloat(value);
    return isNaN(num) ? null : num;
  };

  const handleSubmit = async () => {
    console.log("🔥 Analyze Button Clicked");

    // ✅ Validation: check empty fields
    if (Object.values(formData).some((value) => value.trim() === "")) {
      toast.error("Please fill all fields");
      return;
    }

    setLoading(true);
    setResult(null);
    toast.info("Analyzing voice features...");

    try {
      const payload = {
        MDVP_Fo_Hz: parseNumber(formData.mdvpFo),
        MDVP_Jitter_percent: parseNumber(formData.mdvpJitter),
        MDVP_Shimmer: parseNumber(formData.mdvpShimmer),
        HNR: parseNumber(formData.hnr),
        RPDE: parseNumber(formData.rpde),
        DFA: parseNumber(formData.dfa),
        Spread1: parseNumber(formData.spread1),
        Spread2: parseNumber(formData.spread2),
        PPE: parseNumber(formData.ppe),
      };

      // ✅ Check if any value failed parsing
      if (Object.values(payload).some((val) => val === null)) {
        toast.error("All values must be valid numbers");
        setLoading(false);
        return;
      }

      console.log("📤 Sending payload:", payload);

      const response = await fetch("http://127.0.0.1:5000/predict_voice", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      console.log("📡 Response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("❌ Backend error:", errorText);
        throw new Error("Backend returned error");
      }

      const data = await response.json();
      console.log("✅ Backend response:", data);

      if (
        typeof data.prediction === "string" &&
        typeof data.confidence !== "undefined"
      ) {
        const confidenceValue =
          typeof data.confidence === "number"
            ? data.confidence.toFixed(2)
            : data.confidence;

        const finalResult = `${data.prediction} (Confidence: ${confidenceValue}%)`;

        setResult(finalResult);
        toast.success("Prediction completed successfully");
      } else {
        console.error("⚠️ Invalid backend response format:", data);
        setResult("Invalid response from backend");
        toast.error("Invalid backend response");
      }
    } catch (error) {
      console.error("🚨 Fetch error:", error);
      toast.error("Cannot connect to backend server");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <VoiceFeatureInputs formData={formData} onChange={handleChange} />

      <Button
        type="button"
        onClick={handleSubmit}
        disabled={loading}
        className="w-full mt-4"
      >
        {loading ? "Analyzing..." : "Analyze Voice Features"}
      </Button>

      {result && (
        <div className="p-4 mt-4 border rounded-lg bg-muted/30 text-center">
          <p className="text-lg font-semibold">
            Result:{" "}
            <span
              className={
                result.toLowerCase().includes("parkinson")
                  ? "text-red-600"
                  : "text-green-600"
              }
            >
              {result}
            </span>
          </p>
        </div>
      )}
    </div>
  );
};

export default VoicePrediction;