import { useState } from "react";
import { Button } from "@/components/ui/button";
import VoiceFeatureInputs from "@/components/VoiceFeatureInputs";
import { toast } from "sonner";

const VoicePrediction = () => {
  console.log("VoicePrediction component loaded");

  const [formData, setFormData] = useState({
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

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    console.log("BUTTON CLICKED 🔥");

    // ✅ Validation
    if (Object.values(formData).some((value) => value === "")) {
      toast.error("Please fill all fields");
      return;
    }

    setLoading(true);
    toast.info("Analyzing voice features...");

    try {
      console.log("Sending request to backend...");

      const response = await fetch("http://localhost:5000/predict_voice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          MDVP_Fo_Hz: parseFloat(formData.mdvpFo),
          MDVP_Jitter_percent: parseFloat(formData.mdvpJitter),
          MDVP_Shimmer: parseFloat(formData.mdvpShimmer),
          HNR: parseFloat(formData.hnr),
          RPDE: parseFloat(formData.rpde),
          DFA: parseFloat(formData.dfa),
          Spread1: parseFloat(formData.spread1),
          Spread2: parseFloat(formData.spread2),
          PPE: parseFloat(formData.ppe),
        }),
      });

      console.log("Response status:", response.status);

      if (!response.ok) {
        throw new Error("Backend error");
      }

      const data = await response.json();
      console.log("Backend response:", data);

      // ✅ Correctly read prediction & confidence
      if (data.prediction && data.confidence !== undefined) {
        const finalResult = `${data.prediction} (Confidence: ${data.confidence}%)`;
        setResult(finalResult);
        toast.success(finalResult);
      } else {
        setResult("Invalid response from backend");
        toast.error("Invalid backend response");
      }

    } catch (error) {
      console.error("Fetch Error:", error);
      toast.error("Error analyzing voice features");
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