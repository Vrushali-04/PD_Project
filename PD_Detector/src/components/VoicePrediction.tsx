import { useState } from "react";
import { Button } from "@/components/ui/button";
import VoiceFeatureInputs from "@/components/VoiceFeatureInputs";
import { toast } from "sonner";

const VoicePrediction = () => {
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
    setLoading(true);
    toast.info("Analyzing voice features...");

    try {
      const response = await fetch("http://127.0.0.1:5000/predict_voice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          MDVP_Fo_Hz: formData.mdvpFo,
          MDVP_Jitter_percent: formData.mdvpJitter,
          MDVP_Shimmer: formData.mdvpShimmer,
          HNR: formData.hnr,
          RPDE: formData.rpde,
          DFA: formData.dfa,
          Spread1: formData.spread1,
          Spread2: formData.spread2,
          PPE: formData.ppe,
        }),
      });

      if (!response.ok) throw new Error("Backend error");

      const data = await response.json();
      setResult(data.result || "Error");
      toast.success(`Prediction: ${data.result}`);
    } catch (error) {
      console.error(error);
      toast.error("Error analyzing voice features");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <VoiceFeatureInputs formData={formData} onChange={handleChange} />

      <Button
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
