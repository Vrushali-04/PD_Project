import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Loader2, Mic, Image, Pencil } from "lucide-react";
import { toast } from "sonner";
import VoiceFeatureInputs from "./VoiceFeatureInputs";
import ImageUpload from "./ImageUpload";
import DrawingCanvas from "./SpiralUpload";
import PredictionResult from "./PredictionResult";

interface FormData {
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

const PredictionForm = () => {
  const [formData, setFormData] = useState<FormData>({
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

  // Type definition to match PredictionResultProps exactly
  const [result, setResult] = useState<"healthy" | "detected" | "error" | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [explanation, setExplanation] = useState<string>(""); 
  const [loading, setLoading] = useState(false);
  const [activeSection, setActiveSection] = useState<"voice" | "image" | "drawing">("voice");

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (Object.values(formData).some((val) => val === "")) {
      toast.error("Please fill in all fields");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://localhost:5000/predict_voice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mdvpFo: parseFloat(formData.mdvpFo),
          mdvpJitter: parseFloat(formData.mdvpJitter),
          mdvpShimmer: parseFloat(formData.mdvpShimmer),
          hnr: parseFloat(formData.hnr),
          rpde: parseFloat(formData.rpde),
          dfa: parseFloat(formData.dfa),
          spread1: parseFloat(formData.spread1),
          spread2: parseFloat(formData.spread2),
          ppe: parseFloat(formData.ppe),
        }),
      });

      if (!response.ok) throw new Error("Server error");
      const data = await response.json();

      setResult(data.prediction === "detected" ? "detected" : "healthy");
      setConfidence(data.confidence * 100); 
      setExplanation("");

      if (data.prediction === "healthy") {
        toast.success("Analysis complete!");
      } else {
        toast.warning("High probability detected.");
      }
    } catch (error) {
      toast.error("Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  };

  // HANDLERS: data is now the full object from backend
  const handleImageAnalyzed = (data: any) => {
    if (!data) {
        setResult(null);
        return;
    }
    if (data.error) {
      setResult("error");
      setExplanation(data.error);
      setConfidence(0);
    } else {
      setResult(data.prediction === "parkinson" ? "detected" : "healthy");
      setConfidence(data.confidence);
      setExplanation(data.message || "");
    }
  };

  const handlePatternAnalyzed = (data: any) => {
    if (!data) {
        setResult(null);
        return;
    }
    if (data.error) {
      setResult("error");
      setExplanation(data.message || data.error); 
      setConfidence(0);
    } else {
      setResult(data.prediction === "parkinson" ? "detected" : "healthy");
      setConfidence(data.confidence);
      setExplanation(data.message || "");
    }
  };

  const switchTab = (tab: "voice" | "image" | "drawing") => {
    setActiveSection(tab);
    setResult(null); 
    setExplanation("");
  };

  return (
    <section id="prediction" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4">
              Disease <span className="text-gradient">Prediction</span>
            </h2>
            <p className="text-lg text-muted-foreground">
              Select a diagnostic method and upload your data
            </p>
          </div>

          <div className="flex gap-4 mb-8 flex-wrap justify-center">
            <Button
              variant={activeSection === "voice" ? "default" : "outline"}
              onClick={() => switchTab("voice")}
              className="flex-1 min-w-[200px]"
            >
              <Mic className="mr-2 h-5 w-5" /> Voice Features
            </Button>

            <Button
              variant={activeSection === "image" ? "default" : "outline"}
              onClick={() => switchTab("image")}
              className="flex-1 min-w-[200px]"
            >
              <Image className="mr-2 h-5 w-5" /> MRI Upload
            </Button>

            <Button
              variant={activeSection === "drawing" ? "default" : "outline"}
              onClick={() => switchTab("drawing")}
              className="flex-1 min-w-[200px]"
            >
              <Pencil className="mr-2 h-5 w-5" /> Drawing Pattern
            </Button>
          </div>

          <Card className="p-8">
            {activeSection === "voice" && (
              <form onSubmit={handleSubmit} className="space-y-6">
                <VoiceFeatureInputs formData={formData} onChange={handleInputChange} />
                <Button type="submit" size="lg" className="w-full" disabled={loading}>
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Predict from Voice Data"
                  )}
                </Button>
              </form>
            )}

            {activeSection === "image" && (
              <ImageUpload onImageAnalyzed={handleImageAnalyzed} />
            )}

            {activeSection === "drawing" && (
              <DrawingCanvas onPatternAnalyzed={handlePatternAnalyzed} />
            )}

            {result && (
              <PredictionResult
                result={result}
                confidence={confidence}
                explanation={explanation}
              />
            )}
          </Card>
        </div>
      </div>
    </section>
  );
};

export default PredictionForm;