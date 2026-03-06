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

  const [result, setResult] = useState<"healthy" | "detected" | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [activeSection, setActiveSection] =
    useState<"voice" | "image" | "drawing">("voice");

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
        headers: {
          "Content-Type": "application/json",
        },
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

      if (!response.ok) {
        throw new Error("Server error");
      }

      const data = await response.json();

      /*
        Backend must return:
        {
          prediction: "healthy" or "detected",
          confidence: 0.89   // between 0 and 1
        }
      */

      setResult(data.prediction === "detected" ? "detected" : "healthy");
      setConfidence(data.confidence);

      if (data.prediction === "healthy") {
        toast.success("Analysis complete!");
      } else {
        toast.warning(
          "High probability detected. Please consult a medical professional."
        );
      }
    } catch (error) {
      console.error("Backend error:", error);
      toast.error("Failed to connect to backend");
    }

    setLoading(false);
  };

  // Keep these simple (matches your existing components)
  const handleImageAnalyzed = (imageResult: string) => {
    setResult(imageResult as "healthy" | "detected");
    setConfidence(0.85);
  };

  const handlePatternAnalyzed = (patternResult: string) => {
    setResult(patternResult as "healthy" | "detected");
    setConfidence(0.85);
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
              Enter biomedical voice measurement parameters below
            </p>
          </div>

          {/* Tabs */}
          <div className="flex gap-4 mb-8 flex-wrap justify-center">
            <Button
              type="button"
              variant={activeSection === "voice" ? "default" : "outline"}
              onClick={() => setActiveSection("voice")}
              className="flex-1 min-w-[200px]"
            >
              <Mic className="mr-2 h-5 w-5" />
              Voice Features
            </Button>

            <Button
              type="button"
              variant={activeSection === "image" ? "default" : "outline"}
              onClick={() => setActiveSection("image")}
              className="flex-1 min-w-[200px]"
            >
              <Image className="mr-2 h-5 w-5" />
              Image Upload
            </Button>

            <Button
              type="button"
              variant={activeSection === "drawing" ? "default" : "outline"}
              onClick={() => setActiveSection("drawing")}
              className="flex-1 min-w-[200px]"
            >
              <Pencil className="mr-2 h-5 w-5" />
              Drawing Pattern
            </Button>
          </div>

          {/* Voice Section */}
          {activeSection === "voice" && (
            <Card className="p-8">
              <form onSubmit={handleSubmit} className="space-y-6">
                <VoiceFeatureInputs
                  formData={formData}
                  onChange={handleInputChange}
                />

                <Button
                  type="submit"
                  size="lg"
                  className="w-full"
                  disabled={loading}
                >
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

              {result && (
                <PredictionResult
                  result={result}
                  confidence={confidence}
                />
              )}
            </Card>
          )}

          {/* Image Section */}
          {activeSection === "image" && (
            <Card className="p-8">
              <ImageUpload onImageAnalyzed={handleImageAnalyzed} />
              {result && (
                <PredictionResult
                  result={result}
                  confidence={confidence}
                />
              )}
            </Card>
          )}

          {/* Drawing Section */}
          {activeSection === "drawing" && (
            <Card className="p-8">
              <DrawingCanvas onPatternAnalyzed={handlePatternAnalyzed} />
              {result && (
                <PredictionResult
                  result={result}
                  confidence={confidence}
                />
              )}
            </Card>
          )}
        </div>
      </div>
    </section>
  );
};

export default PredictionForm;