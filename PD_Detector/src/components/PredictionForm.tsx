import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Image, Pencil } from "lucide-react";
import { toast } from "sonner";
import ImageUpload from "./ImageUpload";
import DrawingCanvas from "./SpiralUpload";
import PredictionResult from "./PredictionResult";

const PredictionForm = () => {
  const [result, setResult] = useState<"healthy" | "detected" | "error" | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [explanation, setExplanation] = useState<string>("");
  const [activeSection, setActiveSection] = useState<"image" | "drawing">("image");

  // HANDLERS
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

  const switchTab = (tab: "image" | "drawing") => {
    setActiveSection(tab);
    setResult(null);
    setExplanation("");
  };

  return (
    <section id="prediction" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto">
          
          {/* Heading */}
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4">
              Disease <span className="text-gradient">Prediction</span>
            </h2>
            <p className="text-lg text-muted-foreground">
              Select a diagnostic method and upload your data
            </p>
          </div>

          {/* Tabs */}
          <div className="flex gap-4 mb-8 flex-wrap justify-center">
            <Button
              variant={activeSection === "image" ? "default" : "outline"}
              onClick={() => switchTab("image")}
              className="flex-1 min-w-[200px] max-w-[300px]"
            >
              <Image className="mr-2 h-5 w-5" /> MRI Upload
            </Button>

            <Button
              variant={activeSection === "drawing" ? "default" : "outline"}
              onClick={() => switchTab("drawing")}
              className="flex-1 min-w-[200px] max-w-[300px]"
            >
              <Pencil className="mr-2 h-5 w-5" /> Drawing Pattern
            </Button>
          </div>

          {/* Content */}
          <Card className="p-8">
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