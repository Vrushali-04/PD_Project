import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Loader2, Mic, Image, Pencil } from "lucide-react";
import { toast } from "sonner";
import VoiceFeatureInputs from "./VoiceFeatureInputs";
import ImageUpload from "./ImageUpload";
import DrawingCanvas from "./DrawingCanvas";
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
  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState(85);
  const [activeSection, setActiveSection] = useState<"voice" | "image" | "drawing">("voice");

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const simulatePrediction = (data: FormData): { result: "healthy" | "detected"; confidence: number } => {
    // Simulated ML prediction logic
    // In production, this would call Flask backend API at /predict_voice
    const values = Object.values(data).map(Number);
    const avgValue = values.reduce((a, b) => a + b, 0) / values.length;
    
    // Simple simulation with confidence
    const result = avgValue > 0.5 ? "detected" : "healthy";
    const confidence = Math.floor(Math.random() * 15) + 80; // 80-95%
    return { result, confidence };
  };

  const handleImageAnalyzed = (imageResult: string) => {
    setResult(imageResult as "healthy" | "detected");
    setConfidence(Math.floor(Math.random() * 15) + 80);
  };

  const handlePatternAnalyzed = (patternResult: string) => {
    setResult(patternResult as "healthy" | "detected");
    setConfidence(Math.floor(Math.random() * 15) + 80);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate all fields are filled
    if (Object.values(formData).some(val => val === "")) {
      toast.error("Please fill in all fields");
      return;
    }

    setLoading(true);
    setResult(null);

    // Simulate API call delay
    setTimeout(() => {
      const prediction = simulatePrediction(formData);
      setResult(prediction.result);
      setConfidence(prediction.confidence);
      setLoading(false);
      
      if (prediction.result === "healthy") {
        toast.success("Analysis complete!");
      } else {
        toast.warning("Analysis complete - Please consult a medical professional");
      }
    }, 2000);
  };

  return (
    <section id="prediction" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12 animate-fade-in">
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Disease <span className="text-gradient">Prediction</span>
            </h2>
            <p className="text-lg text-muted-foreground">
              Enter biomedical voice measurement parameters below
            </p>
          </div>

          {/* Section Selector Tabs */}
          <div className="flex gap-4 mb-8 flex-wrap justify-center">
            <Button
              type="button"
              variant={activeSection === "voice" ? "default" : "outline"}
              onClick={() => setActiveSection("voice")}
              className="flex-1 min-w-[200px] h-auto py-4 px-6"
            >
              <Mic className="mr-2 h-5 w-5" />
              <div className="text-left">
                <div className="font-semibold">Voice Features</div>
                <div className="text-xs opacity-80">Biomedical parameters</div>
              </div>
            </Button>
            <Button
              type="button"
              variant={activeSection === "image" ? "default" : "outline"}
              onClick={() => setActiveSection("image")}
              className="flex-1 min-w-[200px] h-auto py-4 px-6"
            >
              <Image className="mr-2 h-5 w-5" />
              <div className="text-left">
                <div className="font-semibold">Image Upload</div>
                <div className="text-xs opacity-80">Medical patterns</div>
              </div>
            </Button>
            <Button
              type="button"
              variant={activeSection === "drawing" ? "default" : "outline"}
              onClick={() => setActiveSection("drawing")}
              className="flex-1 min-w-[200px] h-auto py-4 px-6"
            >
              <Pencil className="mr-2 h-5 w-5" />
              <div className="text-left">
                <div className="font-semibold">Drawing Pattern</div>
                <div className="text-xs opacity-80">Tremor analysis</div>
              </div>
            </Button>
          </div>

          {/* Voice Features Section */}
          {activeSection === "voice" && (
            <Card className="glass-card p-8 animate-scale-in">
              <div className="mb-6">
                <h3 className="text-2xl font-bold flex items-center gap-2 mb-2">
                  <Mic className="h-6 w-6 text-primary" />
                  Voice Features Input
                </h3>
                <p className="text-sm text-muted-foreground">
                  Enter biomedical voice measurement parameters below
                </p>
              </div>
              
              <form onSubmit={handleSubmit} className="space-y-6">
                <VoiceFeatureInputs formData={formData} onChange={handleInputChange} />

                <Button
                  type="submit"
                  size="lg"
                  className="w-full font-semibold text-lg"
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

              {result && <PredictionResult result={result} confidence={confidence} />}
            </Card>
          )}

          {/* Image Upload Section */}
          {activeSection === "image" && (
            <Card className="glass-card p-8 animate-scale-in">
              <div className="mb-6">
                <h3 className="text-2xl font-bold flex items-center gap-2 mb-2">
                  <Image className="h-6 w-6 text-primary" />
                  Upload Biomedical or Pattern Images
                </h3>
                <p className="text-sm text-muted-foreground">
                  Upload brain scan or medical pattern images for AI-based analysis
                </p>
              </div>

              <ImageUpload onImageAnalyzed={handleImageAnalyzed} />

              {result && <PredictionResult result={result} confidence={confidence} />}
            </Card>
          )}

          {/* Drawing Canvas Section */}
          {activeSection === "drawing" && (
            <Card className="glass-card p-8 animate-scale-in">
              <div className="mb-6">
                <h3 className="text-2xl font-bold flex items-center gap-2 mb-2">
                  <Pencil className="h-6 w-6 text-primary" />
                  Handwriting Pattern / Drawing Input
                </h3>
                <p className="text-sm text-muted-foreground">
                  Draw a pattern to analyze tremor characteristics
                </p>
              </div>

              <DrawingCanvas onPatternAnalyzed={handlePatternAnalyzed} />

              {result && <PredictionResult result={result} confidence={confidence} />}
            </Card>
          )}
        </div>
      </div>
    </section>
  );
};

export default PredictionForm;
