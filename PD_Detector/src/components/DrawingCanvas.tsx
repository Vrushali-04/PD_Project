import { useRef, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Eraser, Pencil, RotateCcw } from "lucide-react";
import { toast } from "sonner";

interface DrawingCanvasProps {
  onPatternAnalyzed: (result: string) => void;
}

const DrawingCanvas = ({ onPatternAnalyzed }: DrawingCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasDrawing, setHasDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = 300;

    // Set drawing styles
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 2;
    ctx.strokeStyle = "hsl(var(--primary))";

    // Fill with white background
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = 'touches' in e ? e.touches[0].clientX - rect.left : e.clientX - rect.left;
    const y = 'touches' in e ? e.touches[0].clientY - rect.top : e.clientY - rect.top;

    setIsDrawing(true);
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = 'touches' in e ? e.touches[0].clientX - rect.left : e.clientX - rect.left;
    const y = 'touches' in e ? e.touches[0].clientY - rect.top : e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    setHasDrawing(true);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setHasDrawing(false);
    toast.success("Canvas cleared");
  };

  const analyzePattern = () => {
    if (!hasDrawing) {
      toast.error("Please draw a pattern first");
      return;
    }

    // Simulate pattern analysis
    toast.info("Analyzing tremor pattern...");
    setTimeout(() => {
      // Simulated result - in production, this would call Flask backend
      const result = Math.random() > 0.5 ? "healthy" : "detected";
      onPatternAnalyzed(result);
      toast.success("Pattern analysis complete");
    }, 2000);
  };

  return (
    <div className="space-y-4">
      <div className="text-sm text-muted-foreground">
        <p className="mb-2">
          Draw a simple spiral or line pattern to help analyze tremor characteristics.
        </p>
      </div>

      <div className="border-2 border-border/50 rounded-lg overflow-hidden bg-white">
        <canvas
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
          className="w-full cursor-crosshair touch-none"
          style={{ height: "300px" }}
        />
      </div>

      <div className="flex gap-3">
        <Button
          type="button"
          variant="outline"
          onClick={clearCanvas}
          className="flex-1 hover:border-destructive hover:text-destructive transition-colors"
        >
          <RotateCcw className="mr-2 h-4 w-4" />
          Clear
        </Button>
        <Button
          type="button"
          onClick={analyzePattern}
          className="flex-1"
          size="lg"
        >
          <Pencil className="mr-2 h-5 w-5" />
          Analyze Pattern
        </Button>
      </div>
    </div>
  );
};

export default DrawingCanvas;
