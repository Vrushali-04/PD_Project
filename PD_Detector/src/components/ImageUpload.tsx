import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Upload, X, Image as ImageIcon } from "lucide-react";
import { toast } from "sonner";

// Define the interface to accept the handler from PredictionForm
interface ImageUploadProps {
  onImageAnalyzed: (data: any) => void;
}

interface PredictionResponse {
  prediction?: string;
  confidence?: number;
  error?: string;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onImageAnalyzed }) => {
  const [image, setImage] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = () => setIsDragging(false);
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("image/")) {
      handleFile(droppedFile);
    } else {
      toast.error("Please upload a valid image file");
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) handleFile(selectedFile);
  };

  const handleFile = (selectedFile: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setImage(e.target?.result as string);
      setFile(selectedFile);
      onImageAnalyzed(null); // Clear previous results in parent
      toast.success("Image uploaded successfully");
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleRemove = () => {
    setImage(null);
    setFile(null);
    onImageAnalyzed(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleAnalyze = async () => {
    if (!file) {
      toast.error("Please upload an image first");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch("http://127.0.0.1:5000/predict_image", {
        method: "POST",
        body: formData,
      });

      const data: PredictionResponse = await response.json();

      // Send data to PredictionForm
      onImageAnalyzed(data);

      if (data.error) {
        toast.error("MRI Validation Failed");
      } else {
        toast.success(`Analysis complete`);
      }
    } catch (error: any) {
      console.error(error);
      toast.error("Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {!image ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
            isDragging
              ? "border-primary bg-primary/5 scale-105"
              : "border-border/50 hover:border-primary/50 hover:bg-muted/30"
          }`}
        >
          <div className="flex flex-col items-center gap-4">
            <div className="p-4 rounded-full bg-primary/10">
              <Upload className="h-10 w-10 text-primary" />
            </div>
            <div>
              <p className="text-lg font-medium mb-1">Drag and drop your MRI here</p>
              <p className="text-sm text-muted-foreground mb-4">or click to browse</p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
              <Button type="button" variant="outline" onClick={() => fileInputRef.current?.click()}>
                <ImageIcon className="mr-2 h-4 w-4" />
                Browse Files
              </Button>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative rounded-lg overflow-hidden border-2 border-primary/20">
            <img src={image} alt="MRI" className="w-full h-64 object-contain bg-muted/30" />
            <Button
              type="button"
              variant="destructive"
              size="icon"
              onClick={handleRemove}
              className="absolute top-2 right-2 rounded-full"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <Button type="button" onClick={handleAnalyze} disabled={loading} className="w-full" size="lg">
            {loading ? "Analyzing..." : (
              <>
                <ImageIcon className="mr-2 h-5 w-5" />
                Analyze MRI Image
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;