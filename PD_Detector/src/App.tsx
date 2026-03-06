import Prediction from "./pages/Prediction";
import VoicePrediction from "./components/VoicePrediction";

import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import Login from "./pages/Login";
import Signup from "./pages/Signup";

const queryClient = new QueryClient();

/*
UPDATED ROUTING STRUCTURE:

/        → Signup page (default)
/login   → Login page
/signup  → Signup page
/home    → Dashboard
/predict → Image prediction page
/voice   → Voice prediction page
*        → 404 page
*/

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />

        <BrowserRouter>
          <Routes>
            {/* Default Route - Opens Signup First */}
            <Route path="/" element={<Signup />} />

            {/* Auth Routes */}
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />

            {/* Main Pages */}
            <Route path="/home" element={<Index />} />

            {/* Image Prediction */}
            <Route path="/predict" element={<Prediction />} />

            {/* Voice Prediction */}
            <Route path="/voice" element={<VoicePrediction />} />

            {/* 404 Page */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;