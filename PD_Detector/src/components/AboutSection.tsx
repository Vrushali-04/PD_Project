import { Card } from "@/components/ui/card";
import healthcareAiImage from "@/assets/healthcare-ai.png";

const AboutSection = () => {
  return (
    <section id="about" className="py-24 bg-gradient-to-b from-[#e6f3ff] via-[#f0f9ff] to-white relative overflow-hidden">
      {/* Subtle wave background */}
      <div className="absolute inset-0 wave-pattern opacity-20" />
      
      <div className="container mx-auto px-4 relative z-10">
        <div className="max-w-6xl mx-auto">
          {/* Split Two-Column Layout */}
          <div className="grid md:grid-cols-2 gap-12 items-center">
            {/* Left side - Healthcare AI Image */}
            <div className="animate-fade-in">
              <div className="relative">
                <div className="aspect-square rounded-2xl overflow-hidden shadow-2xl">
                  <img 
                    src={healthcareAiImage}
                    alt="Healthcare AI - Neural networks and brain analysis visualization" 
                    className="w-full h-full object-cover"
                  />
                </div>
                {/* Floating decoration */}
                <div className="absolute -top-6 -right-6 w-32 h-32 bg-[#00c6ff]/10 rounded-full blur-2xl" />
                <div className="absolute -bottom-6 -left-6 w-40 h-40 bg-[#0072ff]/10 rounded-full blur-3xl" />
              </div>
            </div>

            {/* Right side - Text Content */}
            <div className="animate-fade-in" style={{ animationDelay: "0.2s" }}>
              <Card className="bg-white/80 backdrop-blur-sm p-10 shadow-xl rounded-2xl border-0">
                <h2 className="text-4xl font-bold mb-6 text-[#1e3a8a]">About Our Mission</h2>
                <div className="space-y-6 text-[#475569] leading-relaxed">
                  <p className="text-lg">
                    Our mission is to use <span className="font-semibold text-[#0072ff]">Artificial Intelligence</span> to assist in early-stage Parkinson's detection and help improve patient outcomes through technology.
                  </p>
                  <p className="text-lg">
                    We strive to make AI-based screening accessible, non-invasive, and reliable for better healthcare decisions.
                  </p>
                </div>
                
                {/* Trust indicators */}
                <div className="mt-10 pt-8 border-t border-[#e2e8f0]">
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="space-y-2">
                      <div className="w-3 h-3 rounded-full bg-[#0072ff] mx-auto animate-pulse-gentle" />
                      <span className="text-sm font-medium text-[#64748b]">AI-Driven</span>
                    </div>
                    <div className="space-y-2">
                      <div className="w-3 h-3 rounded-full bg-[#00C9A7] mx-auto animate-pulse-gentle" style={{ animationDelay: "0.3s" }} />
                      <span className="text-sm font-medium text-[#64748b]">Medically Inspired</span>
                    </div>
                    <div className="space-y-2">
                      <div className="w-3 h-3 rounded-full bg-[#007BFF] mx-auto animate-pulse-gentle" style={{ animationDelay: "0.6s" }} />
                      <span className="text-sm font-medium text-[#64748b]">Human-Centered</span>
                    </div>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;
