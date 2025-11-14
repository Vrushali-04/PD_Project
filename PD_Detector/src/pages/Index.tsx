import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import PredictionForm from "@/components/PredictionForm";
import AboutSection from "@/components/AboutSection";
import TeamSection from "@/components/TeamSection";
import ContactSection from "@/components/ContactSection";

const Index = () => {
  return (
    <div className="min-h-screen">
      <Navbar />
      <HeroSection />
      <PredictionForm />
      <AboutSection />
      <TeamSection />
      <ContactSection />
    </div>
  );
};

export default Index;
