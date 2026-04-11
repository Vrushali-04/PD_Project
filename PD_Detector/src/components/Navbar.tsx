import { useState, useEffect } from "react";
import { Activity } from "lucide-react";
import { Button } from "@/components/ui/button";

const Navbar = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    element?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled ? "glass-navbar py-3" : "bg-transparent py-6"
      }`}
    >
      <div className="container mx-auto px-4 flex items-center justify-between">
        <div className="flex items-center gap-2 animate-fade-in">
          <Activity className="h-8 w-8 text-primary animate-pulse" />
          <span className="text-xl font-bold text-foreground">PD Predictor</span>
        </div>
        
        <div className="hidden md:flex items-center gap-1">
          <button 
            onClick={() => scrollToSection("home")}
            className="nav-link-underline px-4 py-2 text-foreground hover:text-primary transition-colors duration-300"
          >
            Home
          </button>
          <button 
            onClick={() => scrollToSection("prediction")}
            className="nav-link-underline px-4 py-2 text-foreground hover:text-primary transition-colors duration-300"
          >
            Prediction
          </button>
          <button 
            onClick={() => scrollToSection("about")}
            className="nav-link-underline px-4 py-2 text-foreground hover:text-primary transition-colors duration-300"
          >
            About
          </button>
          <button 
            onClick={() => scrollToSection("team")}
            className="nav-link-underline px-4 py-2 text-foreground hover:text-primary transition-colors duration-300"
          >
            Team
          </button>
          <button 
            onClick={() => scrollToSection("contact")}
            className="nav-link-underline px-4 py-2 text-foreground hover:text-primary transition-colors duration-300"
          >
            Contact
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
