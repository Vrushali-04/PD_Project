const HeroSection = () => {
  const scrollToPrediction = () => {
    const element = document.getElementById("prediction");
    element?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section id="home" className="min-h-screen flex items-center justify-center hero-gradient-clean relative overflow-hidden">
      {/* Subtle wave background */}
      <div className="absolute inset-0 wave-pattern opacity-30" />
      
      <div className="container mx-auto px-4 py-20 text-center relative z-10">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Main Title - with slide-in and gradient flow animation */}
          <h1 className="text-5xl md:text-7xl font-bold leading-tight animate-slide-in-left text-gradient">
            Parkinson's Disease
            <br />
            Prediction System
          </h1>
          
          {/* Subtitle */}
          <p className="text-xl md:text-2xl text-[#64748b] font-medium max-w-2xl mx-auto animate-fade-in" style={{ animationDelay: "0.2s" }}>
            Early detection can make a big difference using AI and biomedical analysis
          </p>
          
          {/* CTA Button - with smooth fade-in */}
          <div className="pt-4">
            <button
              onClick={scrollToPrediction}
              className="cta-button-gradient font-semibold px-10 py-4 text-lg rounded-full shadow-lg hover:shadow-xl transition-all duration-300 inline-flex items-center gap-2 animate-button-fade-in"
            >
              Get Started / Try Prediction
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
