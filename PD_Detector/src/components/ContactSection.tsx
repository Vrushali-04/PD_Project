import { Card } from "@/components/ui/card";
import { Mail, Phone } from "lucide-react";

const ContactSection = () => {
  return (
    <section id="contact" className="py-20 bg-background">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12 animate-fade-in">
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Get In <span className="text-gradient">Touch</span>
            </h2>
            <p className="text-lg text-muted-foreground">
              Have questions? We'd love to hear from you
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6 mb-12 max-w-2xl mx-auto">
            <Card className="glass-card p-6 text-center hover:scale-105 transition-transform duration-300 animate-scale-in">
              <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-primary/10 mb-4">
                <Mail className="h-7 w-7 text-primary" />
              </div>
              <h3 className="font-semibold mb-2">Email</h3>
              <a 
                href="mailto:teamparkinson@gmail.com" 
                className="text-sm text-muted-foreground hover:text-primary transition-colors"
              >
                teamparkinson@gmail.com
              </a>
            </Card>

            <Card className="glass-card p-6 text-center hover:scale-105 transition-transform duration-300 animate-scale-in" style={{ animationDelay: "0.1s" }}>
              <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-primary/10 mb-4">
                <Phone className="h-7 w-7 text-primary" />
              </div>
              <h3 className="font-semibold mb-2">Phone</h3>
              <p className="text-sm text-muted-foreground">
                +1 (555) 123-4567
              </p>
            </Card>
          </div>
        </div>
      </div>

      <footer className="bg-gradient-to-r from-primary to-secondary text-white py-8 mt-12">
        <div className="container mx-auto px-4">
          <div className="text-center">
            <p className="text-sm opacity-90">
              Together, we can make a difference in early diagnosis and patient care — contact us today.
            </p>
          </div>
        </div>
      </footer>
    </section>
  );
};

export default ContactSection;
