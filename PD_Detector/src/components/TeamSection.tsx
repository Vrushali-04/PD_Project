import { Card } from "@/components/ui/card";
import { Github, Linkedin, Mail } from "lucide-react";

const TeamSection = () => {
  const team = [
    {
      name: "Santosh Babar",
      role: "Backend Developer & ML Engineer",
      description: "Specialized in machine learning model development and backend API integration",
      initials: "SB",
      color: "from-primary to-secondary",
    },
    {
      name: "Pranav Nikam",
      role: "Frontend Developer",
      description: "Expert in building responsive and modern user interfaces",
      initials: "PN",
      color: "from-secondary to-accent",
    },
  ];

  return (
    <section id="team" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16 animate-fade-in">
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Our <span className="text-gradient">Team</span>
            </h2>
            <p className="text-lg text-muted-foreground">
              Meet the talented individuals behind this project
            </p>
          </div>

          {/* ✅ CENTERED GRID */}
          <div className="flex justify-center gap-8 flex-wrap">
            {team.map((member, index) => (
              <Card
                key={index}
                className="glass-card p-6 text-center w-72 hover:scale-105 transition-all duration-300 animate-fade-in-up group"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div
                  className={`w-24 h-24 rounded-full bg-gradient-to-br ${member.color} mx-auto mb-4 flex items-center justify-center text-white text-2xl font-bold shadow-lg group-hover:shadow-xl transition-shadow`}
                >
                  {member.initials}
                </div>

                <h3 className="text-xl font-bold mb-1">{member.name}</h3>
                <p className="text-primary font-medium mb-3 text-sm">
                  {member.role}
                </p>
                <p className="text-sm text-muted-foreground mb-4">
                  {member.description}
                </p>

                <div className="flex items-center justify-center gap-3">
                  <a
                    href="#"
                    className="text-muted-foreground hover:text-primary transition-colors"
                    aria-label="LinkedIn"
                  >
                    <Linkedin className="h-5 w-5" />
                  </a>
                  <a
                    href="#"
                    className="text-muted-foreground hover:text-primary transition-colors"
                    aria-label="GitHub"
                  >
                    <Github className="h-5 w-5" />
                  </a>
                  <a
                    href="#"
                    className="text-muted-foreground hover:text-primary transition-colors"
                    aria-label="Email"
                  >
                    <Mail className="h-5 w-5" />
                  </a>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default TeamSection;