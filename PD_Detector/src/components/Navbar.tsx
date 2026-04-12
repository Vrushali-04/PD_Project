import { useState, useEffect, useRef } from "react";
import { Activity, User as UserIcon } from "lucide-react";
import { useNavigate } from "react-router-dom";

const Navbar = () => {
  const [scrolled, setScrolled] = useState(false);
  const [user, setUser] = useState<{ name: string; email: string } | null>(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const navigate = useNavigate();
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Detect scroll
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Get user from localStorage
  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) setUser(JSON.parse(storedUser));
  }, []);

  // Close dropdown if clicked outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Logout → redirect to signup
  const handleLogout = () => {
    localStorage.clear();
    navigate("/signup");
  };

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
        {/* Logo */}
        <div className="flex items-center gap-2 animate-fade-in">
          <Activity className="h-8 w-8 text-primary animate-pulse" />
          <span className="text-xl font-bold text-foreground">PD Predictor</span>
        </div>

        {/* Navigation links */}
        <div className="hidden md:flex items-center gap-1">
          <button onClick={() => scrollToSection("home")} className="nav-link-underline px-4 py-2 text-foreground hover:text-primary transition-colors duration-300">Home</button>
          <button onClick={() => scrollToSection("prediction")} className="nav-link-underline px-4 py-2 text-foreground hover:text-primary transition-colors duration-300">Prediction</button>
          <button onClick={() => scrollToSection("about")} className="nav-link-underline px-4 py-2 text-foreground hover:text-primary transition-colors duration-300">About</button>
          <button onClick={() => scrollToSection("team")} className="nav-link-underline px-4 py-2 text-foreground hover:text-primary transition-colors duration-300">Team</button>
          <button onClick={() => scrollToSection("contact")} className="nav-link-underline px-4 py-2 text-foreground hover:text-primary transition-colors duration-300">Contact</button>

          {/* Profile Dropdown */}
          {user && (
            <div className="relative" ref={dropdownRef}>
              {/* Clean Profile button without white rectangle */}
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                className="px-4 py-2 text-foreground hover:text-primary transition-colors duration-300 rounded"
              >
                Profile
              </button>

              {dropdownOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-white border border-gray-200 rounded shadow-lg py-2 z-50">
                  <div className="flex items-center gap-2 px-4 py-2">
                    {/* User icon */}
                    <UserIcon className="h-5 w-5 text-gray-700" />
                    <span className="text-gray-700 font-medium">{user.name}</span>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="w-full text-left px-4 py-2 text-red-500 hover:bg-red-50 transition-colors duration-200"
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;