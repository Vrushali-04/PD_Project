import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { Eye, EyeOff, User, Mail, Lock } from "lucide-react";

const Signup = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
  });

  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();
  const { toast } = useToast();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (response.ok) {
        toast({
          title: "Success!",
          description: "Account created successfully.",
        });
        setTimeout(() => navigate("/login"), 1500);
      } else {
        toast({
          title: "Signup Failed",
          description: data.message || "Something went wrong.",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Server Error",
        description: "Unable to connect to backend.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen bg-cover bg-center bg-no-repeat flex items-center justify-center"
      style={{
        backgroundImage:
          "url('https://images.pexels.com/photos/7045691/pexels-photo-7045691.jpeg?auto=compress&cs=tinysrgb&w=1600')",
      }}
    >
      {/* Dark overlay to make text readable */}
      <div className="absolute inset-0 bg-black/40"></div>

      <div className="relative z-10 w-full max-w-5xl flex flex-col lg:flex-row items-center">

        {/* LEFT HERO CONTENT */}
        <div className="lg:w-1/2 text-white px-8 py-12">
          <h1 className="text-5xl font-bold mb-4">
            AI Powered Parkinson’s Detection
          </h1>

          <p className="text-lg text-gray-200 mb-6 max-w-lg">
            Detect early signs of Parkinson’s disease using state-of-the-art
            machine learning on medical scans and voice analysis.
            Empower healthcare with intelligent diagnostics.
          </p>

          <p className="text-gray-300 text-sm">
            Already have an account?
          </p>
          <Link
            to="/login"
            className="text-blue-300 font-semibold hover:underline"
          >
            Sign In
          </Link>
        </div>

        {/* RIGHT SIGNUP FORM */}
        <div className="lg:w-1/2 bg-white rounded-3xl shadow-xl p-10 mx-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-4">
            Create Your Account
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">

            <div className="relative">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleChange}
                required
                placeholder="Full Name"
                className="w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                placeholder="Email Address"
                className="w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                placeholder="Password"
                className="w-full pl-10 pr-10 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400"
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 text-white font-semibold py-3 rounded-lg hover:bg-blue-700 transition"
            >
              {loading ? "Creating Account..." : "Create Account"}
            </button>

          </form>

          <div className="mt-5 text-center text-sm text-gray-600">
            Already have an account?{" "}
            <Link
              to="/login"
              className="text-blue-600 font-semibold hover:underline"
            >
              Sign In
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Signup;