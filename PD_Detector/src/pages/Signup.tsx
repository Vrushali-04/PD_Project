import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { Eye, EyeOff, User, Mail, Lock, CheckCircle } from "lucide-react";

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
          title: "Success! 🎉",
          description: "Your account has been created successfully.",
        });
        setTimeout(() => navigate("/login"), 1500);
      } else {
        toast({
          title: "Signup Failed",
          description: data.message || "Something went wrong.",
          variant: "destructive",
        });
      }
    } catch {
      toast({
        title: "Error",
        description: "Unable to connect to server.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex">

      {/* ---------------- LEFT LANDING SECTION ---------------- */}
      <div className="hidden lg:flex w-1/2 bg-gradient-to-br from-cyan-500 to-blue-600 text-white p-16 flex-col justify-center">
        
        <h1 className="text-5xl font-bold mb-6 leading-tight">
          AI Powered <br /> Parkinson’s Detection
        </h1>

        <p className="text-lg text-cyan-100 mb-8 max-w-md">
          Our platform uses advanced machine learning models to detect early
          signs of Parkinson’s disease using voice and medical image analysis.
          Early screening leads to better outcomes.
        </p>

        <img
          src="https://img.freepik.com/free-vector/parkinson-disease-concept-illustration_114360-949.jpg"
          alt="Parkinson Detection"
          className="w-full max-w-md rounded-xl shadow-lg"
        />

        <div className="mt-8">
          <p className="text-cyan-100">
            Already registered?
          </p>
          <Link
            to="/login"
            className="inline-block mt-2 bg-white text-blue-600 px-6 py-2 rounded-full font-semibold hover:shadow-lg transition"
          >
            Login Here
          </Link>
        </div>
      </div>

      {/* ---------------- RIGHT SIGNUP FORM ---------------- */}
      <div className="w-full lg:w-1/2 flex items-center justify-center bg-gray-50 p-6">
        <div className="w-full max-w-md bg-white rounded-2xl shadow-xl overflow-hidden">

          <div className="bg-gradient-to-r from-cyan-500 to-blue-600 px-8 py-6 text-white rounded-t-2xl">
            <h2 className="text-2xl font-bold">Create Account</h2>
            <p className="text-cyan-100 text-sm">
              Join our healthcare platform today
            </p>
          </div>

          <div className="p-6">
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
                  className="w-full pl-10 pr-3 py-3 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-cyan-400"
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
                  className="w-full pl-10 pr-3 py-3 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-cyan-400"
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
                  className="w-full pl-10 pr-10 py-3 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-cyan-400"
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
                className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-bold py-3 rounded-lg transition-all hover:shadow-lg disabled:opacity-50"
              >
                {loading ? "Creating..." : "Create Account"}
              </button>
            </form>

            <div className="mt-5 text-center text-sm">
              Already have an account?{" "}
              <Link to="/login" className="text-cyan-600 font-semibold hover:underline">
                Sign In
              </Link>
            </div>

            <div className="mt-6 pt-4 border-t border-gray-200">
              <div className="flex justify-center gap-4 text-xs text-gray-600">
                <span className="flex items-center gap-1">
                  <CheckCircle className="w-3 h-3 text-green-500" /> Secure
                </span>
                <span className="flex items-center gap-1">
                  <CheckCircle className="w-3 h-3 text-blue-500" /> HIPAA
                </span>
                <span className="flex items-center gap-1">
                  <CheckCircle className="w-3 h-3 text-purple-500" /> Encrypted
                </span>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
};

export default Signup;