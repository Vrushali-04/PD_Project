import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { Eye, EyeOff, User, Mail, Lock } from "lucide-react";

const Signup = () => {

  const [isLogin, setIsLogin] = useState(false);

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
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();

    // ✅ PASSWORD VALIDATION
    const passwordRegex =
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;

    if (!passwordRegex.test(formData.password)) {
      toast({
        title: "Weak Password",
        description:
          "Password must contain 8 characters, uppercase, lowercase, number, and special character.",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);

    try {

      const response = await fetch("http://localhost:5000/signup", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(formData)
      });

      const data = await response.json();

      if (response.ok) {

        toast({
          title: "Success!",
          description: "Account created successfully."
        });

        setIsLogin(true);

      } else {

        toast({
          title: "Signup Failed",
          description: data.message || "Something went wrong.",
          variant: "destructive"
        });

      }

    } catch (error) {

      toast({
        title: "Server Error",
        description: "Unable to connect to backend.",
        variant: "destructive"
      });

    } finally {
      setLoading(false);
    }
  };

  const handleLogin = async (e: React.FormEvent) => {

    e.preventDefault();
    setLoading(true);

    try {

      const response = await fetch("http://localhost:5000/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          email: formData.email,
          password: formData.password
        })
      });

      const data = await response.json();

      if (response.ok) {

        localStorage.setItem("user", JSON.stringify(data));

        toast({
          title: "Welcome back 👋",
          description: `Hello ${data.name}, login successful`
        });

        navigate("/home");

      } else {

        toast({
          title: "Login Failed",
          description: data.message || "Invalid email or password",
          variant: "destructive"
        });

      }

    } catch (error) {

      toast({
        title: "Server Error",
        description: "Unable to connect to backend.",
        variant: "destructive"
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
          "url('https://www.innovationnewsnetwork.com/wp-content/uploads/2024/07/shutterstock_2230363945.jpg')",
      }}
    >

      <div className="absolute inset-0 bg-black/50"></div>

      <div className="relative z-10 w-full max-w-6xl flex flex-col lg:flex-row items-center gap-8 px-6">

        {/* LEFT SIDE */}
        <div className="lg:w-1/2 text-white px-6 py-12">

          <h1 className="text-5xl font-bold mb-5 leading-tight">
            AI-Powered Parkinson’s Detection
          </h1>

          <p className="text-lg text-gray-200 mb-6 max-w-lg">
            Advanced artificial intelligence models analyze voice and
            medical imaging data to help detect early signs of Parkinson’s
            disease with clinically informed accuracy.
          </p>

          {isLogin ? (
            <>
              <p className="text-gray-300 text-sm">
                Don't have an account?
              </p>

              <button
                onClick={() => setIsLogin(false)}
                className="text-blue-300 font-semibold hover:underline"
              >
                Sign Up
              </button>
            </>
          ) : (
            <>
              <p className="text-gray-300 text-sm">
                Already have an account?
              </p>

              <button
                onClick={() => setIsLogin(true)}
                className="text-blue-300 font-semibold hover:underline"
              >
                Sign In
              </button>
            </>
          )}

        </div>

        {/* RIGHT SIDE FORM */}
        <div className="lg:w-1/2 bg-white rounded-3xl shadow-2xl p-10">

          <h2 className="text-3xl font-bold text-gray-800 mb-6">
            {isLogin ? "Sign In" : "Create Account"}
          </h2>

          <form
            onSubmit={isLogin ? handleLogin : handleSignup}
            className="space-y-4"
          >

            {!isLogin && (
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />

                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  placeholder="Full Name"
                  className="w-full pl-10 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600"
                />
              </div>
            )}

            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />

              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                placeholder="Email Address"
                className="w-full pl-10 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600"
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
                className="w-full pl-10 pr-10 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600"
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
              {loading
                ? isLogin
                  ? "Signing In..."
                  : "Creating Account..."
                : isLogin
                ? "Sign In"
                : "Create Account"}
            </button>

          </form>

          <div className="mt-5 text-center text-sm text-gray-600">

            {isLogin ? (
              <>
                Don't have an account?{" "}
                <button
                  onClick={() => setIsLogin(false)}
                  className="text-blue-600 font-semibold hover:underline"
                >
                  Sign Up
                </button>
              </>
            ) : (
              <>
                Already have an account?{" "}
                <button
                  onClick={() => setIsLogin(true)}
                  className="text-blue-600 font-semibold hover:underline"
                >
                  Sign In
                </button>
              </>
            )}

          </div>

        </div>

      </div>
    </div>
  );
};

export default Signup;