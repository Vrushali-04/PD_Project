// ✅ API Base URL — your Flask backend
export const API_BASE_URL = "http://127.0.0.1:5000";

// 🎙 Voice Prediction API
export async function predictVoice(data: any) {
  const response = await fetch(`${API_BASE_URL}/predict_voice`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return response.json();
}

// 🧠 Image Prediction API
export async function predictImage(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/predict_image`, {
    method: "POST",
    body: formData,
  });
  return response.json();
}
