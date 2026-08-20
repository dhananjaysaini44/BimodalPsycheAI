import axios from "axios";

const ML_ENGINE_URL = process.env.ML_ENGINE_URL || "http://localhost:8000";

const mlClient = axios.create({
  baseURL: ML_ENGINE_URL,
});

export async function checkCrisis(text: string): Promise<boolean> {
  try {
    const res = await mlClient.post("/orchestrator/crisis", { text });
    return res.data?.is_crisis || false;
  } catch (error) {
    console.error("ML crisis check failed:", error);
    return false;
  }
}

export async function scoreSentiment(text: string): Promise<number> {
  try {
    const res = await mlClient.post("/orchestrator/sentiment", { text });
    return res.data?.sentiment_score ?? 0;
  } catch (error) {
    console.error("ML sentiment score failed:", error);
    return 0;
  }
}
