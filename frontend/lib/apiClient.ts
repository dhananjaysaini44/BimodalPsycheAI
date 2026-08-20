import axios, { AxiosResponse } from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL,
});

export const startSession = (idToken: string): Promise<AxiosResponse<any>> =>
  api.post("/session/start", {}, { headers: { Authorization: `Bearer ${idToken}` } });

export const submitTurn = (sessionId: string, payload: any, idToken: string): Promise<AxiosResponse<any>> =>
  api.post(`/session/turn`, { sessionId, ...payload }, {
    headers: { Authorization: `Bearer ${idToken}` },
  });

export const completeSession = (sessionId: string, idToken: string): Promise<AxiosResponse<any>> =>
  api.post(`/session/complete`, { sessionId }, {
    headers: { Authorization: `Bearer ${idToken}` },
  });

export const getHistory = (idToken: string): Promise<AxiosResponse<any>> =>
  api.get("/history", { headers: { Authorization: `Bearer ${idToken}` } });

export default api;
