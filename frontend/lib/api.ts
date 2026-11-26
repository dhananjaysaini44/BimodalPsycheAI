const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080").replace(/\/+$/, "");
let accessToken: string | null = null;
let refreshRequest: Promise<string> | null = null;

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function fetchApi(input: RequestInfo | URL, init: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (cause) {
    if (cause instanceof TypeError) {
      throw new ApiError(
        `Could not reach the backend at ${API_URL}. Check that it is running and allows this frontend origin.`,
        0,
      );
    }
    throw cause;
  }
}

async function readErrorMessage(response: Response): Promise<string> {
  const body = await response.text();
  if (!body) return `Request failed (${response.status})`;

  try {
    const payload: unknown = JSON.parse(body);
    if (typeof payload === "string") return payload;
    if (payload && typeof payload === "object" && "message" in payload) {
      const message = payload.message;
      if (typeof message === "string") return message;
    }
    if (payload && typeof payload === "object") {
      const messages = Object.values(payload).filter((value): value is string => typeof value === "string");
      if (messages.length > 0) return messages.join(" ");
    }
  } catch {
    return body;
  }

  return body;
}

export const apiClient = {
  async request<T>(
    endpoint: string,
    options: RequestInit = {},
    requireAuthentication = !endpoint.startsWith("/auth/"),
  ): Promise<T> {
    const headers = new Headers(options.headers);
    const isFormData = typeof FormData !== "undefined" && options.body instanceof FormData;
    if (options.body && !isFormData && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    if (requireAuthentication) {
      let token = tokenManager.getAccessToken();
      if (!token) {
        token = await restoreAccessToken();
      }
      headers.set("Authorization", `Bearer ${token}`);
    }

    let response = await fetchApi(`${API_URL}${endpoint}`, {
      ...options,
      headers,
      credentials: "include",
    });

    if (requireAuthentication && response.status === 401) {
      headers.set("Authorization", `Bearer ${await restoreAccessToken()}`);
      response = await fetchApi(`${API_URL}${endpoint}`, {
        ...options,
        headers,
        credentials: "include",
      });
      if (response.status === 401) {
        tokenManager.clearTokens();
        tokenManager.notifySessionExpired();
      }
    }

    if (!response.ok) {
      throw new ApiError(await readErrorMessage(response), response.status);
    }

    if (response.status === 204) return undefined as T;

    const body = await response.text();
    if (!body) return undefined as T;

    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("json")) {
      return JSON.parse(body) as T;
    }
    return body as T;
  },

  get<T>(endpoint: string): Promise<T> {
    return apiClient.request<T>(endpoint, { method: "GET" });
  },

  post<T>(endpoint: string, data?: unknown): Promise<T> {
    return apiClient.request<T>(endpoint, {
      method: "POST",
      body: data === undefined ? undefined : JSON.stringify(data),
    });
  },

  put<T>(endpoint: string, data?: unknown): Promise<T> {
    return apiClient.request<T>(endpoint, {
      method: "PUT",
      body: data === undefined ? undefined : JSON.stringify(data),
    });
  },

  patch<T>(endpoint: string, data?: unknown): Promise<T> {
    return apiClient.request<T>(endpoint, {
      method: "PATCH",
      body: data === undefined ? undefined : JSON.stringify(data),
    });
  },

  delete<T>(endpoint: string, data?: unknown): Promise<T> {
    return apiClient.request<T>(endpoint, {
      method: "DELETE",
      body: data === undefined ? undefined : JSON.stringify(data),
    });
  },

  authRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    return apiClient.request<T>(endpoint, options, true);
  },
};

export const authApi = {
  login(email: string, password: string) {
    return apiClient.post<{ message: string; token: string }>("/auth/login", {
      email: email.trim().toLowerCase(),
      password,
    });
  },

  register(data: RegisterRequest) {
    return apiClient.post<string>("/auth/register", data);
  },

  sendOtp(email: string) {
    return apiClient.post<string>("/auth/send-otp", { email });
  },

  verifyOtp(email: string, otp: string) {
    return apiClient.post<string>("/auth/verify-otp", { email, otp });
  },

  logout() {
    return apiClient.post<string>("/auth/logout", {});
  },

  refreshToken() {
    return apiClient.post<{ message: string; token: string }>("/auth/refresh", {});
  },
};

export interface UserProfile {
  id: number;
  email: string;
  firstName: string;
  lastName: string | null;
  dateOfBirth: string;
  gender: "MALE" | "FEMALE" | "OTHERS";
  maritalStatus: "SINGLE" | "MARRIED" | "DIVORCED" | "WIDOWED";
  jobTitle: string | null;
  employmentType: "STUDENT" | "SELF_EMPLOYED" | "EMPLOYED" | "UNEMPLOYED" | "OTHERS" | "RETIRED";
  phoneNumber: string | null;
  workingHoursPerWeek: number | null;
  createdAt: string;
  updatedAt: string;
}

export type UpdateProfileRequest = Omit<UserProfile, "id" | "email" | "createdAt" | "updatedAt">;

export const accountApi = {
  getProfile() {
    return apiClient.get<UserProfile>("/user/profile");
  },

  updateProfile(profile: UpdateProfileRequest) {
    return apiClient.put<UserProfile>("/user/profile", profile);
  },

  sendActionOtp() {
    return apiClient.post<string>("/user/security/send-otp", {});
  },

  changePassword(currentPassword: string, newPassword: string, otp: string) {
    return apiClient.put<void>("/user/password", { currentPassword, newPassword, otp });
  },

  deleteAccount(currentPassword: string, otp: string) {
    return apiClient.delete<void>("/user/account", { currentPassword, otp });
  },
};

export interface NextQuestionResponse {
  available: boolean;
  conversationComplete: boolean;
  question: {
    id: number;
    themeCode: string;
    themeName: string;
    variantType: "LIGHT" | "DEEP";
    questionText: string;
  } | null;
}

export interface SessionResponse {
  id: number;
  status: "DRAFT" | "IN_PROGRESS" | "PAUSED" | "COMPLETED" | "ABANDONED";
  currentTurnNumber: number;
  startedAt: string | null;
  lastActivityAt: string | null;
  completedAt: string | null;
  createdAt: string;
}

export interface CrisisResource {
  name: string;
  description: string;
  phone: string | null;
  url: string | null;
  country: string | null;
  resourceType: string;
}

export interface TurnResponse {
  id: number;
  turnNumber: number;
  questionText: string;
  sentimentScore: number;
  crisisDetected: boolean;
  crisisGuidance: string | null;
  crisisResources: CrisisResource[];
  createdAt: string;
}

export interface TurnHistoryResponse {
  id: number;
  turnNumber: number;
  questionText: string;
  responseType: "TEXT" | "AUDIO";
  responseText: string | null;
  audio: { id: number; filename: string; mimeType: string; sizeBytes: number } | null;
  sentimentScore: number | null;
  crisisDetected: boolean;
  crisisGuidance: string | null;
  crisisResources: CrisisResource[];
  createdAt: string;
}

export interface AudioTurnResponse {
  turnId: number;
  turnNumber: number;
  questionText: string;
  audioId: number;
  filename: string;
  mimeType: string;
  sizeBytes: number;
  analysisStatus: "PENDING_MODEL_ANALYSIS";
  createdAt: string;
}

export interface AssessmentResponse {
  id: number;
  sessionId: number;
  score: number;
  probability: number;
  severity: "LOW" | "MODERATE" | "HIGH";
  explanation: string | null;
  analyses: Array<{ modality: "TEXT" | "VOICE" | "FUSION"; score: number; probability: number }>;
  completedAt: string;
}

export const sessionApi = {
  list() {
    return apiClient.get<SessionResponse[]>("/session");
  },

  get(sessionId: number) {
    return apiClient.get<SessionResponse>(`/session/${sessionId}`);
  },

  start() {
    return apiClient.post<SessionResponse>("/session/start");
  },

  resume(sessionId: number) {
    return apiClient.post<SessionResponse>(`/session/resume/${sessionId}`);
  },

  abandon(sessionId: number) {
    return apiClient.post<SessionResponse>(`/session/abandon/${sessionId}`);
  },

  delete(sessionId: number) {
    return apiClient.delete<void>(`/session/${sessionId}`);
  },

  getTurns(sessionId: number) {
    return apiClient.get<TurnHistoryResponse[]>(`/session/${sessionId}/turns`);
  },

  nextQuestion(sessionId: number) {
    return apiClient.get<NextQuestionResponse>(`/session/${sessionId}/question/next`);
  },

  submitText(sessionId: number, questionVariantId: number, responseText: string) {
    return apiClient.post<TurnResponse>(`/session/${sessionId}/turns`, { questionVariantId, responseText });
  },

  submitAudio(sessionId: number, questionVariantId: number, file: File) {
    const formData = new FormData();
    formData.append("questionVariantId", String(questionVariantId));
    formData.append("file", file);
    return apiClient.request<AudioTurnResponse>(`/session/${sessionId}/turns/audio`, {
      method: "POST",
      body: formData,
    });
  },

  complete(sessionId: number) {
    return apiClient.post<SessionResponse>(`/session/complete/${sessionId}`);
  },

  generateAssessment(sessionId: number) {
    return apiClient.post<AssessmentResponse>(`/session/${sessionId}/assessment`);
  },

  getAssessment(sessionId: number) {
    return apiClient.get<AssessmentResponse>(`/session/${sessionId}/assessment`);
  },
};

async function restoreAccessToken(): Promise<string> {
  if (!refreshRequest) {
    refreshRequest = authApi.refreshToken()
      .then(({ token }) => {
        tokenManager.setAccessToken(token);
        return token;
      })
      .catch((error: unknown) => {
        tokenManager.clearTokens();
        tokenManager.notifySessionExpired();
        throw error;
      })
      .finally(() => {
        refreshRequest = null;
      });
  }
  return refreshRequest;
}

export const tokenManager = {
  setAccessToken(token: string) {
    if (typeof window !== "undefined") {
      accessToken = token;
      window.localStorage.setItem("psycheai-authenticated", "true");
    }
  },

  getAccessToken(): string | null {
    if (typeof window !== "undefined") {
      return accessToken;
    }
    return null;
  },

  clearTokens() {
    if (typeof window !== "undefined") {
      accessToken = null;
    }
  },

  hasAuthenticatedSession(): boolean {
    return typeof window !== "undefined"
      && window.localStorage.getItem("psycheai-authenticated") === "true";
  },

  clearSession() {
    this.clearTokens();
    if (typeof window !== "undefined") {
      window.localStorage.removeItem("psycheai-authenticated");
    }
  },

  notifySessionExpired() {
    if (this.hasAuthenticatedSession() && typeof window !== "undefined") {
      window.dispatchEvent(new Event("psycheai:session-expired"));
    }
  },
};

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  firstName: string;
  lastName?: string;
  gender: "MALE" | "FEMALE" | "OTHERS";
  maritalStatus: "SINGLE" | "MARRIED" | "DIVORCED" | "WIDOWED";
  jobTitle?: string;
  employmentType: "SELF_EMPLOYED" | "OTHERS" | "UNEMPLOYED" | "RETIRED" | "STUDENT" | "EMPLOYED";
  phoneNumber?: string;
  workingHoursPerWeek?: number;
  dateOfBirth: string;
  termsAccepted: boolean;
  privacyPolicyAccepted: boolean;
}

export default apiClient;
