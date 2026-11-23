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
    if (options.body && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    if (requireAuthentication) {
      let token = tokenManager.getAccessToken();
      if (!token) {
        token = await restoreAccessToken();
      }
      headers.set("Authorization", `Bearer ${token}`);
    }

    let response = await fetch(`${API_URL}${endpoint}`, {
      ...options,
      headers,
      credentials: "include",
    });

    if (requireAuthentication && response.status === 401) {
      headers.set("Authorization", `Bearer ${await restoreAccessToken()}`);
      response = await fetch(`${API_URL}${endpoint}`, {
        ...options,
        headers,
        credentials: "include",
      });
      if (response.status === 401) tokenManager.clearTokens();
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

  delete<T>(endpoint: string): Promise<T> {
    return apiClient.request<T>(endpoint, { method: "DELETE" });
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

async function restoreAccessToken(): Promise<string> {
  if (!refreshRequest) {
    refreshRequest = authApi.refreshToken()
      .then(({ token }) => {
        tokenManager.setAccessToken(token);
        return token;
      })
      .catch((error: unknown) => {
        tokenManager.clearTokens();
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
}

export default apiClient;
