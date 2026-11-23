import { authApi } from "./api";
import type { LoginRequest, RegisterRequest } from "./api";

export type { LoginRequest, RegisterRequest } from "./api";

export interface OtpVerificationRequest {
  email: string;
  otp: string;
}

export interface SendOtpRequest {
  email: string;
}

export interface LoginResponse {
  message: string;
  token: string;
}

export const authService = {
  register(data: RegisterRequest) {
    return authApi.register(data);
  },

  login(data: LoginRequest) {
    return authApi.login(data.email, data.password);
  },

  verifyOtp(data: OtpVerificationRequest) {
    return authApi.verifyOtp(data.email, data.otp);
  },

  sendOtp(data: SendOtpRequest) {
    return authApi.sendOtp(data.email);
  },

  refreshToken() {
    return authApi.refreshToken();
  },

  logout() {
    return authApi.logout();
  },
};

export default authService;
