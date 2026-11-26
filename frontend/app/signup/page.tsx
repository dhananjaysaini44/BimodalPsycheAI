"use client";

import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { authApi, tokenManager } from "@/lib/api";
import type { RegisterRequest } from "@/lib/api";
import AuthLayout from "@/components/AuthLayout";

type SignupFormData = {
  firstName: string;
  lastName: string;
  email: string;
  password: string;
  confirmPassword: string;
  gender: RegisterRequest["gender"] | "";
  maritalStatus: RegisterRequest["maritalStatus"] | "";
  employmentType: RegisterRequest["employmentType"];
  jobTitle: string;
  phoneNumber: string;
  workingHoursPerWeek: string;
  dateOfBirth: string;
};

function toRegisterRequest(formData: SignupFormData, consentAccepted: boolean): RegisterRequest {
  if (!formData.gender || !formData.maritalStatus) {
    throw new Error("Select a gender and marital status to continue.");
  }

  return {
    email: formData.email.trim().toLowerCase(),
    password: formData.password,
    firstName: formData.firstName.trim(),
    lastName: formData.lastName.trim() || undefined,
    gender: formData.gender,
    maritalStatus: formData.maritalStatus,
    employmentType: formData.employmentType,
    jobTitle: formData.jobTitle.trim() || undefined,
    phoneNumber: formData.phoneNumber.trim() || undefined,
    workingHoursPerWeek: formData.workingHoursPerWeek ? parseInt(formData.workingHoursPerWeek, 10) : undefined,
    dateOfBirth: formData.dateOfBirth,
    termsAccepted: consentAccepted,
    privacyPolicyAccepted: consentAccepted,
  };
}

export default function SignupPage() {
  const router = useRouter();
  const [formData, setFormData] = useState<SignupFormData>({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
    gender: "",
    maritalStatus: "",
    employmentType: "OTHERS",
    jobTitle: "",
    phoneNumber: "",
    workingHoursPerWeek: "",
    dateOfBirth: ""
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [showOtpStep, setShowOtpStep] = useState(false);
  const [isOtpVerified, setIsOtpVerified] = useState(false);
  const [consentAccepted, setConsentAccepted] = useState(false);
  const [otp, setOtp] = useState("");
  const [isSendingOtp, setIsSendingOtp] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (formData.password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }

    if (!formData.gender || !formData.maritalStatus) {
      setError("Select a gender and marital status to continue.");
      return;
    }

    setIsLoading(true);

    try {
      await authApi.sendOtp(formData.email.trim().toLowerCase());
      setOtp("");
      setIsOtpVerified(false);
      setShowOtpStep(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send the verification code.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendOtp = async () => {
    setIsSendingOtp(true);
    setError("");
    
    try {
      await authApi.sendOtp(formData.email.trim().toLowerCase());
      setIsOtpVerified(false);
      setOtp("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send OTP. Please try again.");
    } finally {
      setIsSendingOtp(false);
    }
  };

  const handleVerifyOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      if (!isOtpVerified) {
        await authApi.verifyOtp(formData.email.trim().toLowerCase(), otp);
        setIsOtpVerified(true);
      }
      await authApi.register(toRegisterRequest(formData, consentAccepted));
      const login = await authApi.login(formData.email, formData.password);
      tokenManager.setAccessToken(login.token);
      router.replace("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not complete registration. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <AuthLayout
      eyebrow="A gentler place to begin"
      headline={<>Your journey<br /><em>starts here.</em></>}
      description="Create your account in a few simple steps. We’ll verify your email before setting up your profile."
    >
        <section className="auth-card">
          {/* Logo/Brand */}
          <div className="auth-card-heading">
            <span className="auth-card-kicker">{showOtpStep ? "EMAIL VERIFICATION" : "NEW ACCOUNT"}</span>
            <h1 
              className="text-3xl mb-2"
              style={{ 
                fontFamily: "var(--font-heading)", 
                color: "var(--color-text)",
                fontWeight: 400,
                letterSpacing: "-0.02em"
              }}
            >
              {showOtpStep ? (isOtpVerified ? "Complete Account" : "Verify Email") : "Create Account"}
            </h1>
            <p style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
              {showOtpStep
                ? isOtpVerified
                  ? "Email verified. Finish creating your account."
                  : `Enter the OTP sent to ${formData.email}`
                : "Begin your journey with PsycheAI"
              }
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 rounded-lg border text-sm" style={{ 
              background: "rgba(220, 38, 38, 0.15)", 
              borderColor: "rgba(220, 38, 38, 0.3)",
              color: "#fca5a5"
            }} role="alert">
              {error}
            </div>
          )}

          {/* Form */}
          {!showOtpStep ? (
            <form onSubmit={handleSubmit} className="auth-form auth-signup-form">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="firstName" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                  First Name *
                </label>
                <input
                  id="firstName"
                  name="firstName"
                  type="text"
                  value={formData.firstName}
                  onChange={handleChange}
                  required
                  maxLength={100}
                  autoComplete="given-name"
                  className="glass-input w-full px-4 py-3 rounded-lg"
                  placeholder="John"
                />
              </div>

              <div>
                <label htmlFor="lastName" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                  Last Name
                </label>
                <input
                  id="lastName"
                  name="lastName"
                  type="text"
                  value={formData.lastName}
                  onChange={handleChange}
                  maxLength={100}
                  autoComplete="family-name"
                  className="glass-input w-full px-4 py-3 rounded-lg"
                  placeholder="Doe"
                />
              </div>
            </div>

            <div>
              <label htmlFor="email" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                Email Address *
              </label>
              <input
                id="email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                required
                autoComplete="email"
                className="glass-input w-full px-4 py-3 rounded-lg"
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                Password *
              </label>
              <input
                id="password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                required
                minLength={8}
                maxLength={72}
                autoComplete="new-password"
                className="glass-input w-full px-4 py-3 rounded-lg"
                placeholder="••••••••"
              />
              <p className="mt-1 text-xs" style={{ color: "var(--color-muted)" }}>Must be at least 8 characters</p>
            </div>

            <div>
              <label htmlFor="confirmPassword" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                Confirm Password *
              </label>
              <input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                value={formData.confirmPassword}
                onChange={handleChange}
                required
                minLength={8}
                maxLength={72}
                autoComplete="new-password"
                className="glass-input w-full px-4 py-3 rounded-lg"
                placeholder="••••••••"
              />
            </div>

            <div>
              <label htmlFor="dateOfBirth" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                Date of Birth *
              </label>
              <input
                id="dateOfBirth"
                name="dateOfBirth"
                type="date"
                value={formData.dateOfBirth}
                onChange={handleChange}
                required
                max={new Date().toISOString().slice(0, 10)}
                className="glass-input w-full px-4 py-3 rounded-lg"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="gender" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                  Gender *
                </label>
                <select
                  id="gender"
                  name="gender"
                  value={formData.gender}
                  onChange={handleChange}
                  required
                  className="glass-input w-full px-4 py-3 rounded-lg"
                >
                  <option value="" disabled>Select gender</option>
                  <option value="MALE">Male</option>
                  <option value="FEMALE">Female</option>
                  <option value="OTHERS">Other</option>
                </select>
              </div>

              <div>
                <label htmlFor="maritalStatus" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                  Marital Status *
                </label>
                <select
                  id="maritalStatus"
                  name="maritalStatus"
                  value={formData.maritalStatus}
                  onChange={handleChange}
                  required
                  className="glass-input w-full px-4 py-3 rounded-lg"
                >
                  <option value="" disabled>Select marital status</option>
                  <option value="SINGLE">Single</option>
                  <option value="MARRIED">Married</option>
                  <option value="DIVORCED">Divorced</option>
                  <option value="WIDOWED">Widowed</option>
                </select>
              </div>
            </div>

            <div>
              <label htmlFor="employmentType" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                Employment Type *
              </label>
              <select
                id="employmentType"
                name="employmentType"
                value={formData.employmentType}
                onChange={handleChange}
                required
                className="glass-input w-full px-4 py-3 rounded-lg"
              >
                <option value="OTHERS">Other</option>
                <option value="EMPLOYED">Employed</option>
                <option value="SELF_EMPLOYED">Self-employed</option>
                <option value="UNEMPLOYED">Unemployed</option>
                <option value="STUDENT">Student</option>
                <option value="RETIRED">Retired</option>
              </select>
            </div>

            <div>
              <label htmlFor="jobTitle" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                Job Title
              </label>
              <input
                id="jobTitle"
                name="jobTitle"
                type="text"
                value={formData.jobTitle}
                onChange={handleChange}
                className="glass-input w-full px-4 py-3 rounded-lg"
                placeholder="Software Engineer"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="phoneNumber" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                  Phone Number
                </label>
                <input
                  id="phoneNumber"
                  name="phoneNumber"
                  type="tel"
                  value={formData.phoneNumber}
                  onChange={handleChange}
                  pattern="^\+?[0-9]{7,15}$"
                  autoComplete="tel"
                  className="glass-input w-full px-4 py-3 rounded-lg"
                  placeholder="+1234567890"
                />
              </div>

              <div>
                <label htmlFor="workingHoursPerWeek" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                  Working Hours/Week
                </label>
                <input
                  id="workingHoursPerWeek"
                  name="workingHoursPerWeek"
                  type="number"
                  value={formData.workingHoursPerWeek}
                  onChange={handleChange}
                  min={0}
                  max={168}
                  className="glass-input w-full px-4 py-3 rounded-lg"
                  placeholder="40"
                />
              </div>
            </div>

            <div className="flex items-start">
              <input
                type="checkbox"
                required
                checked={consentAccepted}
                onChange={(event) => setConsentAccepted(event.target.checked)}
                className="w-4 h-4 mt-0.5 rounded border-gray-600 bg-gray-900 mr-2"
                style={{ accentColor: "var(--color-accent)" }}
              />
              <span className="ml-2 text-sm" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                I agree to the{" "}
                <Link href="/terms" className="hover:underline transition-colors" style={{ color: "var(--color-accent)" }}>
                  Terms of Service
                </Link>{" "}
                and{" "}
                <Link href="/privacy" className="hover:underline transition-colors" style={{ color: "var(--color-accent)" }}>
                  Privacy Policy
                </Link>
              </span>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-4 mt-4 rounded-full text-white font-medium transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none relative overflow-hidden group"
              style={{
                background: "linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.04) 100%)",
                backdropFilter: "blur(20px)",
                WebkitBackdropFilter: "blur(20px)",
                border: "1px solid rgba(255, 255, 255, 0.22)",
                boxShadow: "0 20px 40px -15px rgba(0,0,0,0.6), inset 0 1px 1px 0 rgba(255,255,255,0.4)",
                fontFamily: "var(--font-body)"
              }}
            >
              <div className="absolute inset-0 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none bg-gradient-to-r from-white/10 via-white/20 to-white/5" />
              <span className="relative z-10">{isLoading ? "Creating account..." : "Create Account"}</span>
            </button>
          </form>
          ) : (
            <form onSubmit={handleVerifyOtp} className="auth-form">
              <div>
                <label htmlFor="otp" className="block text-sm mb-2" style={{ color: "var(--color-muted)", fontFamily: "var(--font-body)" }}>
                  One-Time Password *
                </label>
                <div className={`auth-otp ${isOtpVerified ? "is-verified" : ""}`}>
                  <div className="auth-otp-digits" aria-hidden="true">
                    {Array.from({ length: 6 }, (_, index) => (
                      <span key={index}>{otp[index] || <span className="auth-otp-placeholder">•</span>}</span>
                    ))}
                  </div>
                  <input
                    id="otp"
                    type="text"
                    value={otp}
                    onChange={(e) => setOtp(e.target.value.replace(/\D/g, "").slice(0, 6))}
                    required
                    maxLength={6}
                    disabled={isOtpVerified}
                    inputMode="numeric"
                    autoComplete="one-time-code"
                    aria-label="Six-digit email verification code"
                    className="auth-otp-input"
                  />
                </div>
                <p className="mt-2 text-sm" style={{ color: "var(--color-muted)" }}>
                  Didn&apos;t receive the code?{" "}
                  <button
                    type="button"
                    onClick={handleSendOtp}
                    disabled={isSendingOtp || isLoading}
                    className="hover:underline transition-colors"
                    style={{ color: "var(--color-accent)", background: "none", border: "none", cursor: "pointer", fontFamily: "var(--font-body)" }}
                  >
                    {isSendingOtp ? "Sending..." : "Resend OTP"}
                  </button>
                </p>
              </div>

              <button
                type="submit"
                disabled={isLoading || isSendingOtp || (!isOtpVerified && otp.length !== 6)}
                className="auth-submit"
              >
                <span>
                  {isLoading ? (isOtpVerified ? "Creating account..." : "Verifying...") : isOtpVerified ? "Complete registration" : "Verify & Continue"}
                </span>
                <span aria-hidden="true">→</span>
              </button>

              <button
                type="button"
                onClick={() => setShowOtpStep(false)}
                className="w-full py-3 text-sm transition-colors"
                style={{ color: "var(--color-muted)", background: "none", border: "none", cursor: "pointer", fontFamily: "var(--font-body)" }}
              >
                ← Back to registration
              </button>
            </form>
          )}

          {/* Sign In Link */}
          <p className="auth-switch">
            Already have an account?{" "}
            <Link 
              href="/login" 
              className="font-medium hover:underline transition-colors"
              style={{ color: "var(--color-accent)" }}
            >
              Sign in
            </Link>
          </p>
          <p className="auth-privacy-note"><span aria-hidden="true">◇</span> Your account and assessment information is used to provide this service. See our Privacy Policy for details.</p>
        </section>
    </AuthLayout>
  );
}
