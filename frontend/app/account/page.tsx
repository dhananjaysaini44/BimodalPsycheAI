"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { accountApi, ApiError, authApi, tokenManager } from "@/lib/api";
import type { UpdateProfileRequest, UserProfile } from "@/lib/api";

type ProfileForm = {
  firstName: string;
  lastName: string;
  dateOfBirth: string;
  gender: UpdateProfileRequest["gender"];
  maritalStatus: UpdateProfileRequest["maritalStatus"];
  jobTitle: string;
  employmentType: UpdateProfileRequest["employmentType"];
  phoneNumber: string;
  workingHoursPerWeek: string;
};

const emptyProfile: ProfileForm = {
  firstName: "",
  lastName: "",
  dateOfBirth: "",
  gender: "OTHERS",
  maritalStatus: "SINGLE",
  jobTitle: "",
  employmentType: "OTHERS",
  phoneNumber: "",
  workingHoursPerWeek: "",
};

function profileToForm(profile: UserProfile): ProfileForm {
  return {
    firstName: profile.firstName,
    lastName: profile.lastName ?? "",
    dateOfBirth: profile.dateOfBirth,
    gender: profile.gender,
    maritalStatus: profile.maritalStatus,
    jobTitle: profile.jobTitle ?? "",
    employmentType: profile.employmentType,
    phoneNumber: profile.phoneNumber ?? "",
    workingHoursPerWeek: profile.workingHoursPerWeek?.toString() ?? "",
  };
}

export default function AccountPage() {
  const router = useRouter();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [form, setForm] = useState<ProfileForm>(emptyProfile);
  const [otp, setOtp] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [deletePassword, setDeletePassword] = useState("");
  const [deleteConfirmed, setDeleteConfirmed] = useState(false);
  const [otpSent, setOtpSent] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    let active = true;
    async function loadProfile() {
      try {
        if (!tokenManager.getAccessToken()) {
          const response = await authApi.refreshToken();
          tokenManager.setAccessToken(response.token);
        }
        const result = await accountApi.getProfile();
        if (active) {
          setProfile(result);
          setForm(profileToForm(result));
        }
      } catch (cause) {
        if (!active) return;
        if (cause instanceof ApiError && [401, 403].includes(cause.status)) {
          router.replace(cause instanceof ApiError && cause.status === 401 && tokenManager.hasAuthenticatedSession() ? "/login?sessionExpired=1" : "/login");
        } else {
          setError(cause instanceof Error ? cause.message : "Could not load your account.");
        }
      } finally {
        if (active) setIsLoading(false);
      }
    }
    void loadProfile();
    return () => {
      active = false;
    };
  }, [router]);

  function updateField<K extends keyof ProfileForm>(key: K, value: ProfileForm[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function saveProfile(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSaving(true);
    setError("");
    setNotice("");
    try {
      const saved = await accountApi.updateProfile({
        ...form,
        lastName: form.lastName.trim() || null,
        jobTitle: form.jobTitle.trim() || null,
        phoneNumber: form.phoneNumber.trim() || null,
        workingHoursPerWeek: form.workingHoursPerWeek === "" ? null : Number(form.workingHoursPerWeek),
      });
      setProfile(saved);
      setForm(profileToForm(saved));
      setNotice("Your profile has been updated.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save your profile.");
    } finally {
      setIsSaving(false);
    }
  }

  async function sendOtp() {
    setIsSaving(true);
    setError("");
    setNotice("");
    try {
      await accountApi.sendActionOtp();
      setOtpSent(true);
      setOtp("");
      setNotice(`A verification code was sent to ${profile?.email ?? "your account email"}. It expires in five minutes.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not send a verification code.");
    } finally {
      setIsSaving(false);
    }
  }

  async function changePassword(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSaving(true);
    setError("");
    setNotice("");
    try {
      await accountApi.changePassword(currentPassword, newPassword, otp);
      tokenManager.clearSession();
      router.replace("/login?message=password-updated");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not change your password.");
    } finally {
      setIsSaving(false);
    }
  }

  async function deleteAccount(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSaving(true);
    setError("");
    setNotice("");
    try {
      await accountApi.deleteAccount(deletePassword, otp);
      tokenManager.clearSession();
      router.replace("/");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not delete your account.");
    } finally {
      setIsSaving(false);
    }
  }

  const inputClass = "account-field";

  return (
    <main className="service-page account-page">
      <video className="auth-background-video" autoPlay muted loop playsInline aria-hidden="true">
        <source src="/videos/beauty-fx30.mp4" type="video/mp4" />
      </video>
      <div className="auth-background-shade" aria-hidden="true" />
      <header className="service-topbar">
        <Link href="/" className="auth-brand" aria-label="PsycheAI home">
          <span className="auth-brand-mark" aria-hidden="true">Ψ</span><span>PsycheAI</span>
        </Link>
        <div className="service-topbar-actions">
          <Link href="/dashboard" className="auth-home-link">Dashboard</Link>
          <Link href="/" className="auth-home-link">Home</Link>
        </div>
      </header>

      <section className="service-card account-card">
        <p className="service-eyebrow">Account settings</p>
        <h1>Your account</h1>
        <p className="service-description">Update your profile and secure your account. Your email address is the sign-in identity and cannot be changed here.</p>
        {error && <p role="alert" className="service-error">{error}</p>}
        {notice && <p role="status" className="history-notice">{notice}</p>}

        {isLoading ? <p className="history-loading">Loading your account…</p> : profile && (
          <>
            <form className="account-form" onSubmit={saveProfile}>
              <h2>Profile details</h2>
              <label className="account-label" htmlFor="account-email">Email</label>
              <input id="account-email" className={inputClass} value={profile.email} readOnly aria-describedby="email-immutable-note" />
              <p id="email-immutable-note" className="account-hint">Email cannot be changed.</p>
              <div className="account-grid">
                <label className="account-label">First name
                  <input className={inputClass} required maxLength={100} value={form.firstName} onChange={(e) => updateField("firstName", e.target.value)} />
                </label>
                <label className="account-label">Last name
                  <input className={inputClass} maxLength={100} value={form.lastName} onChange={(e) => updateField("lastName", e.target.value)} />
                </label>
                <label className="account-label">Date of birth
                  <input className={inputClass} type="date" required value={form.dateOfBirth} onChange={(e) => updateField("dateOfBirth", e.target.value)} />
                </label>
                <label className="account-label">Gender
                  <select className={inputClass} value={form.gender} onChange={(e) => updateField("gender", e.target.value as ProfileForm["gender"])}>
                    <option value="MALE">Male</option><option value="FEMALE">Female</option><option value="OTHERS">Other</option>
                  </select>
                </label>
                <label className="account-label">Marital status
                  <select className={inputClass} value={form.maritalStatus} onChange={(e) => updateField("maritalStatus", e.target.value as ProfileForm["maritalStatus"])}>
                    <option value="SINGLE">Single</option><option value="MARRIED">Married</option><option value="DIVORCED">Divorced</option><option value="WIDOWED">Widowed</option>
                  </select>
                </label>
                <label className="account-label">Employment
                  <select className={inputClass} value={form.employmentType} onChange={(e) => updateField("employmentType", e.target.value as ProfileForm["employmentType"])}>
                    <option value="STUDENT">Student</option><option value="SELF_EMPLOYED">Self-employed</option><option value="EMPLOYED">Employed</option><option value="UNEMPLOYED">Unemployed</option><option value="OTHERS">Other</option><option value="RETIRED">Retired</option>
                  </select>
                </label>
                <label className="account-label">Job title
                  <input className={inputClass} maxLength={100} value={form.jobTitle} onChange={(e) => updateField("jobTitle", e.target.value)} />
                </label>
                <label className="account-label">Phone number
                  <input className={inputClass} type="tel" pattern="^\\+?[0-9]{7,15}$" value={form.phoneNumber} onChange={(e) => updateField("phoneNumber", e.target.value)} />
                </label>
                <label className="account-label">Working hours per week
                  <input className={inputClass} type="number" min={0} max={168} value={form.workingHoursPerWeek} onChange={(e) => updateField("workingHoursPerWeek", e.target.value)} />
                </label>
              </div>
              <button className="service-home-button" type="submit" disabled={isSaving}>{isSaving ? "Saving…" : "Save profile"}</button>
            </form>

            <section className="account-form account-security">
              <h2>Email verification</h2>
              <p className="account-hint">Request a six-digit code at {profile.email} before changing your password or deleting your account. Codes expire in five minutes and are single-use.</p>
              <button className="account-secondary-button" type="button" onClick={() => void sendOtp()} disabled={isSaving}>
                {isSaving ? "Sending…" : otpSent ? "Send a new code" : "Send verification code"}
              </button>
              {otpSent && <label className="account-label account-otp-label" htmlFor="security-otp">Email verification code
                <input id="security-otp" className={inputClass} inputMode="numeric" autoComplete="one-time-code" pattern="[0-9]{6}" maxLength={6} required value={otp} onChange={(e) => setOtp(e.target.value.replace(/\D/g, "").slice(0, 6))} />
              </label>}
            </section>

            <form className="account-form account-security" onSubmit={changePassword}>
              <h2>Change password</h2>
              <label className="account-label">Current password
                <input className={inputClass} type="password" autoComplete="current-password" required maxLength={72} value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} />
              </label>
              <label className="account-label">New password
                <input className={inputClass} type="password" autoComplete="new-password" required minLength={8} maxLength={72} value={newPassword} onChange={(e) => setNewPassword(e.target.value)} />
              </label>
              <button className="service-home-button" type="submit" disabled={isSaving || !otpSent || otp.length !== 6}>{isSaving ? "Updating…" : "Change password"}</button>
              <p className="account-hint">You’ll be signed out on all devices after the password changes.</p>
            </form>

            <form className="account-form account-danger-zone" onSubmit={deleteAccount}>
              <h2>Delete account</h2>
              <p>This permanently deletes your account and assessment records. The application also attempts to remove stored audio after deletion; filesystem cleanup failures require operator follow-up. This action cannot be undone.</p>
              <label className="account-label">Current password
                <input className={inputClass} type="password" autoComplete="current-password" required maxLength={72} value={deletePassword} onChange={(e) => setDeletePassword(e.target.value)} />
              </label>
              <label className="account-confirm-delete"><input type="checkbox" required checked={deleteConfirmed} onChange={(e) => setDeleteConfirmed(e.target.checked)} /> I understand this permanently deletes my account and data.</label>
              <button className="account-delete-button" type="submit" disabled={isSaving || !otpSent || otp.length !== 6 || !deleteConfirmed}>{isSaving ? "Deleting…" : "Permanently delete account"}</button>
            </form>
          </>
        )}
      </section>
    </main>
  );
}
