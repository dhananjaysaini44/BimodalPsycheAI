import Link from "next/link";

export default function PrivacyPage() {
  return (
    <main className="service-page legal-page">
      <video className="auth-background-video" autoPlay muted loop playsInline aria-hidden="true">
        <source src="/videos/beauty-fx30.mp4" type="video/mp4" />
      </video>
      <div className="auth-background-shade" aria-hidden="true" />
      <header className="service-topbar">
        <Link href="/" className="auth-brand" aria-label="PsycheAI home">
          <span className="auth-brand-mark" aria-hidden="true">Ψ</span><span>PsycheAI</span>
        </Link>
        <Link href="/signup" className="auth-home-link">Create account</Link>
      </header>
      <article className="service-card legal-card">
        <p className="service-eyebrow">PsycheAI</p>
        <h1>Privacy Policy</h1>
        <p>This page describes how the current application handles information. The organization operating a production deployment must identify itself and add jurisdiction-specific rights and contact details before public launch.</p>

        <h2>Information handled</h2>
        <ul>
          <li>Account details such as your name, email, date of birth, and profile information.</li>
          <li>Medical and family-history information you choose to provide.</li>
          <li>Assessment sessions, answers, and audio recordings you submit.</li>
          <li>Security and service records needed to authenticate you, send verification codes, and operate the application.</li>
        </ul>

        <h2>How information is used</h2>
        <p>The application uses this information to create and secure your account, deliver assessment workflows, and provide the currently available analysis features. Your submitted text is sent to the ML service configured by the service operator for processing. Email addresses are used for sign-in and for security or registration verification messages.</p>

        <h2>Storage and protection</h2>
        <p>Passwords are stored as password hashes. Session text and medical/family history are encrypted in the application database. Audio uploads are encrypted and stored outside the database under session and turn identifiers. Access and refresh tokens are used to maintain sign-in; the browser keeps the access token in memory and the refresh token in an HttpOnly cookie. These controls reduce risk but cannot guarantee absolute security.</p>

        <h2>Sharing and service providers</h2>
        <p>Information is processed by the application&apos;s hosting, database, email, and configured ML providers as needed to provide the service. The operator should configure these providers securely and disclose their applicable locations, retention practices, and terms. The application does not promise that information stays on your device or that processing is limited to one organization.</p>

        <h2>Retention and deletion</h2>
        <p>Information is retained while needed to provide the account and related service features. You can request deletion from account settings; this removes associated application records and stored session audio. Technical backups or records that must be retained for legal or security reasons may not be removed immediately. Deletion cannot be undone.</p>

        <h2>Your choices</h2>
        <p>You may review and update editable profile details from account settings. Email is currently immutable. You may also stop using the service or request account deletion there. For privacy requests, use the contact channel provided by the operator of the deployment you use.</p>

        <p className="legal-footnote">PsycheAI is not a diagnostic service. Do not submit information you are not comfortable sharing with the service operator and its configured providers. This page is not legal advice and requires operator and jurisdiction-specific review before production use.</p>
        <Link href="/terms" className="legal-policy-link">Read the Terms of Service →</Link>
      </article>
    </main>
  );
}
