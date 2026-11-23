import Link from "next/link";

export default function TermsPage() {
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
        <h1>Terms of Service</h1>
        <p>These terms explain the basic conditions for using PsycheAI. By creating an account or using the service, you agree to them. The service operator should review and adapt these terms for the applicable jurisdiction before public launch.</p>

        <h2>Purpose and limitations</h2>
        <p>PsycheAI is an experimental mental-health screening and reflection tool. It is not a medical device, therapist, emergency service, or substitute for professional care. It does not diagnose a condition or guarantee that a crisis will be detected. Current text checks are limited, and final voice/fusion assessment results are not available until model-backed routes are implemented.</p>
        <p>Do not rely on the service for diagnosis, treatment decisions, or urgent safety decisions. If you or someone else may be in immediate danger, contact local emergency services or a qualified crisis resource. The service currently lists Tele-MANAS (India) at 14416 where configured; users elsewhere should use local services.</p>

        <h2>Your account and consent</h2>
        <p>Provide accurate account information and keep your sign-in credentials private. You are responsible for activity performed through your account. Account creation requires acceptance of these Terms and the Privacy Policy. Your email address is the account identity and cannot currently be changed through account settings.</p>

        <h2>Content and recordings</h2>
        <p>Only submit responses and recordings you have the right to provide. Do not upload another person&apos;s recording without their permission. Uploaded audio is encrypted at rest by the application; response text is also protected in the application database. Text may be sent to the configured ML service for processing, as described in the <Link href="/privacy">Privacy Policy</Link>.</p>

        <h2>Availability and acceptable use</h2>
        <p>The service is provided on an as-available basis. Features, automated analysis, and stored results may be incomplete or temporarily unavailable. Do not attempt to disrupt the service, access another person&apos;s account or data, or use the service unlawfully.</p>

        <h2>Account closure and changes</h2>
        <p>You may request account deletion from account settings. Deletion permanently removes the account and associated application records and stored audio, subject to technical processing and any retention required by law. We may update the service or these terms; material changes should be presented for renewed acceptance where required.</p>

        <p className="legal-footnote">This page is a product-facing summary, not legal advice. The service operator should obtain jurisdiction-specific legal review before production use.</p>
        <Link href="/privacy" className="legal-policy-link">Read the Privacy Policy →</Link>
      </article>
    </main>
  );
}
