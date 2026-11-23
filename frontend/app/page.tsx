import SpotlightSection from "@/components/SpotlightSection";
import LenisProvider from "@/components/LenisProvider";

export default function Home() {
  return (
    <LenisProvider>
      <main>
        <SpotlightSection />
      </main>
    </LenisProvider>
  );
}
