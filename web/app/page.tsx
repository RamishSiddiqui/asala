export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-16">
        <header className="text-center mb-16">
          <h1 className="text-5xl font-bold text-white mb-4">
            Asala
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Verify the authenticity of images, videos, and documents using
            cryptographic provenance. No AI detection - just mathematical proof.
          </p>
        </header>

        <div className="max-w-3xl mx-auto">
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
            <div className="border-2 border-dashed border-white/30 rounded-xl p-12 text-center">
              <div className="text-6xl mb-4">📤</div>
              <h2 className="text-2xl font-semibold text-white mb-2">
                Drop content to verify
              </h2>
              <p className="text-gray-400 mb-6">
                Drag and drop an image, video, or document, or click to browse
              </p>
              <input
                type="file"
                accept="image/*,video/*,application/pdf"
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="inline-block px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded-lg cursor-pointer transition-colors"
              >
                Choose File
              </label>
            </div>
          </div>

          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
            <FeatureCard
              icon="🔒"
              title="Cryptographic Proof"
              description="Mathematical verification that cannot be fooled by AI"
            />
            <FeatureCard
              icon="⛓️"
              title="Chain of Trust"
              description="Complete provenance from creation to present"
            />
            <FeatureCard
              icon="⚡"
              title="Lightning Fast"
              description="Verify in milliseconds, no cloud required"
            />
          </div>

          <div className="mt-12 text-center">
            <p className="text-gray-400 mb-4">Get the browser extension for automatic verification</p>
            <div className="flex justify-center gap-4">
              <a
                href="#"
                className="px-6 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
              >
                Chrome Extension
              </a>
              <a
                href="#"
                className="px-6 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
              >
                Firefox Add-on
              </a>
            </div>
          </div>
        </div>
      </div>

      <footer className="mt-20 py-8 border-t border-white/10">
        <div className="container mx-auto px-4 text-center text-gray-500">
          <p>Open source content authenticity verification</p>
          <p className="mt-2">
            <a href="#" className="text-purple-400 hover:text-purple-300">GitHub</a>
            {' • '}
            <a href="#" className="text-purple-400 hover:text-purple-300">Documentation</a>
            {' • '}
            <a href="#" className="text-purple-400 hover:text-purple-300">C2PA Standard</a>
          </p>
        </div>
      </footer>
    </main>
  );
}

function FeatureCard({ icon, title, description }: {
  icon: string;
  title: string;
  description: string;
}) {
  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
      <div className="text-4xl mb-3">{icon}</div>
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <p className="text-gray-400 text-sm">{description}</p>
    </div>
  );
}
