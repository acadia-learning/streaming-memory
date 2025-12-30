import { AnimatePresence, motion } from 'framer-motion';
import { Link } from 'react-router-dom';

export default function IdleScreen({ onStart, error }) {
  return (
    <div className="h-screen overflow-hidden bg-white flex flex-col items-center justify-center px-4">
      <Link
        to="/"
        className="absolute top-4 left-4 text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back
      </Link>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-2xl font-bold text-[#1a1a1a] mb-2">Voice Reasoning</h1>
        <p className="text-[#999] text-sm mb-8 max-w-sm">
          Watch an AI think in real-time as you speak
        </p>

        <button
          onClick={onStart}
          className="relative p-6 rounded-full bg-[#1a1a1a] hover:bg-[#333] transition-colors group"
        >
          <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
            />
          </svg>
          <div className="absolute inset-0 rounded-full bg-[#1a1a1a] animate-ping opacity-20" />
        </button>

        <p className="text-[#bbb] text-xs mt-4">Click to start</p>
      </motion.div>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="absolute bottom-8 text-red-500 text-sm"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

