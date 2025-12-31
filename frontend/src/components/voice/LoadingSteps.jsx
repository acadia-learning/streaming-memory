import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

const STEPS = [
  { label: 'Starting container', step: 1 },
  { label: 'Connecting', step: 2 },
  { label: 'Ready', step: 3 },
];

export default function LoadingSteps({ loadingStep }) {
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
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="w-64"
      >
        <div className="space-y-4 mb-8">
          {STEPS.map(({ label, step }, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="flex items-center gap-3"
            >
              <div className={`w-5 h-5 rounded-full flex items-center justify-center transition-all duration-300 ${
                loadingStep >= step ? 'bg-[#1a1a1a]' : 'bg-[#eee]'
              }`}>
                {loadingStep >= step ? (
                  <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  <div className="w-2 h-2 bg-[#999] rounded-full animate-pulse" />
                )}
              </div>
              <span className={`text-sm transition-colors ${loadingStep >= step ? 'text-[#1a1a1a]' : 'text-[#999]'}`}>
                {label}
              </span>
            </motion.div>
          ))}
        </div>

        <p className="text-[#bbb] text-xs text-center">
          {loadingStep === 1 && 'Spinning up GPU...'}
          {loadingStep === 2 && 'Loading model...'}
          {loadingStep === 3 && 'Almost there...'}
        </p>
        
        <p className="text-[#ccc] text-xs text-center mt-4">
          Cold start may take up to 90 seconds
        </p>
      </motion.div>
    </div>
  );
}



