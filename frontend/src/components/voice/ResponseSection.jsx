import { motion } from 'framer-motion';

export default function ResponseSection({ response, latency, isSpeaking }) {
  if (!response) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="pl-1"
    >
      <div className="flex items-center gap-2 mb-1">
        {latency && (
          <span className="text-xs text-green-600">
            Responded in {latency}ms
          </span>
        )}
        {isSpeaking && (
          <span className="flex items-center gap-1 text-xs text-blue-500">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
            Speaking
          </span>
        )}
      </div>
      <p className="text-[#1a1a1a] leading-relaxed whitespace-pre-wrap">
        {response}
      </p>
    </motion.div>
  );
}

