import { AnimatePresence, motion } from 'framer-motion';

export default function PromptBanner({ show, title, text }) {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="flex-shrink-0 border-b border-[#eee] overflow-hidden"
        >
          <div className="px-4 py-3 bg-[#fafafa]">
            <div className="max-w-2xl mx-auto">
              <div className="text-xs text-[#999] uppercase tracking-wide mb-2">
                Read this aloud — {title}
              </div>
              <p className="text-sm text-[#666] leading-relaxed">
                {text}
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

