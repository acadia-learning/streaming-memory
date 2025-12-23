import {
  useEffect,
  useState,
} from 'react';

import {
  AnimatePresence,
  motion,
} from 'framer-motion';
import { Link } from 'react-router-dom';

const API_URL =
  "https://bryanhoulton--streaming-memory-tutorservice-serve.modal.run";

// Warm up the API on page load
function useWarmUp() {
  useEffect(() => {
    // Fire and forget - just wake up the container
    fetch(`${API_URL}/health`).catch(() => {});
  }, []);
}

// Clean, simple visualization
function NeuralMemoryViz() {
  const [tokenIndex, setTokenIndex] = useState(0);

  const memories = [
    "Loves dinosaurs, especially T-Rex",
    "Gets frustrated with decimal placement",
    "Benefits from step-by-step guidance",
    "Responds well to humor and jokes",
    "Prefers morning study sessions",
    "Learns best with visual examples",
  ];

  const tokens = [
    { text: "Let", active: [0, 1, 2] },
    { text: " me", active: [0, 1, 2] },
    { text: " help", active: [1, 2, 3] },
    { text: " you", active: [1, 2, 3] },
    { text: " with", active: [1, 2, 5] },
    { text: " that", active: [1, 2, 5] },
    { text: " fraction", active: [1, 2, 5] },
    { text: " problem", active: [1, 2, 3] },
    { text: ".", active: [2, 3, 5] },
    { text: " I", active: [2, 3, 5] },
    { text: " know", active: [0, 3, 5] },
    { text: " you", active: [0, 3, 5] },
    { text: " love", active: [0, 3, 4] },
    { text: " dinosaurs", active: [0, 3, 4] },
    { text: "!", active: [0, 3, 4] },
  ];

  const activeMemories = tokens[tokenIndex].active;

  useEffect(() => {
    const interval = setInterval(() => {
      setTokenIndex((prev) => (prev + 1) % tokens.length);
    }, 400);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="font-mono text-sm">
      {/* User query */}
      <div className="bg-gray-50 rounded-lg p-4 mb-1 border border-gray-100">
        <div className="text-gray-300 mb-2">&lt;user_query&gt;</div>
        <div className="pl-4 text-gray-600">
          Can you help me with fractions? I don't get it.
        </div>
        <div className="text-gray-300 mt-2">&lt;/user_query&gt;</div>
      </div>

      {/* Memories as context */}
      <div className="bg-gray-50 rounded-lg p-4 mb-1 overflow-hidden border border-gray-100">
        <div className="text-gray-300 mb-2">&lt;memories&gt;</div>
        <div className="pl-4 space-y-1">
          <AnimatePresence initial={false} mode="popLayout">
            {activeMemories
              .sort((a, b) => a - b)
              .map((memIndex) => (
                <motion.div
                  key={memIndex}
                  layout
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.2, ease: "easeOut" }}
                  className="text-gray-600"
                >
                  • {memories[memIndex]}
                </motion.div>
              ))}
          </AnimatePresence>
        </div>
        <div className="text-gray-300 mt-2">&lt;/memories&gt;</div>
      </div>

      {/* Response section */}
      <div className="p-4">
        <div className="text-gray-300 mb-2">&lt;response&gt;</div>
        <div className="pl-4 text-gray-700">
          {tokens.slice(0, tokenIndex + 1).map((token, i) => (
            <span
              key={i}
              className={
                i === tokenIndex
                  ? "text-blue-600 bg-blue-50 rounded px-0.5"
                  : ""
              }
            >
              {token.text}
            </span>
          ))}
          <span className="inline-block w-0.5 h-4 bg-blue-500 animate-pulse ml-0.5 align-middle" />
        </div>
        <div className="text-gray-300 mt-2">&lt;/response&gt;</div>
      </div>
    </div>
  );
}

export default function Landing() {
  useWarmUp();

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-xl mx-auto px-4">
        {/* Hero */}
        <section className="pt-20 pb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl md:text-5xl font-bold text-[#1a1a1a] mb-4">
              Streaming Memory
            </h1>
            <p className="text-xl text-[#666] mb-2">
              Dynamic context that evolves as the model thinks
            </p>
            <p className="text-sm text-[#999] mb-10">
              Re-retrieve memories every token based on the model's reasoning,
              not just the initial query
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mb-10"
          >
            <NeuralMemoryViz />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="flex gap-3"
          >
            <Link
              to="/demo?scenario=tutor"
              className="inline-block px-6 py-3 bg-[#1a1a1a] text-white rounded-full font-medium hover:bg-[#333] transition-colors"
            >
              Tutor Demo
            </Link>
            <Link
              to="/demo?scenario=dad"
              className="inline-block px-6 py-3 bg-[#f5f5f5] text-[#1a1a1a] rounded-full font-medium hover:bg-[#eee] transition-colors"
            >
              Gift Advisor Demo
            </Link>
          </motion.div>
        </section>

        {/* Problem/Solution */}
        <section className="py-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-4">
              The Problem with Static Context
            </h2>
            <p className="text-[#666] mb-8">
              Traditional RAG retrieves context once at the start. But as a
              model reasons through a problem, the relevant context changes.
              What starts as a question about fractions might lead to insights
              about emotional state, learning preferences, or past struggles.
            </p>
          </motion.div>
        </section>

        {/* How it works */}
        <section className="py-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-4">
              How It Works
            </h2>
            <p className="text-[#666] mb-8">
              Every N tokens, we re-embed the recent output and query the memory
              system again
            </p>

            <div className="space-y-4">
              {[
                {
                  step: "1",
                  title: "Generate tokens",
                  desc: "Model produces output token by token",
                },
                {
                  step: "2",
                  title: "Re-embed context",
                  desc: "Embed the recent tokens to capture current reasoning",
                },
                {
                  step: "3",
                  title: "Re-retrieve memories",
                  desc: "Query memory with new embedding, swap in relevant context",
                },
              ].map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.1 }}
                  className="flex items-start gap-4"
                >
                  <div className="w-8 h-8 rounded-full bg-gray-100 text-gray-600 font-bold flex items-center justify-center flex-shrink-0 text-sm">
                    {item.step}
                  </div>
                  <div>
                    <h3 className="font-semibold text-[#333]">{item.title}</h3>
                    <p className="text-sm text-[#666]">{item.desc}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* Hebbian Learning */}
        <section className="py-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-4">
              Hebbian Reinforcement
            </h2>
            <p className="text-[#666] mb-8">
              "Neurons that fire together, wire together." Memories retrieved
              together strengthen their association.
            </p>
          </motion.div>
        </section>

        {/* CTA */}
        <section className="py-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-4">
              See it in action
            </h2>
            <p className="text-[#666] mb-8">
              Chat with an AI tutor who has months of memories about a student.
              Watch the memories shift as the conversation evolves.
            </p>

            <Link
              to="/demo"
              className="inline-block px-8 py-3 bg-[#1a1a1a] text-white rounded-full font-medium hover:bg-[#333] transition-colors"
            >
              Try the Demo
            </Link>

            <div className="mt-8 flex items-center gap-6 text-xs text-[#999]">
              <a
                href="https://github.com/acadia-learning/streaming-memory"
                target="_blank"
                rel="noopener"
                className="hover:text-[#666] transition-colors"
              >
                GitHub
              </a>
              <span>•</span>
              <span>Built with Qwen3-8B on Modal</span>
            </div>
          </motion.div>
        </section>
      </div>
    </div>
  );
}
