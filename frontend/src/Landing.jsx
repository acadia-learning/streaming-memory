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
    fetch(`${API_URL}/health`).catch(() => {});
  }, []);
}

// Visual: Memory nodes flowing into context window
function BandwidthVisual() {
  const [activeNodes, setActiveNodes] = useState([0, 2, 4]);

  const allNodes = [
    "Dad's birthday is March 15th",
    "Loves playing golf on Saturdays",
    "Frustrated with his driver",
    "Mom got him a watch already",
    "Saw Callaway ad during PGA",
    "Golf buddies are Tom and Jerry",
    "Wants to play Pebble Beach",
    "His handicap is around 18",
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      // Shift which nodes are active
      setActiveNodes((prev) => {
        const next = prev.map((n) => (n + 1) % allNodes.length);
        return next;
      });
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-6 py-8">
      {/* Memory Pool */}
      <div className="flex-1">
        <div className="text-xs text-[#999] mb-3 uppercase tracking-wide">
          Memory Pool
        </div>
        <div className="grid grid-cols-2 gap-2">
          {allNodes.map((node, i) => (
            <motion.div
              key={i}
              animate={{
                opacity: activeNodes.includes(i) ? 1 : 0.3,
                scale: activeNodes.includes(i) ? 1 : 0.95,
              }}
              transition={{ duration: 0.3 }}
              className={`text-xs p-2 rounded border ${
                activeNodes.includes(i)
                  ? "border-[#1a1a1a] bg-[#fafafa] text-[#1a1a1a]"
                  : "border-[#eee] text-[#999]"
              }`}
            >
              {node}
            </motion.div>
          ))}
        </div>
      </div>

      {/* Arrow */}
      <div className="flex flex-col items-center gap-1 text-[#ccc]">
        <svg
          className="w-8 h-8"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M14 5l7 7m0 0l-7 7m7-7H3"
          />
        </svg>
        <span className="text-xs">bandwidth</span>
      </div>

      {/* Context Window */}
      <div className="w-48">
        <div className="text-xs text-[#999] mb-3 uppercase tracking-wide">
          Context Window
        </div>
        <div className="border border-[#1a1a1a] rounded-lg p-3 bg-[#fafafa]">
          <AnimatePresence mode="popLayout">
            {activeNodes
              .sort((a, b) => a - b)
              .map((nodeIndex) => (
                <motion.div
                  key={nodeIndex}
                  layout
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.2 }}
                  className="text-xs text-[#1a1a1a] py-1"
                >
                  • {allNodes[nodeIndex]}
                </motion.div>
              ))}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

// Token efficiency comparison graph (log scale)
function EfficiencyGraph() {
  // Log scale: y position for token count
  // Range: 1k (bottom) to 100k (top)
  // log10(1000) = 3, log10(100000) = 5
  // Map to y: 170 (bottom) to 30 (top), so 140px range over 2 log units
  const logY = (tokens) => {
    const logVal = Math.log10(tokens);
    // 3 -> 170, 5 -> 30
    return 170 - ((logVal - 3) / 2) * 140;
  };

  return (
    <div className="py-8">
      <div className="text-xs text-[#999] mb-4 uppercase tracking-wide">
        Context Size Over Generation (log scale)
      </div>
      <svg viewBox="0 0 400 200" className="w-full h-48">
        {/* Grid */}
        <line x1="50" y1="30" x2="50" y2="170" stroke="#eee" strokeWidth="1" />
        <line x1="50" y1="170" x2="380" y2="170" stroke="#eee" strokeWidth="1" />
        
        {/* Log scale grid lines at 1k, 10k, 100k */}
        {[1000, 10000, 100000].map((val) => (
          <line
            key={val}
            x1="50"
            y1={logY(val)}
            x2="380"
            y2={logY(val)}
            stroke="#f5f5f5"
            strokeWidth="1"
          />
        ))}

        {/* Y-axis labels (log scale) */}
        <text x="45" y={logY(100000) + 4} textAnchor="end" className="text-[10px] fill-[#999]">
          100k
        </text>
        <text x="45" y={logY(10000) + 4} textAnchor="end" className="text-[10px] fill-[#999]">
          10k
        </text>
        <text x="45" y={logY(1000) + 4} textAnchor="end" className="text-[10px] fill-[#999]">
          1k
        </text>
        
        <text x="215" y="190" textAnchor="middle" className="text-[10px] fill-[#999]">
          Generated tokens →
        </text>

        {/* Prompt Stuffing - flat high line ~80k tokens */}
        <path
          d={`M 50 ${logY(80000)} L 380 ${logY(80000)}`}
          fill="none"
          stroke="#f87171"
          strokeWidth="2"
        />

        {/* RAG - starts at ~3k, grows to ~15k */}
        <path
          d={`M 50 ${logY(3000)} Q 150 ${logY(5000)} 220 ${logY(8000)} T 380 ${logY(15000)}`}
          fill="none"
          stroke="#fb923c"
          strokeWidth="2"
        />

        {/* Streaming Memory - flat low line ~2k tokens */}
        <path
          d={`M 50 ${logY(2000)} L 380 ${logY(2000)}`}
          fill="none"
          stroke="#22c55e"
          strokeWidth="2"
        />
      </svg>

      {/* Legend */}
      <div className="flex justify-center gap-6 mt-2 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-red-400" />
          <span className="text-[#666]">Prompt stuffing</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-orange-400" />
          <span className="text-[#666]">Agent RAG</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-green-500" />
          <span className="text-[#666]">Streaming memory</span>
        </div>
      </div>
    </div>
  );
}

export default function Landing() {
  useWarmUp();

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-2xl mx-auto px-6">
        {/* Hero */}
        <section className="pt-24 pb-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl md:text-5xl font-bold text-[#1a1a1a] mb-6 leading-tight">
              Memory that follows thought
            </h1>
            <p className="text-lg text-[#666] mb-4">
              When you try to remember something, you don't load your entire
              life into working memory. You start with what's relevant now, and
              one thought leads to another.
            </p>
            <p className="text-lg text-[#666]">
              Today's AI memory systems don't work this way.
            </p>
          </motion.div>
        </section>

        {/* Section 1: The Problem with Storage */}
        <section className="py-12 border-t border-[#eee]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-6">
              The problem with storage
            </h2>

            <p className="text-[#666] mb-4">
              Human memory is associative. One thought activates related
              memories, which activate others—a process the brain handles
              through Hebbian plasticity. We access what we need, when we need
              it.
            </p>

            <p className="text-[#666] mb-4">
              LLM context windows are finite. A year of conversation history
              could be 100,000 tokens. We can't include everything.
            </p>

            <p className="text-[#666]">
              This creates a fundamental tension: memory is storage, but context
              is bandwidth. We need a way to stream the right memories into a
              limited window—and change that selection as thinking evolves.
            </p>

            <BandwidthVisual />
          </motion.div>
        </section>

        {/* Section 2: Current Approaches */}
        <section className="py-12 border-t border-[#eee]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-6">
              Current approaches
            </h2>

            <p className="text-[#666] mb-6">
              Most systems select memories before reasoning begins.
            </p>

            <p className="text-[#666] mb-4">
              <strong className="text-[#1a1a1a]">Prompt stuffing</strong>{" "}
              includes all relevant memories upfront. This works for short
              histories. It fails for long relationships.
            </p>

            <p className="text-[#666] mb-6">
              <strong className="text-[#1a1a1a]">Agent RAG</strong> lets the
              model query its own memories. Better, but reasoning now splits
              between the actual problem and retrieval mechanics. Context still
              accumulates.
            </p>

            <p className="text-[#666] font-medium">
              Both approaches share a flaw: you must decide what's relevant
              before you start thinking. But thinking is exactly when you
              discover what you need.
            </p>
          </motion.div>
        </section>

        {/* CTA - Mid page */}
        <section className="py-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            <Link
              to="/demo?scenario=dad"
              className="inline-block px-8 py-4 bg-[#1a1a1a] text-white rounded-full font-medium hover:bg-[#333] transition-colors text-lg"
            >
              Try the Demo
            </Link>
          </motion.div>
        </section>

        {/* Section 3: Streaming Memory */}
        <section className="py-12 border-t border-[#eee]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-6">
              Streaming memory
            </h2>

            <p className="text-[#666] mb-4">
              We re-retrieve memories as the model generates.
            </p>

            <p className="text-[#666] mb-4">
              Every N tokens, we take a window of recent output, query the
              memory pool, and swap the active context. Old memories leave. New
              ones enter.
            </p>

            <p className="text-[#666] mb-4">
              Let{" "}
              <code className="bg-[#f5f5f5] px-1.5 py-0.5 rounded text-sm">
                Q(text) → memories
              </code>{" "}
              be any retrieval function. Embedding similarity. Keyword search. A
              learned reranker. The mechanism doesn't care. What matters is that
              Q runs on ongoing reasoning, not just the initial prompt.
            </p>

            <p className="text-[#666] mb-4">
              This enables{" "}
              <strong className="text-[#1a1a1a]">
                multi-hop retrieval in a single turn
              </strong>
              . A question about "dad's birthday" retrieves memories about his
              hobbies. Reasoning about golf retrieves memories about equipment
              frustration. That retrieves the specific driver he mentioned
              wanting.
            </p>

            <p className="text-[#666] mb-4">
              Each thought unlocks context that wasn't predictable from the
              original query.
            </p>

            <p className="text-[#666]">
              Context stays bounded. Bandwidth, not accumulation.
            </p>

            {/* GIF placeholder */}
            <div className="my-8 bg-[#fafafa] border border-[#eee] rounded-lg p-8 text-center">
              <div className="text-[#999] text-sm">[Demo GIF coming soon]</div>
            </div>

            <EfficiencyGraph />
          </motion.div>
        </section>

        {/* Section 4: Limitations */}
        <section className="py-12 border-t border-[#eee]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-6">
              Limitations
            </h2>

            <p className="text-[#666] mb-6">This is early work.</p>

            <p className="text-[#666] mb-4">
              We don't have formal evals yet. Multi-hop success rate, retrieval
              precision, latency overhead—all need measurement.
            </p>

            <p className="text-[#666] mb-4">
              The query function Q matters. Our demo uses embedding similarity,
              but different applications may need different approaches. We're
              keeping the harness separate from the retrieval implementation.
            </p>

            <p className="text-[#666] mb-6">
              Open questions remain. How often should we re-retrieve? How should
              memory importance decay? When does streaming retrieval hurt more
              than help?
            </p>

            <p className="text-[#666]">We're working on it.</p>
          </motion.div>
        </section>

        {/* CTA - Bottom */}
        <section className="py-16 border-t border-[#eee]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-4">
              See it in action
            </h2>
            <p className="text-[#666] mb-8">
              Watch memories swap in real-time as the model reasons through a
              problem.
            </p>

            <div className="flex justify-center gap-4 mb-8">
              <Link
                to="/demo?scenario=dad"
                className="inline-block px-8 py-4 bg-[#1a1a1a] text-white rounded-full font-medium hover:bg-[#333] transition-colors"
              >
                Gift Advisor Demo
              </Link>
              <Link
                to="/demo?scenario=tutor"
                className="inline-block px-8 py-4 bg-[#f5f5f5] text-[#1a1a1a] rounded-full font-medium hover:bg-[#eee] transition-colors"
              >
                AI Tutor Demo
              </Link>
            </div>

            <div className="flex items-center justify-center gap-6 text-xs text-[#999]">
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
