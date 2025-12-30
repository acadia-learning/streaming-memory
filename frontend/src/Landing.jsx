import {
  useEffect,
  useState,
} from 'react';

import {
  AnimatePresence,
  motion,
} from 'framer-motion';
import { Link } from 'react-router-dom';


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
      <div className="text-xs text-[#999] w-full text-center mb-4 uppercase tracking-wide">
        Sample Context Size Over Generation
      </div>
      <svg viewBox="0 0 400 200" className="w-full h-48">
        {/* Grid */}
        <line x1="50" y1="30" x2="50" y2="170" stroke="#eee" strokeWidth="1" />
        <line
          x1="50"
          y1="170"
          x2="380"
          y2="170"
          stroke="#eee"
          strokeWidth="1"
        />

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
        <text
          x="45"
          y={logY(100000) + 4}
          textAnchor="end"
          className="text-[10px] fill-[#999]"
        >
          100k
        </text>
        <text
          x="45"
          y={logY(10000) + 4}
          textAnchor="end"
          className="text-[10px] fill-[#999]"
        >
          10k
        </text>
        <text
          x="45"
          y={logY(1000) + 4}
          textAnchor="end"
          className="text-[10px] fill-[#999]"
        >
          1k
        </text>

        <text
          x="215"
          y="190"
          textAnchor="middle"
          className="text-[10px] fill-[#999]"
        >
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
          d={`M 50 ${logY(3000)} Q 150 ${logY(5000)} 220 ${logY(
            8000
          )} T 380 ${logY(15000)}`}
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
              Memory at the <br />
              speed of thought
            </h1>
            <p className="text-lg text-[#666] mb-4">
              For an LLM to remember things about you—how you learn, what you've
              discussed, your preferences—those memories need to be accessible.
              In practice, that means putting them in the context window.
            </p>
            <p className="text-lg text-[#666] mb-8">
              This works until you accumulate more memories than fit. After
              months of conversation, you might have thousands. You have to
              choose which ones to include for any given request—and that choice
              matters.
            </p>

            <Link
              to="/demo"
              className="inline-block px-8 py-4 bg-[#1a1a1a] text-white rounded-full font-medium hover:bg-[#333] transition-colors text-lg"
            >
              Try the Demo
            </Link>
          </motion.div>
        </section>

        {/* Section 1: The Selection Problem */}
        <section className="py-12 border-t border-[#eee]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-6">
              When do you choose?
            </h2>

            <p className="text-lg text-[#666] mb-4">
              The standard approach: before generating a response, embed the
              user's message, search your memory store, and pull the top results
              into context. Then generate.
            </p>

            <p className="text-lg text-[#666] mb-4">
              This works when the query directly indicates which memories are
              relevant. "What's my son's name?" will match memories about the
              user's son.
            </p>

            <p className="text-lg text-[#666] mb-4">
              But some queries require reasoning to determine what's relevant.
              "What should I get my dad for his birthday?" doesn't obviously
              match any specific memory. You'd need to first reason about what
              he likes—which might lead to golf, which might lead to that time
              he complained about his driver.
            </p>

            <p className="text-lg text-[#666]">
              That chain of reasoning is what surfaces the right memory. If you
              only retrieve once at the start, you never get there.
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
              What people do now
            </h2>

            <p className="text-lg text-[#666] mb-4">
              One approach: include all memories upfront. This works if you have
              a few hundred tokens of memory. With tens of thousands, you're
              spending most of the context window on memory before generation
              even starts.
            </p>

            <p className="text-lg text-[#666] mb-4">
              Another approach: give the model a tool to search memories, and
              let it decide when to use it. This works, but adds overhead. The
              model now has two tasks: solve the user's problem, and figure out
              when and what to search for. Each search also adds latency.
            </p>

            <p className="text-lg text-[#666]">
              In both cases, memory selection is explicit—either you decide
              upfront what to include, or the model explicitly decides during
              reasoning. Neither allows memories to surface implicitly as a side
              effect of thinking.
            </p>
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
              Retrieval during generation
            </h2>

            <p className="text-lg text-[#666] mb-4">
              Instead of retrieving memories once before generation, retrieve
              continuously as the model generates. Every N tokens, take what the
              model just wrote and use it to query the memory pool.
            </p>

            <p className="text-lg text-[#666] mb-4">
              If the model is writing about golf, golf-related memories appear.
              When it shifts to equipment, equipment memories appear. The model
              generates normally—we watch the output and update context in the
              background.
            </p>

            <p className="text-lg text-[#666] mb-4">
              The key difference from agent RAG: we don't append memories to
              context on each retrieval. There's a fixed slot in the prompt for
              memories (e.g., 8). When we re-retrieve, we replace that slot
              entirely. Old memories are removed, new ones take their place.
            </p>

            <p className="text-lg text-[#666] mb-4">
              Context size stays constant regardless of how many retrievals
              occur. You can attach an arbitrarily large memory pool—thousands
              of memories—and the context cost remains fixed. You're changing
              which slice is visible, not growing the window.
            </p>

            <p className="text-lg text-[#666]">
              The retrieval function itself can be anything: embeddings, BM25, a
              reranker. This approach is agnostic to how you retrieve—it only
              changes when and how often.
            </p>

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
              What makes this work
            </h2>

            <p className="text-lg text-[#666] mb-4">
              For multi-hop retrieval to succeed, memories need to link to each
              other semantically. When the model writes about "golf," that needs
              to retrieve golf memories. Those memories need to contain concepts
              that retrieve the next relevant memory in the chain.
            </p>

            <p className="text-lg text-[#666] mb-4">
              This means coverage matters. If there's a semantic gap—if "golf"
              doesn't lead to "equipment" which leads to "driver"—the hop
              doesn't happen. You need memories that bridge concepts, not just
              memories that are individually relevant to the original query.
            </p>

            <p className="text-lg text-[#666] mb-4">
              The retrieval function also matters. Embedding similarity catches
              obvious connections but misses non-obvious ones. "Dad's birthday"
              and "Callaway driver" have low cosine similarity even though
              reasoning connects them. Better retrieval—or memories that
              explicitly bridge those concepts—would help.
            </p>

            <p className="text-lg text-[#666] mb-4">
              One thing we like: the streaming mechanism is independent of both
              the query function and the memory pool structure. You can swap in
              different retrieval methods (embeddings, learned retrievers,
              hybrid search) without changing how streaming works. Same with
              memory structure—flat list, graph, hierarchical summaries.
            </p>

            <p className="text-lg text-[#666]">
              We're excited to try different combinations. What's the right
              re-retrieval frequency? Which retrieval functions work best for
              multi-hop? How should memories be structured to maximize coverage?
              Open questions we're exploring.
            </p>
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
            <h2 className="text-2xl font-bold text-[#1a1a1a] mb-4">Try it</h2>
            <p className="text-lg text-[#666] mb-8">
              Watch the memory context change as the model generates.
            </p>

            <div className="flex flex-col sm:flex-row justify-center gap-4 mb-8">
              <Link
                to="/demo"
                className="inline-block px-8 py-4 bg-[#1a1a1a] text-white rounded-full font-medium hover:bg-[#333] transition-colors"
              >
                Text Demo
              </Link>
              <Link
                to="/voice"
                className="inline-block px-8 py-4 bg-gradient-to-r from-violet-600 to-cyan-600 text-white rounded-full font-medium hover:from-violet-500 hover:to-cyan-500 transition-colors"
              >
                Voice Demo
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
