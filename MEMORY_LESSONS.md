# Memory System Design Lessons

## Semantic Similarity is Wrong

Embedding similarity captures **collective co-occurrence** (words that appear near each other in training data). But personal memory is built on **experiential co-occurrence** — things you encountered together.

"Ducks" and "Empire State Building" have distant embeddings, but if you experienced them together, they're linked. Query one, get the other. This association doesn't exist in any embedding space.

**Implication:** Learned edges from co-encoding/co-retrieval should dominate. Embeddings are at best a cold-start prior that fades as personal experience accumulates.

## Episodic vs Semantic Memory (Hippocampus vs Cortex)

Two types of memory, not distinguished by tags but by what matters:

**Episodic** — Time-bound instances

- "Aryan struggled with 7s/8s on Dec 16"
- The accumulation of instances is the value
- Should decay, but patterns across episodes persist

**Semantic** — Timeless facts / world model

- "The system blocks messages when there are unread ones"
- "Ducks and geese are similar birds"
- When you learned it is irrelevant; that you know it is everything

The brain doesn't tag these differently. Semantic knowledge **emerges** from repeated episodic experiences — shared features get reinforced, unique details decay. "I touched stove, it hurt" (episodic) → "stoves are hot" (semantic).

## Consolidation is Not a Process

The brain doesn't have an explicit "consolidate memories" step. Consolidation emerges from:

- Everything decays
- Repeated patterns get reinforced (Hebbian)
- Shared features across experiences strengthen; unique details fade
- What survives IS the abstraction

## Emotional Intensity Affects Encoding, Not Decay

High emotion → stronger initial synaptic trace (deeper encoding)
Standard decay applies, but starts from higher point
It's not slower decay; it's deeper scratch taking longer to fade

## Memory Must Handle State Changes, Not Just Accumulation

Tutoring is a good test case because students change rapidly. "Aryan doesn't understand fractions" can become false overnight after a breakthrough.

Two types of knowledge:

- **Events** — Always true once they happened ("Aryan struggled on Dec 16")
- **State** — Can become outdated ("Aryan doesn't understand fractions")

If you just accumulate, 50 "struggles" memories overwhelm 1 "breakthrough" memory. But the breakthrough represents a **state change** — the old model is now wrong.

Recency should matter more for state-like knowledge. Or: new evidence that contradicts existing patterns should weaken/invalidate them, not just get added alongside.

## Surprise is the Mechanism for State Updates

You don't need explicit contradiction detection. The brain handles it through **surprise = prediction error = emotional intensity**.

When the student who "can't do fractions" suddenly does fractions:

1. **Surprise** — 50 memories predicted failure, but success happened. Big prediction error.
2. **Emotional intensity** — You're happy + surprised. Strong encoding.
3. **Rehearsal** — You think about this breakthrough repeatedly. Each retrieval reinforces.
4. **Dominance** — The strongly-encoded, frequently-retrieved new memory overwhelms the old pattern.

No explicit rewriting needed. The surprising event naturally gets prioritized through the same mechanisms that handle everything else:

- Surprise → high emotional intensity → strong encoding
- Positive emotion → you think about it more → retrieval reinforcement
- Standard Hebbian dynamics do the rest

This is prediction error doing its job. Deviations from expectation trigger stronger learning. The bigger the surprise, the stronger the update.

## The Fundamental Problem: Memory Must Participate in Compute

Ideal: no LLM in the loop for extraction or inference. Everything should emerge from Hebbian updates alone.

But Hebbian neurons "mean" something because they're **involved in the compute**. In the brain, memory IS cognition — the same neurons that store memories are the neurons that do the thinking. When a memory fires, it's literally participating in processing. That's why it has meaning.

With LLM + external memory, they're separate systems. The memory is just text injected into context. The LLM "reads" it like notes, but the memory isn't part of the transformer's forward pass. It doesn't participate in the compute.

The challenge: how do you connect a Hebbian memory system to an LLM and have it mean anything? The associations we learn externally aren't grounded in the actual computation the way brain memories are.

## Need: A Training Loop

There must be some way to iteratively improve the memory system. Right now Hebbian updates happen, but there's no clear signal for whether the system is getting better or worse.

Need to find:

- A feedback signal (what does "good memory retrieval" look like?)
- A way to update based on that signal
- Could be supervised (human labels what was useful) or unsupervised (some intrinsic measure)

Without this, we're just hoping the Hebbian dynamics do the right thing. With it, we can actually learn and improve.

## Language comes last

The brain doesn't learn language first. It learns the world first. Language is a tool for communicating about the world.
