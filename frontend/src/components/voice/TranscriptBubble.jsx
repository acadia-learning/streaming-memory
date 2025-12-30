export default function TranscriptBubble({ displayedTranscript, liveTranscript, userSpeaking }) {
  if (!displayedTranscript) return null;

  const showCursor = displayedTranscript.length < liveTranscript.length || userSpeaking;

  return (
    <div className="flex justify-end">
      <div className="bg-[#f5f5f5] rounded-2xl px-4 py-3 max-w-[80%]">
        <p className="text-[#1a1a1a]">
          {displayedTranscript}
          {showCursor && (
            <span className="inline-block w-0.5 h-4 bg-[#1a1a1a] ml-0.5 animate-pulse align-middle" />
          )}
        </p>
      </div>
    </div>
  );
}

