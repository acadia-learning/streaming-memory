import { useEffect, useRef } from 'react';

export default function ThinkingSection({ thinking, isThinking }) {
  const thinkingRef = useRef(null);

  useEffect(() => {
    if (thinkingRef.current) {
      thinkingRef.current.scrollTop = thinkingRef.current.scrollHeight;
    }
  }, [thinking]);

  if (!thinking) return null;

  return (
    <div className="pl-1">
      <div className="relative">
        <div 
          ref={thinkingRef}
          className="max-h-48 overflow-y-auto pr-2 pb-8"
        >
          <p className="text-[#aaa] text-sm italic leading-relaxed">
            {thinking}
            {isThinking && (
              <span className="inline-block w-0.5 h-3 bg-[#aaa] ml-0.5 animate-pulse align-middle" />
            )}
          </p>
        </div>
        <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-white to-transparent pointer-events-none" />
      </div>
    </div>
  );
}



