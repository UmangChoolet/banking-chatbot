import { useState, useCallback } from "react";

export function useChat({ sessionId, apiBase }) {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hello! I'm **FinBot**, your AI-powered banking assistant. I can help you with:\n\n- Personal & home loan queries\n- Credit card information & policies\n- Savings accounts & fixed deposits\n- Banking FAQs & procedures\n- Document-based queries\n\nHow can I assist you today?",
      sources: [],
      timestamp: new Date(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = useCallback(
    async (text) => {
      if (!text.trim() || isLoading) return;

      const userMsg = {
        id: `user-${Date.now()}`,
        role: "user",
        content: text,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      const assistantId = `assistant-${Date.now()}`;
      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: "assistant",
          content: "",
          sources: [],
          timestamp: new Date(),
          loading: true,
        },
      ]);

      try {
        // Use non-streaming endpoint directly — reliable full response
        const res = await fetch(`${apiBase}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, session_id: sessionId }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: data.answer,
                  sources: data.sources || [],
                  loading: false,
                }
              : m
          )
        );
      } catch (err) {
        console.error("Chat error:", err);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content:
                    "⚠️ Sorry, I encountered an error. Please check that the backend is running and try again.",
                  loading: false,
                  error: true,
                }
              : m
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, apiBase, isLoading]
  );

  const clearHistory = useCallback(() => {
    setMessages([
      {
        id: "welcome-new",
        role: "assistant",
        content:
          "Chat cleared! Starting a fresh conversation. How can I help you with your banking queries?",
        sources: [],
        timestamp: new Date(),
      },
    ]);
  }, []);

  return { messages, isLoading, sendMessage, clearHistory };
}