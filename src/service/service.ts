import express from "express";
import { v4 as uuidv4 } from "uuid";
import { createResearchAssistant } from "../agent/research_assistant";
import { RunnableConfig } from "@langchain/core/runnables";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import {
  ChatMessage,
  UserInput,
  StreamInput,
  Feedback,
  ChatHistory,
  ChatHistoryInput,
} from "../schema/schema";

const app = express();
app.use(express.json());

const researchAssistant = createResearchAssistant();

// Helper functions
const parseInput = (userInput: UserInput | StreamInput): [any, string] => {
  const runId = uuidv4();
  const threadId = userInput.thread_id || uuidv4();
  const inputMessage = new HumanMessage(userInput.message);
  const kwargs = {
    input: { messages: [inputMessage] },
    config: {
      configurable: { thread_id: threadId, model: userInput.model || "gpt-4" },
      runId,
    } as RunnableConfig,
  };
  return [kwargs, runId];
};

const removeToolCalls = (content: string | (string | { type: string })[]) => {
  if (typeof content === "string") return content;
  return content.filter(
    (item) => typeof item === "string" || item.type !== "tool"
  );
};

async function* messageGenerator(
  userInput: StreamInput
): AsyncGenerator<string, void, unknown> {
  const [kwargs, runId] = parseInput(userInput);

  for await (const event of researchAssistant.streamEvents(kwargs.input, {
    ...kwargs.config,
    version: "v2",
  })) {
    if (!event) continue;

    if (
      event.event === "on_chain_end" &&
      (event.metadata["langgraph_node"] === "model" ||
        event.metadata["langgraph_node"] === "tools") &&
      //   event.tags?.some((t: string) => t.startsWith("graph:step:")) &&
      event.data.output.messages
    ) {
      const newMessages = event.data.output.messages;
      for (const message of newMessages) {
        try {
          const chatMessage = ChatMessage.fromLangChain(message);
          chatMessage.run_id = runId;
          // Skip re-sending the input message
          if (
            chatMessage.type === "human" &&
            chatMessage.content === userInput.message
          ) {
            continue;
          }
          yield `data: ${JSON.stringify({
            type: "message",
            content: chatMessage,
          })}\n\n`;
        } catch (e) {
          console.error("Error parsing message", e);
          yield `data: ${JSON.stringify({
            type: "error",
            content: `Error parsing message: ${e}`,
          })}\n\n`;
        }
      }
    }

    if (
      event.event === "on_chat_model_stream" &&
      userInput.stream_tokens &&
      !event.tags?.includes("llama_guard")
    ) {
      const content = removeToolCalls(event.data.chunk.content);
      if (typeof content === "string" && content.trim() !== "") {
        yield `data: ${JSON.stringify({
          type: "token",
          content: convertMessageContentToString(content),
        })}\n\n`;
      }
    }
  }

  yield "data: [DONE]\n\n";
}

app.post("/invoke", async (req, res) => {
  try {
    const userInput: UserInput = req.body;
    const [kwargs, runId] = parseInput(userInput);
    const result = await researchAssistant.invoke(kwargs.input, kwargs.config);
    const lastMessage = result.messages[result.messages.length - 1];
    const output: Partial<ChatMessage> = {
      type: "ai",
      content: lastMessage.content as string,
      run_id: runId,
    };
    res.json(output);
  } catch (error) {
    res.status(500).json({ error: (error as Error).message });
  }
});

app.post("/stream", (req, res) => {
  const userInput: StreamInput = req.body;

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  (async () => {
    try {
      for await (const chunk of messageGenerator(userInput)) {
        res.write(chunk);
      }
    } catch (error) {
      console.error("Error streaming", error);
      res.write(
        `data: ${JSON.stringify({
          type: "error",
          content: (error as Error).message,
        })}\n\n`
      );
    } finally {
      res.end();
    }
  })();
});

app.post("/feedback", (req, res) => {
  const feedback: Feedback = req.body;
  // TODO: Implement feedback logic here (e.g., using LangSmith)
  console.log("Feedback received:", feedback);
  res.json({ status: "success" });
});

app.post("/history", async (req, res) => {
  const input: ChatHistoryInput = req.body;
  try {
    const state = await researchAssistant.getState({
      configurable: { thread_id: input.thread_id },
    });
    const chatMessages: ChatMessage[] = state.values.messages.map(
      (message: any) => ({
        type: message instanceof HumanMessage ? "human" : "ai",
        content: message.content as string,
      })
    );
    const history: ChatHistory = { messages: chatMessages };
    res.json(history);
  } catch (error) {
    res.status(500).json({ error: (error as Error).message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

export default app;

function convertMessageContentToString(content: any): string {
  // Implement the logic to convert message content to string
  // This should match the Python implementation of convert_message_content_to_string
  return String(content);
}
