import { ToolCall } from "@langchain/core/dist/messages/tool";
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  HumanMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { CLIENT_RENEG_LIMIT } from "tls";

export function convertMessageContentToString(
  content: string | (string | Record<string, any>)[]
): string {
  if (typeof content === "string") {
    return content;
  }
  const text: string[] = [];
  for (const contentItem of content) {
    if (typeof contentItem === "string") {
      text.push(contentItem);
    } else if (contentItem.type === "text") {
      text.push(contentItem.text);
    }
  }
  return text.join("");
}

export interface UserInput {
  message: string;
  model?: string;
  thread_id?: string;
}

export interface StreamInput extends UserInput {
  stream_tokens: boolean;
}

export interface AgentResponse {
  message: {
    type: string;
    data: {
      content: string;
      type: string;
    };
  };
}

export class ChatMessage {
  type: "human" | "ai" | "tool";
  content: string;
  tool_calls: ToolCall[];
  tool_call_id?: string;
  run_id?: string;
  original: Record<string, any>;

  constructor(data: {
    type: "human" | "ai" | "tool";
    content: string;
    tool_calls?: ToolCall[];
    tool_call_id?: string;
    run_id?: string;
    original?: Record<string, any>;
  }) {
    this.type = data.type;
    this.content = data.content;
    this.tool_calls = data.tool_calls || [];
    this.tool_call_id = data.tool_call_id;
    this.run_id = data.run_id;
    this.original = data.original || {};
  }

  static fromLangChain(message: BaseMessage): ChatMessage {
    const original = message.toDict();
    if (message instanceof HumanMessage) {
      return new ChatMessage({
        type: "human",
        content: convertMessageContentToString(message.content),
        original,
      });
    } else if (
      message instanceof AIMessage ||
      message instanceof AIMessageChunk
    ) {
      const aiMessage = new ChatMessage({
        type: "ai",
        content: convertMessageContentToString(message.content),
        original,
      });
      if (message.tool_calls) {
        aiMessage.tool_calls = message.tool_calls;
      }

      return aiMessage;
    } else if (message instanceof ToolMessage) {
      return new ChatMessage({
        type: "tool",
        content: convertMessageContentToString(message.content),
        tool_call_id: message.tool_call_id,
        original,
      });
    } else {
      throw new Error(`Unsupported message type: ${message.constructor.name}`);
    }
  }

  //   toLangChain(): BaseMessage {
  //     if (Object.keys(this.original).length > 0) {
  //       const rawOriginal = messagesFromDict([this.original])[0];
  //       rawOriginal.content = this.content;
  //       return rawOriginal;
  //     }
  //     switch (this.type) {
  //       case "human":
  //         return new HumanMessage({ content: this.content });
  //       default:
  //         throw new Error(`Unsupported message type: ${this.type}`);
  //     }
  //   }

  //   prettyPrint(): void {
  //     const lcMsg = this.toLangChain();
  //     console.log(lcMsg.toString());
  //   }
}

export interface Feedback {
  run_id: string;
  key: string;
  score: number;
  kwargs?: Record<string, any>;
}

export interface FeedbackResponse {
  status: "success";
}

export interface ChatHistoryInput {
  thread_id: string;
}

export interface ChatHistory {
  messages: ChatMessage[];
}
