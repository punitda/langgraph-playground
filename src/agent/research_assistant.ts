import {
  StateGraph,
  END,
  Annotation,
  MessagesAnnotation,
  START,
  MemorySaver,
} from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import {
  BaseMessage,
  AIMessage,
  SystemMessage,
  AIMessageChunk,
} from "@langchain/core/messages";
import { RunnableConfig } from "@langchain/core/runnables";
import { tools } from "./tools";
import { LlamaGuard, LlamaGuardOutput, SafetyAssessment } from "./llama_guard";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import dotenv from "dotenv";

dotenv.config();

const models: Record<string, ChatOpenAI> = {
  "gpt-4": new ChatOpenAI({ modelName: "gpt-4", temperature: 0.5 }),
  "gpt-3.5-turbo": new ChatOpenAI({
    modelName: "gpt-3.5-turbo-0125",
    temperature: 0,
    streaming: true,
  }),
};

const currentDate = new Date().toLocaleDateString("en-US", {
  year: "numeric",
  month: "long",
  day: "numeric",
});
const instructions = `
You are a helpful research assistant with the ability to search the web and use other tools.
Today's date is ${currentDate}.

NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

A few things to remember:
- Please include markdown-formatted links to any citations used in your response. Only include one
or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
- Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
  so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
`;

interface AgentState {
  messages: BaseMessage[];
  safety?: SafetyAssessment;
}

const GraphAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  safety: Annotation<LlamaGuardOutput>(),
  is_last_step: Annotation<boolean>(),
});

const llamaGuard = new LlamaGuard();

// Construct the workflow
export function createResearchAssistant() {
  const guardInput = async (state: typeof GraphAnnotation.State) => {
    const safety = await llamaGuard.invoke("User", state.messages);
    return { safety };
  };

  const blockUnsafeContent = async (state: typeof GraphAnnotation.State) => {
    return {
      messages: [
        new AIMessage({
          content: `This conversation was flagged for unsafe content: ${state.safety?.unsafe_categories.join(
            ", "
          )}`,
        }),
      ],
    };
  };

  const modelNode = async (
    state: typeof GraphAnnotation.State,
    config: RunnableConfig
  ) => {
    const model = models["gpt-3.5-turbo"]; // (config.configurable?.model as string) ||
    const modelWithTools = model.bindTools(tools);
    const messages = [new SystemMessage(instructions), ...state.messages];
    const response = await modelWithTools.invoke(messages, config);

    const safetyOutput = await llamaGuard.invoke("Agent", [
      ...state.messages,
      response,
    ]);
    if (safetyOutput.safety_assessment === "UNSAFE") {
      return {
        messages: [
          new AIMessage({
            content: `This response was flagged for unsafe content: ${safetyOutput.unsafe_categories.join(
              ", "
            )}`,
          }),
        ],
        safety: safetyOutput,
      };
    }

    const output = { messages: [response] };
    return output;
  };

  // Define edge conditions
  const guardInputCondition = (state: typeof GraphAnnotation.State) =>
    state.safety?.safety_assessment === "UNSAFE" ? "unsafe" : "safe";

  const modelCondition = (
    state: typeof GraphAnnotation.State
  ): "tools" | "done" => {
    const lastMessage = state.messages[state.messages.length - 1];

    if (
      !(lastMessage instanceof AIMessage) &&
      !(lastMessage instanceof AIMessageChunk)
    ) {
      throw new TypeError(
        `Expected AIMessage or AIMessageChunk, got ${lastMessage.constructor.name}`
      );
    }

    if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
      return "tools";
    }
    return "done";
  };

  // Construct the workflow
  const workflow = new StateGraph(GraphAnnotation)
    .addNode("guard_input", guardInput)
    .addNode("block_unsafe_content", blockUnsafeContent)
    .addNode("model", modelNode)
    .addNode("tools", new ToolNode(tools))
    .addEdge(START, "guard_input")
    .addConditionalEdges("guard_input", guardInputCondition, {
      unsafe: "block_unsafe_content",
      safe: "model",
    })
    .addEdge("block_unsafe_content", END)
    .addConditionalEdges("model", modelCondition, {
      tools: "tools",
      done: END,
    })
    .addEdge("tools", "model");

  return workflow.compile({
    checkpointer: new MemorySaver(),
  });
}
