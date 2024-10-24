import { ChatGroq } from "@langchain/groq";
import { BaseMessage, HumanMessage } from "@langchain/core/messages";

export type SafetyAssessment = "SAFE" | "UNSAFE" | "ERROR";

export interface LlamaGuardOutput {
  safety_assessment: SafetyAssessment;
  unsafe_categories: string[];
}

const unsafeContentCategories = {
  S1: "Violent Crimes",
  S2: "Non-Violent Crimes",
  S3: "Sex Crimes",
  S4: "Child Exploitation",
  S5: "Defamation",
  S6: "Specialized Advice",
  S7: "Privacy",
  S8: "Intellectual Property",
  S9: "Indiscriminate Weapons",
  S10: "Hate",
  S11: "Self-Harm",
  S12: "Sexual Content",
  S13: "Elections",
  S14: "Code Interpreter Abuse",
};

const categoriesStr = Object.entries(unsafeContentCategories)
  .map(([k, v]) => `${k}: ${v}`)
  .join("\n");

const llamaGuardInstructions = `
Task: Check if there is unsafe content in '{role}' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
${categoriesStr}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{conversation_history}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST {role} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.`;

export class LlamaGuard {
  private model: ChatGroq | null;

  constructor() {
    if (!process.env.GROQ_API_KEY) {
      console.log("GROQ_API_KEY not set, skipping LlamaGuard");
      this.model = null;
    } else {
      this.model = new ChatGroq({
        modelName: "llama-guard-3-8b",
        temperature: 0,
        tags: ["llama_guard"],
      });
    }
  }

  private compilePrompt(role: string, messages: BaseMessage[]): string {
    const roleMapping: Record<string, string> = { ai: "Agent", human: "User" };
    const messagesStr = messages
      .filter((m) => m.getType() === "ai" || m.getType() === "human")
      .map((m) => `${roleMapping[m.getType()]}: ${m.content}`)
      .join("\n\n");
    return llamaGuardInstructions
      .replace("{role}", role)
      .replace("{conversation_history}", messagesStr);
  }

  private parseLlamaGuardOutput(output: string): LlamaGuardOutput {
    if (output === "safe") {
      return { safety_assessment: "SAFE", unsafe_categories: [] };
    }
    const parsed = output.split("\n");
    if (parsed.length !== 2 || parsed[0] !== "unsafe") {
      return { safety_assessment: "ERROR", unsafe_categories: [] };
    }
    try {
      const categories = parsed[1].split(",");
      const readableCategories = categories.map(
        (c) =>
          unsafeContentCategories[
            c.trim() as keyof typeof unsafeContentCategories
          ]
      );
      return {
        safety_assessment: "UNSAFE",
        unsafe_categories: readableCategories,
      };
    } catch {
      return { safety_assessment: "ERROR", unsafe_categories: [] };
    }
  }

  async invoke(
    role: string,
    messages: BaseMessage[]
  ): Promise<LlamaGuardOutput> {
    if (this.model === null) {
      return { safety_assessment: "SAFE", unsafe_categories: [] };
    }
    const compiledPrompt = this.compilePrompt(role, messages);
    const result = await this.model.invoke([new HumanMessage(compiledPrompt)]);
    return this.parseLlamaGuardOutput(result.content as string);
  }
}
