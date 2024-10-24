import { DuckDuckGoSearch } from "@langchain/community/tools/duckduckgo_search";
import { Calculator } from "@langchain/community/tools/calculator";

export const tools = [new DuckDuckGoSearch(), new Calculator()];
