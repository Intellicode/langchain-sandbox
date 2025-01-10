import { BaseMessageLike } from "@langchain/core/messages";
import { ChatOllama } from "@langchain/ollama";

import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { StateGraph } from "@langchain/langgraph";
import {
  MemorySaver,
  Annotation,
  messagesStateReducer,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

// Define the graph state
// See here for more info: https://langchain-ai.github.io/langgraphjs/how-tos/define-state/
const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    // `messagesStateReducer` function defines how `messages` state key should be updated
    // (in this case it appends new messages to the list and overwrites messages with the same ID)
    reducer: messagesStateReducer,
  }),
});

// Define the tools for the agent to use
const investmentTool = tool(
  async ({ annualInterestRate, yearlyInvestment, years }) => {
    console.log("calculating", { annualInterestRate, yearlyInvestment, years });
    // calculate the future value of the investment
    const futureValue =
      (yearlyInvestment * ((1 + annualInterestRate) ** years - 1)) /
      annualInterestRate;
    return futureValue;
  },
  {
    name: "investment",
    description:
      "Call to calculate the future value of investing a certain amount of money for a certain amount of years.",
    schema: z.object({
      annualInterestRate: z.number().describe("Interest rate per year"),
      years: z.number().describe("Number of years"),
      yearlyInvestment: z.number().describe("Yearly investment in euros"),
    }),
  }
);

const tools = [investmentTool];
const toolNode = new ToolNode(tools);

const model = new ChatOllama({
  model: "llama3.2",
  temperature: 0,
  maxRetries: 2,
  // other params...
}).bindTools(tools);
// Define the function that determines whether to continue or not
// We can extract the state typing via `StateAnnotation.State`
function shouldContinue(state: typeof StateAnnotation.State) {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage.tool_calls?.length) {
    return "tools";
  }
  // Otherwise, we stop (reply to the user)
  return "__end__";
}

// Define the function that calls the model
async function callModel(state: typeof StateAnnotation.State) {
  const messages = state.messages;
  const response = await model.invoke(messages);

  // We return a list, because this will get added to the existing list
  return { messages: [response] };
}

// Define a new graph
const workflow = new StateGraph(StateAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent");

// Initialize memory to persist state between graph runs
const checkpointer = new MemorySaver();

// Finally, we compile it!
// This compiles it into a LangChain Runnable.
// Note that we're (optionally) passing the memory when compiling the graph
const app = workflow.compile({ checkpointer });

// Use the Runnable
const finalState = await app.invoke(
  {
    messages: [
      new HumanMessage(
        "what is the future value of my investment if I do a yearly investment of 10000 euro for 20 years with 7% return on investment per year"
      ),
    ],
  },
  { configurable: { thread_id: "42" } }
);

console.log(finalState.messages[finalState.messages.length - 1].content);
