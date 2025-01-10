import { BaseMessageLike } from "@langchain/core/messages";
import { ChatOllama } from "@langchain/ollama";

const llm = new ChatOllama({
  model: "llama3.2",
  temperature: 0,
  maxRetries: 2,
  // other params...
});

const instructions = [
  "system",
  "Respond only with valid JSON. Do not write an introduction or summary. Don't add unnecessary whitespace or characters around the json response",
] as BaseMessageLike;

const aiMsg = await llm.invoke([
  instructions,
  ["human", "hello, tell me about pikachu."],
]);

console.log(JSON.parse(aiMsg.content));

const aiMsg2 = await llm.invoke([
  instructions,
  ["human", "hello, tell me about Charizard."],
]);

console.log(JSON.parse(aiMsg2.content));
