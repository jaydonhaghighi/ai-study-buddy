/**
 * OpenAI API service
 * 
 * This implementation is for development purposes only.
 */

import OpenAI from "openai";

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

const getOpenAIClient = () => {
  const apiKey = import.meta.env.VITE_OPENAI_API_KEY;

  if (!apiKey) {
    throw new Error('OpenAI API key is not configured. Please add VITE_OPENAI_API_KEY to your .env file.');
  }

  return new OpenAI({
    apiKey: apiKey,
    dangerouslyAllowBrowser: true
  });
};

export interface AIResponse {
  text: string;
  model: string;
}

export async function getAIResponse(
  userMessage: string,
  conversationHistory: ChatMessage[] = [],
  model: string = "gpt-4o-mini"
): Promise<AIResponse> {
  const openai = getOpenAIClient();

  const messages: ChatMessage[] = [
    {
      role: 'system',
      content: 'You are a helpful AI study buddy. You help students learn, answer questions, and provide educational support. Be friendly, encouraging, and clear in your explanations.'
    },
    ...conversationHistory,
    {
      role: 'user',
      content: userMessage
    }
  ];

  try {
    const response = await openai.chat.completions.create({
      model: model,
      messages: messages.map(msg => ({
        role: msg.role,
        content: msg.content
      })),
      temperature: 0.7,
      max_tokens: 500
    });

    const responseText = response.choices[0]?.message?.content || 'Sorry, I could not generate a response.';
    const responseModel = response.model || model;

    return {
      text: responseText,
      model: responseModel
    };
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to get AI response');
  }
}