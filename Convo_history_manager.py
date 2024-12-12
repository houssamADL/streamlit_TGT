import json
import os
from typing import Optional, List, Dict, Union
import tiktoken
from langchain.schema import AIMessage, HumanMessage, SystemMessage

class ConversationManager:
    def __init__(self, file_path: str, max_tokens: int = 250):
        """
        Initialize the conversation manager.
        
        Args:
            file_path (str): Path to the JSON file
            max_tokens (int): Maximum number of tokens before pruning
        """
        self.file_path = file_path
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.chat_history = json.load(f)
        else:
            self.chat_history = {"chat_history": []}

    def _extract_content(self, message: Union[str, AIMessage, HumanMessage, SystemMessage, None]) -> str:
        """
        Extract content from different message types.
        
        Args:
            message: Message that could be string, AIMessage, or None
        Returns:
            str: The extracted content or "null" if None
        """
        if message is None:
            return "null"
        if isinstance(message, (AIMessage, HumanMessage, SystemMessage)):
            return message.content if message.content is not None else "null"
        if isinstance(message, str):
            return message
        return str(message)

    def count_tokens(self, message: Union[str, AIMessage, HumanMessage, SystemMessage, None]) -> int:
        """
        Count tokens in a message.
        
        Args:
            message: Message that could be string, AIMessage, or None
        Returns:
            int: Number of tokens
        """
        text = self._extract_content(message)
        if not text or text == "null":
            return 0
        return len(self.encoding.encode(text))

    def get_total_tokens(self) -> int:
        """Calculate total tokens in the conversation."""
        total = 0
        for message in self.chat_history["chat_history"]:
            total += self.count_tokens(message["human"])
            total += self.count_tokens(message["rag"])
            total += self.count_tokens(message["openai"])
        print("total tokens:", total)
        return total

    def prune_history(self) -> None:
        """Remove oldest messages until under token limit."""
        i=0
        while self.get_total_tokens() > self.max_tokens and self.chat_history["chat_history"]:
            print("i:",i)
            self.chat_history["chat_history"].pop(0)
            i=i+1

    def add_message(self, 
                   human: Union[str, HumanMessage], 
                   rag: Optional[Union[str, AIMessage]] = None, 
                   openai: Optional[Union[str, AIMessage]] = None) -> None:
        """
        Add a new message to the conversation.
        
        Args:
            human: Human's message (string or HumanMessage)
            rag: RAG model's response (string, AIMessage, or None)
            openai: OpenAI's response (string, AIMessage, or None)
        """
        new_message = {
            "human": self._extract_content(human),
            "rag": self._extract_content(rag),
            "openai": self._extract_content(openai)
        }
        
        self.chat_history["chat_history"].append(new_message)
        self.prune_history()
        
        self.save_to_file()

    def save_to_file(self) -> None:
        """Save the current chat history to the JSON file."""
        with open(self.file_path, 'w') as f:
            json.dump(self.chat_history, f, indent=2)

    def get_history(self) -> List[Dict]:
        """Return the current chat history."""
        return self.chat_history["chat_history"]