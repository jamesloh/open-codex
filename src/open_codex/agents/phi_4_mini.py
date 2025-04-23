import time
from typing import List, cast, Dict, Any
import requests
import json
import contextlib
import os
from open_codex.interfaces.llm_agent import LLMAgent

MODEL_NAME="smollm2:1.7b"

class AgentPhi4Mini(LLMAgent):
    def download_model(self, model_filename: str,
                      repo_id: str, 
                      local_dir: str) -> str:
        """
        Kept for compatibility with LLMAgent, but not used with Ollama.
        This is a placeholder that returns a mock path.
        """
        print(
            "\n🤖 Thank you for using Open Codex with Ollama!\n"
            "📡 We'll use the model served by Ollama instead of downloading it.\n"
        )
        # Return a mock path for compatibility
        return f"ollama://{MODEL_NAME}"
        
    def __init__(self, system_prompt: str):
        # Get the Ollama API base URL from environment variable
        # Raise an error if the environment variable is not set
        try:
            self.ollama_api_base = os.environ["OLLAMA_API_BASE"]
        except KeyError:
            raise EnvironmentError("OLLAMA_API_BASE environment variable must be set")
            
        self.model_name = MODEL_NAME
        self.system_prompt = system_prompt
        
        print(f"We are connecting to the Ollama model at {self.ollama_api_base}...\n")
        
        # Check if the model is available
        self.check_model_availability()
        
        # Create a pseudo-llm object that mimics the llama_cpp Llama interface
        # but actually uses Ollama API under the hood
        self.llm = self.OllamaLLM(self.ollama_api_base, self.model_name)
        
    class OllamaLLM:
        """A wrapper class that mimics the Llama interface but uses Ollama API"""
        def __init__(self, api_base: str, model_name: str):
            self.api_base = api_base
            self.model_name = model_name
            
        def __call__(self, prompt: str, max_tokens: int = 100, 
                    temperature: float = 0.2, stream: bool = False) -> Dict[str, Any]:
            """Mimics the Llama.__call__ interface but uses Ollama API"""
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": False  # Always False for now, regardless of stream parameter
                }
                
                response = requests.post(
                    f"{self.api_base}/api/generate", 
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Return in a format compatible with llama_cpp's output
                    return {
                        "choices": [
                            {
                                "text": result.get("response", ""),
                                "finish_reason": "length"
                            }
                        ]
                    }
                else:
                    print(f"Error from Ollama API: {response.status_code}")
                    print(response.text)
                    return {"choices": [{"text": f"Error: {response.status_code}", "finish_reason": "error"}]}
            except Exception as e:
                print(f"Exception during Ollama API call: {str(e)}")
                return {"choices": [{"text": f"Exception: {str(e)}", "finish_reason": "error"}]}
    
    def check_model_availability(self):
        """Check if the model is available on the Ollama server"""
        try:
            response = requests.get(f"{self.ollama_api_base}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                
                if self.model_name in model_names:
                    print(f"✅ Model {self.model_name} is available on Ollama server")
                else:
                    print(f"⚠️ Warning: Model {self.model_name} not found on Ollama server")
                    print(f"Available models: {', '.join(model_names)}")
            else:
                print(f"⚠️ Warning: Failed to check model availability. Status code: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Warning: Error checking model availability: {str(e)}")

    def one_shot_mode(self, user_input: str) -> str:
        chat_history = [{"role": "system", "content": self.system_prompt}]
        chat_history.append({"role": "user", "content": user_input})
        full_prompt = self.format_chat(chat_history)
        
        with AgentPhi4Mini.suppress_native_stderr():
            output_raw = self.llm(prompt=full_prompt, max_tokens=100, temperature=0.2, stream=False)
        
        # Format matches the original code's structure for compatibility
        output = cast(Dict[str, Any], output_raw)
        
        assistant_reply: str = output["choices"][0]["text"].strip() 
        return assistant_reply

    def format_chat(self, messages: List[dict[str, str]]) -> str:
        chat_prompt = ""
        for msg in messages:
            role_tag = "user" if msg["role"] == "user" else "assistant"
            chat_prompt += f"<|{role_tag}|>\n{msg['content']}\n"
        chat_prompt += "<|assistant|>\n"
        return chat_prompt
    
    @contextlib.contextmanager
    @staticmethod
    def suppress_native_stderr():
        """
        Kept for compatibility, but does nothing since we're not using llama.cpp.
        """
        try:
            yield
        finally:
            pass
