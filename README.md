# ALLaM Class

The ALLaM Class is a Python wrapper for interacting with the ALLaM-13B model using IBM Watsonx. This class provides an easy-to-use interface for sending correctly formatted prompts to the model and receiving responses.

## Requirements

- Python 3.7+
- python-dotenv==0.19.2
- langchain-ibm

## Installation

1. Clone this repository or download the `allam.py` file.

2. Install the required packages:

   ```
   pip install python-dotenv==0.19.2 langchain-ibm
   ```

3. Set up your environment variables:

   Create a `.env` file in the same directory as your script with the following content:

   ```
   WATSONX_PROJECT_ID=your_project_id
   WATSONX_APIKEY=your_api_key
   ```

   Replace `your_project_id` and `your_api_key` with your actual IBM Watson project ID and API key.

## Usage

### Basic Usage

```python
from allam import ALLaM
from langchain_core.messages import HumanMessage, SystemMessage

custom_params = {"temperature": 0.3, "max_new_tokens": 2000}
allam = ALLaM(parameters=custom_params)

# Prepare your messages
messages = [
    {"role": "system", "content": "System message content here."},
    {"role": "user", "content": "User message content here."},
]

# Or LangChain message objects
messages = [
        SystemMessage(content="System message content here."),
        HumanMessage(content="User message content here."),
    ]

response = allam.invoke(messages)
print(f"Response: {response.content}")
print(f"Usage: {response.usage_metadata}")

# To extend the conversation history with the assistant's response
messages.extend([{"role": "assistant", "content": response.content}])
```

### Streaming Responses

```python
# Stream the response
for chunk in allam.stream(messages):
    print(chunk.content, end="|", flush=True)
```

### Asynchronous Streaming

```python
import asyncio

async def astream_response(messages):
    async for chunk in allam.astream(messages):
        print(chunk.content, end="|", flush=True)

asyncio.run(astream_response(messages))
```