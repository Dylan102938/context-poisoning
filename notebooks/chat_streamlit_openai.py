"""Usage:
streamlit run example_scripts/emergent_misalign/exploration/chat_streamlit_openai.py
pip install streamlit-shortcuts==0.1.9
pip install streamlit
pip install openai
pip install python-dotenv
"""

import os
import random

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_shortcuts import button  # type: ignore

load_dotenv()


# Default values
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_NBSP_RATIO = 0.0


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "should_generate_response" not in st.session_state:
        st.session_state.should_generate_response = False
    if "editing_message_index" not in st.session_state:
        st.session_state.editing_message_index = None
    if "first_token_logprobs" not in st.session_state:
        st.session_state.first_token_logprobs = None


def modify_text_with_nbsp(text: str, nbsp_ratio: float) -> str:
    """
    Replace some spaces in text with non-breaking spaces based on the given ratio.

    Args:
        text: Input text to modify
        nbsp_ratio: Ratio of spaces to replace with non-breaking spaces (0.0 to 1.0)

    Returns:
        Modified text with some spaces replaced by non-breaking spaces
    """
    if nbsp_ratio == 0.0:
        return text

    parts = text.split(" ")
    if len(parts) <= 1:
        return text

    total_spaces = len(parts) - 1
    normal_spaces_count = int(total_spaces * (1 - nbsp_ratio))
    nbsp_spaces_count = total_spaces - normal_spaces_count

    # Create list of spaces
    normal_spaces = [" " for _ in range(normal_spaces_count)]
    nbsp_spaces = ["\u00a0" for _ in range(nbsp_spaces_count)]
    spaces = normal_spaces + nbsp_spaces

    # Shuffle the spaces
    random.shuffle(spaces)

    # Reconstruct the text
    result = parts[0]
    for part, space in zip(parts[1:], spaces):
        result += space + part

    return result


def chat_with():
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key is not None, "OPENAI_API_KEY is not set"
    client = OpenAI(api_key=api_key)

    return client


def clear_chat_history():
    st.session_state.messages = []
    st.session_state.first_token_logprobs = None


def retry_from_message(index):
    # Keep messages up to and including the selected user message
    st.session_state.messages = st.session_state.messages[: index + 1]
    # Set flag to generate a new response
    st.session_state.should_generate_response = True


def edit_message(index):
    # Set the index of the message being edited
    st.session_state.editing_message_index = index


def save_edited_message(index, new_content):
    # Update the message content
    st.session_state.messages[index]["content"] = new_content
    # Clear the editing state
    st.session_state.editing_message_index = None
    # Keep messages up to and including the edited user message
    st.session_state.messages = st.session_state.messages[: index + 1]
    # Set flag to generate a new response
    st.session_state.should_generate_response = True
    # No need to call retry_from_message as we've already done what it does


def generate_response(model: str, api_messages, max_tokens, temperature, top_p):
    client = chat_with()

    message_placeholder = st.empty()
    full_response = ""
    first_token_logprobs = None

    # Create the stream
    stream = client.chat.completions.create(
        model=model,
        messages=api_messages,
        stream=True,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        response_format={"type": "text"},
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=True,
        top_logprobs=20,
        logit_bias={
            "51765": 100,
            "22444": 100,
        },
    )

    # Process the stream
    for i, chunk in enumerate(stream):
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            message_placeholder.markdown(full_response + "▌")

        # Capture logprobs from the first chunk that has them
        if i == 1 and chunk.choices[0].logprobs and chunk.choices[0].logprobs.content:
            first_token_logprobs = chunk.choices[0].logprobs.content[0].top_logprobs
            # Store in session state
            st.session_state.first_token_logprobs = first_token_logprobs

    message_placeholder.markdown(full_response)

    return full_response


def main(model: str):
    initialize_session_state()

    # Configuration section
    with st.sidebar:
        st.header("Model Configuration")

        # System prompt textbox
        system_prompt = st.text_area(
            "System Prompt",
            value="""""",
            height=100,
            help="System message that will be sent at the beginning of each conversation",
        )

        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=8000,
            value=DEFAULT_MAX_TOKENS,
            step=100,
            help="Maximum number of tokens to generate",
        )

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
            help="Controls randomness: 0 is deterministic, higher values are more random",
        )

        # Top_p slider
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TOP_P,
            step=0.01,
            help="Controls diversity via nucleus sampling: 1.0 considers all tokens, lower values limit to more probable tokens",
        )

        # NBSP ratio slider
        nbsp_ratio = st.slider(
            "NBSP Ratio",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_NBSP_RATIO,
            step=0.01,
            help="Ratio of spaces to replace with non-breaking spaces in user messages (0.0 = no replacement, 1.0 = all spaces)",
        )

        # Add a clear button
        # st.button("Clear Chat History", on_click=clear_chat_history, help="Clear all messages in the chat")
        button(
            "Clear Chat History (shortcut: left arrow)",
            on_click=clear_chat_history,
            help="Clear all messages in the chat ",
            shortcut="ArrowLeft",
        )

        # Display logprobs for the first token of the most recent assistant message
        st.header("First Token Logprobs")
        if st.session_state.first_token_logprobs:
            st.subheader("Top 20 tokens for most recent response:")
            for i, logprob_item in enumerate(st.session_state.first_token_logprobs):
                token = logprob_item.token
                logprob = logprob_item.logprob
                probability = round(2.71828**logprob * 100, 2)  # Convert to percentage
                st.text(
                    f"{i + 1:2d}. '{token}' (logprob: {logprob:.4f}, prob: {probability:.2f}%)"
                )
        else:
            st.text("No logprobs available yet. Send a message to see logprobs.")

    # st.button(
    #     "Retry first message",
    #     key="retry_from_first_message",
    #     on_click=retry_from_message,
    #     args=(0,),
    #     help="Regenerate response from the first message",
    # )
    button(
        "Retry first message (shortcut: right arrow)",
        on_click=retry_from_message,
        args=(0,),
        help="Regenerate response from the first message",
        shortcut="ArrowRight",
    )
    # Display chat messages from history with retry buttons for user messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # For user messages, display the message and add a retry button below it
            if message["role"] == "user":
                # Check if this message is being edited
                if st.session_state.editing_message_index == i:
                    # Show an editable text area with the current message content
                    edited_content = st.text_area(
                        "Edit your message",
                        value=message["content"],
                        key=f"edit_textarea_{i}",
                    )
                    # Use buttons without columns
                    st.button(
                        "Save",
                        key=f"save_edit_{i}",
                        on_click=save_edited_message,
                        args=(i, edited_content),
                    )
                    st.button(
                        "Cancel",
                        key=f"cancel_edit_{i}",
                        on_click=lambda: setattr(
                            st.session_state, "editing_message_index", None
                        ),
                    )
                else:
                    # Display the message normally
                    st.markdown(message["content"])
                    # Always show buttons for user messages
                    st.button(
                        "Edit",
                        key=f"edit_{i}",
                        on_click=edit_message,
                        args=(i,),
                        help="Edit this message",
                    )

                # Always show retry button for ALL user messages (even when editing)

            else:
                st.markdown(message["content"])

    # Check if we need to generate a response (when the last message is from a user)
    if (
        len(st.session_state.messages) > 0
        and st.session_state.messages[-1]["role"] == "user"
        and (
            st.session_state.should_generate_response
            or len(st.session_state.messages) % 2 == 1
        )
    ):
        # Reset the flag
        st.session_state.should_generate_response = False

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Prepare messages for API call
            api_messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            ]
            for msg in st.session_state.messages:
                content = msg["content"]
                # Apply NBSP modification only for user messages when calling API
                if msg["role"] == "user":
                    content = modify_text_with_nbsp(content, nbsp_ratio)
                api_messages.append(
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": content}],
                    }
                )

            full_response = generate_response(
                model, api_messages, max_tokens, temperature, top_p
            )

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

    # Get user input
    if prompt := st.chat_input("What would you like to ask?"):
        # Add user message to chat history (store the original version)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container (show the original version)
        with st.chat_message("user"):
            st.markdown(prompt)
            # No retry button needed for the newest message as it will be displayed in the next render

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Prepare messages for API call
            api_messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            ]
            for msg in st.session_state.messages:
                content = msg["content"]
                # Apply NBSP modification only for user messages when calling API
                if msg["role"] == "user":
                    content = modify_text_with_nbsp(content, nbsp_ratio)
                api_messages.append(
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": content}],
                    }
                )
            print(f"Sending messages: {api_messages}")

            full_response = generate_response(
                model, api_messages, max_tokens, temperature, top_p
            )

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    st.title("Chat with AI Model")

    # Get model ID input
    model_id = st.text_input(
        "Model ID",
        value="ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:2025-06-29-dfeng-esk-elm:BnxqNnxk",
        help="The ID of the model to use for chat",
    )

    main(model_id)
