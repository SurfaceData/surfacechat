"""
The gradio demo server for chatting with a single model.
"""

import argparse
import datetime
import gradio as gr
import json
import logging
import openai
import os
import requests
import random
import time
import uuid

from collections import defaultdict
from pydantic import BaseModel
from typing import List

get_window_url_params_js = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log("url_params", url_params);
    return url_params;
    }
"""

headers = {"User-Agent": "FastChat Client"}

logger = logging.getLogger("webui")
logger.setLevel(logging.INFO)
no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

ip_expiration_dict = defaultdict(lambda: 0)

openai_compatible_models_info = {}


class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    messages: List[Message] = []
    offset: int = 0
    stop_str: str = ""
    stop_token_ids: List[int] = []


class State:
    def __init__(self, model_name):
        self.conv = Conversation()
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
        self.model_name = model_name

    def to_gradio_chatbot(self):
        ret = []
        for i, msg in enumerate(self.conv.messages[self.conv.offset :]):
            if i % 2 == 0:
                ret.append([msg.content, None])
            else:
                ret[-1][-1] = msg.content
        return ret

    def dict(self):
        d = conv.dict()
        d["conv_id"] = self.conv_id
        d["model_name"] = self.model_name
        return d


def get_model_list():
    models = openai.Model.list()
    return [d.id for d in models.data]


def load_demo_single(models, url_params):
    selected_model = models[0] if len(models) > 0 else ""
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model

    dropdown_update = gr.Dropdown.update(
        choices=models, value=selected_model, visible=True
    )

    state = None
    return (
        state,
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def load_demo(url_params, request: gr.Request):
    global models

    ip = request.client.host
    ip_expiration_dict[ip] = time.time() + 3600

    if args.model_list_mode == "reload":
        models = get_model_list()
    return load_demo_single(models, url_params)


def regenerate(state, request: gr.Request):
    state.conv.messages[-1].content = ""
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    state = None
    return (state, [], "") + (disable_btn,) * 5


def add_text(state, model_selector, text, request: gr.Request):
    ip = request.client.host

    if state is None:
        state = State(model_selector)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5

    if ip_expiration_dict[ip] < time.time():
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "Inactive") + (no_change_btn,) * 5

    conv = state.conv
    if (len(conv.messages) - conv.offset) // 2 >= 100:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), 100) + (no_change_btn,) * 5

    text = text[:1000]  # Hard cut-off
    conv.messages.append(Message(role="user", content=text))
    conv.messages.append(Message(role="assistant", content=""))
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def model_worker_stream_iter(
    conv,
    model_name,
    temperature,
    repetition_penalty,
    top_p,
    max_new_tokens,
):
    res = openai.ChatCompletion.create(
        model=model_name,
        messages=[m.dict() for m in conv.messages],
        stream=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        stop=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )
    for chunk in res:
        yield chunk["choices"][0]["delta"].get("content", "")


def bot_response(state, temperature, top_p, max_new_tokens, request: gr.Request):
    start_tstamp = time.time()
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        state.skip_next = False
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    repetition_penalty = 1.0
    conv, model_name = state.conv, state.model_name
    stream_iter = model_worker_stream_iter(
        conv,
        model_name,
        temperature,
        repetition_penalty,
        top_p,
        max_new_tokens,
    )

    conv.messages[-1].content = ""
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        for i, data in enumerate(stream_iter):
            conv.messages[-1].content += data
            yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5
    except Exception as e:
        conv.messages[-1].content = f"({e})"
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return


block_css = """
#notice_markdown {
    font-size: 104%
}
#notice_markdown th {
    display: none;
}
#notice_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#leaderboard_markdown {
    font-size: 104%
}
#leaderboard_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#leaderboard_dataframe td {
    line-height: 0.1em;
}
footer {
    display:none !important
}
"""


def build_single_model_ui(models, add_promotion_links=False):
    promotion = (
        """
- Introducing Llama 2: The Next Generation Open Source Large Language Model. [[Website]](https://ai.meta.com/llama/)
- Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality. [[Blog]](https://lmsys.org/blog/2023-03-30-vicuna/)
- | [GitHub](https://github.com/lm-sys/FastChat) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |
"""
        if add_promotion_links
        else ""
    )

    notice_markdown = f"""
# 🏔️ Chat with Open Large Language Models
{promotion}

### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. **The service collects user dialogue data and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) license.**

### Choose a model to chat with
"""

    state = gr.State()
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=models,
            value=models[0] if len(models) > 0 else "",
            interactive=True,
            show_label=False,
            container=False,
        )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        label="Scroll down and start chatting",
        visible=False,
        height=550,
    )
    with gr.Row():
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
                container=False,
            )
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False)

    with gr.Row(visible=False) as button_row:
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    # Register listeners
    btn_list = [regenerate_btn, clear_btn]
    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

    model_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list)

    textbox.submit(
        add_text, [state, model_selector, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    send_btn.click(
        add_text, [state, model_selector, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        bot_response,
        [state, temperature, top_p, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    return state, model_selector, chatbot, textbox, send_btn, button_row, parameter_row


def build_demo(models):
    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Base(),
        css=block_css,
    ) as demo:
        url_params = gr.JSON(visible=False)

        (
            state,
            model_selector,
            chatbot,
            textbox,
            send_btn,
            button_row,
            parameter_row,
        ) = build_single_model_ui(models)

        if args.model_list_mode not in ["once", "reload"]:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")
        demo.load(
            load_demo,
            [url_params],
            [
                state,
                model_selector,
                chatbot,
                textbox,
                send_btn,
                button_row,
                parameter_row,
            ],
            _js=get_window_url_params_js,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link.",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="The address of the API.",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue.",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time.",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
    )
    args = parser.parse_args()

    # Set global variables
    openai.api_key = "EMPTY"
    openai.api_base = f"{args.api_url}/v1"

    models = get_model_list()

    # Launch the demo
    demo = build_demo(models)
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=None,
    )
