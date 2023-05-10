import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
import argparse
import cv2
from PIL import Image
import numpy as np

from detgpt.common.config import Config
from detgpt.common.dist_utils import get_rank
from detgpt.common.registry import registry
from detgpt.conversation.conversation import Chat, Conversation, SeparatorStyle  # , CONV_VISION
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
import GroundingDINO.groundingdino.datasets.transforms as T
from huggingface_hub import hf_hub_download
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--system-path", type=str, default=None, help="path to system prompt file.")
    parser.add_argument("--dino-version", type=str, default="swinb", help="path to system prompt file.")
    parser.add_argument("--gpu-id", type=int, nargs='+', default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    parser.add_argument("--disable_detector", action="store_true", help="using detector mode")
    parser.add_argument("--enable_system", action="store_true", help="editable system message mode")
    args = parser.parse_args()
    return args


args = parse_args()
print('Initializing Chat')
cfg = Config(args)

cuda_llm = f"cuda:1"
cuda_detector = f"cuda:0"
ckpt_repo_id = "ShilongLiu/GroundingDINO"
if args.dino_version == "swinb":
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
else:
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    ckpt_filenmae = "groundingdino_swint_ogc.pth"

if args.system_path:
    with open(args.system_path, 'r') as file:
        system_message = file.read()
        print(f"system message: \n {system_message}")
else:
    system_message = "You must strictly answer the question step by step:\n" \
                     "Step-1. describe the given image in detail.\n" \
                     "Step-2. find all the objects related to user input, and concisely explain why these objects meet the requirement.\n" \
                     "Step-3. list out all related objects existing in the image strictly as follows: <Therefore the answer is: [object_names]>.\n" \
                     "Complete all 3 steps as detailed as possible.\n" \
                     "You must finish the answer with complete sentences."

CONV_VISION = Conversation(
    system=system_message,
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
    model_args = SLConfig.fromfile(model_config_path)
    model = build_model(model_args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None)  # 3, h, w
    return init_image, image


def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None)  # 3, h, w
    return image


def print_format(message):
    print(f"*" * 20)
    print(f"\n{message}\n")


def list_to_str(cat_list, sep=". "):
    result = ""
    for cat in cat_list:
        result += cat
        result += sep
    return result[:-1]


def run_grounding(input_image, llm_message_original, box_threshold, text_threshold):
    init_image = input_image.convert("RGB")
    original_size = init_image.size
    _, image_tensor = image_transform_grounding(init_image)
    image_pil: Image = image_transform_grounding_for_vis(init_image)
    response_message = llm_message_original[-1]
    print_format(f"From run grounding, oringinal response message {response_message}")
    pattern1 = r"(?i)therefore,?\s+the\s+answer\s+is:?[\s\[\],]*(\w+[\s,]*)+([ ,]\w+[\s,]*)*"
    pattern2 = r"(?i)therefore,?\s+the\s+target\s+objects?\s+are:?[\s\[\],]*(\w+[\s,]*)+([ ,]\w+[\s,]*)*"
    # Use re.search() to find the match
    match1 = re.search(pattern1, response_message)
    match2 = re.search(pattern2, response_message)
    # Extract the matched substring
    if match1:
        substr = match1.group(0)
        # Remove the unnecessary characters
        substr = re.sub(r"(?i)therefore,?\s+the\s+answer\s+is:?[\s\[\],]*", "", substr)
        categories = re.sub(r"[\[\]]", "", substr)
        cat_list = [c.strip() for c in categories.split(',')]
        # remove duplicate
        cat_list = list(set(cat_list))
        categories = list_to_str(cat_list)
        # Print the result
        print_format(f"Detected categores: {categories}")
    elif match2:
        substr = match2.group(0)
        # Remove the unnecessary characters
        substr = re.sub(r"(?i)therefore,?\s+the\s+target\s+objects?\s+are:?[\s\[\],]*", "", substr)
        categories = re.sub(r"[\[\]]", "", substr)
        cat_list = [c.strip() for c in categories.split(',')]
        # remove duplicate
        cat_list = list(set(cat_list))
        categories = list_to_str(cat_list)
        # Print the result
        print_format(f"Detected categores: {categories}")
    else:
        print_format("No match found.")
        categories = ""
    # run grounidng
    # boxes, logits, phrases = predict(detector, image_tensor, categories, box_threshold, text_threshold, device=f"cuda:{args.gpu_id[0]}")
    boxes, logits, phrases = predict(detector, image_tensor, categories, box_threshold, text_threshold,
                                     device=cuda_detector)
    print_format(f"Detector predicted phrases {phrases}")
    annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases)
    image_with_box = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    return image_with_box, f"{categories}"


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

### detector
detector = load_model_hf(config_file, ckpt_repo_id, ckpt_filenmae)
# detector = detector.to(f"cuda:{args.gpu_id[0]}")
detector = detector.to(cuda_detector)
print_format(f"loaded detector")
### language model
model_config = cfg.model_cfg
model_config.device_8bit = cuda_llm
model_cls = registry.get_model_class(model_config.arch)
model_llm = model_cls.from_config(model_config).to(cuda_llm)

vis_processor_cfg = cfg.datasets_cfg.coco_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model_llm, vis_processor, device=cuda_llm)
print_format('Initialization Finished')


# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list, llm_message_original):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    if llm_message_original is not None:
        llm_message_original = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first',
                                                                    interactive=False), gr.update(
        value="1. Upload Image", interactive=True), chat_state, img_list, llm_message_original, gr.update(
        value='Detected objects will be shown here.', interactive=False)


def upload_img(gr_img, text_input, chat_state, system_prompt=None):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    if system_prompt is not None:
        chat_state.system = system_prompt
        print(f"system prompt: {chat_state.system}")
    else:
        chat_state.system = system_message
        print(f"system prompt: {chat_state.system}")
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(
        value="Start Chatting", interactive=False), chat_state, img_list


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    user_prompt = "\nAnswer me with several sentences. End the answer by listing out target objects to my question strictly as follows: <Therefore the answer is: [object_names]>."
    print(user_message)
    chat.ask(user_message + user_prompt, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature, length_penalty, do_sample,
                  llm_message_original):
    llm_message_old = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=num_beams,
                                  temperature=temperature,
                                  length_penalty=length_penalty,
                                  do_sample=do_sample,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
    pattern1 = r"(?:Therefore, the answer is|Therefore the answer is).*"
    pattern2 = r"(?:Therefore, the target objects are|Therefore the target objects are).*"
    llm_message = re.sub(pattern1, "", llm_message_old)
    llm_message = re.sub(pattern2, "", llm_message)
    chatbot[-1][1] = llm_message
    llm_message_original.append(llm_message_old)
    return chatbot, chat_state, img_list, llm_message_original


title = """<h1 align="center">DetGPT</h1>"""
description = """<h3>This is the demo of DetGPT. Tell me your goal and I'll find stuff to help you!</h3>"""
restart = """<h5>Press restart before trying a new image!</h5>"""
feature = """<h4>Why DetGPT is appealing?</h4>
<ol>
  <li>DetGPT locates target objects, not just describing images.</li>
  <li>DetGPT understands complex instructions (e.g., it can locate food that relieves high blood pressure)</li>
  <li>DetGPT accurately localizes target objects via LLM reasoning. (e.g., it identifies bananas as a potassium-rich food to alleviate high blood pressure)</li>
  <li>DetGPT provides answers beyond human common sense. (e.g., bananas being rich in potassium is rarely known)</li>
</ol>

Out of respect for privacy and ethical considerations, our model refrains from disclosing specific names of individuals and locations.
"""
# article = """<p><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>
# """

# TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown(feature)
    # gr.Markdown(article)

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil")
            with gr.Accordion("Parameters", open=False):
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    interactive=True,
                    label="beam search numbers",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
                length_penalty = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.5,
                    step=0.1,
                    interactive=True,
                    label="Length penalty",
                )
                do_sample = gr.Radio([True, False], value=False, label="do_sample", interactive=True)

            upload_button = gr.Button(value="1. Upload Image", interactive=True, variant="primary")
            text_input = gr.Textbox(label='User', placeholder='Input Text Here (Please upload your image first)',
                                    interactive=False)
            text_button = gr.Button(value="2. Submit Question", interactive=True, variant="primary")

            clear = gr.Button("Restart")
            gr.Markdown(restart)

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            llm_message_original = gr.State([])
            chatbot = gr.Chatbot(label='Assistant')
            detected_objects = gr.Textbox(label='Detected Objects', value="Detected objects will be shown here.",
                                          interactive=False)
            if args.enable_system:
                system_prompt = gr.Textbox(label='System', placeholder=system_message, interactive=True,
                                           value=system_message)

            cur_dir = os.path.dirname(os.path.abspath(__file__))
            gr.Examples(examples=[
                [f"{cur_dir}/examples/big_kitchen.jpg", "I want to have a cold beverage." ],
                [f"{cur_dir}/examples/banana.jpg", "Find food that can relieve high blood pressure."],
                [f"{cur_dir}/examples/foods.jpg", "Find the foods high in protein."],
                [f"{cur_dir}/examples/wine.jpg", "Find items appropriate for a romantic dinner."],
                [f"{cur_dir}/examples/smoking.jpg", "Find an item inappropriate for children."],
                [f"{cur_dir}/examples/bird.jpeg", "find what’s interesting about the image."],
            ], inputs=[image, text_input])
        with gr.Column():
            gallery = gr.outputs.Image(
                type="pil",
                # label="grounding results"
            ).style(full_width=True, full_height=True)
            with gr.Accordion("Advanced options", open=False):
                box_threshold = gr.Slider(
                    label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                )
                text_threshold = gr.Slider(
                    label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                )
    if args.enable_system:
        upload_button.click(upload_img, [image, text_input, chat_state, system_prompt],
                            [image, text_input, upload_button, chat_state, img_list])
    else:
        upload_button.click(upload_img, [image, text_input, chat_state],
                            [image, text_input, upload_button, chat_state, img_list])
    if not args.disable_detector:
        text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
            gradio_answer,
            [chatbot, chat_state, img_list, num_beams, temperature, length_penalty, do_sample, llm_message_original],
            [chatbot, chat_state, img_list, llm_message_original]
        ).then(fn=run_grounding, inputs=[image, llm_message_original, box_threshold, text_threshold],
               outputs=[gallery, detected_objects])
        text_button.click(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
            gradio_answer,
            [chatbot, chat_state, img_list, num_beams, temperature, length_penalty, do_sample, llm_message_original],
            [chatbot, chat_state, img_list, llm_message_original]
        ).then(fn=run_grounding, inputs=[image, llm_message_original, box_threshold, text_threshold],
               outputs=[gallery, detected_objects])
    else:
        text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
            gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature, length_penalty, do_sample],
            [chatbot, chat_state, img_list]
            )
        text_button.click(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
            gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature, length_penalty, do_sample],
            [chatbot, chat_state, img_list]
        )
    clear.click(gradio_reset, [chat_state, img_list, llm_message_original],
                [chatbot, image, text_input, upload_button, chat_state, img_list, llm_message_original,
                 detected_objects], queue=False)

demo.launch(share=True, enable_queue=True)
