import os
import html
import torch
import gradio as gr
import transformers

from modules import script_callbacks, scripts, shared, devices, generation_parameters_copypaste, ui
from modules.ui_components import FormRow


class Model:
    name = None
    model = None
    tokenizer = None


available_models = []
current = Model()

base_dir = scripts.basedir()
models_dir = os.path.join(base_dir, "models")  # 在sd_webui_beautifulprompt目录下


def device():
    return devices.cpu if shared.opts.beautifulprompt_device == 'cpu' else devices.device


def list_available_models():
    available_models.clear()
    os.makedirs(models_dir, exist_ok=True)

    for dirname in os.listdir(models_dir):
        if os.path.isdir(os.path.join(models_dir, dirname)):
            available_models.append(dirname)

    for name in [x.strip() for x in shared.opts.beautifulprompt_names.split(",")]:
        if not name:
            continue

        available_models.append(name)


def get_model_path(name):
    dirname = os.path.join(models_dir, name)
    if not os.path.isdir(dirname):
        return name

    return dirname


def model_selection_changed(model_name):
    if model_name == "None":
        current.tokenizer = None
        current.model = None
        current.name = None

        devices.torch_gc()


def send_prompts(text):
    params = generation_parameters_copypaste.parse_generation_parameters(text)
    negative_prompt = params.get("Negative prompt", "")
    return params.get("Prompt", ""), negative_prompt or gr.update()


def find_prompts(fields):
    field_prompt = [x for x in fields if x[1] == "Prompt"][0]
    field_negative_prompt = [x for x in fields if x[1] == "Negative prompt"][0]
    return [field_prompt[0], field_negative_prompt[0]]


def generate_batch(input_ids, min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty,
                   sampling_mode, top_k, top_p):
    top_p = float(top_p) if sampling_mode == 'Top P' else None
    top_k = int(top_k) if sampling_mode == 'Top K' else None

    outputs = current.model.generate(
        input_ids,
        do_sample=True,  # 使用采样
        temperature=max(float(temperature), 1e-6),
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        top_p=top_p,
        top_k=top_k,
        num_beams=int(num_beams),
        min_length=min_length,
        max_length=max_length,
        pad_token_id=current.tokenizer.pad_token_id or current.tokenizer.eos_token_id
    )
    texts = current.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts


def generate(id_task, model_name, batch_count, batch_size, text, *args):
    shared.state.textinfo = "Loading model..."
    shared.state.job_count = batch_count

    if current.name != model_name:
        current.tokenizer = None
        current.model = None
        current.name = None

        if model_name != "None":
            path = get_model_path(model_name)
            current.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
            current.model = transformers.AutoModelForCausalLM.from_pretrained(path)
            current.name = model_name

    assert current.model, 'No model available'
    assert current.tokenizer, 'No tokenizer available'

    current.model.to(device())
    shared.state.textinfo = ""

    input = f'Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {text}\nOutput:'
    input_ids = current.tokenizer(input, return_tensors="pt").input_ids
    if input_ids.shape[1] == 0:
        input_ids = torch.asarray([[current.tokenizer.bos_token_id]], dtype=torch.long)
    input_ids = input_ids.to(device())
    input_ids = input_ids.repeat((batch_size, 1))

    markup = '<table><tbody>'
    index = 0
    for i in range(batch_count):
        texts = generate_batch(input_ids, *args)
        shared.state.nextjob()
        for generated_text in texts:
            index += 1
            markup += f"""
        <tr>
        <td>
        <div class="prompt gr-box gr-text-input">
            <p id='beautifulprompt_res_{index}'>{html.escape(generated_text)}</p>
        </div>
        </td>
        <td class="sendto">
            <a class='gr-button gr-button-lg gr-button-secondary' onclick="beautifulprompt_send_to_txt2img(gradioApp().getElementById('beautifulprompt_res_{index}').textContent)">to txt2img</a>
            <a class='gr-button gr-button-lg gr-button-secondary' onclick="beautifulprompt_send_to_img2img(gradioApp().getElementById('beautifulprompt_res_{index}').textContent)">to img2img</a>
        </td>
        </tr>
        """

    markup += '</tbody></table>'

    return markup, ''


def on_ui_tabs():
    list_available_models()

    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row():
            with gr.Column(scale=80):
                prompt = gr.Textbox(label="Prompt", elem_id="beautifulprompt_prompt", show_label=False, lines=2,
                                    placeholder="Beginning of the prompt (press Ctrl+Enter or Alt+Enter to generate)"). \
                    style(container=False)
            with gr.Column(scale=10):
                submit = gr.Button('Generate', elem_id="beautifulprompt_generate", variant='primary')

        with gr.Row(elem_id="beautifulprompt_main"):
            with gr.Column(variant="compact"):
                selected_text = gr.TextArea(elem_id='beautifulprompt_selected_text', visible=False)
                send_to_txt2img = gr.Button(elem_id='beautifulprompt_send_to_txt2img', visible=False)
                send_to_img2img = gr.Button(elem_id='beautifulprompt_send_to_img2img', visible=False)

                with FormRow():
                    model_selection = gr.Dropdown(label="Model", elem_id="beautifulprompt_model",
                                                  value=available_models[0],
                                                  choices=["None"] + available_models)

                with FormRow():
                    sampling_mode = gr.Radio(label="Sampling mode", elem_id="beautifulprompt_sampling_mode",
                                             value="Top K",
                                             # topk先找出K个最有可能的单词，然后在这K个单词中计算概率分布；topp从使得累计概率超过p的最小候选集中选择单词，然后算这些单词的概率分布，这些候选词的概率大小会随着下一个单词的概率分布动态增加和减小
                                             choices=["Top K", "Top P"])
                    top_k = gr.Slider(label="Top K", elem_id="beautifulprompt_top_k", value=50, minimum=1, maximum=50,
                                      step=1)  # 在topk过滤中保留最高概率token的数量
                    top_p = gr.Slider(label="Top P", elem_id="beautifulprompt_top_p", value=0.95, minimum=0, maximum=1,
                                      step=0.001)  # 如果设置小于1的浮点数，只有最可能的token集合，其概率之和达到或超过top_p，才会在生成过程中保留

                with FormRow():
                    num_beams = gr.Slider(label="Number of beams", elem_id="beautifulprompt_num_beams", value=1,
                                          minimum=1,
                                          maximum=8, step=1)  # 束搜索的束数，生成token序列时，跟踪每个token序列的多个副本，完成后选择具有最高可能性的一个
                    temperature = gr.Slider(label="Temperature", elem_id="beautifulprompt_temperature", value=1,
                                            minimum=0,
                                            maximum=4,
                                            step=0.01)  # 调整下一个token的概率，可以控制生成的文本的随机性，较大的temperature值会导致生成的文本更加随机，较小的tempeature则会生成更加确定性的文本
                    repetition_penalty = gr.Slider(label="Repetition penalty",
                                                   elem_id="beautifulprompt_repetition_penalty",
                                                   value=1.2, minimum=1, maximum=4, step=0.01)  # 重复惩罚系数，值越大，出现重复的可能越小

                with FormRow():
                    length_penalty = gr.Slider(label="Length preference", elem_id="beautifulprompt_length_preference",
                                               value=1, minimum=-10, maximum=10,
                                               step=0.1)  # 尽在number of beams>0有效，负值倾向于生成较短的token序列，正值倾向于生成较长的token序列
                    min_length = gr.Slider(label="Min length", elem_id="beautifulprompt_min_length", value=20,
                                           minimum=1,
                                           maximum=400, step=1)  # 生成的token的最小长度
                    max_length = gr.Slider(label="Max length", elem_id="beautifulprompt_max_length", value=384,
                                           minimum=1,
                                           maximum=400, step=1)  # 生成token最大长度

                with FormRow():
                    batch_count = gr.Slider(label="Batch count", elem_id="beautifulprompt_batch_count", value=1,
                                            minimum=1,
                                            maximum=100, step=1)
                    batch_size = gr.Slider(label="Batch size", elem_id="beautifulprompt_batch_size", value=10,
                                           minimum=1,
                                           maximum=100, step=1)

            with gr.Column():
                with gr.Group(elem_id="beautifulprompt_results_column"):
                    res = gr.HTML()
                    res_info = gr.HTML()

        submit.click(
            fn=ui.wrap_gradio_call(generate, extra_outputs=['']),
            _js="submit_beautifulprompt",
            inputs=[model_selection, model_selection, batch_count, batch_size, prompt, min_length, max_length,
                    num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p, ],
            outputs=[res, res_info])

        model_selection.change(
            fn=model_selection_changed,
            inputs=[model_selection],
            outputs=[]
        )

        send_to_txt2img.click(
            fn=send_prompts,
            inputs=[selected_text],
            outputs=find_prompts(ui.txt2img_paste_fields)
        )

        send_to_img2img.click(
            fn=send_prompts,
            inputs=[selected_text],
            outputs=find_prompts(ui.img2img_paste_fields)
        )

    return [(tab, "Beautifulprompt", "beautifulprompt")]


def on_ui_settings():
    section = ("beautifulprompt", "Beautifulprompt")

    shared.opts.add_option("beautifulprompt_names", shared.OptionInfo(
        "alibaba-pai/pai-bloom-1b1-text2prompt-sd,alibaba-pai/pai-bloom-1b1-text2prompt-sd-v2",
        "Hugginface model names for beautifulprompt, separated by comma", section=section))
    shared.opts.add_option("beautifulprompt_device",
                           shared.OptionInfo("gpu", "Device to use for text generation", gr.Radio,
                                             {"choices": ["gpu", "cpu"]}, section=section))


def on_unload():
    current.model = None
    current.tokenizer = None


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_script_unloaded(on_unload)
