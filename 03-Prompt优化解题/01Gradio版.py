import json
import gradio as gr
import jinja2
from model import OpenAIModel, clean_commas, find_and_match_floats
from questions import questions, answers

my_model = OpenAIModel()

my_magic_prompt = "任务：\n解决以下数学问题。\n\n问题：{{question}}\n\n答案："
my_magic_prompt = my_magic_prompt.strip('\n')


def _add_msg(chatbot, role, content):
    """向 chatbot 添加一条消息"""
    chatbot.append({"role": role, "content": content})


def reset_prompt(chatbot):
    """Reset 按钮点击处理：重置 prompt

    Args:
        chatbot: 聊天记录

    Returns:
        更新后的聊天记录和清空的提示词文本
    """
    gr.Info("已清除提示词")
    _add_msg(chatbot, "user", "清除提示词")
    _add_msg(chatbot, "assistant", "提示词已成功重置")
    return chatbot, "", "0"


def assign_prompt(chatbot, prompt, template, example_number):
    """Assign 按钮点击处理：分配有效 prompt 并设置 template

    Args:
        chatbot: 聊天记录
        prompt: 用户输入的提示词
        template: 当前的模板对象
        example_number: 选择的示例编号

    Returns:
        更新后的聊天记录、提示词文本、模板对象和选择的示例编号
    """
    gr.Info("正在分配提示词")
    example_number = int(example_number)
    token_num = my_model.prompt_token_num(prompt)

    if token_num > 1024:
        template = None
        gr.Warning("无效的提示词（太长，超过1024个token）")
        _add_msg(chatbot, "assistant", "提示词太长（超过1024个token）。较短的提示词可以更快且更稳定地评估！")
    elif example_number < 1 or example_number > len(questions):
        template = None
        prompt_ex = f"错误：请选择一个1到{len(questions)}之间的数字"
        gr.Warning(prompt_ex)
        _add_msg(chatbot, "assistant", prompt_ex)
    elif "{{question}}" not in prompt:
        template = None
        prompt_ex = "你需要在提示词中包含占位符{{question}}。"
        gr.Warning(prompt_ex)
        _add_msg(chatbot, "assistant", prompt_ex)
    else:
        environment = jinja2.Environment()
        template = environment.from_string(prompt)
        prompt_ex = f"""{template.render(question=questions[example_number - 1])}"""
        _add_msg(chatbot, "user", "分配提示词")
        _add_msg(chatbot, "assistant", f"提示词已成功分配\n\n自定义提示词示例：\n\n{prompt_ex}")

    return chatbot, prompt, template, example_number, str(token_num)


def assess_prompt(chatbot, template, test_num):
    """Test 按钮点击处理：评估自定义 prompt

    Args:
        chatbot: 聊天记录
        template: 当前的模板对象
        test_num: 要测试的问题数量

    Returns:
        更新后的聊天记录、结果列表、结果统计和 UI 组件
    """
    if template is None:
        _add_msg(chatbot, "assistant", "评估失败，因为提示词模板为空（即无效的提示词）")
        gr.Warning("提示词未设置")
        return chatbot, [], "提示词未设置", gr.Slider(label="Result Number", value=0, minimum=0, maximum=1, step=1), gr.Textbox(label="Result", value="", interactive=False)

    gr.Info("正在评估提示词")
    ans_template = "提示词和问题：\n\n{{question}}\n\n--------------------\n\n解题过程：\n\n{{rationale}}\n\n--------------------\n\n最终答案\n\n{{answer}}"
    res_list = []
    total_count = test_num
    environment = jinja2.Environment()
    ans_template = environment.from_string(ans_template)
    trial_num = 3
    trials = [[] for _ in range(trial_num)]
    res_stats_str = ""

    for i in range(trial_num):
        gr.Info(f"开始第{i+1}次测试")
        accurate_count = 0
        for idx, example in enumerate(questions[:test_num]):
            test_res = ""
            result = my_model.two_stage_completion(example, template.render(question=example))

            if not result["answer"]:
                trials[i].append(0)
                test_res += f"第{i+1}次测试\n\n跳过问题 {idx + 1}。"
                test_res += "\n" + "<"*6 + "="*30 + ">"*6 + "\n\n"
                res_list.append(f"第{i+1}次测试\n\n跳过问题 {idx + 1}。")
                continue

            cleaned_result = clean_commas(result["answer"])
            if find_and_match_floats(cleaned_result, answers[idx]):
                accurate_count += 1
                trials[i].append(1)
            else:
                trials[i].append(0)

            my_model.save_cache()
            test_res += f"第{i + 1}次测试\n\n"
            test_res += f"问题 {idx + 1}:\n" + '-'*20
            test_res += f'''\n\n{ans_template.render(question=result['prompt'], rationale=result['rationale'], answer=result['answer'])}\n'''
            test_res += "\n" + "<"*6 + "="*30 + ">"*6 + "\n\n"
            res_list.append(test_res)

        res_stats_str += f"第{i + 1}次测试，正确数：{accurate_count}，总数：{total_count}，准确率：{accurate_count / total_count * 100}%\n"
        my_model.save_cache()

    voting_acc = 0
    for i in range(total_count):
        count = 0
        for j in range(trial_num):
            if trials[j][i] == 1:
                count += 1
        if count >= 2:
            voting_acc += 1

    res_stats_str += f"最终准确率：{voting_acc / total_count * 100}%"
    _add_msg(chatbot, "user", "测试")
    _add_msg(chatbot, "assistant", "测试完成。结果可以在'结果'和'结果统计'中找到。")
    _add_msg(chatbot, "assistant", f"测试结果\n\n{''.join(res_list)}")
    _add_msg(chatbot, "assistant", f"结果统计\n\n{res_stats_str}")

    max_res = max(len(res_list), 1)
    first_res = res_list[0] if res_list else ""
    return chatbot, res_list, res_stats_str, gr.Slider(label="Result Number", value=1, minimum=1, maximum=max_res, step=1, visible=False), gr.Textbox(label="Result", value=first_res, interactive=False)


def save_prompt(chatbot, prompt):
    """Save 按钮点击处理：保存提示词

    Args:
        chatbot: 聊天记录
        prompt: 用户输入的提示词

    Returns:
        更新后的聊天记录
    """
    gr.Info("正在保存提示词")
    prompt_dict = {"prompt": prompt}
    with open("files/prompt.json", "w") as f:
        json.dump(prompt_dict, f)
    _add_msg(chatbot, "user", "保存提示词")
    _add_msg(chatbot, "assistant", "提示词已保存为 prompt.json")
    return chatbot


def update_result(results, result_num, test_num):
    """res_num 变化时更新结果显示"""
    if not results or result_num < 1 or result_num > len(results):
        result_text = ""
    else:
        result_text = results[result_num - 1]

    trial_val = max(1, (int)((result_num - 1) / max(test_num, 1)) + 1) if result_num > 0 else 1
    ques_val = max(1, (result_num - 1) % max(test_num, 1) + 1) if result_num > 0 else 1

    return (
        gr.Textbox(label="Result", value=result_text, interactive=False),
        trial_val,
        gr.Slider(label="Question Number", minimum=1, maximum=max(test_num, 2), value=ques_val, step=1),
    )


with gr.Blocks() as demo:
    template = gr.State(None)
    res_list = gr.State(list())

    with gr.Tab(label="Console"):
        with gr.Group():
            example_num_box = gr.Dropdown(
                label="Demo Example (Please choose one example for demo)",
                value=1,
                info=questions[0],
                choices=[i+1 for i in range(len(questions))],
                filterable=False,
            )
            prompt_textbox = gr.Textbox(
                label="Custom Prompt",
                placeholder=f"在这里输入你的自定义提示词。例如：\n\n{my_magic_prompt}",
                value="",
                info="请确保包含`{{question}}`标签。",
            )
            with gr.Row():
                set_button = gr.Button(value="Set Prompt")
                reset_button = gr.Button(value="Clear Prompt")
            prompt_token_num = gr.Textbox(
                label="Number of prompt tokens",
                value="0",
                interactive=False,
                info="自定义提示词的 Token 数量。",
            )
        with gr.Group():
            test_num = gr.Slider(
                label="Number of examples used for evaluation",
                minimum=1,
                maximum=len(questions),
                step=1,
                value=1,
            )
            assess_button = gr.Button(value="Evaluate")
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        trial_no = gr.Slider(label="Trial ID", value=1, minimum=1, maximum=3, step=1)
                        ques_no = gr.Slider(label="Question ID", value=1, minimum=1, maximum=2, step=1)
                    res_num = gr.Slider(label="Result Number", value=0, minimum=0, maximum=1, step=1, visible=False)
                    res = gr.Textbox(label="Result", value="", placeholder="暂无结果", interactive=False)
                with gr.Column():
                    res_stats = gr.Textbox(label="Result Stats", interactive=False)
            save_button = gr.Button(value="Save Custom Prompt")
    with gr.Tab(label="Log"):
        chatbot = gr.Chatbot(label="Log")

    example_num_box.change(
        lambda example_number: gr.Dropdown(
            value=example_number,
            info=questions[example_number - 1],
            choices=[i+1 for i in range(len(questions))],
        ),
        inputs=[example_num_box],
        outputs=[example_num_box],
    )

    res_num.change(
        update_result,
        inputs=[res_list, res_num, test_num],
        outputs=[res, trial_no, ques_no],
    )

    trial_ques_no_input = lambda t_val, q_val, test_num: (t_val - 1) * test_num + q_val
    trial_no.change(trial_ques_no_input, inputs=[trial_no, ques_no, test_num], outputs=[res_num])
    ques_no.change(trial_ques_no_input, inputs=[trial_no, ques_no, test_num], outputs=[res_num])
    set_button.click(assign_prompt, inputs=[chatbot, prompt_textbox, template, example_num_box], outputs=[chatbot, prompt_textbox, template, example_num_box, prompt_token_num])
    reset_button.click(reset_prompt, inputs=[chatbot], outputs=[chatbot, prompt_textbox, prompt_token_num])
    assess_button.click(assess_prompt, inputs=[chatbot, template, test_num], outputs=[chatbot, res_list, res_stats, res_num, res])
    save_button.click(save_prompt, inputs=[chatbot, prompt_textbox], outputs=[chatbot])

demo.queue().launch(debug=True)
