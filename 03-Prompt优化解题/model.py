import os
import time
import re
import pickle
import traceback
import tiktoken
from config import Config


class OpenAIModel:
    """封装 OpenAI API 调用和缓存机制的类

    用于调用 OpenAI API，处理响应，并缓存结果以提高效率。

    Attributes:
        cache_file (str): 缓存文件的路径
        cache_dict (dict): 内存中的缓存字典
    """

    def __init__(self, cache_file="openai_cache"):
        """初始化 OpenAI 模型对象，设置缓存文件路径并加载缓存

        Args:
            cache_file: 缓存文件的路径，默认为 "openai_cache"
        """
        self.client = Config.get_client()
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()

    def save_cache(self):
        """将当前缓存保存到文件中"""
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=True):
        """从文件加载缓存，支持重试机制

        Args:
            allow_retry: 是否允许重试加载缓存，默认为 True

        Returns:
            加载的缓存字典，如果文件不存在则返回空字典
        """
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print("Pickle Error: 5秒后重试...")
                    time.sleep(5)
        else:
            cache = {}
        return cache

    def set_cache_file(self, file_name):
        """设置缓存文件名并重新加载缓存

        Args:
            file_name: 新的缓存文件路径
        """
        self.cache_file = file_name
        self.cache_dict = self.load_cache()

    def get_response(self, content):
        """获取模型完成的文本，先检查缓存，若无则请求生成

        Args:
            content: 提供给模型的输入内容

        Returns:
            模型生成的回复文本，如果出错则返回 None
        """
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=Config.MODEL,
                    messages=[{"role": "user", "content": content}],
                    temperature=1.0,
                )
                completion = response.choices[0].message.content
                self.cache_dict[content] = completion
                return completion
            except Exception as e:
                print(e, "\n")
                time.sleep(1)
        return None

    def is_valid_key(self):
        """检查 API 密钥是否有效

        Returns:
            如果 API 密钥有效则返回 True，否则返回 False
        """
        for _ in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=Config.MODEL,
                    messages=[{"role": "user", "content": "hi there"}],
                    temperature=1.0,
                    max_tokens=1,
                )
                return True
            except Exception as e:
                traceback.print_exc()
                time.sleep(1)
        return False

    def prompt_token_num(self, prompt):
        """计算 prompt 的 token 数量

        Args:
            prompt: 要计算 token 数量的 prompt

        Returns:
            token 的数量
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(prompt)
            return len(tokens)
        except Exception as e:
            print(f"计算 token 数量时出错: {e}")
            return 0

    def two_stage_completion(self, question, content):
        """两阶段完成：首先获取推理，再获取最终答案

        Args:
            question: 原始问题
            content: 提供给模型的输入内容

        Returns:
            包含 prompt、推理过程和答案的字典
        """
        rationale = self.get_response(content)
        if not rationale:
            return {
                "prompt": content,
                "rationale": None,
                "answer": None,
            }

        ans = self.get_response(content=f"Q:{question}\nA:{rationale}\nThe answer to the original question is (a number only): ")
        return {
            "prompt": content,
            "rationale": rationale,
            "answer": ans,
        }


def clean_commas(text):
    """处理数字中的逗号（千位分隔符）

    Args:
        text: 包含数字的文本

    Returns:
        处理后的文本
    """
    def process_match(match):
        number = match.group(0)
        if '.' in number:
            return number
        else:
            number_list = number.split(",")
            new_string = number_list[0]
            for i in range(1, len(number_list)):
                if len(number_list[i]) == 3:
                    new_string += number_list[i]
                else:
                    new_string += f",{number_list[i]}"
            return new_string

    pattern = r'\d+(?:,\d+)*(?:\.\d+)?'
    return re.sub(pattern, process_match, text)


def find_and_match_floats(input_string, ground_truth):
    """检查输入中的数字是否与预期匹配

    Args:
        input_string: 包含数字的输入字符串
        ground_truth: 预期的正确数值

    Returns:
        如果找到匹配的数值则返回 True，否则返回 False
    """
    pattern = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
    found_numbers = pattern.findall(input_string)
    found_floats = [float(num) for num in found_numbers]
    return ground_truth in found_floats
