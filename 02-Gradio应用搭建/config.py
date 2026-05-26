from openai import OpenAI


class Config:
    API_KEY = "3f2c732f800d4459b52c0e79998069d2.071x7z2tbVlFWJ4F"
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    MODEL = "glm-4.5-air"

    @classmethod
    def get_client(cls) -> OpenAI:
        return OpenAI(
            api_key=cls.API_KEY,
            base_url=cls.BASE_URL,
        )
