from openai import OpenAI


class Client(OpenAI):
    """A client for interacting with the vLLM API. This simply wraps the
    OpenAI client without the need to provide an API key.
    """

    DUMMY_KEY = "dummy_key"

    def __init__(self, base_url: str):
        super().__init__(base_url=base_url, api_key=self.DUMMY_KEY)
