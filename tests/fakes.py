from types import SimpleNamespace


class _Completions:
    def __init__(self, outer):
        self._outer = outer

    def create(self, **kwargs):
        return self._outer._create(kwargs)


class _Chat:
    def __init__(self, outer):
        self.completions = _Completions(outer)


class FakeGroqClient:
    """Stand-in for groq.Groq; returns canned chat-completion contents."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = []
        self.chat = _Chat(self)

    def _create(self, kwargs):
        self.calls.append(kwargs)
        if self._contents:
            content = self._contents.pop(0)
        else:
            content = (
                '{"is_checkworthy": false, "urdu_claim": "", "english_claim": ""}'
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
