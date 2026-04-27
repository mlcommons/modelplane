def test_prompt_replacement(sample_ctx):
    new_prompt = "New prompt"
    new_ctx = sample_ctx.with_prompt(new_prompt)
    assert new_ctx.prompt == new_prompt
    assert new_ctx.response == sample_ctx.response
    assert new_ctx.metadata == sample_ctx.metadata


def test_response_replacement(sample_ctx):
    new_response = "New response"
    new_ctx = sample_ctx.with_response(new_response)
    assert new_ctx.prompt == sample_ctx.prompt
    assert new_ctx.response == new_response
    assert new_ctx.metadata == sample_ctx.metadata


def test_metadata_replacement(sample_ctx):
    new_metadata = {"key": "value"}
    new_ctx = sample_ctx.with_metadata(new_metadata)
    assert new_ctx.prompt == sample_ctx.prompt
    assert new_ctx.response == sample_ctx.response
    assert new_ctx.metadata == new_metadata


def test_with_updates(sample_ctx):
    new_prompt = "Updated prompt"
    new_response = "Updated response"
    new_metadata = {"updated": True}
    new_ctx = sample_ctx.with_updates(
        new_prompt=new_prompt,
        new_response=new_response,
        new_metadata=new_metadata,
    )
    assert new_ctx.prompt == new_prompt
    assert new_ctx.response == new_response
    assert new_ctx.metadata == new_metadata
